// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// f620=softmax, f621=scaled_dot_product_attention, f622=mse_loss,
// f623=scaled_dot_product_attention_causal, f624=apply_causal_mask, f625=rope.

use crate::device::{t500, t501};
use anyhow::{Result, ensure};

// --- Softmax (two-pass) ---

/// t514 = SoftmaxParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t514 {
    rows: u32,
    cols: u32,
    _pad: [u32; 2],
}

// Fused subgroup softmax: single dispatch, one workgroup per row.
// 256 threads cooperatively reduce across 4 subgroups via LDS.
// subgroupMax / subgroupAdd replace multi-round tree reductions.
// Requires wgpu::Features::SUBGROUP (s509 == true); selected at runtime.
const SHADER_SOFTMAX_FUSED: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:   array<f32>;

var<workgroup> wg_scratch: array<f32, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)           wg:      vec3<u32>,
    @builtin(local_invocation_index) t:       u32,
    @builtin(subgroup_id)            sg_id:   u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {
    let row = wg.x + wg.y * 65535u;
    if row >= p.rows { return; }
    let base = row * p.cols;

    var local_max: f32 = -1.0e30;
    for (var c: u32 = t; c < p.cols; c += 256u) {
        local_max = max(local_max, input[base + c]);
    }
    let sg_max = subgroupMax(local_max);
    if sg_lane == 0u { wg_scratch[sg_id] = sg_max; }
    workgroupBarrier();
    var global_max: f32 = select(-1.0e30, wg_scratch[t], t < 4u);
    global_max = subgroupMax(global_max);
    if t == 0u { wg_scratch[4] = global_max; }
    workgroupBarrier();
    global_max = wg_scratch[4];

    var local_sum: f32 = 0.0;
    for (var c: u32 = t; c < p.cols; c += 256u) {
        local_sum += exp(input[base + c] - global_max);
    }
    let sg_sum = subgroupAdd(local_sum);
    if sg_lane == 0u { wg_scratch[sg_id] = sg_sum; }
    workgroupBarrier();
    var global_sum: f32 = select(0.0, wg_scratch[t], t < 4u);
    global_sum = subgroupAdd(global_sum);
    if t == 0u { wg_scratch[5] = global_sum; }
    workgroupBarrier();
    global_sum = wg_scratch[5];

    let inv_sum = 1.0 / global_sum;
    for (var c: u32 = t; c < p.cols; c += 256u) {
        out[base + c] = exp(input[base + c] - global_max) * inv_sum;
    }
}
";

// Pass 1: one thread per row. Compute max and sum(exp(x - max)).
const SHADER_SOFTMAX_STATS: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= p.rows { return; }

    let base = row * p.cols;
    var mx: f32 = input[base];
    for (var i: u32 = 1u; i < p.cols; i++) {
        mx = max(mx, input[base + i]);
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < p.cols; i++) {
        sum += exp(input[base + i] - mx);
    }
    stats[row * 2u] = mx;
    stats[row * 2u + 1u] = sum;
}
";

// Pass 2: one thread per element. exp(x - max) / sum.
const SHADER_SOFTMAX_APPLY: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.rows * p.cols { return; }

    let row = idx / p.cols;
    let mx = stats[row * 2u];
    let sum = stats[row * 2u + 1u];
    out[idx] = exp(input[idx] - mx) / sum;
}
";

// --- Head split / merge / repeat_kv (Sprint 7 step 7) ---
//
// Attention projections output [seq, n_heads*head_dim]; SDPA expects [n_heads, seq, head_dim].
// split_heads (f627) and merge_heads (f628) transpose between the two layouts.
// repeat_kv (f629) expands [n_kv_heads, kv_seq, head_dim] → [n_heads, kv_seq, head_dim]
// for Grouped Query Attention (GQA) where n_kv_heads < n_heads.

/// t542 = HeadParams (private).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t542 {
    n_heads: u32,
    seq_len: u32,
    head_dim: u32,
    _pad: u32,
}

/// t543 = RepeatKvParams (private).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t543 {
    n_kv_heads: u32,
    n_rep: u32, // = n_heads / n_kv_heads
    kv_seq: u32,
    head_dim: u32,
}

// f627: out[bh*(seq*hd) + s*hd + d] = src[s*(n_heads*hd) + bh*hd + d]
const SHADER_SPLIT_HEADS: &str = "
struct P { n_heads: u32, seq_len: u32, head_dim: u32, _p0: u32, }
@group(0) @binding(0) var<uniform>             p:   P;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.n_heads * p.seq_len * p.head_dim;
    if idx >= total { return; }
    let bh = idx / (p.seq_len * p.head_dim);
    let s  = (idx / p.head_dim) % p.seq_len;
    let d  = idx % p.head_dim;
    let src_idx = s * (p.n_heads * p.head_dim) + bh * p.head_dim + d;
    dst[idx] = src[src_idx];
}
";

// f628: dst[s*(n_heads*hd) + bh*hd + d] = src[bh*(seq*hd) + s*hd + d]
const SHADER_MERGE_HEADS: &str = "
struct P { n_heads: u32, seq_len: u32, head_dim: u32, _p0: u32, }
@group(0) @binding(0) var<uniform>             p:   P;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.n_heads * p.seq_len * p.head_dim;
    if idx >= total { return; }
    let bh = idx / (p.seq_len * p.head_dim);
    let s  = (idx / p.head_dim) % p.seq_len;
    let d  = idx % p.head_dim;
    let dst_idx = s * (p.n_heads * p.head_dim) + bh * p.head_dim + d;
    dst[dst_idx] = src[idx];
}
";

// f629: out[out_head*(kv_seq*hd) + s*hd + d] = src[(out_head/n_rep)*(kv_seq*hd) + s*hd + d]
const SHADER_REPEAT_KV: &str = "
struct P { n_kv_heads: u32, n_rep: u32, kv_seq: u32, head_dim: u32, }
@group(0) @binding(0) var<uniform>             p:   P;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.n_kv_heads * p.n_rep * p.kv_seq * p.head_dim;
    if idx >= total { return; }
    let out_head = idx / (p.kv_seq * p.head_dim);
    let kv_head  = out_head / p.n_rep;
    let tail     = idx % (p.kv_seq * p.head_dim);
    dst[idx] = src[kv_head * p.kv_seq * p.head_dim + tail];
}
";

// --- MSE Loss ---

/// t513 = ReduceParams. Single-uint uniform for reductions like mse_loss.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t513 {
    n: u32,
    _pad: [u32; 3],
}

// --- Causal mask (Sprint 7 step 2) ---
//
// In-place mask of upper-triangular positions to -INF (effectively zero after softmax).
// Supports asymmetric q_seq_len vs kv_seq_len for KV-cache decoding: position i in Q
// corresponds to absolute position i + (kv_seq_len - q_seq_len) in KV, and can attend
// to j ≤ that. Prefill (q==kv): standard triangular. Decode (q=1, kv=N): no mask.

/// t535 = CausalMaskParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t535 {
    batch_heads: u32,
    q_seq_len: u32,
    kv_seq_len: u32,
    _pad: u32,
}

const SHADER_CAUSAL_MASK: &str = "
struct P { batch_heads: u32, q_seq_len: u32, kv_seq_len: u32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch_heads * p.q_seq_len * p.kv_seq_len;
    if idx >= total { return; }

    let kv_per_q = p.q_seq_len * p.kv_seq_len;
    let rem = idx % kv_per_q;
    let i = rem / p.kv_seq_len;
    let j = rem % p.kv_seq_len;

    // offset = kv_seq_len - q_seq_len (always >= 0; checked on Rust side).
    let offset = p.kv_seq_len - p.q_seq_len;
    if j > i + offset {
        scores[idx] = -1.0e30;
    }
}
";

// --- Rotary Position Embeddings (Sprint 7 step 2) ---
//
// Applies RoPE rotation to a tensor of shape [batch_heads, seq_len, head_dim] where
// head_dim is even. For each pair (2d, 2d+1) at position s in the sequence (with absolute
// position s + start_pos for KV-cache decode), rotates by angle = (s + start_pos) * theta
// where theta = base^(-2d/head_dim). Llama/Mistral/Qwen/Gemma convention with adjacent pairing.

/// t536 = RopeParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t536 {
    batch_heads: u32,
    seq_len: u32,
    head_dim: u32,
    start_pos: u32,
    base: f32,
    _pad: [u32; 3],
}

const SHADER_ROPE: &str = "
struct P {
    batch_heads: u32, seq_len: u32, head_dim: u32, start_pos: u32,
    base: f32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch_heads * p.seq_len * p.head_dim;
    if idx >= total { return; }

    let d = idx % p.head_dim;
    let s = (idx / p.head_dim) % p.seq_len;
    let bh = idx / (p.seq_len * p.head_dim);

    let pair_idx = d / 2u;
    let is_first = (d % 2u) == 0u;
    let pos_f = f32(p.start_pos + s);

    // inv_freq = base^(-2 * pair_idx / head_dim). Computed as exp(log(base) * exponent).
    let exponent = -2.0 * f32(pair_idx) / f32(p.head_dim);
    let inv_freq = pow(p.base, exponent);
    let angle = pos_f * inv_freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    let base_idx = bh * p.seq_len * p.head_dim + s * p.head_dim + pair_idx * 2u;
    let x_a = input[base_idx];
    let x_b = input[base_idx + 1u];

    if is_first {
        out[idx] = x_a * cos_a - x_b * sin_a;
    } else {
        out[idx] = x_a * sin_a + x_b * cos_a;
    }
}
";

// --- Fused causal SDPA — online softmax, no N×N allocation (Sprint 7 step 6) ---
//
// One thread per (batch_head × q_row). Iterates over kv positions sequentially,
// applying online softmax in a single pass. Uses the output buffer as the
// accumulator (zero-initialised at thread start, normalised at thread end).
// Peak extra memory: O(batch_heads × q_seq × head_dim) — no N×N scores matrix.
// Causal mask is implicit: loop breaks when j > abs_qi_pos.

/// t541 = FusedSdpaParams (private).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t541 {
    batch_heads: u32,
    q_seq: u32,
    kv_seq: u32,
    head_dim: u32,
    scale: f32,
    _pad: [u32; 3],
}

// Private per-invocation accumulator — lives in VGPRs, not VRAM.
// Sized for the largest head_dim in common use (LLaMA: 128). The compiler
// allocates exactly this many VGPRs per lane (~148 total with loop vars);
// fits in the 256-VGPR budget without spilling on RDNA1.
// Wave64 fused SDPA: one wave64 (64 threads) per q_row.
// Each lane owns stride-64 elements of head_dim:
//   lane d owns elements {d, d+64, d+128, d+192} up to head_dim.
// QK dot product parallelized across all 64 lanes; subgroupAdd gives
// the full score simultaneously on all lanes (requires wave64 = one subgroup
// per workgroup, i.e. RDNA/RADV with s509==true).
// Accumulator is 2-4 scalars per lane (VGPRs), not 128 — ~10x lower VGPR pressure.
// Gate: f626 dispatches this path when s509 && head_dim ≤ 256 && head_dim % 64 == 0.
const SHADER_FUSED_SDPA_W64: &str = "
struct P {
    batch_heads: u32, q_seq: u32, kv_seq: u32, head_dim: u32,
    scale: f32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>             p:   P;
@group(0) @binding(1) var<storage, read>       q:   array<f32>;
@group(0) @binding(2) var<storage, read>       k:   array<f32>;
@group(0) @binding(3) var<storage, read>       v:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<private> lane_acc: array<f32, 4>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)           wg:   vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let bq = wg.x + wg.y * 65535u;
    if bq >= p.batch_heads * p.q_seq { return; }

    let bh       = bq / p.q_seq;
    let qi       = bq % p.q_seq;
    let abs_pos  = qi + p.kv_seq - p.q_seq;
    let q_base   = bh * p.q_seq  * p.head_dim + qi * p.head_dim;
    let kv_base  = bh * p.kv_seq * p.head_dim;
    let dpl      = p.head_dim / 64u;

    for (var s: u32 = 0u; s < dpl; s++) { lane_acc[s] = 0.0; }
    var m: f32 = -1.0e30;
    var l: f32 = 0.0;

    for (var j: u32 = 0u; j <= abs_pos; j++) {
        let k_row = kv_base + j * p.head_dim;
        var partial: f32 = 0.0;
        for (var s: u32 = 0u; s < dpl; s++) {
            let d = lane + s * 64u;
            partial += q[q_base + d] * k[k_row + d];
        }
        let score = subgroupAdd(partial) * p.scale;

        let m_new = max(m, score);
        let alpha = exp(m - m_new);
        let beta  = exp(score - m_new);

        let v_row = kv_base + j * p.head_dim;
        for (var s: u32 = 0u; s < dpl; s++) {
            let d = lane + s * 64u;
            lane_acc[s] = lane_acc[s] * alpha + beta * v[v_row + d];
        }
        l = alpha * l + beta;
        m = m_new;
    }

    let inv_l = 1.0 / l;
    for (var s: u32 = 0u; s < dpl; s++) {
        out[q_base + lane + s * 64u] = lane_acc[s] * inv_l;
    }
}
";

// Previous version wrote `out[...]` on every KV step (read-modify-write to VRAM
// each iteration), which dominated runtime for long contexts.
const SHADER_FUSED_SDPA: &str = "
struct P {
    batch_heads: u32, q_seq: u32, kv_seq: u32, head_dim: u32,
    scale: f32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>             p:   P;
@group(0) @binding(1) var<storage, read>       q:   array<f32>;
@group(0) @binding(2) var<storage, read>       k:   array<f32>;
@group(0) @binding(3) var<storage, read>       v:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<private> acc_p: array<f32, 128>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bq = gid.x + gid.y * 65535u * 256u;
    if bq >= p.batch_heads * p.q_seq { return; }

    let bh      = bq / p.q_seq;
    let qi      = bq % p.q_seq;
    let abs_pos = qi + p.kv_seq - p.q_seq;

    let q_base  = bh * p.q_seq  * p.head_dim + qi * p.head_dim;
    let kv_base = bh * p.kv_seq * p.head_dim;

    for (var d: u32 = 0u; d < p.head_dim; d++) {
        acc_p[d] = 0.0;
    }

    var m: f32 = -1.0e30;
    var l: f32 = 0.0;

    for (var j: u32 = 0u; j <= abs_pos; j++) {
        var score: f32 = 0.0;
        let k_row = kv_base + j * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d++) {
            score += q[q_base + d] * k[k_row + d];
        }
        score *= p.scale;

        let m_new = max(m, score);
        let alpha = exp(m - m_new);
        let beta  = exp(score - m_new);

        let v_row = kv_base + j * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d++) {
            acc_p[d] = acc_p[d] * alpha + beta * v[v_row + d];
        }
        l = alpha * l + beta;
        m = m_new;
    }

    let inv_l    = 1.0 / l;
    let out_base = bh * p.q_seq * p.head_dim + qi * p.head_dim;
    for (var d: u32 = 0u; d < p.head_dim; d++) {
        out[out_base + d] = acc_p[d] * inv_l;
    }
}
";

// --- MSE Loss ---

const SHADER_MSE_SUM: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pred: array<f32>;
@group(0) @binding(2) var<storage, read> tgt: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < p.n; i++) {
        let d = pred[i] - tgt[i];
        sum += d * d;
    }
    out[0] = sum / f32(p.n);
}
";

// --- Batch-decode SDPA (f630) and batch RoPE (f631) ---
//
// f630: Online-softmax SDPA for a batch of single decode tokens.
//   Q: [B*Nh, Hd], K/V: [max_batch*Nkv, max_kv_seq, Hd] (BatchKvPool layout).
//   kv_lens[b] = how many KV positions are valid for batch element b.
//   One workgroup per query row (bq = b*Nh + h_q).
//
// f631: RoPE for a batch of single decode tokens.
//   Input: [B*Nh, Hd]. start_pos_buf[b] = absolute position for token b.
//   b = bh / num_heads.

/// t554 = BatchDecodeSdpaParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t554 {
    total_q_heads: u32, // B * Nh
    num_heads: u32,     // Nh (per request)
    num_kv_heads: u32,  // Nkv (per request)
    max_kv_seq: u32,    // stride: positions in KV pool per (slot, head)
    head_dim: u32,      // Hd
    slot_stride: u32,   // Nkv * max_kv_seq * Hd (elements per slot in K/V buffer)
    scale: f32,         // 1/sqrt(Hd)
    _pad: u32,
}

/// t555 = RopeBatchParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t555 {
    total_heads: u32, // B * Nh (or B * Nkv for K)
    num_heads: u32,   // Nh (used to compute b = bh / num_heads)
    head_dim: u32,
    _pad: u32,
    base: f32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

// Wave64 path: one 64-thread workgroup per query row (bq = b*Nh + h_q).
// Lane `d` handles elements {d, d+64, d+128, ...} of head_dim (dpl = head_dim/64).
// kv_lens[b] controls the attention window (no causal mask needed: decode has q_seq=1,
// so the new token always attends to all previous tokens including itself).
const SHADER_FUSED_SDPA_BATCH_DECODE_W64: &str = "
struct P {
    total_q_heads: u32, num_heads: u32, num_kv_heads: u32, max_kv_seq: u32,
    head_dim: u32, slot_stride: u32, scale: f32, _pad: u32,
}
@group(0) @binding(0) var<uniform>             p:       P;
@group(0) @binding(1) var<storage, read>       q:       array<f32>;
@group(0) @binding(2) var<storage, read>       k_all:   array<f32>;
@group(0) @binding(3) var<storage, read>       v_all:   array<f32>;
@group(0) @binding(4) var<storage, read>       kv_lens: array<f32>;
@group(0) @binding(5) var<storage, read_write> out:     array<f32>;

var<private> lane_acc: array<f32, 4>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)           wg:   vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let bq = wg.x + wg.y * 65535u;
    if bq >= p.total_q_heads { return; }

    let b     = bq / p.num_heads;
    let h_q   = bq % p.num_heads;
    let n_rep = p.num_heads / p.num_kv_heads;
    let h_kv  = h_q / n_rep;
    let kv_len = u32(kv_lens[b]);
    if kv_len == 0u { return; }

    let q_base  = bq * p.head_dim;
    let kv_base = b * p.slot_stride + h_kv * p.max_kv_seq * p.head_dim;
    let dpl     = p.head_dim / 64u;

    for (var s: u32 = 0u; s < dpl; s++) { lane_acc[s] = 0.0; }
    var m: f32 = -1.0e30;
    var l: f32 = 0.0;

    for (var j: u32 = 0u; j < kv_len; j++) {
        let k_row = kv_base + j * p.head_dim;
        var partial: f32 = 0.0;
        for (var s: u32 = 0u; s < dpl; s++) {
            partial += q[q_base + lane + s * 64u] * k_all[k_row + lane + s * 64u];
        }
        let score = subgroupAdd(partial) * p.scale;

        let m_new = max(m, score);
        let alpha = exp(m - m_new);
        let beta  = exp(score - m_new);
        let v_row = kv_base + j * p.head_dim;
        for (var s: u32 = 0u; s < dpl; s++) {
            lane_acc[s] = lane_acc[s] * alpha + beta * v_all[v_row + lane + s * 64u];
        }
        l = alpha * l + beta;
        m = m_new;
    }

    let inv_l = 1.0 / l;
    for (var s: u32 = 0u; s < dpl; s++) {
        out[bq * p.head_dim + lane + s * 64u] = lane_acc[s] * inv_l;
    }
}
";

// Fallback (non-wave64): one thread per query row, scalar dot product.
// head_dim must be ≤ 128 (fits in private VGPR array).
const SHADER_FUSED_SDPA_BATCH_DECODE: &str = "
struct P {
    total_q_heads: u32, num_heads: u32, num_kv_heads: u32, max_kv_seq: u32,
    head_dim: u32, slot_stride: u32, scale: f32, _pad: u32,
}
@group(0) @binding(0) var<uniform>             p:       P;
@group(0) @binding(1) var<storage, read>       q:       array<f32>;
@group(0) @binding(2) var<storage, read>       k_all:   array<f32>;
@group(0) @binding(3) var<storage, read>       v_all:   array<f32>;
@group(0) @binding(4) var<storage, read>       kv_lens: array<f32>;
@group(0) @binding(5) var<storage, read_write> out:     array<f32>;

var<private> acc_p: array<f32, 128>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bq = gid.x + gid.y * 65535u * 256u;
    if bq >= p.total_q_heads { return; }

    let b     = bq / p.num_heads;
    let h_q   = bq % p.num_heads;
    let n_rep = p.num_heads / p.num_kv_heads;
    let h_kv  = h_q / n_rep;
    let kv_len = u32(kv_lens[b]);
    if kv_len == 0u { return; }

    let q_base  = bq * p.head_dim;
    let kv_base = b * p.slot_stride + h_kv * p.max_kv_seq * p.head_dim;

    for (var d: u32 = 0u; d < p.head_dim; d++) { acc_p[d] = 0.0; }
    var m: f32 = -1.0e30;
    var l: f32 = 0.0;

    for (var j: u32 = 0u; j < kv_len; j++) {
        var score: f32 = 0.0;
        let k_row = kv_base + j * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d++) {
            score += q[q_base + d] * k_all[k_row + d];
        }
        score *= p.scale;

        let m_new = max(m, score);
        let alpha = exp(m - m_new);
        let beta  = exp(score - m_new);
        let v_row = kv_base + j * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d++) {
            acc_p[d] = acc_p[d] * alpha + beta * v_all[v_row + d];
        }
        l = alpha * l + beta;
        m = m_new;
    }

    let inv_l = 1.0 / l;
    for (var d: u32 = 0u; d < p.head_dim; d++) {
        out[bq * p.head_dim + d] = acc_p[d] * inv_l;
    }
}
";

// RoPE for a batch of single decode tokens.
// Input: [total_heads, head_dim]. start_pos_buf[b] = absolute position for batch element b.
// b = bh / num_heads. seq_len is implicitly 1 (decode), so pos = start_pos + 0 = start_pos.
const SHADER_ROPE_BATCH: &str = "
struct P {
    total_heads: u32, num_heads: u32, head_dim: u32, _pad: u32,
    base: f32, _p1: u32, _p2: u32, _p3: u32,
}
@group(0) @binding(0) var<uniform>             p:             P;
@group(0) @binding(1) var<storage, read>       input:         array<f32>;
@group(0) @binding(2) var<storage, read>       start_pos_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:           array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.total_heads * p.head_dim;
    if idx >= total { return; }

    let bh  = idx / p.head_dim;
    let d   = idx % p.head_dim;
    let b   = bh / p.num_heads;
    let start_pos = u32(start_pos_buf[b]);
    let pos_f = f32(start_pos);

    let pair_idx = d / 2u;
    let is_first = (d % 2u) == 0u;
    let exponent = -2.0 * f32(pair_idx) / f32(p.head_dim);
    let inv_freq = pow(p.base, exponent);
    let angle = pos_f * inv_freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    let base_idx = bh * p.head_dim + pair_idx * 2u;
    let x_a = input[base_idx];
    let x_b = input[base_idx + 1u];

    if is_first {
        out[idx] = x_a * cos_a - x_b * sin_a;
    } else {
        out[idx] = x_a * sin_a + x_b * cos_a;
    }
}
";

impl t500 {
    /// f620 = softmax. Softmax along the last dimension. Input shape: [rows, cols].
    /// Uses fused subgroup reduction when s509 (SUBGROUP feature) is available,
    /// otherwise falls back to the two-pass (stats + apply) approach.
    pub fn f620(&self, p0: &t501, p1: u32, p2: u32) -> Result<t501> {
        ensure!(p0.s507 == (p1 * p2) as usize);

        let v0 = t514 {
            rows: p1,
            cols: p2,
            _pad: [0; 2],
        };

        if self.s509 {
            // Fused path: one workgroup per row, subgroupMax/subgroupAdd for reductions.
            // Overflow: wg.x + wg.y * 65535 encodes row when p1 > 65535.
            let v1 = self.f503((p1 * p2) as usize);
            let (vx, vy) = if p1 <= 65535 {
                (p1, 1)
            } else {
                (65535, p1.div_ceil(65535))
            };
            self.f543(
                SHADER_SOFTMAX_FUSED,
                Some("softmax_fused"),
                &v0,
                &[p0],
                &v1,
                (vx, vy, 1),
            );
            return Ok(v1);
        }

        // Two-pass fallback: stats pass then normalize pass.
        let v1 = self.f503((p1 * 2) as usize);
        self.f543(
            SHADER_SOFTMAX_STATS,
            Some("softmax_stats"),
            &v0,
            &[p0],
            &v1,
            super::f540(p1),
        );

        let v2 = p1 * p2;
        let v3 = self.f503(v2 as usize);

        let v4 = self.f506(&v0);
        let v5 = self
            .s500
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("softmax_apply"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SOFTMAX_APPLY.into()),
            });
        let v6 = self
            .s500
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("softmax_apply"),
                layout: None,
                module: &v5,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let v7 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &v6.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: v4.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p0.s505.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v1.s505.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v3.s505.as_entire_binding(),
                },
            ],
        });
        let (v8, v9, v10) = super::f540(v2);
        let mut v11 = self
            .s500
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut v12 = v11.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_apply"),
                timestamp_writes: None,
            });
            v12.set_pipeline(&v6);
            v12.set_bind_group(0, &v7, &[]);
            v12.dispatch_workgroups(v8, v9, v10);
        }
        self.s501.submit(Some(v11.finish()));

        Ok(v3)
    }

    /// f621 = scaled_dot_product_attention. softmax(Q @ K^T / sqrt(d_k)) @ V.
    /// Q,K,V: [batch_heads, seq_len, d_k]. Returns [batch_heads, seq_len, d_k].
    pub fn f621(&self, p0: &t501, p1: &t501, p2: &t501, p3: u32, p4: u32, p5: u32) -> Result<t501> {
        // 1. K^T: [batch_heads, d_k, seq_len]
        let v0 = self.f641(p1, p3, p4, p5, 1)?;

        // 2. scores = Q @ K^T: [batch_heads, seq_len, seq_len]
        let v1 = self.f581(p0, &v0, p3, p4, p4, p5)?;

        // 3. Scale by 1/sqrt(d_k)
        let v2 = 1.0 / (p5 as f32).sqrt();
        let v3 = self.f553(&v1, v2)?;

        // 4. Softmax over last dim (each row of seq_len)
        let v4 = self.f620(&v3, p3 * p4, p4)?;

        // 5. attn @ V: [batch_heads, seq_len, d_k]
        self.f581(&v4, p2, p3, p4, p5, p4)
    }

    /// f624 = apply_causal_mask. In-place sets `scores[bh, i, j] = -1e30` for j > i + (kv-q).
    /// `scores` shape: [batch_heads, q_seq_len, kv_seq_len]. Requires kv_seq_len >= q_seq_len.
    pub fn f624(&self, p0: &t501, p1: u32, p2: u32, p3: u32) -> Result<()> {
        ensure!(
            p3 >= p2,
            "apply_causal_mask: kv_seq_len ({}) must be >= q_seq_len ({})",
            p3,
            p2
        );
        ensure!(
            p0.s507 == (p1 * p2 * p3) as usize,
            "apply_causal_mask: size mismatch"
        );
        let v0 = t535 {
            batch_heads: p1,
            q_seq_len: p2,
            kv_seq_len: p3,
            _pad: 0,
        };
        self.f543(
            SHADER_CAUSAL_MASK,
            Some("apply_causal_mask"),
            &v0,
            &[],
            p0,
            super::f540(p1 * p2 * p3),
        );
        Ok(())
    }

    /// f623 = scaled_dot_product_attention_causal. SDPA with a causal mask.
    /// Supports asymmetric q_seq_len <= kv_seq_len for KV-cache decode steps.
    /// Q: [batch_heads, q_seq_len, d_k]. K,V: [batch_heads, kv_seq_len, d_k].
    /// Returns: [batch_heads, q_seq_len, d_k].
    pub fn f623(
        &self,
        p0: &t501,
        p1: &t501,
        p2: &t501,
        p3: u32,
        p4: u32,
        p5: u32,
        p6: u32,
    ) -> Result<t501> {
        ensure!(
            p5 >= p4,
            "causal SDPA: kv_seq_len ({}) must be >= q_seq_len ({})",
            p5,
            p4
        );
        ensure!(
            p0.s507 == (p3 * p4 * p6) as usize,
            "causal SDPA: Q size mismatch"
        );
        ensure!(
            p1.s507 == (p3 * p5 * p6) as usize,
            "causal SDPA: K size mismatch"
        );
        ensure!(
            p2.s507 == (p3 * p5 * p6) as usize,
            "causal SDPA: V size mismatch"
        );

        // 1. K^T: [batch_heads, d_k, kv_seq_len]
        let v0 = self.f641(p1, p3, p5, p6, 1)?;

        // 2. scores = Q @ K^T: [batch_heads, q_seq_len, kv_seq_len]
        let v1 = self.f581(p0, &v0, p3, p4, p5, p6)?;

        // 3. Scale by 1/sqrt(d_k)
        let v2 = 1.0 / (p6 as f32).sqrt();
        let v3 = self.f553(&v1, v2)?;

        // 4. Apply causal mask in place
        self.f624(&v3, p3, p4, p5)?;

        // 5. Softmax over last dim (each row of kv_seq_len)
        let v4 = self.f620(&v3, p3 * p4, p5)?;

        // 6. attn @ V: [batch_heads, q_seq_len, d_k]
        self.f581(&v4, p2, p3, p4, p6, p5)
    }

    /// f625 = rope. Rotary position embeddings on a tensor [batch_heads, seq_len, head_dim].
    /// `head_dim` must be even. `start_pos` shifts absolute position for KV-cache decode.
    /// `base` is the rotation base (Llama/Mistral default: 10000.0).
    pub fn f625(&self, p0: &t501, p1: u32, p2: u32, p3: u32, p4: u32, p5: f32) -> Result<t501> {
        ensure!(p3 % 2 == 0, "rope: head_dim ({}) must be even", p3);
        ensure!(
            p0.s507 == (p1 * p2 * p3) as usize,
            "rope: input size mismatch"
        );
        ensure!(p5 > 0.0, "rope: base must be positive");

        let v0 = p1 * p2 * p3;
        let v1 = self.f503(v0 as usize);
        let v2 = t536 {
            batch_heads: p1,
            seq_len: p2,
            head_dim: p3,
            start_pos: p4,
            base: p5,
            _pad: [0; 3],
        };
        self.f543(SHADER_ROPE, Some("rope"), &v2, &[p0], &v1, super::f540(v0));
        Ok(v1)
    }

    /// f627 = split_heads. [seq, n_heads*head_dim] → [n_heads, seq, head_dim].
    /// Prepares Q/K/V output of linear projection for multi-head attention.
    pub fn f627(&self, p0: &t501, p1: u32, p2: u32, p3: u32) -> Result<t501> {
        ensure!(
            p0.s507 == (p2 * p1 * p3) as usize,
            "split_heads: size mismatch"
        );
        let v0 = self.f503(p0.s507);
        let v1 = t542 {
            n_heads: p1,
            seq_len: p2,
            head_dim: p3,
            _pad: 0,
        };
        self.f543(
            SHADER_SPLIT_HEADS,
            Some("split_heads"),
            &v1,
            &[p0],
            &v0,
            super::f540(p0.s507 as u32),
        );
        Ok(v0)
    }

    /// f628 = merge_heads. [n_heads, seq, head_dim] → [seq, n_heads*head_dim].
    /// Collapses attention output back to the projection input shape.
    pub fn f628(&self, p0: &t501, p1: u32, p2: u32, p3: u32) -> Result<t501> {
        ensure!(
            p0.s507 == (p1 * p2 * p3) as usize,
            "merge_heads: size mismatch"
        );
        let v0 = self.f503(p0.s507);
        let v1 = t542 {
            n_heads: p1,
            seq_len: p2,
            head_dim: p3,
            _pad: 0,
        };
        self.f543(
            SHADER_MERGE_HEADS,
            Some("merge_heads"),
            &v1,
            &[p0],
            &v0,
            super::f540(p0.s507 as u32),
        );
        Ok(v0)
    }

    /// f629 = repeat_kv. [n_kv_heads, kv_seq, head_dim] → [n_heads, kv_seq, head_dim].
    /// Expands GQA key/value heads to match the full query head count.
    /// When n_rep = 1 (MHA), this is a zero-copy clone.
    pub fn f629(&self, p0: &t501, p1: u32, p2: u32, p3: u32, p4: u32) -> Result<t501> {
        ensure!(
            p2 > 0 && p1 % p2 == 0,
            "repeat_kv: n_heads ({}) must be divisible by n_kv_heads ({})",
            p1,
            p2
        );
        ensure!(
            p0.s507 == (p2 * p3 * p4) as usize,
            "repeat_kv: size mismatch"
        );
        let v_rep = p1 / p2;
        let v_out = (p1 * p3 * p4) as usize;
        let v0 = self.f503(v_out);
        let v1 = t543 {
            n_kv_heads: p2,
            n_rep: v_rep,
            kv_seq: p3,
            head_dim: p4,
        };
        self.f543(
            SHADER_REPEAT_KV,
            Some("repeat_kv"),
            &v1,
            &[p0],
            &v0,
            super::f540((p1 * p3 * p4) as u32),
        );
        Ok(v0)
    }

    /// f626 = scaled_dot_product_attention_fused. Online-softmax causal SDPA.
    /// Identical semantics to f623 but never allocates an N×N scores matrix.
    /// Safe for long contexts: peak extra memory is O(batch_heads × q_seq × head_dim).
    /// Q: [batch_heads, q_seq, head_dim]. K,V: [batch_heads, kv_seq, head_dim].
    /// Requires kv_seq >= q_seq (same as f623). Returns [batch_heads, q_seq, head_dim].
    pub fn f626(
        &self,
        p0: &t501,
        p1: &t501,
        p2: &t501,
        p3: u32,
        p4: u32,
        p5: u32,
        p6: u32,
    ) -> Result<t501> {
        ensure!(
            p5 >= p4,
            "fused SDPA: kv_seq ({}) must be >= q_seq ({})",
            p5,
            p4
        );
        ensure!(
            p0.s507 == (p3 * p4 * p6) as usize,
            "fused SDPA: Q size mismatch"
        );
        ensure!(
            p1.s507 == (p3 * p5 * p6) as usize,
            "fused SDPA: K size mismatch"
        );
        ensure!(
            p2.s507 == (p3 * p5 * p6) as usize,
            "fused SDPA: V size mismatch"
        );

        let v0 = self.f503((p3 * p4 * p6) as usize); // output [bh, q_seq, head_dim]
        let v1 = t541 {
            batch_heads: p3,
            q_seq: p4,
            kv_seq: p5,
            head_dim: p6,
            scale: 1.0 / (p6 as f32).sqrt(),
            _pad: [0; 3],
        };
        // Wave64 path: one 64-thread workgroup per q_row; subgroupAdd parallelises
        // the head_dim dot product. Requires wave64 (s509 == true on RDNA/RADV).
        // Falls back to single-thread-per-q-row for non-wave64 or odd head_dim.
        if self.s509 && p6 % 64 == 0 && p6 <= 256 {
            let total = p3 * p4;
            let (vx, vy) = if total <= 65535 {
                (total, 1)
            } else {
                (65535, total.div_ceil(65535))
            };
            self.f543(
                SHADER_FUSED_SDPA_W64,
                Some("fused_sdpa_w64"),
                &v1,
                &[p0, p1, p2],
                &v0,
                (vx, vy, 1),
            );
        } else {
            self.f543(
                SHADER_FUSED_SDPA,
                Some("fused_sdpa"),
                &v1,
                &[p0, p1, p2],
                &v0,
                super::f540(p3 * p4),
            );
        }
        Ok(v0)
    }

    /// f622 = mse_loss. mean((pred - target)^2). Returns a 1-element buffer.
    pub fn f622(&self, p0: &t501, p1: &t501) -> Result<t501> {
        ensure!(p0.s507 == p1.s507, "mse: length mismatch");
        let v0 = self.f503(1);
        let v1 = t513 {
            n: p0.s507 as u32,
            _pad: [0; 3],
        };
        self.f543(SHADER_MSE_SUM, Some("mse"), &v1, &[p0, p1], &v0, (1, 1, 1));
        Ok(v0)
    }

    /// f630 = fused_sdpa_batch_decode. Online-softmax SDPA for a batch of decode steps.
    /// Each of `p5` (active_batch) requests contributes exactly one query token (q_seq=1).
    /// Q:     [active_batch * num_heads,    head_dim]  flat (= [B*Nh, Hd])
    /// K/V:   [max_batch * num_kv_heads, max_kv_seq, head_dim] flat  (BatchKvPool buffers)
    /// kv_lens: [active_batch] as f32  (valid KV length per slot, including the new token)
    /// Returns: [active_batch * num_heads, head_dim]  (same shape as Q)
    pub fn f630(
        &self,
        p0: &t501, // Q
        p1: &t501, // K_all (batch KV pool K buffer)
        p2: &t501, // V_all (batch KV pool V buffer)
        p3: &t501, // kv_lens [active_batch] as f32
        p4: u32,   // active_batch
        p5: u32,   // num_heads (Nh, per request)
        p6: u32,   // num_kv_heads (Nkv, per request)
        p7: u32,   // max_kv_seq (pool stride per head row)
        p8: u32,   // head_dim
    ) -> Result<t501> {
        let total_q = p4 * p5;
        ensure!(
            p0.s507 == (total_q * p8) as usize,
            "f630: Q size mismatch: got {} want {}",
            p0.s507,
            total_q * p8
        );
        ensure!(
            p5 % p6 == 0,
            "f630: num_heads ({}) not divisible by num_kv_heads ({})",
            p5,
            p6
        );
        ensure!(
            p8 % 64 == 0 || p8 <= 128,
            "f630: head_dim ({}) must be multiple of 64 or ≤128",
            p8
        );

        let slot_stride = p6 * p7 * p8;
        let out = self.f503((total_q * p8) as usize);
        let params = t554 {
            total_q_heads: total_q,
            num_heads: p5,
            num_kv_heads: p6,
            max_kv_seq: p7,
            head_dim: p8,
            slot_stride,
            scale: 1.0 / (p8 as f32).sqrt(),
            _pad: 0,
        };

        if self.s509 && p8 % 64 == 0 && p8 <= 256 {
            let (wx, wy) = if total_q <= 65535 {
                (total_q, 1)
            } else {
                (65535, total_q.div_ceil(65535))
            };
            self.f543(
                SHADER_FUSED_SDPA_BATCH_DECODE_W64,
                Some("sdpa_batch_decode_w64"),
                &params,
                &[p0, p1, p2, p3],
                &out,
                (wx, wy, 1),
            );
        } else {
            self.f543(
                SHADER_FUSED_SDPA_BATCH_DECODE,
                Some("sdpa_batch_decode"),
                &params,
                &[p0, p1, p2, p3],
                &out,
                super::f540(total_q),
            );
        }
        Ok(out)
    }

    /// f631 = rope_batch_decode. RoPE for a batch of single decode tokens.
    /// Input: [total_heads, head_dim]  where total_heads = active_batch * num_heads.
    /// start_pos_buf: [active_batch] as f32 — absolute position for token of each batch element.
    /// b = bh / p3 (num_heads) gives the batch element index.
    /// Returns: [total_heads, head_dim]  (same shape).
    pub fn f631(
        &self,
        p0: &t501, // input [total_heads * head_dim]
        p1: &t501, // start_pos_buf [active_batch] as f32
        p2: u32,   // total_heads = active_batch * num_heads
        p3: u32,   // num_heads per batch element (to compute b = bh / p3)
        p4: u32,   // head_dim
        p5: f32,   // rope_base
    ) -> Result<t501> {
        ensure!(p4 % 2 == 0, "f631: head_dim ({}) must be even", p4);
        ensure!(
            p0.s507 == (p2 * p4) as usize,
            "f631: input size mismatch: got {} want {}",
            p0.s507,
            p2 * p4
        );
        ensure!(p5 > 0.0, "f631: base must be positive");

        let out = self.f503((p2 * p4) as usize);
        let params = t555 {
            total_heads: p2,
            num_heads: p3,
            head_dim: p4,
            _pad: 0,
            base: p5,
            _p1: 0,
            _p2: 0,
            _p3: 0,
        };
        self.f543(
            SHADER_ROPE_BATCH,
            Some("rope_batch"),
            &params,
            &[p0, p1],
            &out,
            super::f540(p2 * p4),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;

    fn dev() -> &'static t500 {
        &crate::ops::TEST_DEV
    }

    // CPU reference softmax
    fn cpu_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        for r in 0..rows {
            let row = &input[r * cols..(r + 1) * cols];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
            for c in 0..cols {
                out[r * cols + c] = (row[c] - mx).exp() / sum;
            }
        }
        out
    }

    // CPU reference attention
    fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, dk: usize) -> Vec<f32> {
        let scale = 1.0 / (dk as f32).sqrt();
        let mut scores = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let mut s = 0.0;
                for d in 0..dk {
                    s += q[i * dk + d] * k[j * dk + d];
                }
                scores[i * seq + j] = s * scale;
            }
        }
        let attn = cpu_softmax(&scores, seq, seq);
        let mut out = vec![0.0f32; seq * dk];
        for i in 0..seq {
            for d in 0..dk {
                let mut s = 0.0;
                for j in 0..seq {
                    s += attn[i * seq + j] * v[j * dk + d];
                }
                out[i * dk + d] = s;
            }
        }
        out
    }

    #[test]
    fn f620_vs_cpu() {
        let v0: Vec<f32> = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 5.0, 5.0, 5.0];
        let v1 = cpu_softmax(&v0, 3, 3);
        let v2 = dev()
            .f504(&dev().f620(&dev().f502(&v0), 3, 3).unwrap())
            .unwrap();
        f544(&v2, &v1, 1e-4);
        for v3 in 0..3 {
            let v4: f32 = v2[v3 * 3..(v3 + 1) * 3].iter().sum();
            assert!((v4 - 1.0).abs() < 1e-4, "row {v3} sum = {v4}");
        }
    }

    #[test]
    fn f620_large_values() {
        let v0 = vec![1000.0, 1001.0, 1002.0];
        let v1 = cpu_softmax(&v0, 1, 3);
        let v2 = dev()
            .f504(&dev().f620(&dev().f502(&v0), 1, 3).unwrap())
            .unwrap();
        f544(&v2, &v1, 1e-4);
        let v3: f32 = v2.iter().sum();
        assert!((v3 - 1.0).abs() < 1e-4, "sum = {v3}");
    }

    #[test]
    fn f620_single_element() {
        let v0 = dev()
            .f504(&dev().f620(&dev().f502(&[42.0]), 1, 1).unwrap())
            .unwrap();
        f544(&v0, &[1.0], 1e-5);
    }

    #[test]
    fn f621_vs_cpu() {
        let v0: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let v1: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let v2: Vec<f32> = (0..12).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let v3 = cpu_attention(&v0, &v1, &v2, 3, 4);
        let v4 = dev()
            .f504(
                &dev()
                    .f621(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        3,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&v4, &v3, 1e-3);
    }

    #[test]
    fn f621_uniform_qk() {
        let v0 = vec![1.0, 1.0, 1.0, 1.0];
        let v1 = v0.clone();
        let v2 = vec![0.0, 10.0, 20.0, 30.0];
        let v3 = cpu_attention(&v0, &v1, &v2, 2, 2);
        let v4 = dev()
            .f504(
                &dev()
                    .f621(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        2,
                        2,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&v4, &v3, 1e-3);
    }

    #[test]
    fn f620_known_uniform_input() {
        // Analytical: softmax over k equal values = uniform 1/k.
        // softmax([0,0,0]) = [1/3, 1/3, 1/3]. Independent of our cpu_softmax helper.
        let v0 = dev()
            .f504(&dev().f620(&dev().f502(&[0.0f32; 3]), 1, 3).unwrap())
            .unwrap();
        let v1 = 1.0_f32 / 3.0;
        f544(&v0, &[v1, v1, v1], 1e-6);
    }

    #[test]
    fn f620_known_binary_input() {
        // softmax([0, 1]) = [1/(1+e), e/(1+e)]. With e at f64 precision:
        //   1/(1+e) ≈ 0.26894142, e/(1+e) ≈ 0.73105858 (same numbers as sigmoid(1)/sigmoid(-1)).
        let v0 = dev()
            .f504(&dev().f620(&dev().f502(&[0.0f32, 1.0]), 1, 2).unwrap())
            .unwrap();
        f544(&v0, &[0.26894142, 0.73105858], 1e-5);
    }

    #[test]
    fn f621_known_uniform_attention() {
        // When Q == K (any uniform value), all scores are equal -> softmax = uniform 1/seq_len
        // -> each output row = mean(V over kv dim). Independent of cpu_attention helper.
        // Q = K = zero, V = [[1, 5], [3, 7]] -> output[i, :] = mean(V) per col = [(1+3)/2, (5+7)/2] = [2, 6].
        let v0 = vec![0.0f32; 4]; // [1 head, 2 seq, d_k=2]
        let v1 = v0.clone();
        let v2 = vec![1.0f32, 5.0, 3.0, 7.0];
        let v3 = dev()
            .f504(
                &dev()
                    .f621(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        2,
                        2,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&v3, &[2.0, 6.0, 2.0, 6.0], 1e-4);
    }

    #[test]
    fn f621_multi_batch() {
        // 3 batch_heads, seq=2, dk=2. Each head is independent.
        // Use zero Q/K → uniform attention → output row = mean(V rows).
        let bh = 3u32;
        let seq = 2u32;
        let dk = 2u32;
        let q = vec![0.0f32; (bh * seq * dk) as usize];
        let k = q.clone();
        let v: Vec<f32> = (0..(bh * seq * dk) as usize).map(|i| i as f32).collect();
        let got = dev()
            .f504(
                &dev()
                    .f621(
                        &dev().f502(&q),
                        &dev().f502(&k),
                        &dev().f502(&v),
                        bh,
                        seq,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        // For each head h: V rows are [h*4, h*4+1, h*4+2, h*4+3].
        // Mean per col: col0 = (h*4 + h*4+2)/2, col1 = (h*4+1 + h*4+3)/2.
        for h in 0..bh as usize {
            let base_v = h * (seq * dk) as usize;
            let mean0 = (v[base_v] + v[base_v + 2]) / 2.0;
            let mean1 = (v[base_v + 1] + v[base_v + 3]) / 2.0;
            let base_o = h * (seq * dk) as usize;
            f544(&got[base_o..base_o + 2], &[mean0, mean1], 1e-4);
            f544(&got[base_o + 2..base_o + 4], &[mean0, mean1], 1e-4);
        }
    }

    #[test]
    fn f622_basic() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0]);
        let v1 = dev().f502(&[1.5, 2.5, 3.5]);
        let v2 = dev().f504(&dev().f622(&v0, &v1).unwrap()).unwrap();
        f544(&v2, &[0.25], 1e-5);
    }

    #[test]
    fn f622_zero() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0]);
        let v1 = dev().f504(&dev().f622(&v0, &v0).unwrap()).unwrap();
        f544(&v1, &[0.0], 1e-6);
    }

    #[test]
    fn f622_known_value() {
        let v0 = dev().f502(&[0.0, 0.0, 0.0]);
        let v1 = dev().f502(&[1.0, 2.0, 3.0]);
        let v2 = dev().f504(&dev().f622(&v0, &v1).unwrap()).unwrap();
        f544(&v2, &[14.0 / 3.0], 1e-5);
    }

    #[test]
    fn f620_size_mismatch() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0]);
        assert!(dev().f620(&v0, 2, 3).is_err());
    }

    #[test]
    fn f622_length_mismatch() {
        let v0 = dev().f502(&[1.0, 2.0]);
        let v1 = dev().f502(&[1.0, 2.0, 3.0]);
        assert!(dev().f622(&v0, &v1).is_err());
    }

    #[test]
    fn f622_single_element() {
        let v0 = dev()
            .f504(
                &dev()
                    .f622(&dev().f502(&[5.0]), &dev().f502(&[3.0]))
                    .unwrap(),
            )
            .unwrap();
        f544(&v0, &[4.0], 1e-5);
    }

    #[test]
    fn f622_negative_values() {
        let v0 = dev().f502(&[-1.0, -2.0]);
        let v1 = dev().f502(&[1.0, 2.0]);
        let v2 = dev().f504(&dev().f622(&v0, &v1).unwrap()).unwrap();
        f544(&v2, &[10.0], 1e-5);
    }

    // --- f624 = apply_causal_mask ---

    #[test]
    fn f624_square_triangular() {
        // 1 head, q=kv=3, all-zero scores -> upper triangle becomes -1e30, diagonal+lower stay 0
        let v0 = dev().f502(&[0.0f32; 9]);
        dev().f624(&v0, 1, 3, 3).unwrap();
        let v1 = dev().f504(&v0).unwrap();
        // Row 0: j=0 keep, j=1 masked, j=2 masked
        // Row 1: j=0 keep, j=1 keep, j=2 masked
        // Row 2: j=0 keep, j=1 keep, j=2 keep
        assert_eq!(v1[0], 0.0);
        assert!(v1[1] < -1e29);
        assert!(v1[2] < -1e29);
        assert_eq!(v1[3], 0.0);
        assert_eq!(v1[4], 0.0);
        assert!(v1[5] < -1e29);
        assert_eq!(v1[6], 0.0);
        assert_eq!(v1[7], 0.0);
        assert_eq!(v1[8], 0.0);
    }

    #[test]
    fn f624_decode_no_mask() {
        // q=1, kv=4: position 0 in Q is at absolute kv index 3, can attend to all of {0,1,2,3}.
        // No score should be masked.
        let v0 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0]);
        dev().f624(&v0, 1, 1, 4).unwrap();
        let v1 = dev().f504(&v0).unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f624_partial_offset() {
        // q=2, kv=4: q rows are at absolute kv indices {2, 3}.
        // Row 0 (abs 2): j<=2 keep, j=3 masked.
        // Row 1 (abs 3): all keep.
        let v0 = dev().f502(&[0.0f32; 8]); // 1 head, 2*4 = 8 scores
        dev().f624(&v0, 1, 2, 4).unwrap();
        let v1 = dev().f504(&v0).unwrap();
        assert_eq!(v1[0], 0.0);
        assert_eq!(v1[1], 0.0);
        assert_eq!(v1[2], 0.0);
        assert!(v1[3] < -1e29);
        assert_eq!(v1[4], 0.0);
        assert_eq!(v1[5], 0.0);
        assert_eq!(v1[6], 0.0);
        assert_eq!(v1[7], 0.0);
    }

    #[test]
    fn f624_kv_smaller_than_q_errors() {
        let v0 = dev().f502(&[0.0f32; 6]);
        assert!(dev().f624(&v0, 1, 3, 2).is_err());
    }

    // --- f623 = causal SDPA ---
    // CPU reference: scores masked, then softmax + V product.

    fn cpu_causal_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_seq: usize,
        kv_seq: usize,
        dk: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (dk as f32).sqrt();
        let offset = kv_seq - q_seq;
        // scores: [q_seq, kv_seq]
        let mut scores = vec![0.0f32; q_seq * kv_seq];
        for i in 0..q_seq {
            for j in 0..kv_seq {
                let mut s = 0.0;
                for d in 0..dk {
                    s += q[i * dk + d] * k[j * dk + d];
                }
                scores[i * kv_seq + j] = s * scale;
                if j > i + offset {
                    scores[i * kv_seq + j] = -1.0e30;
                }
            }
        }
        // softmax each row
        let mut attn = vec![0.0f32; q_seq * kv_seq];
        for i in 0..q_seq {
            let row = &scores[i * kv_seq..(i + 1) * kv_seq];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
            for j in 0..kv_seq {
                attn[i * kv_seq + j] = (row[j] - mx).exp() / sum;
            }
        }
        // out = attn @ V: [q_seq, dk]
        let mut out = vec![0.0f32; q_seq * dk];
        for i in 0..q_seq {
            for d in 0..dk {
                let mut s = 0.0;
                for j in 0..kv_seq {
                    s += attn[i * kv_seq + j] * v[j * dk + d];
                }
                out[i * dk + d] = s;
            }
        }
        out
    }

    #[test]
    fn f623_prefill_vs_cpu() {
        // q_seq = kv_seq = 3, d_k = 4, 1 head — standard triangular mask
        let v0: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let v1: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let v2: Vec<f32> = (0..12).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let v3 = cpu_causal_attention(&v0, &v1, &v2, 3, 3, 4);
        let v4 = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        3,
                        3,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&v4, &v3, 1e-3);
    }

    #[test]
    fn f623_decode_step_vs_unmasked() {
        // q_seq = 1, kv_seq = 4 — no mask should apply, so f623 should match unmasked f621
        // if we run f621 with seq_len=4 and only look at the first row of output (which corresponds
        // to the Q row). To compare apples to apples, build a Q with the same single row replicated
        // 4 times, run unmasked f621, and verify f623's single-row output matches f621's first row.
        let v0_one = vec![0.1f32, -0.2, 0.3, -0.4]; // Q's single row [1, d_k=4]
        let v1: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.1).collect(); // K [4, 4]
        let v2: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect(); // V [4, 4]

        // f623 with q_seq=1, kv_seq=4
        let v3 = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&v0_one),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        1,
                        4,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        // CPU reference: causal mask with q=1 kv=4 -> no mask
        let v4 = cpu_causal_attention(&v0_one, &v1, &v2, 1, 4, 4);
        f544(&v3, &v4, 1e-3);
    }

    #[test]
    fn f623_partial_decode_vs_cpu() {
        // q_seq = 2, kv_seq = 5, d_k = 4 — partial mask. Q's rows are at abs positions {3, 4}.
        let v0: Vec<f32> = (0..8).map(|i| (i as f32) * 0.07).collect();
        let v1: Vec<f32> = (0..20).map(|i| (i as f32) * 0.04 - 0.2).collect();
        let v2: Vec<f32> = (0..20).map(|i| (i as f32) * 0.11).collect();
        let v3 = cpu_causal_attention(&v0, &v1, &v2, 2, 5, 4);
        let v4 = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        2,
                        5,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&v4, &v3, 1e-3);
    }

    #[test]
    fn f623_first_row_only_sees_first_kv() {
        // Construct V where each kv position has a one-hot at a unique slot.
        // Row 0 of Q with causal mask sees only kv position 0, so output[0, :] = V[0, :].
        let v_data: Vec<f32> = (0..16)
            .map(|i| if i < 4 { 100.0 } else { (i as f32) - 4.0 })
            .collect();
        // Q row 0: any vector; doesn't matter for first row's masked attention (only j=0 in distribution)
        let q_data = vec![
            1.0f32, 0.5, -0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
        let v_out = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&q_data),
                        &dev().f502(&k_data),
                        &dev().f502(&v_data),
                        1,
                        4,
                        4,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        // First row of output (positions [0..4]) must equal V[0, :] (since the mask leaves only j=0)
        f544(&v_out[..4], &v_data[..4], 1e-3);
    }

    // --- f625 = rope ---
    // RoPE: pos 0 -> identity. Non-zero pos -> hand-computed cos/sin rotation.

    #[test]
    fn f625_position_zero_is_identity() {
        // start_pos=0, seq_len=1 -> position 0 -> angle = 0 -> cos=1, sin=0 -> output = input
        let v0 = vec![1.0f32, 2.0, 3.0, 4.0]; // 1 head, 1 seq, head_dim=4
        let v1 = dev()
            .f504(&dev().f625(&dev().f502(&v0), 1, 1, 4, 0, 10000.0).unwrap())
            .unwrap();
        f544(&v1, &v0, 1e-5);
    }

    #[test]
    fn f625_known_rotation() {
        // 1 head, seq_len=1, head_dim=2, start_pos=1, base=10000.
        // pair_idx=0, exponent=0 -> inv_freq=1.0. pos=1 -> angle=1.0 -> cos(1), sin(1).
        // Input [1.0, 0.0] (x_a=1, x_b=0):
        //   out[0] = 1*cos(1) - 0*sin(1) = cos(1) ≈ 0.5403
        //   out[1] = 1*sin(1) + 0*cos(1) = sin(1) ≈ 0.8415
        let v0 = vec![1.0f32, 0.0];
        let v1 = dev()
            .f504(&dev().f625(&dev().f502(&v0), 1, 1, 2, 1, 10000.0).unwrap())
            .unwrap();
        f544(&v1, &[0.54030231, 0.84147098], 1e-4);
    }

    #[test]
    fn f625_rotation_preserves_norm() {
        // RoPE rotations are orthogonal — input/output norms (per pair) must match.
        let v0: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.0).collect(); // 1 head, 4 seq, head_dim=8
        let v1 = dev()
            .f504(&dev().f625(&dev().f502(&v0), 1, 4, 8, 0, 10000.0).unwrap())
            .unwrap();
        // For each (seq, pair), the (x_a, x_b) norm must equal (out_a, out_b) norm.
        for s in 0..4usize {
            for pair in 0..4usize {
                let base_i = s * 8 + pair * 2;
                let in_norm_sq = v0[base_i].powi(2) + v0[base_i + 1].powi(2);
                let out_norm_sq = v1[base_i].powi(2) + v1[base_i + 1].powi(2);
                assert!(
                    (in_norm_sq - out_norm_sq).abs() < 1e-3,
                    "norm mismatch at s={s} pair={pair}: in={in_norm_sq} out={out_norm_sq}"
                );
            }
        }
    }

    #[test]
    fn f625_start_pos_shift() {
        // RoPE(input, start_pos=5, seq_len=1) should equal RoPE(input padded by 5 zeros, start_pos=0)[5]
        // Easier: rope at seq position s with start_pos=0 should equal rope at seq position 0 with start_pos=s.
        let v0 = vec![0.7f32, -0.3, 0.5, 0.2]; // [1, 1, 4]
        let result_a = dev()
            .f504(&dev().f625(&dev().f502(&v0), 1, 1, 4, 5, 10000.0).unwrap())
            .unwrap();

        // Build a 6-seq input where position 5 is v0, others zero; rope at start_pos=0; take row 5
        let mut padded = vec![0.0f32; 24];
        padded[20..24].copy_from_slice(&v0); // seq=5, head_dim=4
        let result_b_full = dev()
            .f504(
                &dev()
                    .f625(&dev().f502(&padded), 1, 6, 4, 0, 10000.0)
                    .unwrap(),
            )
            .unwrap();
        let result_b = &result_b_full[20..24];
        f544(&result_a, result_b, 1e-4);
    }

    // CPU reference RoPE: for each (bh, s, pair_idx): angle = (start_pos + s) / base^(2*pair_idx/head_dim).
    // Rotation: [x0*cos - x1*sin, x0*sin + x1*cos].
    fn cpu_rope(
        input: &[f32],
        batch_heads: usize,
        seq_len: usize,
        head_dim: usize,
        start_pos: usize,
        base: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch_heads * seq_len * head_dim];
        for bh in 0..batch_heads {
            for s in 0..seq_len {
                for pair_idx in 0..(head_dim / 2) {
                    let exponent = -2.0 * pair_idx as f32 / head_dim as f32;
                    let inv_freq = base.powf(exponent);
                    let angle = (start_pos + s) as f32 * inv_freq;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let base_idx = bh * seq_len * head_dim + s * head_dim + pair_idx * 2;
                    let x0 = input[base_idx];
                    let x1 = input[base_idx + 1];
                    out[base_idx] = x0 * cos_a - x1 * sin_a;
                    out[base_idx + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
        out
    }

    #[test]
    fn f625_multi_position_vs_cpu() {
        // seq_len=4, head_dim=4, batch_heads=2, start_pos=0, theta=10000.
        // Exercises the full multi-position, multi-head RoPE path against a CPU reference.
        let batch_heads = 2usize;
        let seq_len = 4usize;
        let head_dim = 4usize;
        let start_pos = 0u32;
        let base = 10000.0f32;

        let input: Vec<f32> = (0..batch_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.1 - 1.5)
            .collect();

        let cpu_out = cpu_rope(
            &input,
            batch_heads,
            seq_len,
            head_dim,
            start_pos as usize,
            base,
        );

        let gpu_out = dev()
            .f504(
                &dev()
                    .f625(
                        &dev().f502(&input),
                        batch_heads as u32,
                        seq_len as u32,
                        head_dim as u32,
                        start_pos,
                        base,
                    )
                    .unwrap(),
            )
            .unwrap();

        f544(&gpu_out, &cpu_out, 1e-4);
    }

    #[test]
    fn f625_start_pos_continuity() {
        // Verify that RoPE on tokens [0..4] with start_pos=0 produces the same rotations
        // as RoPE on tokens [2..4] with start_pos=2 (the subsequence must match).
        // This tests that the start_pos offset is correctly applied per-sequence-position.
        let batch_heads = 1usize;
        let seq_len_full = 4usize;
        let head_dim = 4usize;
        let base = 10000.0f32;

        let input_full: Vec<f32> = (0..batch_heads * seq_len_full * head_dim)
            .map(|i| (i as f32) * 0.13 - 0.5)
            .collect();

        // Full sequence: tokens [0..4], start_pos=0.
        let full_out = dev()
            .f504(
                &dev()
                    .f625(
                        &dev().f502(&input_full),
                        batch_heads as u32,
                        seq_len_full as u32,
                        head_dim as u32,
                        0,
                        base,
                    )
                    .unwrap(),
            )
            .unwrap();

        // Subsequence: tokens [2..4] extracted from the same input, start_pos=2.
        let sub_seq_len = 2usize;
        let sub_start = sub_seq_len * head_dim; // offset of seq position 2 in the flat input
        let input_sub: Vec<f32> =
            input_full[sub_start..sub_start + sub_seq_len * head_dim].to_vec();

        let sub_out = dev()
            .f504(
                &dev()
                    .f625(
                        &dev().f502(&input_sub),
                        batch_heads as u32,
                        sub_seq_len as u32,
                        head_dim as u32,
                        2,
                        base,
                    )
                    .unwrap(),
            )
            .unwrap();

        // The last two sequence positions from the full run must equal the subsequence output.
        let full_tail = &full_out[sub_start..sub_start + sub_seq_len * head_dim];
        f544(&sub_out, full_tail, 1e-4);
    }

    #[test]
    fn f625_odd_head_dim_errors() {
        let v0 = vec![1.0f32; 5];
        assert!(dev().f625(&dev().f502(&v0), 1, 1, 5, 0, 10000.0).is_err());
    }

    #[test]
    fn f625_zero_base_errors() {
        let v0 = vec![1.0f32; 4];
        assert!(dev().f625(&dev().f502(&v0), 1, 1, 4, 0, 0.0).is_err());
    }

    // --- f626 = fused causal SDPA (online softmax, no N×N alloc) ---

    // f626 must agree with f623 on a standard prefill (q_seq == kv_seq).
    #[test]
    fn f626_prefill_matches_f623() {
        let v0: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let v1: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let v2: Vec<f32> = (0..12).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let ref_out = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        3,
                        3,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        let got = dev()
            .f504(
                &dev()
                    .f626(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        3,
                        3,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&got, &ref_out, 1e-3);
    }

    // f626 decode step: q_seq=1, kv_seq=4. Must match f623.
    #[test]
    fn f626_decode_matches_f623() {
        let v0 = vec![0.1f32, -0.2, 0.3, -0.4];
        let v1: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let v2: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
        let ref_out = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        1,
                        4,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        let got = dev()
            .f504(
                &dev()
                    .f626(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        1,
                        4,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&got, &ref_out, 1e-3);
    }

    // f626: first output row must attend only to V[0] (causal mask eliminates all j>0).
    #[test]
    fn f626_first_row_is_v0() {
        let v_data: Vec<f32> = (0..16)
            .map(|i| if i < 4 { 100.0 } else { i as f32 })
            .collect();
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
        let got = dev()
            .f504(
                &dev()
                    .f626(
                        &dev().f502(&q_data),
                        &dev().f502(&k_data),
                        &dev().f502(&v_data),
                        1,
                        4,
                        4,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&got[..4], &v_data[..4], 1e-3);
    }

    // f626: multiple heads — each head must produce the same result as f623.
    #[test]
    fn f626_multi_head_matches_f623() {
        let bh = 2u32;
        let seq = 3u32;
        let dk = 4u32;
        let n = (bh * seq * dk) as usize;
        let v0: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07 - 0.5).collect();
        let v1: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 + 0.2).collect();
        let v2: Vec<f32> = (0..n).map(|i| (i as f32) * 0.11 - 0.3).collect();
        let ref_out = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        bh,
                        seq,
                        seq,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        let got = dev()
            .f504(
                &dev()
                    .f626(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        bh,
                        seq,
                        seq,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&got, &ref_out, 1e-3);
    }

    // f626: partial decode (q_seq=2, kv_seq=5) must match f623.
    #[test]
    fn f626_partial_decode_matches_f623() {
        let v0: Vec<f32> = (0..8).map(|i| (i as f32) * 0.07).collect();
        let v1: Vec<f32> = (0..20).map(|i| (i as f32) * 0.04 - 0.2).collect();
        let v2: Vec<f32> = (0..20).map(|i| (i as f32) * 0.11).collect();
        let ref_out = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        2,
                        5,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        let got = dev()
            .f504(
                &dev()
                    .f626(
                        &dev().f502(&v0),
                        &dev().f502(&v1),
                        &dev().f502(&v2),
                        1,
                        2,
                        5,
                        4,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&got, &ref_out, 1e-3);
    }

    // f626: kv_seq < q_seq must error (same constraint as f623).
    #[test]
    fn f626_kv_smaller_than_q_errors() {
        let v0 = dev().f502(&vec![0.0f32; 12]);
        assert!(dev().f626(&v0, &v0, &v0, 1, 3, 2, 4).is_err());
    }

    // f626 wave64 path: head_dim=64, prefill — exercises dpl=1, single-element-per-lane.
    #[test]
    fn f626_w64_hd64_matches_f623() {
        let bh = 2u32;
        let seq = 4u32;
        let dk = 64u32;
        let n = (bh * seq * dk) as usize;
        let nkv = (bh * seq * dk) as usize;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let k: Vec<f32> = (0..nkv).map(|i| (i as f32) * 0.007 + 0.1).collect();
        let v: Vec<f32> = (0..nkv).map(|i| (i as f32) * 0.013 - 0.3).collect();
        let ref_out = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&q),
                        &dev().f502(&k),
                        &dev().f502(&v),
                        bh,
                        seq,
                        seq,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        let got = dev()
            .f504(
                &dev()
                    .f626(
                        &dev().f502(&q),
                        &dev().f502(&k),
                        &dev().f502(&v),
                        bh,
                        seq,
                        seq,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&got, &ref_out, 1e-3);
    }

    // f626 wave64 path: head_dim=128, decode (q_seq=1, kv_seq=16).
    #[test]
    fn f626_w64_hd128_decode_matches_f623() {
        let bh = 4u32;
        let kv = 16u32;
        let dk = 128u32;
        let q: Vec<f32> = (0..(bh * 1 * dk) as usize)
            .map(|i| (i as f32) * 0.005 - 0.3)
            .collect();
        let k: Vec<f32> = (0..(bh * kv * dk) as usize)
            .map(|i| (i as f32) * 0.003 + 0.1)
            .collect();
        let v: Vec<f32> = (0..(bh * kv * dk) as usize)
            .map(|i| (i as f32) * 0.007 - 0.2)
            .collect();
        let ref_out = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&q),
                        &dev().f502(&k),
                        &dev().f502(&v),
                        bh,
                        1,
                        kv,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        let got = dev()
            .f504(
                &dev()
                    .f626(
                        &dev().f502(&q),
                        &dev().f502(&k),
                        &dev().f502(&v),
                        bh,
                        1,
                        kv,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&got, &ref_out, 1e-3);
    }

    // f626 wave64 path: head_dim=128, large prefill — 32 heads, 64 queries, 64 KV.
    #[test]
    fn f626_w64_hd128_prefill_matches_f623() {
        let bh = 32u32;
        let seq = 64u32;
        let dk = 128u32;
        let n = (bh * seq * dk) as usize;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0008 + 0.05).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0012 - 0.1).collect();
        let ref_out = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&q),
                        &dev().f502(&k),
                        &dev().f502(&v),
                        bh,
                        seq,
                        seq,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        let got = dev()
            .f504(
                &dev()
                    .f626(
                        &dev().f502(&q),
                        &dev().f502(&k),
                        &dev().f502(&v),
                        bh,
                        seq,
                        seq,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();
        f544(&got, &ref_out, 1e-3);
    }

    // --- f627 / f628 = split_heads / merge_heads ---

    // split_heads then merge_heads must be the identity.
    #[test]
    fn f627_f628_roundtrip() {
        let n_heads = 2u32;
        let seq = 3u32;
        let hd = 4u32;
        let v0: Vec<f32> = (0..(seq * n_heads * hd)).map(|i| i as f32).collect(); // [seq, n*hd]
        let split = dev().f627(&dev().f502(&v0), n_heads, seq, hd).unwrap();
        let merged = dev().f628(&split, n_heads, seq, hd).unwrap();
        let got = dev().f504(&merged).unwrap();
        f544(&got, &v0, 1e-6);
    }

    // split_heads: element (s=1, bh=0, d=2) should come from position s*(n*hd)+bh*hd+d.
    #[test]
    fn f627_known_index() {
        // n_heads=2, seq=2, head_dim=3. Input [seq, n*hd] = [2, 6].
        // Row 0: [0,1,2,3,4,5]. Row 1: [10,11,12,13,14,15].
        // split output layout [n_heads, seq, head_dim]:
        //   head 0: [input[0,0..3], input[1,0..3]] = [[0,1,2],[10,11,12]]
        //   head 1: [input[0,3..6], input[1,3..6]] = [[3,4,5],[13,14,15]]
        let v0 = vec![0.0f32, 1., 2., 3., 4., 5., 10., 11., 12., 13., 14., 15.];
        let split = dev().f627(&dev().f502(&v0), 2, 2, 3).unwrap();
        let got = dev().f504(&split).unwrap();
        // [head0_seq0, head0_seq1, head1_seq0, head1_seq1] each of len 3
        assert_eq!(got[0..3], [0.0, 1., 2.]); // head 0, seq 0
        assert_eq!(got[3..6], [10., 11., 12.]); // head 0, seq 1
        assert_eq!(got[6..9], [3., 4., 5.]); // head 1, seq 0
        assert_eq!(got[9..12], [13., 14., 15.]); // head 1, seq 1
    }

    // --- f628 = merge_heads dedicated tests ---

    #[test]
    fn f628_known_index() {
        // Inverse of f627_known_index: input is [n_heads, seq, head_dim], output is [seq, n*hd].
        // n_heads=2, seq=2, head_dim=3.
        // head 0: seq0=[0,1,2], seq1=[10,11,12]. head 1: seq0=[3,4,5], seq1=[13,14,15].
        let v0 = vec![0.0f32, 1., 2., 10., 11., 12., 3., 4., 5., 13., 14., 15.]; // [2,2,3]
        let got = dev()
            .f504(&dev().f628(&dev().f502(&v0), 2, 2, 3).unwrap())
            .unwrap();
        // Output [seq, n*hd] = [2, 6]: row0=[0,1,2,3,4,5], row1=[10,11,12,13,14,15]
        assert_eq!(got[0..6], [0.0, 1., 2., 3., 4., 5.]);
        assert_eq!(got[6..12], [10., 11., 12., 13., 14., 15.]);
    }

    #[test]
    fn f628_size_mismatch() {
        let v0 = dev().f502(&[1.0f32; 11]); // 11 ≠ 2*3*2=12
        assert!(dev().f628(&v0, 2, 3, 2).is_err());
    }

    // --- f629 = repeat_kv ---

    // repeat_kv with n_rep=1 (MHA) must be a copy.
    #[test]
    fn f629_rep1_is_copy() {
        let v0: Vec<f32> = (0..12).map(|i| i as f32).collect(); // [2 kv_heads, 2 seq, 3 hd]
        let out = dev().f629(&dev().f502(&v0), 2, 2, 2, 3).unwrap(); // n_heads=n_kv_heads=2
        let got = dev().f504(&out).unwrap();
        f544(&got, &v0, 1e-6);
    }

    // repeat_kv with n_rep=2: [1 kv_head, 2 seq, 2 hd] → [2 heads, 2 seq, 2 hd].
    #[test]
    fn f629_rep2_known() {
        let v0 = vec![1.0f32, 2., 3., 4.]; // [1, 2, 2]
        let out = dev().f629(&dev().f502(&v0), 2, 1, 2, 2).unwrap(); // n_heads=2, n_kv=1
        let got = dev().f504(&out).unwrap();
        // Both head 0 and head 1 should be copies of the single kv head
        f544(&got[..4], &v0, 1e-6);
        f544(&got[4..], &v0, 1e-6);
    }

    // --- f631 = rope_batch_decode ---

    #[test]
    fn f631_matches_serial_f625() {
        // Two batch elements (B=2), num_heads=2, head_dim=4.
        // Element 0 start_pos=3, element 1 start_pos=7.
        // Compare f631 output to running f625 twice with respective start_pos.
        let nh = 2u32;
        let hd = 4u32;
        let b = 2u32;
        let base = 10000.0f32;
        let input: Vec<f32> = (0..(b * nh * hd) as usize)
            .map(|i| (i as f32) * 0.13 - 1.0)
            .collect();
        let start_pos_buf = dev().f502(&[3.0f32, 7.0]);

        let batch_out = dev()
            .f504(
                &dev()
                    .f631(&dev().f502(&input), &start_pos_buf, b * nh, nh, hd, base)
                    .unwrap(),
            )
            .unwrap();

        // Serial reference: run f625 for each batch element's heads separately.
        // Batch element 0: heads 0..nh → input[0..nh*hd], start_pos=3
        let serial_0 = dev()
            .f504(
                &dev()
                    .f625(
                        &dev().f502(&input[..(nh * hd) as usize]),
                        nh,
                        1,
                        hd,
                        3,
                        base,
                    )
                    .unwrap(),
            )
            .unwrap();
        // Batch element 1: heads nh..2nh → input[nh*hd..2nh*hd], start_pos=7
        let serial_1 = dev()
            .f504(
                &dev()
                    .f625(
                        &dev().f502(&input[(nh * hd) as usize..]),
                        nh,
                        1,
                        hd,
                        7,
                        base,
                    )
                    .unwrap(),
            )
            .unwrap();

        let (lo, hi) = batch_out.split_at((nh * hd) as usize);
        f544(lo, &serial_0, 1e-4);
        f544(hi, &serial_1, 1e-4);
    }

    #[test]
    fn f631_position_zero_is_identity() {
        // start_pos=0 for all → RoPE at position 0 is identity.
        let nh = 2u32;
        let hd = 4u32;
        let b = 2u32;
        let input: Vec<f32> = (0..(b * nh * hd) as usize)
            .map(|i| i as f32 * 0.1)
            .collect();
        let sp = dev().f502(&[0.0f32, 0.0]);
        let out = dev()
            .f504(
                &dev()
                    .f631(&dev().f502(&input), &sp, b * nh, nh, hd, 10000.0)
                    .unwrap(),
            )
            .unwrap();
        f544(&out, &input, 1e-5);
    }

    // --- f630 = fused_sdpa_batch_decode ---

    // CPU reference for batch decode SDPA: for each (b, h_q), full attention over kv_len positions.
    fn cpu_batch_decode_sdpa(
        q: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        kv_lens: &[u32],
        active_batch: usize,
        num_heads: usize,
        num_kv_heads: usize,
        max_kv_seq: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_rep = num_heads / num_kv_heads;
        let slot_stride = num_kv_heads * max_kv_seq * head_dim;
        let mut out = vec![0.0f32; active_batch * num_heads * head_dim];
        for b in 0..active_batch {
            let kv_len = kv_lens[b] as usize;
            for h_q in 0..num_heads {
                let h_kv = h_q / n_rep;
                let q_base = (b * num_heads + h_q) * head_dim;
                let kv_base = b * slot_stride + h_kv * max_kv_seq * head_dim;
                let mut scores: Vec<f32> = (0..kv_len)
                    .map(|j| {
                        let k_row = kv_base + j * head_dim;
                        (0..head_dim)
                            .map(|d| q[q_base + d] * k_all[k_row + d])
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|&s| (s - mx).exp()).sum();
                let attn: Vec<f32> = scores.iter().map(|&s| (s - mx).exp() / sum).collect();
                let out_base = (b * num_heads + h_q) * head_dim;
                for d in 0..head_dim {
                    out[out_base + d] = (0..kv_len)
                        .map(|j| {
                            let v_row = kv_base + j * head_dim;
                            attn[j] * v_all[v_row + d]
                        })
                        .sum();
                }
            }
        }
        out
    }

    #[test]
    fn f630_batch_vs_serial_mha() {
        // B=2, Nh=Nkv=2 (MHA), max_kv_seq=8, head_dim=64 (wave64 path).
        // kv_lens=[3,5]. Compare against CPU reference.
        let (b, nh, nkv, mkv, hd) = (2usize, 2usize, 2usize, 8usize, 64usize);
        let total_q = b * nh;
        let slot_stride = nkv * mkv * hd;
        let q: Vec<f32> = (0..total_q * hd).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let k_all: Vec<f32> = (0..b * slot_stride)
            .map(|i| (i as f32) * 0.005 - 0.3)
            .collect();
        let v_all: Vec<f32> = (0..b * slot_stride)
            .map(|i| (i as f32) * 0.007 + 0.1)
            .collect();
        let kv_lens_cpu = vec![3u32, 5u32];
        let kv_lens_f32 = vec![3.0f32, 5.0];

        let cpu_out = cpu_batch_decode_sdpa(&q, &k_all, &v_all, &kv_lens_cpu, b, nh, nkv, mkv, hd);
        let gpu_out = dev()
            .f504(
                &dev()
                    .f630(
                        &dev().f502(&q),
                        &dev().f502(&k_all),
                        &dev().f502(&v_all),
                        &dev().f502(&kv_lens_f32),
                        b as u32,
                        nh as u32,
                        nkv as u32,
                        mkv as u32,
                        hd as u32,
                    )
                    .unwrap(),
            )
            .unwrap();

        f544(&gpu_out, &cpu_out, 1e-2);
    }

    #[test]
    fn f630_batch_vs_serial_gqa() {
        // B=2, Nh=4, Nkv=2 (GQA n_rep=2), max_kv_seq=6, head_dim=64.
        // kv_lens=[4,2].
        let (b, nh, nkv, mkv, hd) = (2usize, 4usize, 2usize, 6usize, 64usize);
        let total_q = b * nh;
        let slot_stride = nkv * mkv * hd;
        let q: Vec<f32> = (0..total_q * hd).map(|i| (i as f32) * 0.02 - 1.0).collect();
        let k_all: Vec<f32> = (0..b * slot_stride)
            .map(|i| (i as f32) * 0.01 - 0.4)
            .collect();
        let v_all: Vec<f32> = (0..b * slot_stride)
            .map(|i| (i as f32) * 0.015 + 0.2)
            .collect();
        let kv_lens_cpu = vec![4u32, 2u32];
        let kv_lens_f32 = vec![4.0f32, 2.0];

        let cpu_out = cpu_batch_decode_sdpa(&q, &k_all, &v_all, &kv_lens_cpu, b, nh, nkv, mkv, hd);
        let gpu_out = dev()
            .f504(
                &dev()
                    .f630(
                        &dev().f502(&q),
                        &dev().f502(&k_all),
                        &dev().f502(&v_all),
                        &dev().f502(&kv_lens_f32),
                        b as u32,
                        nh as u32,
                        nkv as u32,
                        mkv as u32,
                        hd as u32,
                    )
                    .unwrap(),
            )
            .unwrap();

        f544(&gpu_out, &cpu_out, 1e-2);
    }
}
