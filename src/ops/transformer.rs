// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// f670=embedding_lookup, f671=argmax. Transformer inference primitives.
// t534=KVCache + f672..f677 KV cache methods (Sprint 7 step 2).
//
// Token IDs are passed as f32 — exact for vocab sizes up to 2^24 (~16.7M), which covers
// every published tokenizer. A typed u32 buffer is a later refactor; the f32 path keeps
// the t501 surface uniform.

use crate::device::{t500, t501};
use anyhow::{Result, ensure};

/// t524 = EmbedParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t524 {
    n: u32,
    n_ids: u32,
    d_model: u32,
    vocab_size: u32,
}

// One thread per output element. Reads token id, looks up that row of the weight matrix.
// Out-of-range ids are clamped to the last vocab entry rather than corrupting memory.
const SHADER_EMBED: &str = "
struct P { n: u32, n_ids: u32, d_model: u32, vocab_size: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ids: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.n { return; }
    let id_idx = idx / p.d_model;
    let d = idx % p.d_model;
    let token_id = min(u32(ids[id_idx]), p.vocab_size - 1u);
    out[idx] = weights[token_id * p.d_model + d];
}
";

/// t525 = ArgmaxParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t525 {
    rows: u32,
    cols: u32,
    _pad: [u32; 2],
}

// One thread per row. Scans cols, returns the index of the max as f32.
// Tie-breaking: lowest index wins (only `>` updates the running best).
const SHADER_ARGMAX: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x + gid.y * 65535u * 256u;
    if r >= p.rows { return; }
    let base = r * p.cols;
    var best_idx: u32 = 0u;
    var best_val: f32 = input[base];
    for (var c: u32 = 1u; c < p.cols; c++) {
        let v = input[base + c];
        if v > best_val {
            best_val = v;
            best_idx = c;
        }
    }
    out[r] = f32(best_idx);
}
";

/// t526 = EmbedBwdParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t526 {
    n_ids: u32,
    vocab_size: u32,
    d_model: u32,
    _pad: u32,
}

/// t527 = SoftmaxBwdParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t527 {
    rows: u32,
    cols: u32,
    _pad: [u32; 2],
}

/// t528 = RopeBwdParams. Same layout as attention::t536.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t528 {
    batch_heads: u32,
    seq_len: u32,
    head_dim: u32,
    start_pos: u32,
    base: f32,
    _pad: [u32; 3],
}

// Embedding backward: scatter-add grad_output into grad_weight using f32 CAS atomics.
// Output declared as array<atomic<u32>> so f32 bits are written via bitcast + CAS loop.
const SHADER_EMBED_BWD: &str = "
struct P { n_ids: u32, vocab_size: u32, d_model: u32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read> ids:      array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_w: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.n_ids * p.d_model;
    if idx >= total { return; }
    let id_idx    = idx / p.d_model;
    let d         = idx % p.d_model;
    let token_id  = min(u32(ids[id_idx]), p.vocab_size - 1u);
    let w_idx     = token_id * p.d_model + d;
    let val       = grad_out[idx];
    var old_bits  = atomicLoad(&grad_w[w_idx]);
    loop {
        let new_val  = bitcast<f32>(old_bits) + val;
        let new_bits = bitcast<u32>(new_val);
        let cas      = atomicCompareExchangeWeak(&grad_w[w_idx], old_bits, new_bits);
        if cas.exchanged { break; }
        old_bits = cas.old_value;
    }
}
";

// Softmax backward pass 1: one thread per row, compute dot = sum(grad * p, over cols).
const SHADER_SM_BWD_DOT: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> grad:    array<f32>;
@group(0) @binding(2) var<storage, read> prob:    array<f32>;
@group(0) @binding(3) var<storage, read_write> dot_buf: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= p.rows { return; }
    let base = r * p.cols;
    var d: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cols; c++) {
        d += grad[base + c] * prob[base + c];
    }
    dot_buf[r] = d;
}
";

// Softmax backward pass 2: one thread per element.
const SHADER_SM_BWD_DX: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> grad:    array<f32>;
@group(0) @binding(2) var<storage, read> prob:    array<f32>;
@group(0) @binding(3) var<storage, read> dot_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_input: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.rows * p.cols;
    if idx >= total { return; }
    let r = idx / p.cols;
    d_input[idx] = prob[idx] * (grad[idx] - dot_buf[r]);
}
";

// RoPE backward: identical to forward but sin terms negated (conjugate rotation).
const SHADER_ROPE_BWD: &str = "
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

    let exponent = -2.0 * f32(pair_idx) / f32(p.head_dim);
    let inv_freq = pow(p.base, exponent);
    let angle = pos_f * inv_freq;
    let cos_a =  cos(angle);
    let sin_a = -sin(angle);

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

impl t500 {
    /// f670 = embedding_lookup. For each token id in `p0`, return that row of `p1`.
    /// `p1`: [vocab_size, d_model]. `p0`: [n_ids] (f32 holding integer values).
    /// Returns [n_ids, d_model]. For a transformer with input shape [batch, seq],
    /// pass n_ids = batch * seq.
    pub fn f670(&self, p0: &t501, p1: &t501, p2: u32, p3: u32, p4: u32) -> Result<t501> {
        ensure!(
            p0.s507 == p2 as usize,
            "embedding: ids len {} != n_ids {}",
            p0.s507,
            p2
        );
        ensure!(
            p1.s507 == (p3 * p4) as usize,
            "embedding: weights len {} != vocab_size*d_model {}",
            p1.s507,
            p3 * p4,
        );
        ensure!(p3 > 0, "embedding: vocab_size must be > 0");

        let v0 = p2 * p4;
        let v1 = self.f503(v0 as usize);
        let v2 = t524 {
            n: v0,
            n_ids: p2,
            d_model: p4,
            vocab_size: p3,
        };
        self.f543(
            SHADER_EMBED,
            Some("embedding_lookup"),
            &v2,
            &[p0, p1],
            &v1,
            super::f540(v0),
        );
        Ok(v1)
    }

    /// f671 = argmax. Along the last dim: input [rows, cols] -> indices [rows] (as f32).
    /// For LM-head logits with shape [batch*seq, vocab_size], returns one token id per row.
    pub fn f671(&self, p0: &t501, p1: u32, p2: u32) -> Result<t501> {
        ensure!(p0.s507 == (p1 * p2) as usize, "argmax: input size mismatch");
        ensure!(p2 > 0, "argmax: cols must be > 0");
        let v0 = self.f503(p1 as usize);
        let v1 = t525 {
            rows: p1,
            cols: p2,
            _pad: [0; 2],
        };
        self.f543(
            SHADER_ARGMAX,
            Some("argmax"),
            &v1,
            &[p0],
            &v0,
            super::f540(p1),
        );
        Ok(v0)
    }

    /// f793 = embedding_backward. Scatter-add grad_output into grad_weight.
    /// p0=grad_output[n_ids,d_model], p1=ids[n_ids] (f32), p2=n_ids, p3=vocab_size, p4=d_model.
    pub fn f793(&self, p0: &t501, p1: &t501, p2: u32, p3: u32, p4: u32) -> Result<t501> {
        ensure!(
            p0.s507 == (p2 * p4) as usize,
            "embed_bwd: grad_output size mismatch"
        );
        ensure!(p1.s507 == p2 as usize, "embed_bwd: ids size mismatch");
        ensure!(p3 > 0, "embed_bwd: vocab_size must be > 0");

        let v0 = (p3 * p4) as usize;
        let v1 = self.f503(v0);
        // wgpu guarantees zero-initialization on create_buffer — no explicit clear needed.

        let v3 = t526 {
            n_ids: p2,
            vocab_size: p3,
            d_model: p4,
            _pad: 0,
        };
        let v4 = self.f506(&v3);
        let v5 = self.f507(SHADER_EMBED_BWD, Some("embed_bwd"));
        let v6 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &v5.get_bind_group_layout(0),
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
                    resource: p1.s505.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v1.s505.as_entire_binding(),
                },
            ],
        });
        let (v7, v8, v9) = super::f540(p2 * p4);
        let mut v10 = self
            .s500
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut v11 = v10.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed_bwd"),
                timestamp_writes: None,
            });
            v11.set_pipeline(&v5);
            v11.set_bind_group(0, &v6, &[]);
            v11.dispatch_workgroups(v7, v8, v9);
        }
        self.s501.submit(Some(v10.finish()));
        Ok(v1)
    }

    /// f794 = softmax_backward. p0=grad_output[rows,cols], p1=fwd_softmax_output[rows,cols].
    pub fn f794(&self, p0: &t501, p1: &t501, p2: u32, p3: u32) -> Result<t501> {
        ensure!(
            p0.s507 == (p2 * p3) as usize,
            "sm_bwd: grad_output size mismatch"
        );
        ensure!(
            p1.s507 == (p2 * p3) as usize,
            "sm_bwd: fwd_output size mismatch"
        );

        let v0 = t527 {
            rows: p2,
            cols: p3,
            _pad: [0; 2],
        };

        // Pass 1: dot product per row.
        let v1 = self.f503(p2 as usize);
        self.f543(
            SHADER_SM_BWD_DOT,
            Some("sm_bwd_dot"),
            &v0,
            &[p0, p1],
            &v1,
            super::f540(p2),
        );

        // Pass 2: grad_input per element. 3 storage inputs → raw dispatch.
        let v2 = self.f503((p2 * p3) as usize);
        {
            let v3 = self.f506(&v0);
            let v4 = self.f507(SHADER_SM_BWD_DX, Some("sm_bwd_dx"));
            let v5 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &v4.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: v3.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: p0.s505.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p1.s505.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: v1.s505.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: v2.s505.as_entire_binding(),
                    },
                ],
            });
            let (v6, v7, v8) = super::f540(p2 * p3);
            let mut v9 = self
                .s500
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut v10 = v9.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sm_bwd_dx"),
                    timestamp_writes: None,
                });
                v10.set_pipeline(&v4);
                v10.set_bind_group(0, &v5, &[]);
                v10.dispatch_workgroups(v6, v7, v8);
            }
            self.s501.submit(Some(v9.finish()));
        }
        Ok(v2)
    }

    /// f796 = rope_backward. Conjugate RoPE rotation applied to grad_output.
    /// Same signature as f625. p0=grad_output[batch_heads,seq,head_dim].
    pub fn f796(&self, p0: &t501, p1: u32, p2: u32, p3: u32, p4: u32, p5: f32) -> Result<t501> {
        ensure!(p3 % 2 == 0, "rope_bwd: head_dim ({}) must be even", p3);
        ensure!(
            p0.s507 == (p1 * p2 * p3) as usize,
            "rope_bwd: input size mismatch"
        );
        ensure!(p5 > 0.0, "rope_bwd: base must be positive");

        let v0 = p1 * p2 * p3;
        let v1 = self.f503(v0 as usize);
        let v2 = t528 {
            batch_heads: p1,
            seq_len: p2,
            head_dim: p3,
            start_pos: p4,
            base: p5,
            _pad: [0; 3],
        };
        self.f543(
            SHADER_ROPE_BWD,
            Some("rope_bwd"),
            &v2,
            &[p0],
            &v1,
            super::f540(v0),
        );
        Ok(v1)
    }
}

// --- KVCache (Sprint 7 step 2) ---
//
// Append-only persistent K and V buffers for autoregressive decoding. Layout for both:
// `[batch_heads, max_seq, head_dim]` flat. `s513` (cursor) tracks how many tokens are valid.
// Prefill writes the full prompt in one append; each subsequent decode step appends one row.

/// t537 = KVAppendParams. Uniform for the f673 (append) shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t537 {
    batch_heads: u32,
    max_seq: u32,
    head_dim: u32,
    cursor: u32,
    new_tokens: u32,
    _pad: [u32; 3],
}

// Strided write: copies [bh, new_tokens, head_dim] from new_data into cache at
// position [bh, cursor..cursor+new_tokens, :].
const SHADER_KV_APPEND: &str = "
struct P {
    batch_heads: u32, max_seq: u32, head_dim: u32, cursor: u32,
    new_tokens: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> new_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch_heads * p.new_tokens * p.head_dim;
    if idx >= total { return; }

    let d = idx % p.head_dim;
    let s = (idx / p.head_dim) % p.new_tokens;
    let bh = idx / (p.new_tokens * p.head_dim);

    let cache_idx = bh * p.max_seq * p.head_dim + (p.cursor + s) * p.head_dim + d;
    cache[cache_idx] = new_data[idx];
}
";

/// t534 = KVCache. Append-only K and V storage for autoregressive decoding.
pub struct t534 {
    pub(crate) s511: t501, // k
    pub(crate) s512: t501, // v
    pub(crate) s513: u32,  // cursor (tokens stored)
    pub(crate) s514: u32,  // max_seq
    pub(crate) s515: u32,  // batch_heads
    pub(crate) s516: u32,  // head_dim
}

impl t534 {
    /// f672 = KVCache::new. Allocate empty cache for up to `p1` (max_seq) tokens.
    /// `p2` = batch_heads, `p3` = head_dim.
    pub fn f672(p0: &t500, p1: u32, p2: u32, p3: u32) -> Self {
        let v0 = (p2 * p1 * p3) as usize;
        let v1 = p0.f503(v0);
        let v2 = p0.f503(v0);
        Self {
            s511: v1,
            s512: v2,
            s513: 0,
            s514: p1,
            s515: p2,
            s516: p3,
        }
    }

    /// f673 = KVCache::append. Write `p1` (new K) and `p2` (new V), each
    /// shaped [batch_heads, p3 new_tokens, head_dim], at the current cursor; advance.
    pub fn f673(&mut self, p0: &t500, p1: &t501, p2: &t501, p3: u32) -> Result<()> {
        ensure!(
            self.s513 + p3 <= self.s514,
            "KVCache::append: cursor {} + new {} exceeds max_seq {}",
            self.s513,
            p3,
            self.s514
        );
        let v0 = (self.s515 * p3 * self.s516) as usize;
        ensure!(
            p1.s507 == v0,
            "KVCache::append: K size mismatch (got {}, expected {})",
            p1.s507,
            v0
        );
        ensure!(
            p2.s507 == v0,
            "KVCache::append: V size mismatch (got {}, expected {})",
            p2.s507,
            v0
        );

        let v1 = t537 {
            batch_heads: self.s515,
            max_seq: self.s514,
            head_dim: self.s516,
            cursor: self.s513,
            new_tokens: p3,
            _pad: [0; 3],
        };
        let v2 = self.s515 * p3 * self.s516;

        // Append into K
        p0.f543(
            SHADER_KV_APPEND,
            Some("kv_append_k"),
            &v1,
            &[p1],
            &self.s511,
            super::f540(v2),
        );
        // Append into V
        p0.f543(
            SHADER_KV_APPEND,
            Some("kv_append_v"),
            &v1,
            &[p2],
            &self.s512,
            super::f540(v2),
        );

        self.s513 += p3;
        Ok(())
    }

    /// f674 = KVCache::reset. Move the cursor back to zero. Underlying buffers are
    /// not zeroed — subsequent appends overwrite from offset 0.
    pub fn f674(&mut self) {
        self.s513 = 0;
    }

    /// f675 = KVCache::cursor. Number of tokens currently stored.
    pub fn f675(&self) -> u32 {
        self.s513
    }

    /// f676 = KVCache::k_buffer. Borrow the underlying K buffer.
    pub fn f676(&self) -> &t501 {
        &self.s511
    }

    /// f677 = KVCache::v_buffer. Borrow the underlying V buffer.
    pub fn f677(&self) -> &t501 {
        &self.s512
    }
}

// --- BatchKvPool (t551) ---
//
// t551 = BatchKvPool. Pooled K/V storage for up to `max_batch` concurrent decode requests.
// Layout: K (and V) are [max_batch * num_kv_heads, max_seq, head_dim] flat.
//   Slot `s`, kv-head `h` occupies row index `s * num_kv_heads + h`.
// Cursors are CPU-tracked; GPU writes use the cursor to find the insertion point.
//
// f800 = BatchKvPool::new
// f801 = BatchKvPool::copy_from_kvcache  -- copy a prefilled t534 into a slot
// f802 = BatchKvPool::reset_slot         -- zero cursor for a slot
// f803 = BatchKvPool::cursor             -- get cursor for a slot
// f804k / f804v = BatchKvPool::k_buf / v_buf
// f805 = BatchKvPool::batch_decode_append -- append one decode token per active slot
// f806 = BatchKvPool::kv_lens_buf        -- upload cursors as f32 GPU buffer
// f807 = BatchKvPool::migrate_slot       -- copy KV from slot src→dst (for compaction)

/// t552 = CopyToSlotParams. Copies a single-slot KV (t534) into a BatchKvPool slot.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t552 {
    num_kv_heads: u32,
    cursor: u32,
    head_dim: u32,
    max_seq_src: u32,
    max_seq_dst: u32,
    slot_idx: u32,
    _p0: u32,
    _p1: u32,
}

// src layout: kh * max_seq_src * hd + pos * hd + d
// dst layout: (slot * nkv_h + kh) * max_seq_dst * hd + pos * hd + d
const SHADER_COPY_TO_SLOT: &str = "
struct P {
    num_kv_heads: u32, cursor: u32, head_dim: u32, max_seq_src: u32,
    max_seq_dst: u32, slot_idx: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.num_kv_heads * p.cursor * p.head_dim;
    if idx >= total { return; }
    let d   = idx % p.head_dim;
    let pos = (idx / p.head_dim) % p.cursor;
    let kh  = idx / (p.cursor * p.head_dim);
    let src_idx = kh * p.max_seq_src * p.head_dim + pos * p.head_dim + d;
    let dst_kh  = p.slot_idx * p.num_kv_heads + kh;
    let dst_idx = dst_kh * p.max_seq_dst * p.head_dim + pos * p.head_dim + d;
    dst[dst_idx] = src[src_idx];
}
";

/// t553 = BatchDecodeAppendParams. Appends one decode token per active slot.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t553 {
    active_batch: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

// new_data: [active_batch * num_kv_heads * head_dim]  (one token per slot, all heads)
// cursors:  [active_batch] as f32  (cursor position for each slot before this append)
// cache:    [max_batch * num_kv_heads * max_seq * head_dim]
const SHADER_BATCH_DECODE_APPEND: &str = "
struct P {
    active_batch: u32, num_kv_heads: u32, head_dim: u32, max_seq: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       new_data: array<f32>;
@group(0) @binding(2) var<storage, read>       cursors:  array<f32>;
@group(0) @binding(3) var<storage, read_write> cache:    array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.active_batch * p.num_kv_heads * p.head_dim;
    if idx >= total { return; }
    let d    = idx % p.head_dim;
    let kh   = (idx / p.head_dim) % p.num_kv_heads;
    let slot = idx / (p.num_kv_heads * p.head_dim);
    let cursor = u32(cursors[slot]);
    let cache_idx = (slot * p.num_kv_heads + kh) * p.max_seq * p.head_dim
                  + cursor * p.head_dim + d;
    cache[cache_idx] = new_data[idx];
}
";

/// t557 = CopySlotParams. Shared params for slot-to-staging / staging-to-slot shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t557 {
    num_kv_heads: u32,
    cursor: u32,
    head_dim: u32,
    max_seq: u32,
    slot_idx: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

// pool[(slot_idx * nkv_h + kh) * max_seq * hd + pos * hd + d] -> staging[kh * max_seq * hd + ...]
const SHADER_SLOT_TO_STAGING: &str = "
struct P { num_kv_heads: u32, cursor: u32, head_dim: u32, max_seq: u32,
           slot_idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       pool:    array<f32>;
@group(0) @binding(2) var<storage, read_write> staging: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.num_kv_heads * p.cursor * p.head_dim;
    if idx >= total { return; }
    let d   = idx % p.head_dim;
    let pos = (idx / p.head_dim) % p.cursor;
    let kh  = idx / (p.cursor * p.head_dim);
    let pool_kh = p.slot_idx * p.num_kv_heads + kh;
    staging[idx] = pool[pool_kh * p.max_seq * p.head_dim + pos * p.head_dim + d];
}
";

// staging[kh * max_seq * hd + pos * hd + d] -> pool[(slot_idx * nkv_h + kh) * max_seq * hd + ...]
const SHADER_STAGING_TO_SLOT: &str = "
struct P { num_kv_heads: u32, cursor: u32, head_dim: u32, max_seq: u32,
           slot_idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       staging: array<f32>;
@group(0) @binding(2) var<storage, read_write> pool:    array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.num_kv_heads * p.cursor * p.head_dim;
    if idx >= total { return; }
    let d   = idx % p.head_dim;
    let pos = (idx / p.head_dim) % p.cursor;
    let kh  = idx / (p.cursor * p.head_dim);
    let pool_kh = p.slot_idx * p.num_kv_heads + kh;
    pool[pool_kh * p.max_seq * p.head_dim + pos * p.head_dim + d] = staging[idx];
}
";

/// t551 = BatchKvPool. Pooled KV storage for concurrent decode requests.
pub struct t551 {
    pub(crate) s530: t501,     // K: [max_batch * num_kv_heads * max_seq * head_dim]
    pub(crate) s531: t501,     // V: same shape
    pub(crate) s532: u32,      // max_batch
    pub(crate) s533: u32,      // num_kv_heads
    pub(crate) s534: u32,      // max_seq
    pub(crate) s535: u32,      // head_dim
    pub(crate) s536: Vec<u32>, // cursors[max_batch] -- CPU-side KV length per slot
    pub(crate) s537: t501,     // staging K: [num_kv_heads * max_seq * head_dim] for f807
    pub(crate) s538: t501,     // staging V: same
}

impl t551 {
    /// f800 = BatchKvPool::new. Allocate zeroed K/V for up to `p1` (max_batch) slots.
    /// `p2` = num_kv_heads, `p3` = max_seq, `p4` = head_dim.
    pub fn f800(p0: &t500, p1: u32, p2: u32, p3: u32, p4: u32) -> Self {
        let len = (p1 * p2 * p3 * p4) as usize;
        let staging_len = (p2 * p3 * p4) as usize; // one slot worth
        Self {
            s530: p0.f503(len),
            s531: p0.f503(len),
            s532: p1,
            s533: p2,
            s534: p3,
            s535: p4,
            s536: vec![0u32; p1 as usize],
            s537: p0.f503(staging_len),
            s538: p0.f503(staging_len),
        }
    }

    /// f801 = copy_from_kvcache. Copy a prefilled `p2` (t534) into slot `p1`.
    /// After this call, cursor(p1) == p2.cursor().
    pub fn f801(
        &mut self,
        p0: &t500,
        p1: usize,
        k_src: &t501,
        v_src: &t501,
        cursor: u32,
        max_seq_src: u32,
    ) -> Result<()> {
        ensure!(
            p1 < self.s532 as usize,
            "copy_from_kvcache: slot {} >= max_batch {}",
            p1,
            self.s532
        );
        ensure!(
            cursor <= self.s534,
            "copy_from_kvcache: cursor {} > pool max_seq {}",
            cursor,
            self.s534
        );
        if cursor == 0 {
            self.s536[p1] = 0;
            return Ok(());
        }
        let params = t552 {
            num_kv_heads: self.s533,
            cursor,
            head_dim: self.s535,
            max_seq_src,
            max_seq_dst: self.s534,
            slot_idx: p1 as u32,
            _p0: 0,
            _p1: 0,
        };
        let total = self.s533 * cursor * self.s535;
        p0.f543(
            SHADER_COPY_TO_SLOT,
            Some("copy_to_slot_k"),
            &params,
            &[k_src],
            &self.s530,
            super::f540(total),
        );
        p0.f543(
            SHADER_COPY_TO_SLOT,
            Some("copy_to_slot_v"),
            &params,
            &[v_src],
            &self.s531,
            super::f540(total),
        );
        self.s536[p1] = cursor;
        Ok(())
    }

    /// f802 = reset_slot. Zero the cursor for slot `p1`.
    pub fn f802(&mut self, p1: usize) {
        self.s536[p1] = 0;
    }

    /// f803 = cursor. Number of KV tokens stored in slot `p1`.
    pub fn f803(&self, p1: usize) -> u32 {
        self.s536[p1]
    }

    /// f804k = k_buf. Borrow the K buffer.
    pub fn f804k(&self) -> &t501 {
        &self.s530
    }

    /// f804v = v_buf. Borrow the V buffer.
    pub fn f804v(&self) -> &t501 {
        &self.s531
    }

    /// f805 = batch_decode_append. Append one new token per active slot (slots 0..`p3`).
    /// `p1` = new K: [p3 * num_kv_heads * head_dim] (i.e., [p3*nkv_h, head_dim] flat)
    /// `p2` = new V: same shape.
    /// Reads cursors from self.s536[0..p3], writes to cache, then advances each cursor by 1.
    pub fn f805(&mut self, p0: &t500, p1: &t501, p2: &t501, p3: usize) -> Result<()> {
        ensure!(
            p3 <= self.s532 as usize,
            "batch_decode_append: active_batch {} > max_batch {}",
            p3,
            self.s532
        );
        let cursor_data: Vec<f32> = self.s536[..p3].iter().map(|&c| c as f32).collect();
        let cursor_buf = p0.f502(&cursor_data);
        let params = t553 {
            active_batch: p3 as u32,
            num_kv_heads: self.s533,
            head_dim: self.s535,
            max_seq: self.s534,
            _p0: 0,
            _p1: 0,
            _p2: 0,
            _p3: 0,
        };
        let total = (p3 as u32) * self.s533 * self.s535;
        p0.f543(
            SHADER_BATCH_DECODE_APPEND,
            Some("batch_append_k"),
            &params,
            &[p1, &cursor_buf],
            &self.s530,
            super::f540(total),
        );
        p0.f543(
            SHADER_BATCH_DECODE_APPEND,
            Some("batch_append_v"),
            &params,
            &[p2, &cursor_buf],
            &self.s531,
            super::f540(total),
        );
        for s in 0..p3 {
            self.s536[s] += 1;
        }
        Ok(())
    }

    /// f806 = kv_lens_buf. Upload current cursors[0..`p1`] as a [p1] f32 GPU buffer.
    /// Returns the kv lengths AFTER any recent f805 call — pass to fused_sdpa_batch_decode.
    pub fn f806(&self, p0: &t500, p1: usize) -> t501 {
        let data: Vec<f32> = self.s536[..p1].iter().map(|&c| c as f32).collect();
        p0.f502(&data)
    }

    /// f807 = migrate_slot. Copy KV from pool slot `p1` (src) to pool slot `p2` (dst).
    /// Used by serve-side compaction when a slot completes out of order.
    /// Stages through `s537`/`s538` to avoid aliasing within the same buffer.
    pub fn f807(&mut self, p0: &t500, p1: usize, p2: usize) {
        if p1 == p2 {
            return;
        }
        let cursor = self.s536[p1];
        if cursor == 0 {
            self.s536[p2] = 0;
            return;
        }
        let total = self.s533 * cursor * self.s535;
        let params_src = t557 {
            num_kv_heads: self.s533,
            cursor,
            head_dim: self.s535,
            max_seq: self.s534,
            slot_idx: p1 as u32,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        let params_dst = t557 {
            slot_idx: p2 as u32,
            ..params_src
        };
        // pool[src] → staging
        p0.f543(
            SHADER_SLOT_TO_STAGING,
            Some("slot_to_staging_k"),
            &params_src,
            &[&self.s530],
            &self.s537,
            super::f540(total),
        );
        p0.f543(
            SHADER_SLOT_TO_STAGING,
            Some("slot_to_staging_v"),
            &params_src,
            &[&self.s531],
            &self.s538,
            super::f540(total),
        );
        // staging → pool[dst]
        p0.f543(
            SHADER_STAGING_TO_SLOT,
            Some("staging_to_slot_k"),
            &params_dst,
            &[&self.s537],
            &self.s530,
            super::f540(total),
        );
        p0.f543(
            SHADER_STAGING_TO_SLOT,
            Some("staging_to_slot_v"),
            &params_dst,
            &[&self.s538],
            &self.s531,
            super::f540(total),
        );
        self.s536[p2] = cursor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;
    fn dev() -> &'static t500 {
        &crate::ops::TEST_DEV
    }

    // --- embedding_lookup ---

    #[test]
    fn f670_basic() {
        let v0 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let v1 = vec![2.0f32, 0.0, 3.0];
        let v2 = dev()
            .f504(
                &dev()
                    .f670(&dev().f502(&v1), &dev().f502(&v0), 3, 4, 3)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(v2, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn f670_single_token() {
        let v0: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let v1 = vec![7.0f32];
        let v2 = dev()
            .f504(
                &dev()
                    .f670(&dev().f502(&v1), &dev().f502(&v0), 1, 10, 5)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(v2, vec![35.0, 36.0, 37.0, 38.0, 39.0]);
    }

    #[test]
    fn f670_repeat_token() {
        let v0 = vec![10.0, 20.0, 99.0, 88.0];
        let v1 = vec![1.0f32, 1.0, 1.0];
        let v2 = dev()
            .f504(
                &dev()
                    .f670(&dev().f502(&v1), &dev().f502(&v0), 3, 2, 2)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(v2, vec![99.0, 88.0, 99.0, 88.0, 99.0, 88.0]);
    }

    #[test]
    fn f670_out_of_range_clamped() {
        let v0 = vec![1.0, 2.0, 3.0, 4.0];
        let v1 = vec![5.0f32];
        let v2 = dev()
            .f504(
                &dev()
                    .f670(&dev().f502(&v1), &dev().f502(&v0), 1, 2, 2)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(v2, vec![3.0, 4.0]);
    }

    #[test]
    fn f670_transformer_shape() {
        let v0 = 1024;
        let v1 = 128;
        let v2 = 4;
        let v3: Vec<f32> = (0..v0 * v1).map(|i| (i as f32) * 0.001).collect();
        let v4 = vec![10.0f32, 500.0, 0.0, 1023.0];
        let v5 = dev()
            .f504(
                &dev()
                    .f670(&dev().f502(&v4), &dev().f502(&v3), v2, v0 as u32, v1 as u32)
                    .unwrap(),
            )
            .unwrap();
        for (v6, &v7) in (0..v2 as usize).zip(&v4) {
            let v8 = v7 as usize;
            for v9 in 0..v1 {
                let v10 = v3[v8 * v1 + v9];
                let v11 = v5[v6 * v1 + v9];
                assert!(
                    (v11 - v10).abs() < 1e-6,
                    "row {v6} col {v9}: got {v11} expected {v10}"
                );
            }
        }
    }

    #[test]
    fn f670_size_mismatch() {
        let v0 = vec![1.0; 6];
        let v1 = vec![0.0f32, 1.0];
        assert!(
            dev()
                .f670(&dev().f502(&v1), &dev().f502(&v0), 2, 3, 3)
                .is_err()
        );
    }

    #[test]
    fn f670_ids_length_mismatch() {
        let v0 = vec![1.0; 6];
        let v1 = vec![0.0f32, 1.0, 1.0];
        assert!(
            dev()
                .f670(&dev().f502(&v1), &dev().f502(&v0), 2, 2, 3)
                .is_err()
        );
    }

    // --- argmax ---

    #[test]
    fn f671_basic() {
        let v0 = vec![
            1.0f32, 5.0, 2.0, 3.0, 7.0, 1.0, 9.0, 8.0, -1.0, -5.0, -3.0, -2.0,
        ];
        let v1 = dev()
            .f504(&dev().f671(&dev().f502(&v0), 3, 4).unwrap())
            .unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn f671_single_row() {
        let v0 = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let v1 = dev()
            .f504(&dev().f671(&dev().f502(&v0), 1, 7).unwrap())
            .unwrap();
        assert_eq!(v1, vec![5.0]);
    }

    #[test]
    fn f671_single_column() {
        let v0 = vec![42.0f32, -7.0, 100.0];
        let v1 = dev()
            .f504(&dev().f671(&dev().f502(&v0), 3, 1).unwrap())
            .unwrap();
        assert_eq!(v1, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn f671_ties_pick_lowest_index() {
        let v0 = vec![5.0f32, 5.0, 5.0, 5.0];
        let v1 = dev()
            .f504(&dev().f671(&dev().f502(&v0), 1, 4).unwrap())
            .unwrap();
        assert_eq!(v1, vec![0.0]);
    }

    #[test]
    fn f671_lm_head_shape() {
        let v0 = 8u32;
        let v1 = 32u32;
        let mut v2 = vec![0.0f32; (v0 * v1) as usize];
        let mut v3 = Vec::with_capacity(v0 as usize);
        for v4 in 0..v0 {
            let v5 = (v4 * 3 + 7) % v1;
            for v6 in 0..v1 {
                v2[(v4 * v1 + v6) as usize] = (v6 as f32) * 0.01;
            }
            v2[(v4 * v1 + v5) as usize] = 99.0;
            v3.push(v5 as f32);
        }
        let v7 = dev()
            .f504(&dev().f671(&dev().f502(&v2), v0, v1).unwrap())
            .unwrap();
        f544(&v7, &v3, 1e-6);
    }

    #[test]
    fn f671_size_mismatch() {
        let v0 = vec![1.0f32; 10];
        assert!(dev().f671(&dev().f502(&v0), 3, 4).is_err());
    }

    #[test]
    fn f671_zero_cols() {
        let v0 = vec![1.0f32; 0];
        assert!(dev().f671(&dev().f502(&v0), 3, 0).is_err());
    }

    // --- KVCache (f672..f677) ---

    #[test]
    fn f672_starts_empty() {
        let v0 = t534::f672(dev(), 16, 2, 4);
        assert_eq!(v0.f675(), 0);
        // K and V buffers should be alloc'd at full size and zero-initialized
        assert_eq!(v0.f676().s507, 2 * 16 * 4);
        assert_eq!(v0.f677().s507, 2 * 16 * 4);
    }

    #[test]
    fn f673_append_single_token() {
        // 1 head, max_seq=4, head_dim=2 -> cache layout flat: 8 floats per K/V
        let mut v0 = t534::f672(dev(), 4, 1, 2);
        let v1 = dev().f502(&[7.0f32, 8.0]); // [bh=1, new=1, dim=2]
        let v2 = dev().f502(&[70.0f32, 80.0]);
        v0.f673(dev(), &v1, &v2, 1).unwrap();
        assert_eq!(v0.f675(), 1);
        let v3 = dev().f504(v0.f676()).unwrap();
        let v4 = dev().f504(v0.f677()).unwrap();
        // First two slots of K = [7, 8], rest stays zero
        assert_eq!(&v3[..2], &[7.0, 8.0]);
        assert!(v3[2..].iter().all(|&x| x == 0.0));
        assert_eq!(&v4[..2], &[70.0, 80.0]);
        assert!(v4[2..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn f673_append_multi_then_single() {
        // Prefill 3 tokens, then decode-step 1 token. Verify layout.
        let mut v0 = t534::f672(dev(), 8, 1, 2);
        let v1 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]); // 3 tokens
        let v2 = dev().f502(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]);
        v0.f673(dev(), &v1, &v2, 3).unwrap();
        assert_eq!(v0.f675(), 3);

        let v3 = dev().f502(&[99.0f32, 100.0]); // 1 token
        let v4 = dev().f502(&[999.0f32, 1000.0]);
        v0.f673(dev(), &v3, &v4, 1).unwrap();
        assert_eq!(v0.f675(), 4);

        let v5 = dev().f504(v0.f676()).unwrap();
        assert_eq!(&v5[..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0, 100.0]);
        // Slots 8..16 (positions 4..7) still zero
        assert!(v5[8..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn f673_multi_head_layout() {
        // 2 batch_heads, max_seq=3, head_dim=2 -> 12 floats per K/V.
        // Append [bh=2, new=1, dim=2] = [head0: 1,2  head1: 3,4]
        // Cache layout: [bh=0, max=3, dim=2] flat = [1,2,0,0,0,0,3,4,0,0,0,0]
        let mut v0 = t534::f672(dev(), 3, 2, 2);
        let v1 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0]);
        let v2 = dev().f502(&[10.0f32, 20.0, 30.0, 40.0]);
        v0.f673(dev(), &v1, &v2, 1).unwrap();
        let v3 = dev().f504(v0.f676()).unwrap();
        assert_eq!(
            v3,
            vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn f674_reset_zeroes_cursor() {
        let mut v0 = t534::f672(dev(), 4, 1, 2);
        let v1 = dev().f502(&[1.0f32, 2.0]);
        let v2 = dev().f502(&[1.0f32, 2.0]);
        v0.f673(dev(), &v1, &v2, 1).unwrap();
        assert_eq!(v0.f675(), 1);
        v0.f674();
        assert_eq!(v0.f675(), 0);
    }

    #[test]
    fn f673_overflow_errors() {
        let mut v0 = t534::f672(dev(), 2, 1, 2);
        let v1 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]); // 3 tokens > max_seq=2
        let v2 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(v0.f673(dev(), &v1, &v2, 3).is_err());
    }

    #[test]
    fn f673_kv_cache_then_causal_attention() {
        // End-to-end: prefill 2 tokens, decode 1 token, run causal SDPA over the cached buffers.
        // Compare against a direct CPU reference using the full 3-token K/V.
        let dk = 4u32;
        let max_seq = 8u32;
        let mut v0 = t534::f672(dev(), max_seq, 1, dk);

        // Prefill: K_prefill [2, 4], V_prefill [2, 4]
        let k_prefill: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
        let v_prefill: Vec<f32> = (0..8).map(|i| (i as f32) * 0.2 - 0.5).collect();
        v0.f673(dev(), &dev().f502(&k_prefill), &dev().f502(&v_prefill), 2)
            .unwrap();

        // Decode step: append 1 more token
        let k_decode = vec![0.7f32, 0.8, 0.9, 1.0];
        let v_decode = vec![-0.1f32, 0.3, 0.6, 0.9];
        v0.f673(dev(), &dev().f502(&k_decode), &dev().f502(&v_decode), 1)
            .unwrap();
        assert_eq!(v0.f675(), 3);

        // Build Q for the current decode step (1 row)
        let q_now = vec![0.5f32, -0.2, 0.3, 0.1];

        // Run causal SDPA: q_seq=1, kv_seq=3
        // The cache buffer is sized [max_seq=8, dk=4] = 32 floats. To use it as a kv_seq=3 input,
        // we need to slice the first 3 tokens out. For this test we read the cache buffers,
        // slice on CPU, and re-upload — verifying both the cache writes and causal SDPA work.
        let k_full = dev().f504(v0.f676()).unwrap();
        let v_full = dev().f504(v0.f677()).unwrap();
        let k_used = &k_full[..(3 * dk as usize)];
        let v_used = &v_full[..(3 * dk as usize)];

        let attn_gpu = dev()
            .f504(
                &dev()
                    .f623(
                        &dev().f502(&q_now),
                        &dev().f502(k_used),
                        &dev().f502(v_used),
                        1,
                        1,
                        3,
                        dk,
                    )
                    .unwrap(),
            )
            .unwrap();

        // CPU reference using the full prefilled+decoded K/V
        let mut k_all = k_prefill.clone();
        k_all.extend_from_slice(&k_decode);
        let mut v_all = v_prefill.clone();
        v_all.extend_from_slice(&v_decode);
        let attn_cpu = cpu_decode_attention(&q_now, &k_all, &v_all, 3, dk as usize);
        f544(&attn_gpu, &attn_cpu, 1e-3);
    }

    // CPU reference for the decode case (q_seq=1, kv_seq=N, no masking applies).
    fn cpu_decode_attention(q: &[f32], k: &[f32], v: &[f32], kv: usize, dk: usize) -> Vec<f32> {
        let scale = 1.0 / (dk as f32).sqrt();
        let mut scores = vec![0.0f32; kv];
        for j in 0..kv {
            let mut s = 0.0;
            for d in 0..dk {
                s += q[d] * k[j * dk + d];
            }
            scores[j] = s * scale;
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter().map(|&x| (x - mx).exp()).sum();
        let attn: Vec<f32> = scores.iter().map(|&x| (x - mx).exp() / sum).collect();
        let mut out = vec![0.0f32; dk];
        for d in 0..dk {
            for j in 0..kv {
                out[d] += attn[j] * v[j * dk + d];
            }
        }
        out
    }

    #[test]
    fn f793_scatter_add() {
        // vocab=3, d_model=2, n_ids=2. Tokens: [1, 0]. grad_out rows: [[1,2],[3,4]].
        // Expected grad_weight[0]=[3,4], grad_weight[1]=[1,2], grad_weight[2]=[0,0].
        let v0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let v1 = vec![1.0f32, 0.0];
        let v2 = dev()
            .f504(
                &dev()
                    .f793(&dev().f502(&v0), &dev().f502(&v1), 2, 3, 2)
                    .unwrap(),
            )
            .unwrap();
        f544(&v2, &[3.0, 4.0, 1.0, 2.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    fn f793_repeated_token() {
        // vocab=2, d_model=2, n_ids=3. Tokens: [0, 1, 0]. grad_out: [[1,1],[2,2],[3,3]].
        // grad_weight[0] = [1+3, 1+3] = [4,4], grad_weight[1] = [2,2].
        let v0 = vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        let v1 = vec![0.0f32, 1.0, 0.0];
        let v2 = dev()
            .f504(
                &dev()
                    .f793(&dev().f502(&v0), &dev().f502(&v1), 3, 2, 2)
                    .unwrap(),
            )
            .unwrap();
        f544(&v2, &[4.0, 4.0, 2.0, 2.0], 1e-5);
    }

    #[test]
    fn f794_numeric_grad_check() {
        // 2×4 softmax backward.
        let rows = 2u32;
        let cols = 4u32;
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.5];
        let prob = {
            let v: Vec<f32> = dev()
                .f504(&dev().f620(&dev().f502(&input), rows, cols).unwrap())
                .unwrap();
            v
        };
        let grad_out = vec![0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];

        // CPU reference: d_x = p * (g - dot(g, p))
        let mut d_x_cpu = vec![0.0f32; (rows * cols) as usize];
        for r in 0..(rows as usize) {
            let base = r * cols as usize;
            let dot: f32 = (0..cols as usize)
                .map(|c| grad_out[base + c] * prob[base + c])
                .sum();
            for c in 0..cols as usize {
                d_x_cpu[base + c] = prob[base + c] * (grad_out[base + c] - dot);
            }
        }

        let v0 = dev()
            .f504(
                &dev()
                    .f794(&dev().f502(&grad_out), &dev().f502(&prob), rows, cols)
                    .unwrap(),
            )
            .unwrap();

        f544(&v0, &d_x_cpu, 1e-5);
    }

    #[test]
    fn f796_inverse_of_forward() {
        // RoPE backward (conjugate) applied to RoPE forward output should recover original.
        let bh = 1u32;
        let seq = 2u32;
        let hdim = 4u32;
        let start = 0u32;
        let base = 10000.0f32;
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.5];

        let v0 = dev()
            .f625(&dev().f502(&input), bh, seq, hdim, start, base)
            .unwrap();
        let v1 = dev().f796(&v0, bh, seq, hdim, start, base).unwrap();
        let v2 = dev().f504(&v1).unwrap();

        f544(&v2, &input, 1e-5);
    }

    // --- BatchKvPool (t551) ---

    #[test]
    fn f800_new_pool_starts_empty() {
        // max_batch=2, num_kv_heads=1, max_seq=4, head_dim=2 → 16 floats per K/V
        let v0 = t551::f800(dev(), 2, 1, 4, 2);
        assert_eq!(v0.s532, 2);
        assert_eq!(v0.s536, vec![0u32, 0u32]);
        assert_eq!(v0.s530.s507, 2 * 1 * 4 * 2); // 16 elements
        assert_eq!(v0.s531.s507, 16);
        assert_eq!(v0.f803(0), 0);
        assert_eq!(v0.f803(1), 0);
    }

    #[test]
    fn f801_copy_from_kvcache_single_slot() {
        // Prefill 2 tokens into a t534, then copy into slot 0 of a BatchKvPool.
        // num_kv_heads=1, max_seq=4, head_dim=2.
        let mut kv = t534::f672(dev(), 4, 1, 2);
        let k_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [1 head, 2 tokens, dim=2]
        let v_data = vec![10.0f32, 20.0, 30.0, 40.0];
        kv.f673(dev(), &dev().f502(&k_data), &dev().f502(&v_data), 2)
            .unwrap();
        assert_eq!(kv.f675(), 2);

        let mut pool = t551::f800(dev(), 2, 1, 4, 2);
        // Clone the t501 buffers (cheap Arc clone) to avoid split-borrow conflict
        let k_src = kv.s511.clone();
        let v_src = kv.s512.clone();
        pool.f801(dev(), 0, &k_src, &v_src, kv.s513, kv.s514)
            .unwrap();
        assert_eq!(pool.f803(0), 2);
        assert_eq!(pool.f803(1), 0); // slot 1 untouched

        // Verify pool K layout: slot=0, head=0 occupies rows 0..max_seq
        // slot_offset = 0 * 1 * max_seq * dim = 0
        // position 0: [1.0, 2.0] at pool.k[0*4*2 + 0*2 .. 0*4*2 + 0*2 + 2]
        // position 1: [3.0, 4.0] at pool.k[0*4*2 + 1*2 .. 0*4*2 + 1*2 + 2]
        let k_pool = dev().f504(pool.f804k()).unwrap();
        assert_eq!(&k_pool[0..4], &[1.0, 2.0, 3.0, 4.0]); // positions 0,1 of slot 0
        assert!(k_pool[4..8].iter().all(|&x| x == 0.0)); // positions 2,3 still zero
        // Slot 1 (rows 8..16) should be zero
        assert!(k_pool[8..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn f801_two_slots_independent() {
        // Copy different data into slot 0 and slot 1; verify no cross-contamination.
        let mut pool = t551::f800(dev(), 2, 1, 4, 2);

        let mut kv0 = t534::f672(dev(), 4, 1, 2);
        kv0.f673(
            dev(),
            &dev().f502(&[1.0f32, 2.0]),
            &dev().f502(&[10.0f32, 20.0]),
            1,
        )
        .unwrap();
        let k0 = kv0.s511.clone();
        let v0 = kv0.s512.clone();
        pool.f801(dev(), 0, &k0, &v0, kv0.s513, kv0.s514).unwrap();

        let mut kv1 = t534::f672(dev(), 4, 1, 2);
        kv1.f673(
            dev(),
            &dev().f502(&[5.0f32, 6.0, 7.0, 8.0]),
            &dev().f502(&[50.0f32, 60.0, 70.0, 80.0]),
            2,
        )
        .unwrap();
        let k1 = kv1.s511.clone();
        let v1 = kv1.s512.clone();
        pool.f801(dev(), 1, &k1, &v1, kv1.s513, kv1.s514).unwrap();

        assert_eq!(pool.f803(0), 1);
        assert_eq!(pool.f803(1), 2);

        let k_pool = dev().f504(pool.f804k()).unwrap();
        // Slot 0: position 0 = [1,2], positions 1..3 = 0
        assert_eq!(&k_pool[0..2], &[1.0, 2.0]);
        assert!(k_pool[2..8].iter().all(|&x| x == 0.0));
        // Slot 1 (offset = 1 * max_seq * hd = 8): positions 0,1 = [5,6,7,8], positions 2,3 = 0
        assert_eq!(&k_pool[8..12], &[5.0, 6.0, 7.0, 8.0]);
        assert!(k_pool[12..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn f805_batch_decode_append_two_slots() {
        // pool: max_batch=2, nkvh=1, max_seq=4, hd=2
        // Pre-set cursor for slot 0 = 1, slot 1 = 2.
        // Append one token per slot; verify correct positions in cache.
        let mut pool = t551::f800(dev(), 2, 1, 4, 2);
        pool.s536[0] = 1;
        pool.s536[1] = 2;

        // new K: [2 * 1 * 2] = [slot0_h0: 9,10   slot1_h0: 11,12]
        let new_k = dev().f502(&[9.0f32, 10.0, 11.0, 12.0]);
        let new_v = dev().f502(&[90.0f32, 100.0, 110.0, 120.0]);
        pool.f805(dev(), &new_k, &new_v, 2).unwrap();

        assert_eq!(pool.f803(0), 2); // cursor advanced
        assert_eq!(pool.f803(1), 3);

        let k_pool = dev().f504(pool.f804k()).unwrap();
        // Slot 0: position 1 (cursor was 1) = [9, 10]
        assert_eq!(&k_pool[2..4], &[9.0, 10.0]);
        // Slot 1 (offset=8): position 2 (cursor was 2) = [11, 12]
        assert_eq!(&k_pool[12..14], &[11.0, 12.0]);
    }

    #[test]
    fn f802_reset_slot() {
        let mut pool = t551::f800(dev(), 2, 1, 4, 2);
        pool.s536[0] = 3;
        pool.s536[1] = 1;
        pool.f802(0);
        assert_eq!(pool.f803(0), 0);
        assert_eq!(pool.f803(1), 1); // slot 1 unaffected
    }

    #[test]
    fn f807_migrate_slot_copies_kv_and_cursor() {
        // Fill slot 1 with known data; slot 0 should be empty.
        // After f807(1 -> 0), slot 0 should have the same data and cursor.
        let nkv_h = 1u32;
        let max_seq = 4u32;
        let hd = 2u32;
        let mut pool = t551::f800(dev(), 2, nkv_h, max_seq, hd);

        // Manually set slot 1 cursor and write known K/V values via f801.
        // Build a source t534 with 2 KV positions.
        let cursor = 2u32;
        // K for slot 1: kh=0, pos=0: [1.0, 2.0]; pos=1: [3.0, 4.0]
        let k_data = vec![1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]; // [nkv_h * max_seq * hd]
        let v_data = vec![5.0f32, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0];
        let k_src = dev().f502(&k_data);
        let v_src = dev().f502(&v_data);
        pool.f801(dev(), 1, &k_src, &v_src, cursor, max_seq)
            .unwrap();
        assert_eq!(pool.f803(1), 2);

        // Migrate slot 1 → slot 0.
        pool.f807(dev(), 1, 0);
        assert_eq!(pool.f803(0), 2);
        assert_eq!(pool.f803(1), 2); // src cursor unchanged

        // Read back pool K buffer and verify slot 0 has the right values.
        let pool_k = dev().f504(&pool.s530).unwrap();
        // Slot 0, kh=0 row starts at index 0 * nkv_h * max_seq * hd = 0
        assert_eq!(pool_k[0], 1.0);
        assert_eq!(pool_k[1], 2.0);
        assert_eq!(pool_k[2], 3.0);
        assert_eq!(pool_k[3], 4.0);
    }
}
