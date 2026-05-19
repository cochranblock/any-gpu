// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Group / layer / RMS normalization (two-pass: compute stats, then normalize+affine).
// f600=group_norm, f601=group_norm_backward, f602=layer_norm, f603=rms_norm.

use crate::device::{t500, t501};
use anyhow::{ensure, Result};

// --- LayerNorm / RMSNorm shared params ---
// Both ops treat input as a 2-D matrix [rows, cols] and normalize along cols.
// For a transformer with shape [batch, seq, d_model], pass rows = batch*seq, cols = d_model.

/// t523 = LNParams. Uniform for layer_norm / rms_norm shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t523 {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

// LayerNorm fused: one workgroup per row, subgroupAdd for sum and sum_sq, single pass.
// Requires s509 (SUBGROUP feature). Gate: f602 checks self.s509 at dispatch time.
// LayerNorm fused: one workgroup per row, 2 barriers (not 4).
// Both sum and sum_sq are reduced in the same barrier sequence:
//   wg_scratch[0..3] = per-subgroup partial sums
//   wg_scratch[4..7] = per-subgroup partial sum_sq
// Cross-subgroup reduce reads both sets simultaneously, thread-0 writes
// mean → wg_scratch[0] and inv_std → wg_scratch[1] before the final barrier.
const SHADER_LN_NORM_FUSED: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       input: array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       beta:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;

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

    var local_sum: f32 = 0.0;
    var local_sq:  f32 = 0.0;
    for (var c: u32 = t; c < p.cols; c += 256u) {
        let v = input[base + c];
        local_sum += v;
        local_sq  += v * v;
    }

    // Intra-subgroup reduction for both accumulators simultaneously.
    let sg_sum = subgroupAdd(local_sum);
    let sg_sq  = subgroupAdd(local_sq);
    if sg_lane == 0u {
        wg_scratch[sg_id]      = sg_sum;   // [0..3]
        wg_scratch[sg_id + 4u] = sg_sq;    // [4..7]
    }
    workgroupBarrier();

    // Cross-subgroup reduce: first 4 threads hold one partial each.
    var g_sum: f32 = 0.0;
    var g_sq:  f32 = 0.0;
    if t < 4u {
        g_sum = wg_scratch[t];
        g_sq  = wg_scratch[t + 4u];
    }
    g_sum = subgroupAdd(g_sum);
    g_sq  = subgroupAdd(g_sq);
    if t == 0u {
        let mean = g_sum / f32(p.cols);
        wg_scratch[0] = mean;
        wg_scratch[1] = 1.0 / sqrt(g_sq / f32(p.cols) - mean * mean + p.eps);
    }
    workgroupBarrier();
    let mean    = wg_scratch[0];
    let inv_std = wg_scratch[1];

    for (var c: u32 = t; c < p.cols; c += 256u) {
        out[base + c] = (input[base + c] - mean) * inv_std * gamma[c] + beta[c];
    }
}
";

// LayerNorm pass 1: one thread per row, compute mean and inv_std.
const SHADER_LN_STATS: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= p.rows { return; }
    let base = r * p.cols;
    var sum: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cols; c++) {
        let v = input[base + c];
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / f32(p.cols);
    let var_ = sum_sq / f32(p.cols) - mean * mean;
    stats[r * 2u] = mean;
    stats[r * 2u + 1u] = 1.0 / sqrt(var_ + p.eps);
}
";

// LayerNorm pass 2: one thread per element, apply normalize + affine.
const SHADER_LN_NORM: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.rows * p.cols;
    if idx >= total { return; }
    let r = idx / p.cols;
    let c = idx % p.cols;
    let mean = stats[r * 2u];
    let inv_std = stats[r * 2u + 1u];
    out[idx] = (input[idx] - mean) * inv_std * gamma[c] + beta[c];
}
";

// RMSNorm fused: one workgroup per row, subgroupAdd for sum_sq, single pass.
// Requires s509 (SUBGROUP feature). Gate: f603 checks self.s509 at dispatch time.
const SHADER_RMS_NORM_FUSED: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       input:  array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

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

    var local_sq: f32 = 0.0;
    for (var c: u32 = t; c < p.cols; c += 256u) {
        let v = input[base + c];
        local_sq += v * v;
    }
    let sg_sq = subgroupAdd(local_sq);
    if sg_lane == 0u { wg_scratch[sg_id] = sg_sq; }
    workgroupBarrier();
    var global_sq: f32 = select(0.0, wg_scratch[t], t < 4u);
    global_sq = subgroupAdd(global_sq);
    if t == 0u { wg_scratch[4] = 1.0 / sqrt(global_sq / f32(p.cols) + p.eps); }
    workgroupBarrier();
    let inv_rms = wg_scratch[4];

    for (var c: u32 = t; c < p.cols; c += 256u) {
        out[base + c] = input[base + c] * inv_rms * weight[c];
    }
}
";

// RMSNorm pass 1: one thread per row, compute inv_rms = 1/sqrt(mean(x^2) + eps).
const SHADER_RMS_STATS: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= p.rows { return; }
    let base = r * p.cols;
    var sum_sq: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cols; c++) {
        let v = input[base + c];
        sum_sq += v * v;
    }
    stats[r] = 1.0 / sqrt(sum_sq / f32(p.cols) + p.eps);
}
";

// RMSNorm pass 2: one thread per element, x * inv_rms * weight (no bias).
const SHADER_RMS_NORM: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.rows * p.cols;
    if idx >= total { return; }
    let r = idx / p.cols;
    let c = idx % p.cols;
    out[idx] = input[idx] * stats[r] * weight[c];
}
";

/// t522 = GNStatsParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t522 {
    batch: u32,
    channels: u32,
    spatial: u32,
    groups: u32,
    channels_per_group: u32,
    eps: f32,
    _pad: [u32; 2],
}

// Pass 1: one thread per (batch, group). Computes mean and variance.
const SHADER_GN_STATS: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.batch * p.groups;
    if idx >= total { return; }

    let g = idx % p.groups;
    let n = idx / p.groups;

    let count = p.cpg * p.spatial;
    var sum: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cpg; c++) {
        let ch = g * p.cpg + c;
        let base = n * (p.channels * p.spatial) + ch * p.spatial;
        for (var s: u32 = 0u; s < p.spatial; s++) {
            let v = input[base + s];
            sum += v;
            sum_sq += v * v;
        }
    }
    let mean = sum / f32(count);
    let variance = sum_sq / f32(count) - mean * mean;
    stats[idx * 2u] = mean;
    stats[idx * 2u + 1u] = 1.0 / sqrt(variance + p.eps);
}
";

// Pass 2: one thread per element. Normalize and apply affine.
const SHADER_GN_NORM: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.channels * p.spatial;
    if idx >= total { return; }

    let s = idx % p.spatial;
    let ch = (idx / p.spatial) % p.channels;
    let n = idx / (p.spatial * p.channels);
    let g = ch / p.cpg;

    let stat_idx = n * p.groups + g;
    let mean = stats[stat_idx * 2u];
    let inv_std = stats[stat_idx * 2u + 1u];

    out[idx] = (input[idx] - mean) * inv_std * gamma[ch] + beta[ch];
}
";

// --- GroupNorm backward ---
// Pass A: recompute stats (mean, inv_std) per (n, g) — reuse SHADER_GN_STATS.
// Pass B: compute per-(n, g) reduction sums for d_input:
//         s1 = sum over group of (grad_out * gamma)
//         s2 = sum over group of (grad_out * gamma * x_hat)
// Pass C: compute d_input per element using stats and sums.
// Pass D: compute d_gamma and d_beta per channel by reduction over batch and spatial.

const SHADER_GN_BACK_SUMS: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> sums: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.batch * p.groups;
    if idx >= total { return; }
    let g = idx % p.groups;
    let n = idx / p.groups;

    let mean = stats[idx * 2u];
    let inv_std = stats[idx * 2u + 1u];

    var s1: f32 = 0.0;
    var s2: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cpg; c++) {
        let ch = g * p.cpg + c;
        let base = n * (p.channels * p.spatial) + ch * p.spatial;
        for (var s: u32 = 0u; s < p.spatial; s++) {
            let dx_hat = grad_out[base + s] * gamma[ch];
            let x_hat = (input[base + s] - mean) * inv_std;
            s1 += dx_hat;
            s2 += dx_hat * x_hat;
        }
    }
    sums[idx * 2u] = s1;
    sums[idx * 2u + 1u] = s2;
}
";

const SHADER_GN_BACK_DINPUT: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read> sums: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_input: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.channels * p.spatial;
    if idx >= total { return; }

    let s_idx = idx % p.spatial;
    let ch = (idx / p.spatial) % p.channels;
    let n = idx / (p.spatial * p.channels);
    let g = ch / p.cpg;

    let stat_idx = n * p.groups + g;
    let mean = stats[stat_idx * 2u];
    let inv_std = stats[stat_idx * 2u + 1u];
    let x_hat = (input[idx] - mean) * inv_std;
    let dx_hat = grad_out[idx] * gamma[ch];

    let s1 = sums[stat_idx * 2u];
    let s2 = sums[stat_idx * 2u + 1u];
    let count = f32(p.cpg * p.spatial);
    d_input[idx] = inv_std * (dx_hat - (s1 + x_hat * s2) / count);
}
";

const SHADER_GN_BACK_DAFFINE: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_gamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_beta: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if ch >= p.channels { return; }
    let g = ch / p.cpg;

    var dg: f32 = 0.0;
    var db: f32 = 0.0;
    for (var n: u32 = 0u; n < p.batch; n++) {
        let stat_idx = n * p.groups + g;
        let mean = stats[stat_idx * 2u];
        let inv_std = stats[stat_idx * 2u + 1u];
        let base = n * (p.channels * p.spatial) + ch * p.spatial;
        for (var s: u32 = 0u; s < p.spatial; s++) {
            let x_hat = (input[base + s] - mean) * inv_std;
            dg += grad_out[base + s] * x_hat;
            db += grad_out[base + s];
        }
    }
    d_gamma[ch] = dg;
    d_beta[ch] = db;
}
";

// LayerNorm backward pass 1: one thread per row.
// Writes 4 floats per row: [mean, inv_std, S1, S2] where
//   S1 = (1/n)*sum(grad_y * gamma), S2 = (1/n)*sum(grad_y * gamma * xhat).
const SHADER_LN_BWD_STATS: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input:    array<f32>;
@group(0) @binding(2) var<storage, read> grad_y:   array<f32>;
@group(0) @binding(3) var<storage, read> gamma:    array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= p.rows { return; }
    let base = r * p.cols;
    var sum: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cols; c++) {
        let v = input[base + c];
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / f32(p.cols);
    let inv_std = 1.0 / sqrt(sum_sq / f32(p.cols) - mean * mean + p.eps);
    var s1: f32 = 0.0;
    var s2: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cols; c++) {
        let gy_g = grad_y[base + c] * gamma[c];
        let xhat = (input[base + c] - mean) * inv_std;
        s1 += gy_g;
        s2 += gy_g * xhat;
    }
    let n = f32(p.cols);
    stats[r * 4u]      = mean;
    stats[r * 4u + 1u] = inv_std;
    stats[r * 4u + 2u] = s1 / n;
    stats[r * 4u + 3u] = s2 / n;
}
";

// LayerNorm backward pass 2: one thread per element.
const SHADER_LN_BWD_DINPUT: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input:  array<f32>;
@group(0) @binding(2) var<storage, read> grad_y: array<f32>;
@group(0) @binding(3) var<storage, read> gamma:  array<f32>;
@group(0) @binding(4) var<storage, read> stats:  array<f32>;
@group(0) @binding(5) var<storage, read_write> d_input: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.rows * p.cols;
    if idx >= total { return; }
    let r = idx / p.cols;
    let c = idx % p.cols;
    let mean    = stats[r * 4u];
    let inv_std = stats[r * 4u + 1u];
    let s1      = stats[r * 4u + 2u];
    let s2      = stats[r * 4u + 3u];
    let xhat    = (input[idx] - mean) * inv_std;
    d_input[idx] = inv_std * (grad_y[idx] * gamma[c] - s1 - xhat * s2);
}
";

// LayerNorm backward pass 3: one thread per col, sums over all rows.
const SHADER_LN_BWD_DAFFINE: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input:   array<f32>;
@group(0) @binding(2) var<storage, read> grad_y:  array<f32>;
@group(0) @binding(3) var<storage, read> stats:   array<f32>;
@group(0) @binding(4) var<storage, read_write> d_gamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_beta:  array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if c >= p.cols { return; }
    var dg: f32 = 0.0;
    var db: f32 = 0.0;
    for (var r: u32 = 0u; r < p.rows; r++) {
        let mean    = stats[r * 4u];
        let inv_std = stats[r * 4u + 1u];
        let xhat    = (input[r * p.cols + c] - mean) * inv_std;
        dg += grad_y[r * p.cols + c] * xhat;
        db += grad_y[r * p.cols + c];
    }
    d_gamma[c] = dg;
    d_beta[c]  = db;
}
";

// RMSNorm backward pass 1: one thread per row.
// Writes 2 floats per row: [inv_rms, S] where S = (1/n)*sum(grad_y * gamma * x).
const SHADER_RN_BWD_STATS: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input:  array<f32>;
@group(0) @binding(2) var<storage, read> grad_y: array<f32>;
@group(0) @binding(3) var<storage, read> gamma:  array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= p.rows { return; }
    let base = r * p.cols;
    var sum_sq: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cols; c++) {
        let v = input[base + c];
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / sqrt(sum_sq / f32(p.cols) + p.eps);
    var s: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cols; c++) {
        s += grad_y[base + c] * gamma[c] * input[base + c];
    }
    stats[r * 2u]      = inv_rms;
    stats[r * 2u + 1u] = s / f32(p.cols);
}
";

// RMSNorm backward pass 2a: one thread per element.
const SHADER_RN_BWD_DINPUT: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input:  array<f32>;
@group(0) @binding(2) var<storage, read> grad_y: array<f32>;
@group(0) @binding(3) var<storage, read> gamma:  array<f32>;
@group(0) @binding(4) var<storage, read> stats:  array<f32>;
@group(0) @binding(5) var<storage, read_write> d_input: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.rows * p.cols;
    if idx >= total { return; }
    let r = idx / p.cols;
    let c = idx % p.cols;
    let inv_rms = stats[r * 2u];
    let s       = stats[r * 2u + 1u];
    d_input[idx] = inv_rms * gamma[c] * grad_y[idx] - input[idx] * (s * inv_rms * inv_rms * inv_rms);
}
";

// RMSNorm backward pass 2b: one thread per col, sums over all rows for grad_gamma.
const SHADER_RN_BWD_DGAMMA: &str = "
struct P { rows: u32, cols: u32, eps: f32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input:  array<f32>;
@group(0) @binding(2) var<storage, read> grad_y: array<f32>;
@group(0) @binding(3) var<storage, read> stats:  array<f32>;
@group(0) @binding(4) var<storage, read_write> d_gamma: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if c >= p.cols { return; }
    var dg: f32 = 0.0;
    for (var r: u32 = 0u; r < p.rows; r++) {
        let inv_rms = stats[r * 2u];
        dg += grad_y[r * p.cols + c] * input[r * p.cols + c] * inv_rms;
    }
    d_gamma[c] = dg;
}
";

impl t500 {
    /// f600 = group_norm. input[N,C,*spatial] with C/groups groups.
    /// gamma[C] and beta[C] are learnable affine params.
    pub fn f600(
        &self,
        p0: &t501,
        p1: &t501,
        p2: &t501,
        p3: u32, p4: u32, p5: u32, p6: u32,
        p7: f32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p3 * p4 * p5) as usize);
        ensure!(p1.s507 == p4 as usize);
        ensure!(p2.s507 == p4 as usize);
        ensure!(p4 % p6 == 0, "channels must be divisible by groups");

        let v0 = p4 / p6;
        let v1 = t522 { batch: p3, channels: p4, spatial: p5, groups: p6, channels_per_group: v0, eps: p7, _pad: [0; 2] };

        // Pass 1: compute per-group mean and inv_std
        let v2 = self.f503((p3 * p6 * 2) as usize);
        self.f543(
            SHADER_GN_STATS, Some("gn_stats"),
            &v1, &[p0], &v2,
            super::f540(p3 * p6),
        );

        // Pass 2: normalize + affine
        let v3 = p3 * p4 * p5;
        let v4 = self.f503(v3 as usize);

        // For pass 2 we need 5 storage bindings + params. Use raw dispatch.
        let v5 = self.f506(&v1);
        let v6 = self.s500.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gn_norm"),
            source: wgpu::ShaderSource::Wgsl(SHADER_GN_NORM.into()),
        });
        let v7 = self.s500.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gn_norm"),
            layout: None,
            module: &v6,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let v8 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &v7.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v5.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p0.s505.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v2.s505.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p1.s505.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p2.s505.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: v4.s505.as_entire_binding() },
            ],
        });
        let (v9, v10, v11) = super::f540(v3);
        let mut v12 = self.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut v13 = v12.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gn_norm"),
                timestamp_writes: None,
            });
            v13.set_pipeline(&v7);
            v13.set_bind_group(0, &v8, &[]);
            v13.dispatch_workgroups(v9, v10, v11);
        }
        self.s501.submit(Some(v12.finish()));

        Ok(v4)
    }

    /// f602 = layer_norm. input[rows, cols] normalized along cols with learnable
    /// gamma/beta[cols]. For a transformer activation [batch, seq, d_model],
    /// pass rows = batch*seq, cols = d_model.
    pub fn f602(
        &self,
        p0: &t501,
        p1: &t501,
        p2: &t501,
        p3: u32, p4: u32, p5: f32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p3 * p4) as usize, "layer_norm: input size mismatch");
        ensure!(p1.s507 == p4 as usize, "layer_norm: gamma must be [cols]");
        ensure!(p2.s507 == p4 as usize, "layer_norm: beta must be [cols]");

        let v0 = t523 { rows: p3, cols: p4, eps: p5, _pad: 0 };

        if self.s509 {
            let v1 = self.f503((p3 * p4) as usize);
            let (vx, vy) = if p3 <= 65535 { (p3, 1) } else { (65535, p3.div_ceil(65535)) };
            self.f543(SHADER_LN_NORM_FUSED, Some("ln_norm_fused"), &v0, &[p0, p1, p2], &v1, (vx, vy, 1));
            return Ok(v1);
        }

        // Two-pass fallback: per-row mean and inv_std
        let v1 = self.f503((p3 * 2) as usize);
        self.f543(
            SHADER_LN_STATS, Some("ln_stats"),
            &v0, &[p0], &v1,
            super::f540(p3),
        );

        // Pass 2: normalize + affine
        let v2 = p3 * p4;
        let v3 = self.f503(v2 as usize);
        self.f543(
            SHADER_LN_NORM, Some("ln_norm"),
            &v0, &[p0, &v1, p1, p2], &v3,
            super::f540(v2),
        );
        Ok(v3)
    }

    /// f603 = rms_norm. input[rows, cols] scaled by 1/sqrt(mean(x^2) + eps),
    /// then per-col weight. Used by Llama / Mistral / Qwen / Gemma. No bias term.
    pub fn f603(
        &self,
        p0: &t501,
        p1: &t501,
        p2: u32, p3: u32, p4: f32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p2 * p3) as usize, "rms_norm: input size mismatch");
        ensure!(p1.s507 == p3 as usize, "rms_norm: weight must be [cols]");

        let v0 = t523 { rows: p2, cols: p3, eps: p4, _pad: 0 };

        if self.s509 {
            let v1 = self.f503((p2 * p3) as usize);
            let (vx, vy) = if p2 <= 65535 { (p2, 1) } else { (65535, p2.div_ceil(65535)) };
            self.f543(SHADER_RMS_NORM_FUSED, Some("rms_norm_fused"), &v0, &[p0, p1], &v1, (vx, vy, 1));
            return Ok(v1);
        }

        // Two-pass fallback: per-row inv_rms
        let v1 = self.f503(p2 as usize);
        self.f543(
            SHADER_RMS_STATS, Some("rms_stats"),
            &v0, &[p0], &v1,
            super::f540(p2),
        );

        // Pass 2: normalize + per-col weight
        let v2 = p2 * p3;
        let v3 = self.f503(v2 as usize);
        self.f543(
            SHADER_RMS_NORM, Some("rms_norm"),
            &v0, &[p0, &v1, p1], &v3,
            super::f540(v2),
        );
        Ok(v3)
    }

    /// f601 = group_norm_backward. Backward pass for f600. Returns (d_input, d_gamma, d_beta).
    /// Recomputes stats from input internally to avoid storing them on the tape.
    pub fn f601(
        &self,
        p0: &t501,
        p1: &t501,
        p2: &t501,
        p3: u32, p4: u32, p5: u32, p6: u32, p7: f32,
    ) -> Result<(t501, t501, t501)> {
        ensure!(p1.s507 == (p3 * p4 * p5) as usize);
        ensure!(p0.s507 == (p3 * p4 * p5) as usize);
        ensure!(p2.s507 == p4 as usize);
        ensure!(p4 % p6 == 0);

        let v0 = p4 / p6;
        let v1 = t522 {
            batch: p3, channels: p4, spatial: p5, groups: p6,
            channels_per_group: v0, eps: p7, _pad: [0; 2],
        };

        // Pass A: recompute stats per (n, g) (reuse forward stats shader).
        let v2 = self.f503((p3 * p6 * 2) as usize);
        self.f543(
            SHADER_GN_STATS, Some("gn_back_stats"),
            &v1, &[p1], &v2,
            super::f540(p3 * p6),
        );

        // Pass B: compute reduction sums (s1, s2) per (n, g). 5 storage bindings → raw dispatch.
        let v3 = self.f503((p3 * p6 * 2) as usize);
        {
            let v4 = self.f506(&v1);
            let v5 = self.s500.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gn_back_sums"),
                source: wgpu::ShaderSource::Wgsl(SHADER_GN_BACK_SUMS.into()),
            });
            let v6 = self.s500.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gn_back_sums"),
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
                    wgpu::BindGroupEntry { binding: 0, resource: v4.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: p1.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: p0.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: p2.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: v2.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: v3.s505.as_entire_binding() },
                ],
            });
            let (v8, v9, v10) = super::f540(p3 * p6);
            let mut v11 = self.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut v12 = v11.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gn_back_sums"),
                    timestamp_writes: None,
                });
                v12.set_pipeline(&v6);
                v12.set_bind_group(0, &v7, &[]);
                v12.dispatch_workgroups(v8, v9, v10);
            }
            self.s501.submit(Some(v11.finish()));
        }

        // Pass C: compute d_input per element. 6 storage bindings → raw dispatch.
        let v13 = p3 * p4 * p5;
        let v14 = self.f503(v13 as usize);
        {
            let v15 = self.f506(&v1);
            let v16 = self.s500.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gn_back_dinput"),
                source: wgpu::ShaderSource::Wgsl(SHADER_GN_BACK_DINPUT.into()),
            });
            let v17 = self.s500.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gn_back_dinput"),
                layout: None,
                module: &v16,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let v18 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &v17.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: v15.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: p1.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: p0.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: p2.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: v2.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: v3.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: v14.s505.as_entire_binding() },
                ],
            });
            let (v19, v20, v21) = super::f540(v13);
            let mut v22 = self.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut v23 = v22.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gn_back_dinput"),
                    timestamp_writes: None,
                });
                v23.set_pipeline(&v17);
                v23.set_bind_group(0, &v18, &[]);
                v23.dispatch_workgroups(v19, v20, v21);
            }
            self.s501.submit(Some(v22.finish()));
        }

        // Pass D: compute d_gamma and d_beta per channel. 5 storage bindings → raw dispatch.
        let v24 = self.f503(p4 as usize);
        let v25 = self.f503(p4 as usize);
        {
            let v26 = self.f506(&v1);
            let v27 = self.s500.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gn_back_daffine"),
                source: wgpu::ShaderSource::Wgsl(SHADER_GN_BACK_DAFFINE.into()),
            });
            let v28 = self.s500.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gn_back_daffine"),
                layout: None,
                module: &v27,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let v29 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &v28.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: v26.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: p1.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: p0.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: v2.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: v24.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: v25.s505.as_entire_binding() },
                ],
            });
            let (v30, v31, v32) = super::f540(p4);
            let mut v33 = self.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut v34 = v33.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gn_back_daffine"),
                    timestamp_writes: None,
                });
                v34.set_pipeline(&v28);
                v34.set_bind_group(0, &v29, &[]);
                v34.dispatch_workgroups(v30, v31, v32);
            }
            self.s501.submit(Some(v33.finish()));
        }

        Ok((v14, v24, v25))
    }

    /// f791 = layer_norm_backward. Returns (grad_input, grad_gamma, grad_beta).
    /// p0=grad_output[rows,cols], p1=fwd_input[rows,cols], p2=gamma[cols].
    pub fn f791(
        &self,
        p0: &t501,
        p1: &t501,
        p2: &t501,
        p3: u32, p4: u32, p5: f32,
    ) -> Result<(t501, t501, t501)> {
        ensure!(p0.s507 == (p3 * p4) as usize, "ln_bwd: grad_output size mismatch");
        ensure!(p1.s507 == (p3 * p4) as usize, "ln_bwd: fwd_input size mismatch");
        ensure!(p2.s507 == p4 as usize, "ln_bwd: gamma size mismatch");

        let v0 = t523 { rows: p3, cols: p4, eps: p5, _pad: 0 };

        // Pass 1: per-row stats — mean, inv_std, S1, S2.
        let v1 = self.f503((p3 * 4) as usize);
        self.f543(
            SHADER_LN_BWD_STATS, Some("ln_bwd_stats"),
            &v0, &[p1, p0, p2], &v1,
            super::f540(p3),
        );

        // Pass 2: grad_input per element.
        let v2 = self.f503((p3 * p4) as usize);
        self.f543(
            SHADER_LN_BWD_DINPUT, Some("ln_bwd_dinput"),
            &v0, &[p1, p0, p2, &v1], &v2,
            super::f540(p3 * p4),
        );

        // Pass 3: grad_gamma and grad_beta per col.
        let v3 = self.f503(p4 as usize);
        let v4 = self.f503(p4 as usize);
        {
            let v5 = self.f506(&v0);
            let v6 = self.f507(SHADER_LN_BWD_DAFFINE, Some("ln_bwd_daffine"));
            let v7 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &v6.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: v5.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: p1.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: p0.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: v1.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: v3.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: v4.s505.as_entire_binding() },
                ],
            });
            let (v8, v9, v10) = super::f540(p4);
            let mut v11 = self.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut v12 = v11.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ln_bwd_daffine"),
                    timestamp_writes: None,
                });
                v12.set_pipeline(&v6);
                v12.set_bind_group(0, &v7, &[]);
                v12.dispatch_workgroups(v8, v9, v10);
            }
            self.s501.submit(Some(v11.finish()));
        }

        Ok((v2, v3, v4))
    }

    /// f792 = rms_norm_backward. Returns (grad_input, grad_gamma).
    /// p0=grad_output[rows,cols], p1=fwd_input[rows,cols], p2=gamma[cols].
    pub fn f792(
        &self,
        p0: &t501,
        p1: &t501,
        p2: &t501,
        p3: u32, p4: u32, p5: f32,
    ) -> Result<(t501, t501)> {
        ensure!(p0.s507 == (p3 * p4) as usize, "rn_bwd: grad_output size mismatch");
        ensure!(p1.s507 == (p3 * p4) as usize, "rn_bwd: fwd_input size mismatch");
        ensure!(p2.s507 == p4 as usize, "rn_bwd: gamma size mismatch");

        let v0 = t523 { rows: p3, cols: p4, eps: p5, _pad: 0 };

        // Pass 1: per-row stats — inv_rms, S.
        let v1 = self.f503((p3 * 2) as usize);
        self.f543(
            SHADER_RN_BWD_STATS, Some("rn_bwd_stats"),
            &v0, &[p1, p0, p2], &v1,
            super::f540(p3),
        );

        // Pass 2a: grad_input per element.
        let v2 = self.f503((p3 * p4) as usize);
        self.f543(
            SHADER_RN_BWD_DINPUT, Some("rn_bwd_dinput"),
            &v0, &[p1, p0, p2, &v1], &v2,
            super::f540(p3 * p4),
        );

        // Pass 2b: grad_gamma per col.
        let v3 = self.f503(p4 as usize);
        self.f543(
            SHADER_RN_BWD_DGAMMA, Some("rn_bwd_dgamma"),
            &v0, &[p1, p0, &v1], &v3,
            super::f540(p4),
        );

        Ok((v2, v3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;
    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    // CPU reference group_norm
    fn cpu_group_norm(
        input: &[f32], gamma: &[f32], beta: &[f32],
        batch: usize, channels: usize, spatial: usize, groups: usize, eps: f32,
    ) -> Vec<f32> {
        let cpg = channels / groups;
        let mut out = vec![0.0f32; input.len()];
        for n in 0..batch {
            for g in 0..groups {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                let count = (cpg * spatial) as f32;
                for c in 0..cpg {
                    let ch = g * cpg + c;
                    for s in 0..spatial {
                        let v = input[n * channels * spatial + ch * spatial + s];
                        sum += v;
                        sum_sq += v * v;
                    }
                }
                let mean = sum / count;
                let var = sum_sq / count - mean * mean;
                let inv_std = 1.0 / (var + eps).sqrt();
                for c in 0..cpg {
                    let ch = g * cpg + c;
                    for s in 0..spatial {
                        let idx = n * channels * spatial + ch * spatial + s;
                        out[idx] = (input[idx] - mean) * inv_std * gamma[ch] + beta[ch];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn f600_per_channel() {
        // groups=channels: each channel is its own group
        let v0 = dev().f502(&[1.0, 3.0, 2.0, 4.0]);
        let v1 = dev().f502(&[1.0, 1.0]);
        let v2 = dev().f502(&[0.0, 0.0]);
        let v3 = dev().f504(&dev().f600(&v0, &v1, &v2, 1, 2, 2, 2, 1e-5).unwrap()).unwrap();
        f544(&v3, &[-1.0, 1.0, -1.0, 1.0], 1e-3);
    }

    #[test]
    fn f600_single_group() {
        // groups=1: all channels normalized together
        let v0 = vec![1.0, 2.0, 3.0, 4.0];
        let v1 = vec![1.0; 4];
        let v2 = vec![0.0; 4];
        let v3 = cpu_group_norm(&v0, &v1, &v2, 1, 4, 1, 1, 1e-5);
        let v4 = dev().f504(&dev().f600(
            &dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2),
            1, 4, 1, 1, 1e-5
        ).unwrap()).unwrap();
        f544(&v4, &v3, 1e-3);
    }

    #[test]
    fn f600_with_affine() {
        let v0 = dev().f502(&[1.0, 3.0, 2.0, 4.0]);
        let v1 = dev().f502(&[2.0, 0.5]);
        let v2 = dev().f502(&[1.0, -1.0]);
        let v3 = dev().f504(&dev().f600(&v0, &v1, &v2, 1, 2, 2, 2, 1e-5).unwrap()).unwrap();
        f544(&v3, &[-1.0, 3.0, -1.5, -0.5], 1e-3);
    }

    #[test]
    fn f600_batched_vs_cpu() {
        let v0: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let v1 = vec![1.0, 2.0, 0.5, 1.5];
        let v2 = vec![0.0, 1.0, -1.0, 0.5];
        let v3 = cpu_group_norm(&v0, &v1, &v2, 2, 4, 3, 2, 1e-5);
        let v4 = dev().f504(&dev().f600(
            &dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2),
            2, 4, 3, 2, 1e-5
        ).unwrap()).unwrap();
        f544(&v4, &v3, 1e-3);
    }

    // CPU reference for group_norm_backward — returns (d_input, d_gamma, d_beta).
    fn cpu_group_norm_backward(
        grad_out: &[f32], input: &[f32], gamma: &[f32],
        batch: usize, channels: usize, spatial: usize, groups: usize, eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let cpg = channels / groups;
        let count = (cpg * spatial) as f32;
        let n_elem = batch * channels * spatial;
        let mut d_input = vec![0.0f32; n_elem];
        let mut d_gamma = vec![0.0f32; channels];
        let mut d_beta  = vec![0.0f32; channels];

        for n in 0..batch {
            for g in 0..groups {
                // Recompute stats for this (n, g) group.
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                for c in 0..cpg {
                    let ch = g * cpg + c;
                    for s in 0..spatial {
                        let v = input[n * channels * spatial + ch * spatial + s];
                        sum += v;
                        sum_sq += v * v;
                    }
                }
                let mean = sum / count;
                let var = sum_sq / count - mean * mean;
                let inv_std = 1.0 / (var + eps).sqrt();

                // Compute s1 = sum(dx_hat), s2 = sum(dx_hat * x_hat) over the group.
                let mut s1 = 0.0f32;
                let mut s2 = 0.0f32;
                for c in 0..cpg {
                    let ch = g * cpg + c;
                    for s in 0..spatial {
                        let idx = n * channels * spatial + ch * spatial + s;
                        let dx_hat = grad_out[idx] * gamma[ch];
                        let x_hat  = (input[idx] - mean) * inv_std;
                        s1 += dx_hat;
                        s2 += dx_hat * x_hat;
                    }
                }

                // d_input per element.
                for c in 0..cpg {
                    let ch = g * cpg + c;
                    for s in 0..spatial {
                        let idx = n * channels * spatial + ch * spatial + s;
                        let dx_hat = grad_out[idx] * gamma[ch];
                        let x_hat  = (input[idx] - mean) * inv_std;
                        d_input[idx] = inv_std * (dx_hat - (s1 + x_hat * s2) / count);
                    }
                }
            }
        }

        // d_gamma and d_beta: reduce over batch and spatial.
        for n in 0..batch {
            for g in 0..groups {
                // Need stats again for x_hat.
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                let c_range = (g * cpg)..((g + 1) * cpg);
                for ch in c_range.clone() {
                    for s in 0..spatial {
                        let v = input[n * channels * spatial + ch * spatial + s];
                        sum += v;
                        sum_sq += v * v;
                    }
                }
                let mean = sum / count;
                let var = sum_sq / count - mean * mean;
                let inv_std = 1.0 / (var + eps).sqrt();
                for ch in c_range {
                    for s in 0..spatial {
                        let idx = n * channels * spatial + ch * spatial + s;
                        let x_hat = (input[idx] - mean) * inv_std;
                        d_gamma[ch] += grad_out[idx] * x_hat;
                        d_beta[ch]  += grad_out[idx];
                    }
                }
            }
        }

        (d_input, d_gamma, d_beta)
    }

    #[test]
    fn f601_dinput_vs_numeric() {
        // Numeric gradient check for d_input using finite differences.
        // batch=1, channels=2, spatial=4, groups=1, eps=1e-5
        let batch: u32 = 1;
        let channels: u32 = 2;
        let spatial: u32 = 4;
        let groups: u32 = 1;
        let eps = 1e-5f32;
        let h = 1e-3f32;  // finite-difference step

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma_data: Vec<f32> = vec![1.0, 1.0];
        let beta_data: Vec<f32>  = vec![0.0, 0.0];

        // Compute forward pass to get output, then derive grad_out = 2*output/n (MSE vs zeros).
        let v0 = dev().f502(&input_data);
        let v1 = dev().f502(&gamma_data);
        let v2 = dev().f502(&beta_data);
        let v3 = dev().f504(&dev().f600(&v0, &v1, &v2, batch, channels, spatial, groups, eps).unwrap()).unwrap();
        let n = v3.len() as f32;
        let grad_out_data: Vec<f32> = v3.iter().map(|&x| 2.0 * x / n).collect();

        // GPU backward.
        let v4 = dev().f502(&grad_out_data);
        let v5 = dev().f502(&gamma_data);
        let (v6, _, _) = dev().f601(&v4, &v0, &v5, batch, channels, spatial, groups, eps).unwrap();
        let v7 = dev().f504(&v6).unwrap();

        // Numeric gradient: perturb each input element by +/-h and recompute loss.
        let n_elem = input_data.len();
        let mut numeric_grad = vec![0.0f32; n_elem];
        for i in 0..n_elem {
            let mut inp_plus  = input_data.clone();
            let mut inp_minus = input_data.clone();
            inp_plus[i]  += h;
            inp_minus[i] -= h;

            let vp = dev().f502(&inp_plus);
            let out_plus = dev().f504(&dev().f600(&vp, &v1, &v2, batch, channels, spatial, groups, eps).unwrap()).unwrap();
            let loss_plus: f32 = out_plus.iter().map(|&x| x * x).sum::<f32>() / n;

            let vm = dev().f502(&inp_minus);
            let out_minus = dev().f504(&dev().f600(&vm, &v1, &v2, batch, channels, spatial, groups, eps).unwrap()).unwrap();
            let loss_minus: f32 = out_minus.iter().map(|&x| x * x).sum::<f32>() / n;

            numeric_grad[i] = (loss_plus - loss_minus) / (2.0 * h);
        }

        f544(&v7, &numeric_grad, 1e-2);
    }

    #[test]
    fn f601_dgamma_dbeta_known() {
        // Test d_gamma and d_beta with grad_out = all-ones.
        // batch=1, channels=2, spatial=2, groups=1
        // input = [1.0, -1.0, 1.0, -1.0]
        // Each channel: [1, -1] within the single group.
        // With grad_out = all-ones:
        //   d_beta[c] = sum of grad_out for channel c over (batch, spatial) = 1*2 = 2.0 each.
        //   d_gamma[c] = sum of (grad_out * x_hat) for channel c.
        let batch: u32 = 1;
        let channels: u32 = 2;
        let spatial: u32 = 2;
        let groups: u32 = 1;
        let eps = 1e-5f32;

        let input_data: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0];
        let gamma_data: Vec<f32> = vec![1.0, 1.0];
        let grad_out_data: Vec<f32> = vec![1.0; 4];

        // CPU reference.
        let (cpu_din, cpu_dg, cpu_db) = cpu_group_norm_backward(
            &grad_out_data, &input_data, &gamma_data,
            batch as usize, channels as usize, spatial as usize, groups as usize, eps,
        );

        // GPU backward.
        let v0 = dev().f502(&grad_out_data);
        let v1 = dev().f502(&input_data);
        let v2 = dev().f502(&gamma_data);
        let (v3, v4, v5) = dev().f601(&v0, &v1, &v2, batch, channels, spatial, groups, eps).unwrap();
        let gpu_din = dev().f504(&v3).unwrap();
        let gpu_dg  = dev().f504(&v4).unwrap();
        let gpu_db  = dev().f504(&v5).unwrap();

        f544(&gpu_din, &cpu_din, 1e-4);
        f544(&gpu_dg,  &cpu_dg,  1e-4);
        // d_beta[c] = spatial * batch = 2 for each channel.
        f544(&gpu_db,  &[2.0, 2.0], 1e-4);
        // CPU should agree with the known value too.
        assert!((cpu_db[0] - 2.0).abs() < 1e-4 && (cpu_db[1] - 2.0).abs() < 1e-4,
            "cpu_db: {cpu_db:?}");
    }

    #[test]
    fn f601_groups2() {
        // Numeric gradient check with groups=2 (1 channel per group).
        // batch=1, channels=2, spatial=3, groups=2, eps=1e-5
        let batch: u32 = 1;
        let channels: u32 = 2;
        let spatial: u32 = 3;
        let groups: u32 = 2;
        let eps = 1e-5f32;
        let h = 1e-3f32;

        let input_data: Vec<f32> = vec![0.5, -0.5, 1.5,  -1.0, 2.0, 0.0];
        let gamma_data: Vec<f32> = vec![1.0, 2.0];
        let beta_data: Vec<f32>  = vec![0.0, 0.0];

        // Forward for grad_out derivation (MSE vs zeros).
        let v0 = dev().f502(&input_data);
        let v1 = dev().f502(&gamma_data);
        let v2 = dev().f502(&beta_data);
        let v3 = dev().f504(&dev().f600(&v0, &v1, &v2, batch, channels, spatial, groups, eps).unwrap()).unwrap();
        let n = v3.len() as f32;
        let grad_out_data: Vec<f32> = v3.iter().map(|&x| 2.0 * x / n).collect();

        // GPU backward.
        let v4 = dev().f502(&grad_out_data);
        let v5 = dev().f502(&gamma_data);
        let (v6, _, _) = dev().f601(&v4, &v0, &v5, batch, channels, spatial, groups, eps).unwrap();
        let v7 = dev().f504(&v6).unwrap();

        // Numeric gradient.
        let n_elem = input_data.len();
        let mut numeric_grad = vec![0.0f32; n_elem];
        for i in 0..n_elem {
            let mut inp_plus  = input_data.clone();
            let mut inp_minus = input_data.clone();
            inp_plus[i]  += h;
            inp_minus[i] -= h;

            let vp = dev().f502(&inp_plus);
            let out_plus = dev().f504(&dev().f600(&vp, &v1, &v2, batch, channels, spatial, groups, eps).unwrap()).unwrap();
            let loss_plus: f32 = out_plus.iter().map(|&x| x * x).sum::<f32>() / n;

            let vm = dev().f502(&inp_minus);
            let out_minus = dev().f504(&dev().f600(&vm, &v1, &v2, batch, channels, spatial, groups, eps).unwrap()).unwrap();
            let loss_minus: f32 = out_minus.iter().map(|&x| x * x).sum::<f32>() / n;

            numeric_grad[i] = (loss_plus - loss_minus) / (2.0 * h);
        }

        f544(&v7, &numeric_grad, 1e-2);
    }

    #[test]
    fn f600_constant_input() {
        let v0 = vec![5.0; 8];
        let v1 = vec![1.0, 1.0];
        let v2 = vec![0.0, 0.0];
        let v3 = dev().f504(&dev().f600(
            &dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2),
            1, 2, 4, 2, 1e-5
        ).unwrap()).unwrap();
        f544(&v3, &[0.0; 8], 1e-3);
    }

    #[test]
    fn f600_channels_not_divisible() {
        let v0 = vec![1.0; 5];
        let v1 = vec![1.0; 5];
        let v2 = vec![0.0; 5];
        assert!(dev().f600(&dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2), 1, 5, 1, 2, 1e-5).is_err());
    }

    #[test]
    fn f600_input_size_mismatch() {
        let v0 = vec![1.0; 10];
        let v1 = vec![1.0; 4];
        let v2 = vec![0.0; 4];
        assert!(dev().f600(&dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2), 1, 4, 4, 2, 1e-5).is_err());
    }

    // --- LayerNorm tests ---

    fn cpu_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        for r in 0..rows {
            let base = r * cols;
            let mean: f32 = input[base..base + cols].iter().sum::<f32>() / cols as f32;
            let var: f32 = input[base..base + cols].iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in 0..cols {
                out[base + c] = (input[base + c] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
        out
    }

    #[test]
    fn f602_unit_affine() {
        let v0 = vec![1.0f32, 3.0, 5.0, 7.0];
        let v1 = vec![1.0, 1.0];
        let v2 = vec![0.0, 0.0];
        let v3 = dev().f504(&dev().f602(
            &dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2),
            2, 2, 1e-5,
        ).unwrap()).unwrap();
        f544(&v3, &[-1.0, 1.0, -1.0, 1.0], 1e-3);
    }

    #[test]
    fn f602_with_affine() {
        let v0 = vec![1.0f32, 3.0];
        let v1 = vec![2.0, 0.5];
        let v2 = vec![1.0, -1.0];
        let v3 = cpu_layer_norm(&v0, &v1, &v2, 1, 2, 1e-5);
        let v4 = dev().f504(&dev().f602(
            &dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2),
            1, 2, 1e-5,
        ).unwrap()).unwrap();
        f544(&v4, &v3, 1e-3);
    }

    #[test]
    fn f602_transformer_shape() {
        let v0 = 4;
        let v1 = 16;
        let v2: Vec<f32> = (0..v0 * v1).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let v3: Vec<f32> = (0..v1).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let v4: Vec<f32> = (0..v1).map(|i| -0.5 + (i as f32) * 0.03).collect();
        let v5 = cpu_layer_norm(&v2, &v3, &v4, v0, v1, 1e-5);
        let v6 = dev().f504(&dev().f602(
            &dev().f502(&v2), &dev().f502(&v3), &dev().f502(&v4),
            v0 as u32, v1 as u32, 1e-5,
        ).unwrap()).unwrap();
        f544(&v6, &v5, 1e-3);
    }

    #[test]
    fn f602_constant_input() {
        let v0 = vec![7.0f32; 8];
        let v1 = vec![1.0; 4];
        let v2 = vec![0.0; 4];
        let v3 = dev().f504(&dev().f602(
            &dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2),
            2, 4, 1e-5,
        ).unwrap()).unwrap();
        f544(&v3, &[0.0; 8], 1e-3);
    }

    #[test]
    fn f602_size_mismatch() {
        let v0 = vec![1.0f32; 8];
        let v1 = vec![1.0; 4];
        let v2 = vec![0.0; 4];
        assert!(dev().f602(&dev().f502(&v0), &dev().f502(&v1), &dev().f502(&v2), 3, 4, 1e-5).is_err());
    }

    // --- RMSNorm tests ---

    fn cpu_rms_norm(input: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        for r in 0..rows {
            let base = r * cols;
            let mean_sq: f32 = input[base..base + cols].iter().map(|&x| x * x).sum::<f32>() / cols as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for c in 0..cols {
                out[base + c] = input[base + c] * inv_rms * weight[c];
            }
        }
        out
    }

    #[test]
    fn f603_unit_weight() {
        let v0 = vec![3.0f32, 4.0];
        let v1 = vec![1.0, 1.0];
        let v2 = cpu_rms_norm(&v0, &v1, 1, 2, 1e-5);
        let v3 = dev().f504(&dev().f603(
            &dev().f502(&v0), &dev().f502(&v1), 1, 2, 1e-5,
        ).unwrap()).unwrap();
        f544(&v3, &v2, 1e-3);
    }

    #[test]
    fn f603_with_weight() {
        let v0 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v1 = vec![2.0, 0.5, 1.5];
        let v2 = cpu_rms_norm(&v0, &v1, 2, 3, 1e-5);
        let v3 = dev().f504(&dev().f603(
            &dev().f502(&v0), &dev().f502(&v1), 2, 3, 1e-5,
        ).unwrap()).unwrap();
        f544(&v3, &v2, 1e-3);
    }

    #[test]
    fn f603_llama_shape() {
        let v0 = 8;
        let v1 = 64;
        let v2: Vec<f32> = (0..v0 * v1).map(|i| ((i as f32) * 0.013 - 0.5).sin()).collect();
        let v3: Vec<f32> = (0..v1).map(|i| 0.9 + (i as f32) * 0.001).collect();
        let v4 = cpu_rms_norm(&v2, &v3, v0, v1, 1e-6);
        let v5 = dev().f504(&dev().f603(
            &dev().f502(&v2), &dev().f502(&v3), v0 as u32, v1 as u32, 1e-6,
        ).unwrap()).unwrap();
        f544(&v5, &v4, 1e-3);
    }

    #[test]
    fn f603_zero_input() {
        let v0 = vec![0.0f32; 6];
        let v1 = vec![1.0; 3];
        let v2 = dev().f504(&dev().f603(
            &dev().f502(&v0), &dev().f502(&v1), 2, 3, 1e-5,
        ).unwrap()).unwrap();
        f544(&v2, &[0.0; 6], 1e-3);
    }

    #[test]
    fn f603_known_analytical_values() {
        // For input [3, 4] with weight [1, 1], the RMS is sqrt((9+16)/2) = sqrt(12.5).
        // Output: [3/sqrt(12.5), 4/sqrt(12.5)] = [0.84852814, 1.13137085].
        // Computed at f64 precision from the math definition — independent of both
        // our cpu_rms_norm helper and the shader (which use the same formula).
        let v0 = vec![3.0f32, 4.0];
        let v1 = vec![1.0f32, 1.0];
        let v2 = dev().f504(&dev().f603(
            &dev().f502(&v0), &dev().f502(&v1), 1, 2, 0.0,
        ).unwrap()).unwrap();
        f544(&v2, &[0.84852814, 1.13137085], 1e-4);
    }

    #[test]
    fn f603_known_with_per_col_weight() {
        // Input [1, 1, 1, 1] (rms = 1), weight [0.5, 1.0, 2.0, 4.0].
        // Output = input * 1 * weight = weight directly.
        let v0 = vec![1.0f32; 4];
        let v1 = vec![0.5f32, 1.0, 2.0, 4.0];
        let v2 = dev().f504(&dev().f603(
            &dev().f502(&v0), &dev().f502(&v1), 1, 4, 0.0,
        ).unwrap()).unwrap();
        f544(&v2, &[0.5, 1.0, 2.0, 4.0], 1e-5);
    }

    #[test]
    fn f603_size_mismatch() {
        let v0 = vec![1.0f32; 8];
        let v1 = vec![1.0; 4];
        assert!(dev().f603(&dev().f502(&v0), &dev().f502(&v1), 3, 4, 1e-5).is_err());
    }

    #[test]
    fn f603_weight_mismatch() {
        let v0 = vec![1.0f32; 12];
        let v1 = vec![1.0; 3];
        assert!(dev().f603(&dev().f502(&v0), &dev().f502(&v1), 3, 4, 1e-5).is_err());
    }

    // --- fused-path stride coverage (cols > 256 forces multiple iterations per thread) ---

    #[test]
    fn f603_fused_stride_cols512() {
        // cols=512: each of 256 threads iterates twice (c, c+256). Cross-validates CPU.
        let v0 = 4usize;
        let v1 = 512usize;
        let inp: Vec<f32> = (0..v0 * v1).map(|i| ((i as f32) * 0.007 - 1.0).cos()).collect();
        let w: Vec<f32> = (0..v1).map(|i| 0.5 + (i as f32) * 0.001).collect();
        let expected = cpu_rms_norm(&inp, &w, v0, v1, 1e-6);
        let got = dev().f504(&dev().f603(
            &dev().f502(&inp), &dev().f502(&w), v0 as u32, v1 as u32, 1e-6,
        ).unwrap()).unwrap();
        f544(&got, &expected, 1e-3);
    }

    #[test]
    fn f602_fused_stride_cols512() {
        // cols=512: stride loop for both sum and sum_sq reductions. Cross-validates CPU.
        let v0 = 4usize;
        let v1 = 512usize;
        let inp: Vec<f32> = (0..v0 * v1).map(|i| ((i as f32) * 0.007 - 1.0).sin()).collect();
        let gamma: Vec<f32> = (0..v1).map(|i| 0.8 + (i as f32) * 0.0005).collect();
        let beta: Vec<f32> = (0..v1).map(|i| -0.1 + (i as f32) * 0.0002).collect();
        let expected = cpu_layer_norm(&inp, &gamma, &beta, v0, v1, 1e-6);
        let got = dev().f504(&dev().f602(
            &dev().f502(&inp), &dev().f502(&gamma), &dev().f502(&beta),
            v0 as u32, v1 as u32, 1e-6,
        ).unwrap()).unwrap();
        f544(&got, &expected, 1e-3);
    }

    #[test]
    fn f603_fused_many_rows() {
        // Many rows (128) with moderate cols (64): exercises workgroup dispatch.
        let rows = 128usize;
        let cols = 64usize;
        let inp: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.031 - 2.0).tanh()).collect();
        let w: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let expected = cpu_rms_norm(&inp, &w, rows, cols, 1e-5);
        let got = dev().f504(&dev().f603(
            &dev().f502(&inp), &dev().f502(&w), rows as u32, cols as u32, 1e-5,
        ).unwrap()).unwrap();
        f544(&got, &expected, 1e-3);
    }

    #[test]
    fn f602_fused_many_rows() {
        // Many rows (128) with moderate cols (64): exercises workgroup dispatch.
        // Tolerance 2e-3: GPU uses E[x²]-E[x]² variance formula (more cancellation error
        // vs CPU's Σ(x-μ)²/n when input is not zero-centered).
        let rows = 128usize;
        let cols = 64usize;
        let inp: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.031 - 2.0).tanh()).collect();
        let gamma: Vec<f32> = vec![1.0f32; cols];
        let beta: Vec<f32> = vec![0.0f32; cols];
        let expected = cpu_layer_norm(&inp, &gamma, &beta, rows, cols, 1e-5);
        let got = dev().f504(&dev().f602(
            &dev().f502(&inp), &dev().f502(&gamma), &dev().f502(&beta),
            rows as u32, cols as u32, 1e-5,
        ).unwrap()).unwrap();
        f544(&got, &expected, 2e-3);
    }

    // CPU reference: layer_norm_backward.
    fn cpu_ln_bwd(
        grad_y: &[f32], input: &[f32], gamma: &[f32],
        rows: usize, cols: usize, eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut d_input = vec![0.0f32; rows * cols];
        let mut d_gamma = vec![0.0f32; cols];
        let mut d_beta  = vec![0.0f32; cols];
        for r in 0..rows {
            let base = r * cols;
            let mean: f32 = input[base..base + cols].iter().sum::<f32>() / cols as f32;
            let var: f32  = input[base..base + cols].iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            for c in 0..cols {
                let gy_g = grad_y[base + c] * gamma[c];
                let xhat = (input[base + c] - mean) * inv_std;
                s1 += gy_g;
                s2 += gy_g * xhat;
            }
            s1 /= cols as f32;
            s2 /= cols as f32;
            for c in 0..cols {
                let xhat = (input[base + c] - mean) * inv_std;
                d_input[base + c] = inv_std * (grad_y[base + c] * gamma[c] - s1 - xhat * s2);
                d_gamma[c] += grad_y[base + c] * xhat;
                d_beta[c]  += grad_y[base + c];
            }
        }
        (d_input, d_gamma, d_beta)
    }

    #[test]
    fn f791_numeric_grad_check() {
        let rows = 2usize;
        let cols = 4usize;
        let eps = 1e-5f32;
        let input: Vec<f32> = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.2, -0.4, 0.9];
        let gamma: Vec<f32> = vec![1.2, 0.8, 1.5, 0.6];
        let beta:  Vec<f32> = vec![0.1, -0.2, 0.3, -0.1];
        let grad_y: Vec<f32> = vec![1.0, -0.5, 0.8, -0.3, 0.6, -1.0, 0.4, -0.7];

        let (d_input_cpu, d_gamma_cpu, d_beta_cpu) = cpu_ln_bwd(&grad_y, &input, &gamma, rows, cols, eps);

        let (v0, v1, v2) = dev().f791(
            &dev().f502(&grad_y), &dev().f502(&input), &dev().f502(&gamma),
            rows as u32, cols as u32, eps,
        ).unwrap();

        f544(&dev().f504(&v0).unwrap(), &d_input_cpu, 1e-3);
        f544(&dev().f504(&v1).unwrap(), &d_gamma_cpu, 1e-3);
        f544(&dev().f504(&v2).unwrap(), &d_beta_cpu,  1e-3);
    }

    #[test]
    fn f791_finite_diff() {
        let rows = 2usize;
        let cols = 4usize;
        let eps = 1e-5f32;
        let h = 1e-3f32;
        let input: Vec<f32> = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.2, -0.4, 0.9];
        let gamma: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let beta:  Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

        let fwd = |inp: &[f32]| -> Vec<f32> {
            dev().f504(&dev().f602(
                &dev().f502(inp), &dev().f502(&gamma), &dev().f502(&beta),
                rows as u32, cols as u32, eps,
            ).unwrap()).unwrap()
        };

        let fwd_out = fwd(&input);
        let grad_y: Vec<f32> = fwd_out.iter().map(|&v| v * 0.1 + 0.05).collect();

        let (d_input_gpu, _, _) = dev().f791(
            &dev().f502(&grad_y), &dev().f502(&input), &dev().f502(&gamma),
            rows as u32, cols as u32, eps,
        ).unwrap();
        let d_input_gpu = dev().f504(&d_input_gpu).unwrap();

        for i in 0..rows * cols {
            let mut inp_p = input.clone();
            let mut inp_m = input.clone();
            inp_p[i] += h;
            inp_m[i] -= h;
            let fp: f32 = fwd(&inp_p).iter().zip(&grad_y).map(|(a, b)| a * b).sum();
            let fm: f32 = fwd(&inp_m).iter().zip(&grad_y).map(|(a, b)| a * b).sum();
            let numeric = (fp - fm) / (2.0 * h);
            assert!(
                (d_input_gpu[i] - numeric).abs() < 5e-2,
                "ln_bwd d_input[{i}]: gpu={} numeric={}", d_input_gpu[i], numeric
            );
        }
    }

    // CPU reference: rms_norm_backward.
    fn cpu_rn_bwd(
        grad_y: &[f32], input: &[f32], gamma: &[f32],
        rows: usize, cols: usize, eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut d_input = vec![0.0f32; rows * cols];
        let mut d_gamma = vec![0.0f32; cols];
        for r in 0..rows {
            let base = r * cols;
            let sum_sq: f32 = input[base..base + cols].iter().map(|&v| v * v).sum::<f32>() / cols as f32;
            let inv_rms = 1.0 / (sum_sq + eps).sqrt();
            let s: f32 = (0..cols).map(|c| grad_y[base + c] * gamma[c] * input[base + c]).sum::<f32>() / cols as f32;
            for c in 0..cols {
                d_input[base + c] = inv_rms * gamma[c] * grad_y[base + c] - input[base + c] * (s * inv_rms * inv_rms * inv_rms);
                d_gamma[c] += grad_y[base + c] * input[base + c] * inv_rms;
            }
        }
        (d_input, d_gamma)
    }

    #[test]
    fn f792_numeric_grad_check() {
        let rows = 2usize;
        let cols = 4usize;
        let eps = 1e-5f32;
        let input: Vec<f32> = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.2, -0.4, 0.9];
        let gamma: Vec<f32> = vec![1.2, 0.8, 1.5, 0.6];
        let grad_y: Vec<f32> = vec![1.0, -0.5, 0.8, -0.3, 0.6, -1.0, 0.4, -0.7];

        let (d_input_cpu, d_gamma_cpu) = cpu_rn_bwd(&grad_y, &input, &gamma, rows, cols, eps);

        let (v0, v1) = dev().f792(
            &dev().f502(&grad_y), &dev().f502(&input), &dev().f502(&gamma),
            rows as u32, cols as u32, eps,
        ).unwrap();

        f544(&dev().f504(&v0).unwrap(), &d_input_cpu, 1e-3);
        f544(&dev().f504(&v1).unwrap(), &d_gamma_cpu, 1e-3);
    }

    #[test]
    fn f792_finite_diff() {
        let rows = 2usize;
        let cols = 4usize;
        let eps = 1e-5f32;
        let h = 1e-3f32;
        let input: Vec<f32> = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.2, -0.4, 0.9];
        let gamma: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let fwd = |inp: &[f32]| -> Vec<f32> {
            dev().f504(&dev().f603(
                &dev().f502(inp), &dev().f502(&gamma),
                rows as u32, cols as u32, eps,
            ).unwrap()).unwrap()
        };

        let fwd_out = fwd(&input);
        let grad_y: Vec<f32> = fwd_out.iter().map(|&v| v * 0.1 + 0.05).collect();

        let (d_input_gpu, _) = dev().f792(
            &dev().f502(&grad_y), &dev().f502(&input), &dev().f502(&gamma),
            rows as u32, cols as u32, eps,
        ).unwrap();
        let d_input_gpu = dev().f504(&d_input_gpu).unwrap();

        for i in 0..rows * cols {
            let mut inp_p = input.clone();
            let mut inp_m = input.clone();
            inp_p[i] += h;
            inp_m[i] -= h;
            let fp: f32 = fwd(&inp_p).iter().zip(&grad_y).map(|(a, b)| a * b).sum();
            let fm: f32 = fwd(&inp_m).iter().zip(&grad_y).map(|(a, b)| a * b).sum();
            let numeric = (fp - fm) / (2.0 * h);
            assert!(
                (d_input_gpu[i] - numeric).abs() < 5e-2,
                "rn_bwd d_input[{i}]: gpu={} numeric={}", d_input_gpu[i], numeric
            );
        }
    }
}
