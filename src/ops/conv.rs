// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// f580=matmul, f581=batch_matmul, f582=conv2d, f583=conv_transpose2d,
// f584=conv2d_grad_weight, f585=conv2d_grad_bias.

use crate::device::{t500, t501};
use anyhow::{ensure, Result};

// --- Matmul ---

/// t528 = MatmulDims. Dims uniform for tiled matmul.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t528 {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

// Single-wavefront 4x4 register-blocked tiled matmul.
// @workgroup_size(64) = exactly one wave64 on RDNA1/RDNA2.
// workgroupBarrier inside a single wavefront is a wavefront-local fence
// (no cross-wavefront coordination), eliminating ~75% of barrier overhead
// vs the previous 4-wavefront (16x16) workgroup.
// Thread layout within the workgroup: tr = t/8 (0..8), tc = t%8 (0..8).
// Each thread owns a 4x4 output subblock → 64 threads × 16 outputs = 1024 = 32x32.
// LDS: 2 × 32x32 × 4B = 8 KB/workgroup (fits 8 per RDNA1 CU at 64 KB LDS).
// Tile load: 64 threads × 16 elements = 1024 elements per A-tile and B-tile.
const SHADER_MATMUL: &str = "
const TILE: u32 = 32u;
struct Dims { m: u32, n: u32, k: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> tile_a: array<f32, 1024>;
var<workgroup> tile_b: array<f32, 1024>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)          wg: vec3<u32>,
    @builtin(local_invocation_index) t:  u32,
) {
    let tr   = t / 8u;
    let tc   = t % 8u;
    let row0 = wg.x * TILE + tr * 4u;
    let col0 = wg.y * TILE + tc * 4u;

    var acc0:  f32 = 0.0; var acc1:  f32 = 0.0; var acc2:  f32 = 0.0; var acc3:  f32 = 0.0;
    var acc4:  f32 = 0.0; var acc5:  f32 = 0.0; var acc6:  f32 = 0.0; var acc7:  f32 = 0.0;
    var acc8:  f32 = 0.0; var acc9:  f32 = 0.0; var acc10: f32 = 0.0; var acc11: f32 = 0.0;
    var acc12: f32 = 0.0; var acc13: f32 = 0.0; var acc14: f32 = 0.0; var acc15: f32 = 0.0;

    let num_tiles = (dims.k + TILE - 1u) / TILE;

    for (var tile: u32 = 0u; tile < num_tiles; tile++) {
        // 64 threads load 16 elements each = 1024 elements per A-tile and B-tile.
        for (var i: u32 = 0u; i < 16u; i++) {
            let e  = t + i * 64u;
            let er = e / TILE;
            let ec = e % TILE;

            let ga_r = wg.x * TILE + er;
            let ga_c = tile * TILE + ec;
            if ga_r < dims.m && ga_c < dims.k {
                tile_a[e] = a[ga_r * dims.k + ga_c];
            } else {
                tile_a[e] = 0.0;
            }

            let gb_r = tile * TILE + er;
            let gb_c = wg.y * TILE + ec;
            if gb_r < dims.k && gb_c < dims.n {
                tile_b[e] = b[gb_r * dims.n + gb_c];
            } else {
                tile_b[e] = 0.0;
            }
        }

        workgroupBarrier();

        // 4x4 register-block: 8 LDS reads feed 16 FMAs per k-step.
        // a0..a3 each reused across 4 b values; b0..b3 each reused across 4 a values.
        for (var k: u32 = 0u; k < TILE; k++) {
            let a0 = tile_a[(tr * 4u     ) * TILE + k];
            let a1 = tile_a[(tr * 4u + 1u) * TILE + k];
            let a2 = tile_a[(tr * 4u + 2u) * TILE + k];
            let a3 = tile_a[(tr * 4u + 3u) * TILE + k];
            let b0 = tile_b[k * TILE + tc * 4u     ];
            let b1 = tile_b[k * TILE + tc * 4u + 1u];
            let b2 = tile_b[k * TILE + tc * 4u + 2u];
            let b3 = tile_b[k * TILE + tc * 4u + 3u];
            acc0  = fma(a0, b0, acc0);  acc1  = fma(a0, b1, acc1);
            acc2  = fma(a0, b2, acc2);  acc3  = fma(a0, b3, acc3);
            acc4  = fma(a1, b0, acc4);  acc5  = fma(a1, b1, acc5);
            acc6  = fma(a1, b2, acc6);  acc7  = fma(a1, b3, acc7);
            acc8  = fma(a2, b0, acc8);  acc9  = fma(a2, b1, acc9);
            acc10 = fma(a2, b2, acc10); acc11 = fma(a2, b3, acc11);
            acc12 = fma(a3, b0, acc12); acc13 = fma(a3, b1, acc13);
            acc14 = fma(a3, b2, acc14); acc15 = fma(a3, b3, acc15);
        }

        workgroupBarrier();
    }

    let r0 = row0; let r1 = row0 + 1u; let r2 = row0 + 2u; let r3 = row0 + 3u;
    let c0 = col0; let c1 = col0 + 1u; let c2 = col0 + 2u; let c3 = col0 + 3u;
    if r0 < dims.m {
        if c0 < dims.n { out[r0 * dims.n + c0] = acc0;  }
        if c1 < dims.n { out[r0 * dims.n + c1] = acc1;  }
        if c2 < dims.n { out[r0 * dims.n + c2] = acc2;  }
        if c3 < dims.n { out[r0 * dims.n + c3] = acc3;  }
    }
    if r1 < dims.m {
        if c0 < dims.n { out[r1 * dims.n + c0] = acc4;  }
        if c1 < dims.n { out[r1 * dims.n + c1] = acc5;  }
        if c2 < dims.n { out[r1 * dims.n + c2] = acc6;  }
        if c3 < dims.n { out[r1 * dims.n + c3] = acc7;  }
    }
    if r2 < dims.m {
        if c0 < dims.n { out[r2 * dims.n + c0] = acc8;  }
        if c1 < dims.n { out[r2 * dims.n + c1] = acc9;  }
        if c2 < dims.n { out[r2 * dims.n + c2] = acc10; }
        if c3 < dims.n { out[r2 * dims.n + c3] = acc11; }
    }
    if r3 < dims.m {
        if c0 < dims.n { out[r3 * dims.n + c0] = acc12; }
        if c1 < dims.n { out[r3 * dims.n + c1] = acc13; }
        if c2 < dims.n { out[r3 * dims.n + c2] = acc14; }
        if c3 < dims.n { out[r3 * dims.n + c3] = acc15; }
    }
}
";

// --- Batched Matmul ---

/// t529 = BatchMatmulDims. Dims uniform for batched matmul.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t529 {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
}

const SHADER_BATCH_MATMUL: &str = "
const TILE: u32 = 16u;
struct Dims { batch: u32, m: u32, n: u32, k: u32, }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;
    let bat = gid.z;
    if bat >= dims.batch { return; }
    let lr = lid.x;
    let lc = lid.y;
    let a_off = bat * dims.m * dims.k;
    let b_off = bat * dims.k * dims.n;
    let o_off = bat * dims.m * dims.n;

    var sum: f32 = 0.0;
    let num_tiles = (dims.k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        let a_col = t * TILE + lc;
        if row < dims.m && a_col < dims.k {
            tile_a[lr * TILE + lc] = a[a_off + row * dims.k + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        let b_row = t * TILE + lr;
        if b_row < dims.k && col < dims.n {
            tile_b[lr * TILE + lc] = b[b_off + b_row * dims.n + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i++) {
            sum += tile_a[lr * TILE + i] * tile_b[i * TILE + lc];
        }

        workgroupBarrier();
    }

    if row < dims.m && col < dims.n {
        out[o_off + row * dims.n + col] = sum;
    }
}
";

// --- Conv2d ---

/// t530 = Conv2dParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t530 {
    batch: u32,
    in_c: u32,
    out_c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    groups: u32,
}

const SHADER_CONV2D: &str = "
struct P {
    batch: u32, in_c: u32, out_c: u32, in_h: u32,
    in_w: u32, out_h: u32, out_w: u32, kh: u32,
    kw: u32, stride_h: u32, stride_w: u32, pad_h: u32,
    pad_w: u32, dilation_h: u32, dilation_w: u32, groups: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.out_c * p.out_h * p.out_w;
    if idx >= total { return; }

    let ow = idx % p.out_w;
    let oh = (idx / p.out_w) % p.out_h;
    let oc = (idx / (p.out_w * p.out_h)) % p.out_c;
    let n  = idx / (p.out_w * p.out_h * p.out_c);

    let group_in = p.in_c / p.groups;
    let group_out = p.out_c / p.groups;
    let g = oc / group_out;

    var sum: f32 = bias[oc];
    for (var ic: u32 = 0u; ic < group_in; ic++) {
        for (var kh: u32 = 0u; kh < p.kh; kh++) {
            for (var kw: u32 = 0u; kw < p.kw; kw++) {
                let ih = oh * p.stride_h + kh * p.dilation_h - p.pad_h;
                let iw = ow * p.stride_w + kw * p.dilation_w - p.pad_w;
                if ih < p.in_h && iw < p.in_w {
                    let in_idx = n * (p.in_c * p.in_h * p.in_w)
                               + (g * group_in + ic) * (p.in_h * p.in_w)
                               + ih * p.in_w + iw;
                    let w_idx = oc * (group_in * p.kh * p.kw)
                              + ic * (p.kh * p.kw)
                              + kh * p.kw + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    out[idx] = sum;
}
";

// --- TransposeConv2d ---

/// t531 = ConvTranspose2dParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t531 {
    batch: u32,
    in_c: u32,
    out_c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    groups: u32,
}

const SHADER_CONV_TRANSPOSE2D: &str = "
struct P {
    batch: u32, in_c: u32, out_c: u32, in_h: u32,
    in_w: u32, out_h: u32, out_w: u32, kh: u32,
    kw: u32, stride_h: u32, stride_w: u32, pad_h: u32,
    pad_w: u32, dilation_h: u32, dilation_w: u32, groups: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.out_c * p.out_h * p.out_w;
    if idx >= total { return; }

    let ow = idx % p.out_w;
    let oh = (idx / p.out_w) % p.out_h;
    let oc = (idx / (p.out_w * p.out_h)) % p.out_c;
    let n  = idx / (p.out_w * p.out_h * p.out_c);

    let group_in = p.in_c / p.groups;
    let group_out = p.out_c / p.groups;
    let g = oc / group_out;
    let oc_local = oc % group_out;

    var sum: f32 = bias[oc];
    for (var ic: u32 = 0u; ic < group_in; ic++) {
        for (var kh: u32 = 0u; kh < p.kh; kh++) {
            for (var kw: u32 = 0u; kw < p.kw; kw++) {
                // Transposed conv: output pixel (oh,ow) is affected by input pixel (ih,iw)
                // where oh = ih * stride - pad + kh * dilation
                // so ih = (oh + pad - kh * dilation) / stride, must be exact division
                let oh_off = oh + p.pad_h - kh * p.dilation_h;
                let ow_off = ow + p.pad_w - kw * p.dilation_w;
                // Check exact divisibility and bounds (unsigned wraparound handles negatives)
                if oh_off % p.stride_h == 0u && ow_off % p.stride_w == 0u {
                    let ih = oh_off / p.stride_h;
                    let iw = ow_off / p.stride_w;
                    if ih < p.in_h && iw < p.in_w {
                        let in_idx = n * (p.in_c * p.in_h * p.in_w)
                                   + (g * group_in + ic) * (p.in_h * p.in_w)
                                   + ih * p.in_w + iw;
                        // weight: [in_c, out_c/groups, kh, kw]
                        let w_idx = (g * group_in + ic) * (group_out * p.kh * p.kw)
                                  + oc_local * (p.kh * p.kw)
                                  + kh * p.kw + kw;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    out[idx] = sum;
}
";

// --- Conv2d gradient shaders ---

/// t532 = Conv2dGradParams. Used for both grad_weight and matches forward layout.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t532 {
    batch: u32,
    in_c: u32,
    out_c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    groups: u32,
}

/// t533 = Conv2dGradBiasParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t533 {
    batch: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
}

// grad_weight[oc, ic_local, kh, kw] = sum_{n,oh,ow} grad_out[n,oc,oh,ow] * input[n, g*group_in+ic_local, oh*sh+kh*dh-ph, ow*sw+kw*dw-pw]
const SHADER_CONV2D_GRAD_WEIGHT: &str = "
struct P {
    batch: u32, in_c: u32, out_c: u32, in_h: u32,
    in_w: u32, out_h: u32, out_w: u32, kh: u32,
    kw: u32, stride_h: u32, stride_w: u32, pad_h: u32,
    pad_w: u32, dilation_h: u32, dilation_w: u32, groups: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_w: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let group_in = p.in_c / p.groups;
    let group_out = p.out_c / p.groups;
    let total = p.out_c * group_in * p.kh * p.kw;
    if idx >= total { return; }

    let kw_i = idx % p.kw;
    let kh_i = (idx / p.kw) % p.kh;
    let ic = (idx / (p.kw * p.kh)) % group_in;
    let oc = idx / (p.kw * p.kh * group_in);

    let g = oc / group_out;

    var sum: f32 = 0.0;
    for (var n: u32 = 0u; n < p.batch; n++) {
        for (var oh: u32 = 0u; oh < p.out_h; oh++) {
            for (var ow: u32 = 0u; ow < p.out_w; ow++) {
                let ih = oh * p.stride_h + kh_i * p.dilation_h;
                let iw = ow * p.stride_w + kw_i * p.dilation_w;
                if ih >= p.pad_h && iw >= p.pad_w {
                    let ih2 = ih - p.pad_h;
                    let iw2 = iw - p.pad_w;
                    if ih2 < p.in_h && iw2 < p.in_w {
                        let in_idx = n * (p.in_c * p.in_h * p.in_w)
                                   + (g * group_in + ic) * (p.in_h * p.in_w)
                                   + ih2 * p.in_w + iw2;
                        let go_idx = n * (p.out_c * p.out_h * p.out_w)
                                   + oc * (p.out_h * p.out_w)
                                   + oh * p.out_w + ow;
                        sum += grad_out[go_idx] * input[in_idx];
                    }
                }
            }
        }
    }
    grad_w[idx] = sum;
}
";

// grad_bias[oc] = sum_{n,oh,ow} grad_out[n,oc,oh,ow]
const SHADER_CONV2D_GRAD_BIAS: &str = "
struct P { batch: u32, out_c: u32, out_h: u32, out_w: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_bias: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oc = gid.x + gid.y * 65535u * 256u;
    if oc >= p.out_c { return; }
    var sum: f32 = 0.0;
    for (var n: u32 = 0u; n < p.batch; n++) {
        for (var oh: u32 = 0u; oh < p.out_h; oh++) {
            for (var ow: u32 = 0u; ow < p.out_w; ow++) {
                let idx = n * (p.out_c * p.out_h * p.out_w)
                        + oc * (p.out_h * p.out_w)
                        + oh * p.out_w + ow;
                sum += grad_out[idx];
            }
        }
    }
    grad_bias[oc] = sum;
}
";

impl t500 {
    /// f580 = matmul. A(m,k) x B(k,n) = C(m,n). Row-major layout.
    pub fn f580(&self, p0: &t501, p1: &t501, p2: u32, p3: u32, p4: u32) -> Result<t501> {
        ensure!(p0.s507 == (p2 * p4) as usize, "matmul: A has {} elems, expected {}", p0.s507, p2 * p4);
        ensure!(p1.s507 == (p4 * p3) as usize, "matmul: B has {} elems, expected {}", p1.s507, p4 * p3);
        let v0 = self.f503((p2 * p3) as usize);
        let v1 = t528 { m: p2, n: p3, k: p4, _pad: 0 };
        self.f543(SHADER_MATMUL, Some("matmul"), &v1, &[p0, p1], &v0, (p2.div_ceil(32), p3.div_ceil(32), 1));
        Ok(v0)
    }

    /// f581 = batch_matmul. A[batch,m,k] x B[batch,k,n] = C[batch,m,n].
    pub fn f581(&self, p0: &t501, p1: &t501, p2: u32, p3: u32, p4: u32, p5: u32) -> Result<t501> {
        ensure!(p0.s507 == (p2 * p3 * p5) as usize);
        ensure!(p1.s507 == (p2 * p5 * p4) as usize);
        let v0 = self.f503((p2 * p3 * p4) as usize);
        let v1 = t529 { batch: p2, m: p3, n: p4, k: p5 };
        self.f543(SHADER_BATCH_MATMUL, Some("batch_matmul"), &v1, &[p0, p1], &v0, (p3.div_ceil(16), p4.div_ceil(16), p2));
        Ok(v0)
    }

    /// f582 = conv2d. input[N,C_in,H,W] * weight[C_out,C_in/groups,kH,kW] + bias[C_out].
    /// NCHW layout. Returns output[N,C_out,out_H,out_W].
    pub fn f582(
        &self,
        p0: &t501,
        p1: &t501,
        p2: Option<&t501>,
        p3: u32, p4: u32, p5: u32, p6: u32,
        p7: u32, p8: u32, p9: u32,
        p10: (u32, u32), p11: (u32, u32),
        p12: (u32, u32), p13: u32,
    ) -> Result<t501> {
        let v0 = (p5 + 2 * p11.0 - p12.0 * (p8 - 1) - 1) / p10.0 + 1;
        let v1 = (p6 + 2 * p11.1 - p12.1 * (p9 - 1) - 1) / p10.1 + 1;
        let v2 = p3 * p7 * v0 * v1;

        ensure!(p0.s507 == (p3 * p4 * p5 * p6) as usize);
        ensure!(p1.s507 == (p7 * (p4 / p13) * p8 * p9) as usize);

        let v3;
        let v4 = match p2 {
            Some(v5) => {
                ensure!(v5.s507 == p7 as usize);
                v5
            }
            None => {
                v3 = self.f502(&vec![0.0f32; p7 as usize]);
                &v3
            }
        };

        let v6 = self.f503(v2 as usize);
        let v7 = t530 {
            batch: p3, in_c: p4, out_c: p7, in_h: p5, in_w: p6, out_h: v0, out_w: v1,
            kh: p8, kw: p9, stride_h: p10.0, stride_w: p10.1,
            pad_h: p11.0, pad_w: p11.1,
            dilation_h: p12.0, dilation_w: p12.1, groups: p13,
        };

        self.f543(
            SHADER_CONV2D, Some("conv2d"),
            &v7, &[p0, p1, v4], &v6,
            super::f540(v2),
        );
        Ok(v6)
    }

    /// f583 = conv_transpose2d. input[N,C_in,H,W] -> output[N,C_out,out_H,out_W].
    /// Weight layout: [C_in, C_out/groups, kH, kW].
    pub fn f583(
        &self,
        p0: &t501,
        p1: &t501,
        p2: Option<&t501>,
        p3: u32, p4: u32, p5: u32, p6: u32,
        p7: u32, p8: u32, p9: u32,
        p10: (u32, u32), p11: (u32, u32),
        p12: (u32, u32),
        p13: (u32, u32), p14: u32,
    ) -> Result<t501> {
        let v0 = (p5 - 1) * p10.0 - 2 * p11.0 + p13.0 * (p8 - 1) + p12.0 + 1;
        let v1 = (p6 - 1) * p10.1 - 2 * p11.1 + p13.1 * (p9 - 1) + p12.1 + 1;
        let v2 = p3 * p7 * v0 * v1;

        ensure!(p0.s507 == (p3 * p4 * p5 * p6) as usize);
        ensure!(p1.s507 == (p4 * (p7 / p14) * p8 * p9) as usize);

        let v3;
        let v4 = match p2 {
            Some(v5) => {
                ensure!(v5.s507 == p7 as usize);
                v5
            }
            None => {
                v3 = self.f502(&vec![0.0f32; p7 as usize]);
                &v3
            }
        };

        let v6 = self.f503(v2 as usize);
        let v7 = t531 {
            batch: p3, in_c: p4, out_c: p7, in_h: p5, in_w: p6, out_h: v0, out_w: v1,
            kh: p8, kw: p9, stride_h: p10.0, stride_w: p10.1,
            pad_h: p11.0, pad_w: p11.1,
            dilation_h: p13.0, dilation_w: p13.1, groups: p14,
        };

        self.f543(
            SHADER_CONV_TRANSPOSE2D, Some("conv_transpose2d"),
            &v7, &[p0, p1, v4], &v6,
            super::f540(v2),
        );
        Ok(v6)
    }

    /// f584 = conv2d_grad_weight. Backward gradient w.r.t. conv2d weights.
    pub(crate) fn f584(
        &self, p0: &t501, p1: &t501,
        p2: u32, p3: u32, p4: u32, p5: u32,
        p6: u32, p7: u32, p8: u32, p9: u32, p10: u32,
        p11: u32, p12: u32, p13: u32, p14: u32,
        p15: u32, p16: u32, p17: u32,
    ) -> Result<t501> {
        let v0 = p3 / p17;
        let v1 = p6 * v0 * p9 * p10;
        let v2 = self.f503(v1 as usize);
        let v3 = t532 {
            batch: p2, in_c: p3, out_c: p6, in_h: p4, in_w: p5, out_h: p7, out_w: p8,
            kh: p9, kw: p10, stride_h: p11, stride_w: p12, pad_h: p13, pad_w: p14,
            dilation_h: p15, dilation_w: p16, groups: p17,
        };
        self.f543(
            SHADER_CONV2D_GRAD_WEIGHT, Some("conv2d_grad_weight"),
            &v3, &[p0, p1], &v2,
            super::f540(v1),
        );
        Ok(v2)
    }

    /// f585 = conv2d_grad_bias. Backward gradient w.r.t. conv2d bias.
    pub(crate) fn f585(
        &self, p0: &t501,
        p1: u32, p2: u32, p3: u32, p4: u32,
    ) -> Result<t501> {
        let v0 = self.f503(p2 as usize);
        let v1 = t533 { batch: p1, out_c: p2, out_h: p3, out_w: p4 };
        self.f543(
            SHADER_CONV2D_GRAD_BIAS, Some("conv2d_grad_bias"),
            &v1, &[p0], &v0,
            super::f540(p2),
        );
        Ok(v0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;

    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    // CPU reference matmul for cross-validation
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0;
                for i in 0..k { sum += a[row * k + i] * b[i * n + col]; }
                out[row * n + col] = sum;
            }
        }
        out
    }

    // CPU reference conv2d for cross-validation
    fn cpu_conv2d(
        input: &[f32], weight: &[f32], bias: &[f32],
        batch: usize, in_c: usize, in_h: usize, in_w: usize,
        out_c: usize, kh: usize, kw: usize,
        stride: (usize, usize), padding: (usize, usize), groups: usize,
    ) -> Vec<f32> {
        let out_h = (in_h + 2 * padding.0 - kh) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kw) / stride.1 + 1;
        let group_in = in_c / groups;
        let group_out = out_c / groups;
        let mut out = vec![0.0f32; batch * out_c * out_h * out_w];
        for n in 0..batch {
            for oc in 0..out_c {
                let g = oc / group_out;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = bias[oc];
                        for ic in 0..group_in {
                            for kr in 0..kh {
                                for kc in 0..kw {
                                    let ih = oh * stride.0 + kr;
                                    let iw = ow * stride.1 + kc;
                                    let ih = ih as isize - padding.0 as isize;
                                    let iw = iw as isize - padding.1 as isize;
                                    if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                        let in_idx = n * in_c * in_h * in_w
                                            + (g * group_in + ic) * in_h * in_w
                                            + ih as usize * in_w + iw as usize;
                                        let w_idx = oc * group_in * kh * kw + ic * kh * kw + kr * kw + kc;
                                        sum += input[in_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                        out[n * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = sum;
                    }
                }
            }
        }
        out
    }

    // --- Matmul tests ---

    #[test]
    fn f580_2x2() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0]);
        let v1 = dev().f502(&[5.0, 6.0, 7.0, 8.0]);
        let v2 = dev().f504(&dev().f580(&v0, &v1, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(v2, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn f580_nonsquare_vs_cpu() {
        // 3x4 @ 4x2 = 3x2
        let v0: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let v1: Vec<f32> = (1..=8).map(|x| x as f32 * 0.1).collect();
        let v2 = cpu_matmul(&v0, &v1, 3, 2, 4);
        let v3 = dev().f504(&dev().f580(&dev().f502(&v0), &dev().f502(&v1), 3, 2, 4).unwrap()).unwrap();
        f544(&v3, &v2, 1e-4);
    }

    #[test]
    fn f580_1x1() {
        let v0 = dev().f504(&dev().f580(&dev().f502(&[3.0]), &dev().f502(&[7.0]), 1, 1, 1).unwrap()).unwrap();
        assert_eq!(v0, vec![21.0]);
    }

    #[test]
    fn f580_17x13_vs_cpu() {
        // Odd dims that don't align to 16x16 workgroup
        let v0 = 17; let v1 = 13; let v2 = 11;
        let v3: Vec<f32> = (0..v0*v2).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let v4: Vec<f32> = (0..v2*v1).map(|i| (i as f32 * 0.01) - 0.3).collect();
        let v5 = cpu_matmul(&v3, &v4, v0, v1, v2);
        let v6 = dev().f504(&dev().f580(
            &dev().f502(&v3), &dev().f502(&v4), v0 as u32, v1 as u32, v2 as u32
        ).unwrap()).unwrap();
        f544(&v6, &v5, 1e-3);
    }

    // --- Batch matmul tests ---

    #[test]
    fn f581_basic() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let v1 = dev().f502(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let v2 = dev().f504(&dev().f581(&v0, &v1, 2, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(v2, vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn f581_nonsquare() {
        // batch=1, 2x3 @ 3x1 = 2x1
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v1 = dev().f502(&[1.0, 1.0, 1.0]);
        let v2 = dev().f504(&dev().f581(&v0, &v1, 1, 2, 1, 3).unwrap()).unwrap();
        assert_eq!(v2, vec![6.0, 15.0]);
    }

    // --- Conv2d tests ---

    #[test]
    fn f582_3x3_vs_cpu() {
        // Verify GPU conv2d matches CPU reference
        let v0: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let v1 = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
        let v2 = vec![0.0];
        let v3 = cpu_conv2d(&v0, &v1, &v2, 1, 1, 4, 4, 1, 3, 3, (1,1), (0,0), 1);
        let v4 = dev().f504(&dev().f582(
            &dev().f502(&v0), &dev().f502(&v1), Some(&dev().f502(&v2)),
            1, 1, 4, 4, 1, 3, 3, (1,1), (0,0), (1,1), 1
        ).unwrap()).unwrap();
        f544(&v4, &v3, 1e-5);
    }

    #[test]
    fn f582_1x1_kernel() {
        // 1x1 conv = per-pixel channel mixing
        // 2 in channels, 3 out channels, 2x2 spatial
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        // weight[out_c, in_c, 1, 1]: 3 output channels, 2 input channels
        let v1 = dev().f502(&[1.0, 0.5, 0.0, 1.0, -1.0, 2.0]);
        let v2 = dev().f502(&[0.0, 0.0, 0.0]);
        let v3 = dev().f504(&dev().f582(&v0, &v1, Some(&v2),
            1, 2, 2, 2, 3, 1, 1, (1,1), (0,0), (1,1), 1).unwrap()).unwrap();
        // out_c=0: 1.0*in0 + 0.5*in1
        f544(&v3[0..4], &[6.0, 12.0, 18.0, 24.0], 1e-5);
        // out_c=1: 0.0*in0 + 1.0*in1
        f544(&v3[4..8], &[10.0, 20.0, 30.0, 40.0], 1e-5);
    }

    #[test]
    fn f582_padding_vs_cpu() {
        let v0: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let v1 = vec![1.0; 9];
        let v2 = vec![0.0];
        let v3 = cpu_conv2d(&v0, &v1, &v2, 1, 1, 3, 3, 1, 3, 3, (1,1), (1,1), 1);
        let v4 = dev().f504(&dev().f582(
            &dev().f502(&v0), &dev().f502(&v1), None,
            1, 1, 3, 3, 1, 3, 3, (1,1), (1,1), (1,1), 1
        ).unwrap()).unwrap();
        f544(&v4, &v3, 1e-5);
    }

    #[test]
    fn f582_stride2() {
        let v0: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let v1 = dev().f504(&dev().f582(
            &dev().f502(&v0), &dev().f502(&[1.0]), None,
            1, 1, 4, 4, 1, 1, 1, (2,2), (0,0), (1,1), 1
        ).unwrap()).unwrap();
        assert_eq!(v1, vec![1.0, 3.0, 9.0, 11.0]);
    }

    #[test]
    fn f582_5x5_kernel_vs_cpu() {
        // 1x1x8x8 input, 1x1x5x5 kernel, padding=2 -> 8x8 output
        let v0: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let v1: Vec<f32> = (0..25).map(|i| if i == 12 { 1.0 } else { 0.0 }).collect();
        let v2 = vec![0.0];
        let v3 = cpu_conv2d(&v0, &v1, &v2, 1, 1, 8, 8, 1, 5, 5, (1,1), (2,2), 1);
        let v4 = dev().f504(&dev().f582(
            &dev().f502(&v0), &dev().f502(&v1), None,
            1, 1, 8, 8, 1, 5, 5, (1,1), (2,2), (1,1), 1
        ).unwrap()).unwrap();
        f544(&v4, &v3, 1e-4);
    }

    #[test]
    fn f582_multichannel_vs_cpu() {
        // batch=2, in_c=3, out_c=2, 4x4, 3x3 kernel, padding=1
        let v0 = 2; let v1 = 3; let v2 = 2; let v3 = 4; let v4 = 4;
        let v5: Vec<f32> = (0..v0*v1*v3*v4).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let v6: Vec<f32> = (0..v2*v1*3*3).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let v7 = vec![0.1, -0.2];
        let v8 = cpu_conv2d(&v5, &v6, &v7, v0, v1, v3, v4, v2, 3, 3, (1,1), (1,1), 1);
        let v9 = dev().f504(&dev().f582(
            &dev().f502(&v5), &dev().f502(&v6), Some(&dev().f502(&v7)),
            v0 as u32, v1 as u32, v3 as u32, v4 as u32, v2 as u32, 3, 3, (1,1), (1,1), (1,1), 1
        ).unwrap()).unwrap();
        f544(&v9, &v8, 1e-3);
    }

    #[test]
    fn f582_with_bias() {
        // Verify bias is added correctly
        let v0 = dev().f502(&[0.0; 4]);
        let v1 = dev().f502(&[0.0]);
        let v2 = dev().f502(&[42.0]);
        let v3 = dev().f504(&dev().f582(&v0, &v1, Some(&v2),
            1, 1, 2, 2, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap()).unwrap();
        assert_eq!(v3, vec![42.0, 42.0, 42.0, 42.0]);
    }

    // --- Transpose conv2d tests ---

    #[test]
    fn f583_stride2() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0]);
        let v1 = dev().f502(&[1.0]);
        let v2 = dev().f504(&dev().f583(&v0, &v1, None,
            1, 1, 2, 2, 1, 1, 1, (2,2), (0,0), (0,0), (1,1), 1).unwrap()).unwrap();
        assert_eq!(v2.len(), 9);
        f544(&v2, &[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0], 1e-5);
    }

    #[test]
    fn f583_3x3_kernel() {
        let v0 = dev().f502(&[5.0]);
        let v1 = dev().f502(&[1.0; 9]);
        let v2 = dev().f504(&dev().f583(&v0, &v1, None,
            1, 1, 1, 1, 1, 3, 3, (1,1), (0,0), (0,0), (1,1), 1).unwrap()).unwrap();
        assert_eq!(v2.len(), 9);
        f544(&v2, &[5.0; 9], 1e-5);
    }

    // --- Error path tests ---

    #[test]
    fn f580_a_size_mismatch() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0]);
        let v1 = dev().f502(&[1.0, 2.0, 3.0, 4.0]);
        assert!(dev().f580(&v0, &v1, 2, 2, 2).is_err());
    }

    #[test]
    fn f580_b_size_mismatch() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0]);
        let v1 = dev().f502(&[1.0, 2.0, 3.0]);
        assert!(dev().f580(&v0, &v1, 2, 2, 2).is_err());
    }

    #[test]
    fn f582_input_size_mismatch() {
        let v0 = dev().f502(&[1.0; 10]);
        let v1 = dev().f502(&[1.0; 9]);
        assert!(dev().f582(&v0, &v1, None, 1, 1, 4, 4, 1, 3, 3, (1,1), (0,0), (1,1), 1).is_err());
    }

    #[test]
    fn f582_weight_size_mismatch() {
        let v0 = dev().f502(&[1.0; 16]);
        let v1 = dev().f502(&[1.0; 5]);
        assert!(dev().f582(&v0, &v1, None, 1, 1, 4, 4, 1, 3, 3, (1,1), (0,0), (1,1), 1).is_err());
    }

    #[test]
    fn f582_bias_size_mismatch() {
        let v0 = dev().f502(&[1.0; 16]);
        let v1 = dev().f502(&[1.0; 9]);
        let v2 = dev().f502(&[1.0, 2.0]);
        assert!(dev().f582(&v0, &v1, Some(&v2), 1, 1, 4, 4, 1, 3, 3, (1,1), (0,0), (1,1), 1).is_err());
    }

    // --- CPU cross-validation for batch_matmul ---

    #[test]
    fn f581_vs_cpu() {
        let v0 = 3; let v1 = 4; let v2 = 3; let v3 = 5;
        let v4: Vec<f32> = (0..v0*v1*v3).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let v5: Vec<f32> = (0..v0*v3*v2).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let mut v6 = vec![0.0f32; v0 * v1 * v2];
        for bat in 0..v0 {
            for row in 0..v1 {
                for col in 0..v2 {
                    let mut sum = 0.0;
                    for i in 0..v3 {
                        sum += v4[bat*v1*v3 + row*v3 + i] * v5[bat*v3*v2 + i*v2 + col];
                    }
                    v6[bat*v1*v2 + row*v2 + col] = sum;
                }
            }
        }
        let v7 = dev().f504(&dev().f581(
            &dev().f502(&v4), &dev().f502(&v5), v0 as u32, v1 as u32, v2 as u32, v3 as u32
        ).unwrap()).unwrap();
        f544(&v7, &v6, 1e-3);
    }

    // --- Conv2d grad weight numeric check ---

    #[test]
    fn f584_vs_numeric() {
        // 1x1x4x4 input, 1x1x2x2 kernel, stride 1, pad 0
        let v0: Vec<f32> = (1..=16).map(|x| x as f32 * 0.1).collect();
        let v1 = vec![0.5f32, -0.5, 0.3, -0.3];
        let v2 = 1e-3f32;

        // Compute analytical grad_weight
        let v3 = dev().f502(&v0);
        let v4 = dev().f502(&v1);
        let v5 = dev().f582(&v3, &v4, None, 1, 1, 4, 4, 1, 2, 2, (1,1), (0,0), (1,1), 1).unwrap();
        let v6 = vec![1.0f32; 9];
        let v7 = dev().f502(&v6);
        let v8 = dev().f584(&v3, &v7, 1, 1, 4, 4, 1, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 1).unwrap();
        let v9 = dev().f504(&v8).unwrap();

        // Numeric gradient check
        for v10 in 0..4 {
            let mut v11 = v1.clone();
            let mut v12 = v1.clone();
            v11[v10] += v2;
            v12[v10] -= v2;
            let v13 = dev().f502(&v11);
            let v14 = dev().f502(&v12);
            let v15 = dev().f504(&dev().f582(&v3, &v13, None, 1, 1, 4, 4, 1, 2, 2, (1,1), (0,0), (1,1), 1).unwrap()).unwrap();
            let v16 = dev().f504(&dev().f582(&v3, &v14, None, 1, 1, 4, 4, 1, 2, 2, (1,1), (0,0), (1,1), 1).unwrap()).unwrap();
            let v17: f32 = (v15.iter().sum::<f32>() - v16.iter().sum::<f32>()) / (2.0 * v2);
            assert!((v9[v10] - v17).abs() < 1e-2,
                "grad_w[{v10}]: analytical={}, numeric={}", v9[v10], v17);
        }
        let _ = v5;
    }

    #[test]
    fn f585_basic() {
        // grad_out = [[1,2],[3,4]], batch=1, out_c=1, out_h=2, out_w=2
        // grad_bias[0] should = 1+2+3+4 = 10
        let v0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let v1 = dev().f502(&v0);
        let v2 = dev().f585(&v1, 1, 1, 2, 2).unwrap();
        let v3 = dev().f504(&v2).unwrap();
        f544(&v3, &[10.0], 1e-5);

        // 2 output channels, batch=1, out_h=2, out_w=2
        let v4 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v5 = dev().f585(&dev().f502(&v4), 1, 2, 2, 2).unwrap();
        let v6 = dev().f504(&v5).unwrap();
        f544(&v6, &[10.0, 26.0], 1e-5);
    }
}
