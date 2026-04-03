// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Matmul, batched matmul, conv2d, transpose_conv2d.

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

// --- Matmul ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulDims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

// Tiled matmul: 16x16 tiles in workgroup shared memory.
// Each tile of A and B is loaded from global memory once per tile iteration,
// not once per output element. Reduces global memory reads by ~16x.
const SHADER_MATMUL: &str = "
const TILE: u32 = 16u;
struct Dims { m: u32, n: u32, k: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> tile_a: array<f32, 256>;  // TILE * TILE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;
    let lr = lid.x;
    let lc = lid.y;

    var sum: f32 = 0.0;
    let num_tiles = (dims.k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        // Load tile of A: row from global row, col from tile offset
        let a_col = t * TILE + lc;
        if row < dims.m && a_col < dims.k {
            tile_a[lr * TILE + lc] = a[row * dims.k + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        // Load tile of B: row from tile offset, col from global col
        let b_row = t * TILE + lr;
        if b_row < dims.k && col < dims.n {
            tile_b[lr * TILE + lc] = b[b_row * dims.n + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        // Accumulate dot product from shared memory
        for (var i: u32 = 0u; i < TILE; i++) {
            sum += tile_a[lr * TILE + i] * tile_b[i * TILE + lc];
        }

        workgroupBarrier();
    }

    if row < dims.m && col < dims.n {
        out[row * dims.n + col] = sum;
    }
}
";

// --- Batched Matmul ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchMatmulDims {
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv2dParams {
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvTranspose2dParams {
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

impl GpuDevice {
    /// Matrix multiply: A(m,k) x B(k,n) = C(m,n). Row-major layout.
    pub fn matmul(&self, a: &GpuBuffer, b: &GpuBuffer, m: u32, n: u32, k: u32) -> Result<GpuBuffer> {
        ensure!(a.len == (m * k) as usize, "matmul: A has {} elems, expected {}", a.len, m * k);
        ensure!(b.len == (k * n) as usize, "matmul: B has {} elems, expected {}", b.len, k * n);
        let out = self.alloc((m * n) as usize);
        let dims = MatmulDims { m, n, k, _pad: 0 };
        self.dispatch_shader(SHADER_MATMUL, Some("matmul"), &dims, &[a, b], &out, (m.div_ceil(16), n.div_ceil(16), 1));
        Ok(out)
    }

    /// Batched matmul: A[batch,m,k] x B[batch,k,n] = C[batch,m,n].
    pub fn batch_matmul(&self, a: &GpuBuffer, b: &GpuBuffer, batch: u32, m: u32, n: u32, k: u32) -> Result<GpuBuffer> {
        ensure!(a.len == (batch * m * k) as usize);
        ensure!(b.len == (batch * k * n) as usize);
        let out = self.alloc((batch * m * n) as usize);
        let dims = BatchMatmulDims { batch, m, n, k };
        self.dispatch_shader(SHADER_BATCH_MATMUL, Some("batch_matmul"), &dims, &[a, b], &out, (m.div_ceil(16), n.div_ceil(16), batch));
        Ok(out)
    }

    /// Conv2d: input[N,C_in,H,W] * weight[C_out,C_in/groups,kH,kW] + bias[C_out].
    /// NCHW layout. Returns output[N,C_out,out_H,out_W].
    pub fn conv2d(
        &self,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        bias: Option<&GpuBuffer>,
        batch: u32, in_c: u32, in_h: u32, in_w: u32,
        out_c: u32, kh: u32, kw: u32,
        stride: (u32, u32), padding: (u32, u32),
        dilation: (u32, u32), groups: u32,
    ) -> Result<GpuBuffer> {
        let out_h = (in_h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;
        let total = batch * out_c * out_h * out_w;

        ensure!(input.len == (batch * in_c * in_h * in_w) as usize);
        ensure!(weight.len == (out_c * (in_c / groups) * kh * kw) as usize);

        let zero_bias;
        let bias_buf = match bias {
            Some(b) => {
                ensure!(b.len == out_c as usize);
                b
            }
            None => {
                zero_bias = self.upload(&vec![0.0f32; out_c as usize]);
                &zero_bias
            }
        };

        let out = self.alloc(total as usize);
        let params = Conv2dParams {
            batch, in_c, out_c, in_h, in_w, out_h, out_w,
            kh, kw, stride_h: stride.0, stride_w: stride.1,
            pad_h: padding.0, pad_w: padding.1,
            dilation_h: dilation.0, dilation_w: dilation.1, groups,
        };

        self.dispatch_shader(
            SHADER_CONV2D, Some("conv2d"),
            &params, &[input, weight, bias_buf], &out,
            super::dispatch_1d(total),
        );
        Ok(out)
    }

    /// Transposed conv2d (deconvolution): input[N,C_in,H,W] -> output[N,C_out,out_H,out_W].
    /// Weight layout: [C_in, C_out/groups, kH, kW].
    pub fn conv_transpose2d(
        &self,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        bias: Option<&GpuBuffer>,
        batch: u32, in_c: u32, in_h: u32, in_w: u32,
        out_c: u32, kh: u32, kw: u32,
        stride: (u32, u32), padding: (u32, u32),
        output_padding: (u32, u32),
        dilation: (u32, u32), groups: u32,
    ) -> Result<GpuBuffer> {
        let out_h = (in_h - 1) * stride.0 - 2 * padding.0 + dilation.0 * (kh - 1) + output_padding.0 + 1;
        let out_w = (in_w - 1) * stride.1 - 2 * padding.1 + dilation.1 * (kw - 1) + output_padding.1 + 1;
        let total = batch * out_c * out_h * out_w;

        ensure!(input.len == (batch * in_c * in_h * in_w) as usize);
        ensure!(weight.len == (in_c * (out_c / groups) * kh * kw) as usize);

        let zero_bias;
        let bias_buf = match bias {
            Some(b) => {
                ensure!(b.len == out_c as usize);
                b
            }
            None => {
                zero_bias = self.upload(&vec![0.0f32; out_c as usize]);
                &zero_bias
            }
        };

        let out = self.alloc(total as usize);
        let params = ConvTranspose2dParams {
            batch, in_c, out_c, in_h, in_w, out_h, out_w,
            kh, kw, stride_h: stride.0, stride_w: stride.1,
            pad_h: padding.0, pad_w: padding.1,
            dilation_h: dilation.0, dilation_w: dilation.1, groups,
        };

        self.dispatch_shader(
            SHADER_CONV_TRANSPOSE2D, Some("conv_transpose2d"),
            &params, &[input, weight, bias_buf], &out,
            super::dispatch_1d(total),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::assert_approx;

    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

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
    fn test_matmul_2x2() {
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = dev().upload(&[5.0, 6.0, 7.0, 8.0]);
        let result = dev().read(&dev().matmul(&a, &b, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_nonsquare_vs_cpu() {
        // 3x4 @ 4x2 = 3x2
        let a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=8).map(|x| x as f32 * 0.1).collect();
        let expected = cpu_matmul(&a, &b, 3, 2, 4);
        let result = dev().read(&dev().matmul(&dev().upload(&a), &dev().upload(&b), 3, 2, 4).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_1x1() {
        let result = dev().read(&dev().matmul(&dev().upload(&[3.0]), &dev().upload(&[7.0]), 1, 1, 1).unwrap()).unwrap();
        assert_eq!(result, vec![21.0]);
    }

    #[test]
    fn test_matmul_17x13_vs_cpu() {
        // Odd dims that don't align to 16x16 workgroup
        let m = 17; let n = 13; let k = 11;
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.01) - 0.3).collect();
        let expected = cpu_matmul(&a, &b, m, n, k);
        let result = dev().read(&dev().matmul(
            &dev().upload(&a), &dev().upload(&b), m as u32, n as u32, k as u32
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-3);
    }

    // --- Batch matmul tests ---

    #[test]
    fn test_batch_matmul() {
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = dev().upload(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let result = dev().read(&dev().batch_matmul(&a, &b, 2, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_batch_matmul_nonsquare() {
        // batch=1, 2x3 @ 3x1 = 2x1
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = dev().upload(&[1.0, 1.0, 1.0]);
        let result = dev().read(&dev().batch_matmul(&a, &b, 1, 2, 1, 3).unwrap()).unwrap();
        assert_eq!(result, vec![6.0, 15.0]);
    }

    // --- Conv2d tests ---

    #[test]
    fn test_conv2d_3x3_vs_cpu() {
        // Verify GPU conv2d matches CPU reference
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let weight = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
        let bias = vec![0.0];
        let expected = cpu_conv2d(&input, &weight, &bias, 1, 1, 4, 4, 1, 3, 3, (1,1), (0,0), 1);
        let result = dev().read(&dev().conv2d(
            &dev().upload(&input), &dev().upload(&weight), Some(&dev().upload(&bias)),
            1, 1, 4, 4, 1, 3, 3, (1,1), (0,0), (1,1), 1
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_1x1_kernel() {
        // 1x1 conv = per-pixel channel mixing
        // 2 in channels, 3 out channels, 2x2 spatial
        let input = dev().upload(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        // weight[out_c, in_c, 1, 1]: 3 output channels, 2 input channels
        let weight = dev().upload(&[1.0, 0.5, 0.0, 1.0, -1.0, 2.0]);
        let bias = dev().upload(&[0.0, 0.0, 0.0]);
        let result = dev().read(&dev().conv2d(&input, &weight, Some(&bias),
            1, 2, 2, 2, 3, 1, 1, (1,1), (0,0), (1,1), 1).unwrap()).unwrap();
        // out_c=0: 1.0*in0 + 0.5*in1
        assert_approx(&result[0..4], &[6.0, 12.0, 18.0, 24.0], 1e-5);
        // out_c=1: 0.0*in0 + 1.0*in1
        assert_approx(&result[4..8], &[10.0, 20.0, 30.0, 40.0], 1e-5);
    }

    #[test]
    fn test_conv2d_padding_vs_cpu() {
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let weight = vec![1.0; 9];
        let bias = vec![0.0];
        let expected = cpu_conv2d(&input, &weight, &bias, 1, 1, 3, 3, 1, 3, 3, (1,1), (1,1), 1);
        let result = dev().read(&dev().conv2d(
            &dev().upload(&input), &dev().upload(&weight), None,
            1, 1, 3, 3, 1, 3, 3, (1,1), (1,1), (1,1), 1
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_stride2() {
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let result = dev().read(&dev().conv2d(
            &dev().upload(&input), &dev().upload(&[1.0]), None,
            1, 1, 4, 4, 1, 1, 1, (2,2), (0,0), (1,1), 1
        ).unwrap()).unwrap();
        assert_eq!(result, vec![1.0, 3.0, 9.0, 11.0]);
    }

    #[test]
    fn test_conv2d_5x5_kernel_vs_cpu() {
        // 1x1x8x8 input, 1x1x5x5 kernel, padding=2 -> 8x8 output
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..25).map(|i| if i == 12 { 1.0 } else { 0.0 }).collect(); // center=1, rest=0 -> identity
        let bias = vec![0.0];
        let expected = cpu_conv2d(&input, &weight, &bias, 1, 1, 8, 8, 1, 5, 5, (1,1), (2,2), 1);
        let result = dev().read(&dev().conv2d(
            &dev().upload(&input), &dev().upload(&weight), None,
            1, 1, 8, 8, 1, 5, 5, (1,1), (2,2), (1,1), 1
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-4);
    }

    #[test]
    fn test_conv2d_multichannel_vs_cpu() {
        // batch=2, in_c=3, out_c=2, 4x4, 3x3 kernel, padding=1
        let batch = 2; let in_c = 3; let out_c = 2; let h = 4; let w = 4;
        let input: Vec<f32> = (0..batch*in_c*h*w).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let weight: Vec<f32> = (0..out_c*in_c*3*3).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let bias = vec![0.1, -0.2];
        let expected = cpu_conv2d(&input, &weight, &bias, batch, in_c, h, w, out_c, 3, 3, (1,1), (1,1), 1);
        let result = dev().read(&dev().conv2d(
            &dev().upload(&input), &dev().upload(&weight), Some(&dev().upload(&bias)),
            batch as u32, in_c as u32, h as u32, w as u32, out_c as u32, 3, 3, (1,1), (1,1), (1,1), 1
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-3);
    }

    #[test]
    fn test_conv2d_with_bias() {
        // Verify bias is added correctly
        let input = dev().upload(&[0.0; 4]); // 1x1x2x2 zeros
        let weight = dev().upload(&[0.0]); // 1x1x1x1 zero kernel
        let bias = dev().upload(&[42.0]);
        let result = dev().read(&dev().conv2d(&input, &weight, Some(&bias),
            1, 1, 2, 2, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap()).unwrap();
        assert_eq!(result, vec![42.0, 42.0, 42.0, 42.0]);
    }

    // --- Transpose conv2d tests ---

    #[test]
    fn test_conv_transpose2d_stride2() {
        let input = dev().upload(&[1.0, 2.0, 3.0, 4.0]);
        let weight = dev().upload(&[1.0]);
        let result = dev().read(&dev().conv_transpose2d(&input, &weight, None,
            1, 1, 2, 2, 1, 1, 1, (2,2), (0,0), (0,0), (1,1), 1).unwrap()).unwrap();
        assert_eq!(result.len(), 9);
        assert_approx(&result, &[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0], 1e-5);
    }

    #[test]
    fn test_conv_transpose2d_3x3_kernel() {
        // 1x1x1x1 input (single pixel), 1x1x3x3 all-ones kernel, stride=1 -> 3x3 output all same value
        let input = dev().upload(&[5.0]);
        let weight = dev().upload(&[1.0; 9]);
        let result = dev().read(&dev().conv_transpose2d(&input, &weight, None,
            1, 1, 1, 1, 1, 3, 3, (1,1), (0,0), (0,0), (1,1), 1).unwrap()).unwrap();
        assert_eq!(result.len(), 9);
        assert_approx(&result, &[5.0; 9], 1e-5);
    }
}
