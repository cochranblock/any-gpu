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

const SHADER_MATMUL: &str = "
struct Dims { m: u32, n: u32, k: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if row >= dims.m || col >= dims.n { return; }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        sum = sum + a[row * dims.k + i] * b[i * dims.n + col];
    }
    out[row * dims.n + col] = sum;
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
struct Dims { batch: u32, m: u32, n: u32, k: u32, }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let bat = gid.z;
    if row >= dims.m || col >= dims.n || bat >= dims.batch { return; }
    let a_off = bat * dims.m * dims.k;
    let b_off = bat * dims.k * dims.n;
    let o_off = bat * dims.m * dims.n;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        sum = sum + a[a_off + row * dims.k + i] * b[b_off + i * dims.n + col];
    }
    out[o_off + row * dims.n + col] = sum;
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
    use std::sync::LazyLock;

    static DEV: LazyLock<GpuDevice> = LazyLock::new(|| GpuDevice::gpu().expect("need a GPU"));

    #[test]
    fn test_batch_matmul() {
        // batch=2, each 2x2 matmul
        let a = DEV.upload(&[
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ]);
        let b = DEV.upload(&[
            1.0, 0.0, 0.0, 1.0, // identity batch 0
            2.0, 0.0, 0.0, 2.0, // 2x identity batch 1
        ]);
        let c = DEV.batch_matmul(&a, &b, 2, 2, 2, 2).unwrap();
        let result = DEV.read(&c).unwrap();
        assert_eq!(result, vec![
            1.0, 2.0, 3.0, 4.0,   // A * I = A
            10.0, 12.0, 14.0, 16.0, // A * 2I = 2A
        ]);
    }

    #[test]
    fn test_conv2d_3x3_no_pad() {
        // 1 batch, 1 channel, 4x4 input, 1 output channel, 3x3 kernel
        // Vertical edge detector
        let input = DEV.upload(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);
        let weight = DEV.upload(&[
            1.0, 0.0, -1.0,
            1.0, 0.0, -1.0,
            1.0, 0.0, -1.0,
        ]);
        let bias = DEV.upload(&[0.0]);
        let out = DEV.conv2d(&input, &weight, Some(&bias),
            1, 1, 4, 4, 1, 3, 3, (1, 1), (0, 0), (1, 1), 1).unwrap();
        let result = DEV.read(&out).unwrap();
        // Output is 2x2. Manual calculation:
        // (0,0): 1*1+2*0+3*(-1)+5*1+6*0+7*(-1)+9*1+10*0+11*(-1) = 1-3+5-7+9-11 = -6
        assert_eq!(result.len(), 4);
        assert_approx(&result, &[-6.0, -6.0, -6.0, -6.0], 1e-5);
    }

    #[test]
    fn test_conv2d_with_padding() {
        // 1x1x3x3 input, 1x1x3x3 kernel, padding=1 -> 3x3 output
        let input = DEV.upload(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        // All-ones kernel: sum of neighborhood
        let weight = DEV.upload(&[1.0; 9]);
        let out = DEV.conv2d(&input, &weight, None,
            1, 1, 3, 3, 1, 3, 3, (1, 1), (1, 1), (1, 1), 1).unwrap();
        let result = DEV.read(&out).unwrap();
        assert_eq!(result.len(), 9);
        // Center pixel: sum of all 9 = 45
        assert_approx(&result[4..5], &[45.0], 1e-5);
    }

    #[test]
    fn test_conv2d_stride2() {
        // 1x1x4x4 input, 1x1x1x1 kernel (identity), stride=2 -> 2x2 output
        let input = DEV.upload(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);
        let weight = DEV.upload(&[1.0]); // 1x1 kernel
        let out = DEV.conv2d(&input, &weight, None,
            1, 1, 4, 4, 1, 1, 1, (2, 2), (0, 0), (1, 1), 1).unwrap();
        let result = DEV.read(&out).unwrap();
        assert_eq!(result, vec![1.0, 3.0, 9.0, 11.0]);
    }

    #[test]
    fn test_conv_transpose2d_basic() {
        // Transpose of a 1x1 conv with stride=2 is an upsampling
        // input: 1x1x2x2, weight: 1x1x1x1 (value 1.0), stride=2 -> output 1x1x3x3
        let input = DEV.upload(&[1.0, 2.0, 3.0, 4.0]);
        let weight = DEV.upload(&[1.0]);
        let out = DEV.conv_transpose2d(&input, &weight, None,
            1, 1, 2, 2, 1, 1, 1, (2, 2), (0, 0), (0, 0), (1, 1), 1).unwrap();
        let result = DEV.read(&out).unwrap();
        // 3x3 output: values at stride positions, zeros elsewhere
        assert_eq!(result.len(), 9);
        assert_approx(&result, &[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0], 1e-5);
    }
}
