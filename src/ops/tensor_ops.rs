// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Tensor manipulation: concat, transpose.

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

// --- Concat ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConcatParams {
    n: u32,
    outer: u32,
    a_inner: u32,
    b_inner: u32,
}

const SHADER_CONCAT: &str = "
struct P { n: u32, outer: u32, a_inner: u32, b_inner: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.n { return; }

    let combined = p.a_inner + p.b_inner;
    let outer_idx = idx / combined;
    let inner_idx = idx % combined;

    if inner_idx < p.a_inner {
        out[idx] = a[outer_idx * p.a_inner + inner_idx];
    } else {
        out[idx] = b[outer_idx * p.b_inner + (inner_idx - p.a_inner)];
    }
}
";

// --- Transpose (swap two dims) ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeParams {
    n: u32,
    d0: u32,
    d1: u32,
    inner: u32,
    outer_stride: u32,
    _pad: [u32; 3],
}

const SHADER_TRANSPOSE: &str = "
struct P { n: u32, d0: u32, d1: u32, inner: u32, outer_stride: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.n { return; }

    // Decompose output index: outer * (d1 * d0 * inner) + i1 * (d0 * inner) + i0 * inner + inner_idx
    let block = p.d0 * p.d1 * p.inner;
    let outer = idx / block;
    let rem = idx % block;
    let i1 = rem / (p.d0 * p.inner);
    let rem2 = rem % (p.d0 * p.inner);
    let i0 = rem2 / p.inner;
    let inner_idx = rem2 % p.inner;

    // In input, dims are (d0, d1) so source index swaps i0 and i1
    let src = outer * block + i0 * (p.d1 * p.inner) + i1 * p.inner + inner_idx;
    out[idx] = a[src];
}
";

impl GpuDevice {
    /// Concat two buffers along a given axis.
    /// `outer_size` = product of dims before concat axis.
    /// `a_inner` = a's size along concat axis * product of dims after.
    /// `b_inner` = same for b.
    pub fn concat(
        &self,
        a: &GpuBuffer, b: &GpuBuffer,
        outer_size: u32, a_inner: u32, b_inner: u32,
    ) -> Result<GpuBuffer> {
        ensure!(a.len == (outer_size * a_inner) as usize);
        ensure!(b.len == (outer_size * b_inner) as usize);
        let total = outer_size * (a_inner + b_inner);
        let out = self.alloc(total as usize);
        let params = ConcatParams { n: total, outer: outer_size, a_inner, b_inner };
        self.dispatch_shader(
            SHADER_CONCAT, Some("concat"),
            &params, &[a, b], &out,
            super::dispatch_1d(total),
        );
        Ok(out)
    }

    /// Transpose two dimensions of a tensor.
    /// Shape is [..., d0, d1, ...inner_dims].
    /// `outer_size` = product of dims before d0.
    /// `inner` = product of dims after d1.
    pub fn transpose(
        &self,
        a: &GpuBuffer,
        outer_size: u32, d0: u32, d1: u32, inner: u32,
    ) -> Result<GpuBuffer> {
        let total = outer_size * d0 * d1 * inner;
        ensure!(a.len == total as usize);
        let out = self.alloc(total as usize);
        let params = TransposeParams {
            n: total, d0, d1, inner,
            outer_stride: d0 * d1 * inner, _pad: [0; 3],
        };
        self.dispatch_shader(
            SHADER_TRANSPOSE, Some("transpose"),
            &params, &[a], &out,
            super::dispatch_1d(total),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    static DEV: LazyLock<GpuDevice> = LazyLock::new(|| GpuDevice::gpu().expect("need a GPU"));

    #[test]
    fn test_concat_flat() {
        let a = DEV.upload(&[1.0, 2.0, 3.0]);
        let b = DEV.upload(&[4.0, 5.0, 6.0]);
        // outer=1, a_inner=3, b_inner=3 -> simple append
        let result = DEV.read(&DEV.concat(&a, &b, 1, 3, 3).unwrap()).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_channel_axis() {
        // Two 1x2x2 tensors (NCHW: N=1, C=1, H=2, W=2), concat along C
        // a: [[1,2],[3,4]], b: [[5,6],[7,8]]
        let a = DEV.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = DEV.upload(&[5.0, 6.0, 7.0, 8.0]);
        // outer=N=1, a_inner=C_a*H*W=4, b_inner=4
        let result = DEV.read(&DEV.concat(&a, &b, 1, 4, 4).unwrap()).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_transpose_2d() {
        // 2x3 matrix -> 3x2
        let a = DEV.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // outer=1, d0=2, d1=3, inner=1
        let result = DEV.read(&DEV.transpose(&a, 1, 2, 3, 1).unwrap()).unwrap();
        // Transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_batched() {
        // batch=2, 2x3 matrices -> batch=2, 3x2 matrices
        let a = DEV.upload(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
        ]);
        let result = DEV.read(&DEV.transpose(&a, 2, 2, 3, 1).unwrap()).unwrap();
        assert_eq!(result, vec![
            1.0, 4.0, 2.0, 5.0, 3.0, 6.0,
            7.0, 10.0, 8.0, 11.0, 9.0, 12.0,
        ]);
    }
}
