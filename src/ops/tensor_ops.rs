// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Tensor manipulation: concat, transpose, broadcast add, slice.

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

// --- Slice (concat backward) ---
// Extracts a contiguous slice from each outer block of a combined buffer.
// Used for concat backward: grad_a = slice(grad_out, offset=0, size=a_inner),
//                           grad_b = slice(grad_out, offset=a_inner, size=b_inner).

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SliceParams {
    n: u32,
    outer: u32,
    slice_size: u32,
    slice_offset: u32,
    combined: u32,
    _pad: [u32; 3],
}

const SHADER_SLICE: &str = "
struct P { n: u32, outer: u32, slice_size: u32, slice_offset: u32, combined: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.n { return; }
    let outer_idx = idx / p.slice_size;
    let inner_idx = idx % p.slice_size;
    dst[idx] = src[outer_idx * p.combined + p.slice_offset + inner_idx];
}
";

// --- Broadcast add ---
// Adds b[outer] broadcast across inner dim to a[outer, inner]: out[o, i] = a[o, i] + b[o]
// Used for bias adds and time conditioning in UNet.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BroadcastAddParams {
    n: u32,
    outer: u32,
    inner: u32,
    _pad: u32,
}

const SHADER_BROADCAST_ADD: &str = "
struct P { n: u32, outer: u32, inner: u32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.n { return; }
    let outer_idx = idx / p.inner;
    out[idx] = a[idx] + b[outer_idx];
}
";

// --- Sum reduction along inner dim (broadcast add backward for b) ---
// For each outer index, sums inner elements: out[o] = sum_i(src[o, i]).

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SumInnerParams {
    outer: u32,
    inner: u32,
    _pad: [u32; 2],
}

const SHADER_SUM_INNER: &str = "
struct P { outer: u32, inner: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x + gid.y * 65535u * 256u;
    if o >= p.outer { return; }
    var s: f32 = 0.0;
    let base = o * p.inner;
    for (var i: u32 = 0u; i < p.inner; i++) {
        s += src[base + i];
    }
    dst[o] = s;
}
";

// --- Add per-column: y[rows, cols] += bias[cols] ---
// For Linear layer bias: adds b[n] to every row's n-th column.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AddPerColParams {
    n: u32,
    rows: u32,
    cols: u32,
    _pad: u32,
}

const SHADER_ADD_PER_COL: &str = "
struct P { n: u32, rows: u32, cols: u32, _p0: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.n { return; }
    let col = idx % p.cols;
    out[idx] = a[idx] + b[col];
}
";

// --- Sum reduction over rows: out[c] = sum_r(src[r*cols + c]) ---
// Backward for add_per_col w.r.t. the column bias.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SumRowsParams {
    rows: u32,
    cols: u32,
    _pad: [u32; 2],
}

const SHADER_SUM_ROWS: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x + gid.y * 65535u * 256u;
    if c >= p.cols { return; }
    var s: f32 = 0.0;
    for (var r: u32 = 0u; r < p.rows; r++) {
        s += src[r * p.cols + c];
    }
    dst[c] = s;
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

    /// Extract a contiguous slice from each outer block of a buffer.
    /// For combined buffer shape [outer, combined], returns [outer, slice_size]
    /// where slice_size runs from `slice_offset` to `slice_offset + slice_size - 1`.
    /// Used for concat backward.
    pub(crate) fn slice_per_block(
        &self,
        src: &GpuBuffer,
        outer: u32, slice_size: u32, slice_offset: u32, combined: u32,
    ) -> Result<GpuBuffer> {
        ensure!(src.len == (outer * combined) as usize);
        ensure!(slice_offset + slice_size <= combined);
        let total = outer * slice_size;
        let out = self.alloc(total as usize);
        let params = SliceParams {
            n: total, outer, slice_size, slice_offset, combined, _pad: [0; 3],
        };
        self.dispatch_shader(
            SHADER_SLICE, Some("slice"),
            &params, &[src], &out,
            super::dispatch_1d(total),
        );
        Ok(out)
    }

    /// Add b[outer] broadcast across inner dim to a[outer, inner].
    /// For bias add: inner = batch*spatial, outer = channels.
    /// For time conditioning: inner = spatial, outer = batch*channels.
    pub fn add_broadcast(
        &self,
        a: &GpuBuffer, b: &GpuBuffer,
        outer: u32, inner: u32,
    ) -> Result<GpuBuffer> {
        ensure!(a.len == (outer * inner) as usize);
        ensure!(b.len == outer as usize);
        let total = outer * inner;
        let out = self.alloc(total as usize);
        let params = BroadcastAddParams { n: total, outer, inner, _pad: 0 };
        self.dispatch_shader(
            SHADER_BROADCAST_ADD, Some("bcast_add"),
            &params, &[a, b], &out,
            super::dispatch_1d(total),
        );
        Ok(out)
    }

    /// Sum along inner dim: out[o] = sum_i(src[o, i]).
    /// Used for broadcast add backward (grad_b = sum over inner).
    pub(crate) fn sum_inner(
        &self,
        src: &GpuBuffer,
        outer: u32, inner: u32,
    ) -> Result<GpuBuffer> {
        ensure!(src.len == (outer * inner) as usize);
        let out = self.alloc(outer as usize);
        let params = SumInnerParams { outer, inner, _pad: [0; 2] };
        self.dispatch_shader(
            SHADER_SUM_INNER, Some("sum_inner"),
            &params, &[src], &out,
            super::dispatch_1d(outer),
        );
        Ok(out)
    }

    /// Add per-column bias: out[rows, cols] = a[rows, cols] + b[cols].
    /// For Linear layer bias.
    pub fn add_per_col(
        &self,
        a: &GpuBuffer, b: &GpuBuffer,
        rows: u32, cols: u32,
    ) -> Result<GpuBuffer> {
        ensure!(a.len == (rows * cols) as usize);
        ensure!(b.len == cols as usize);
        let total = rows * cols;
        let out = self.alloc(total as usize);
        let params = AddPerColParams { n: total, rows, cols, _pad: 0 };
        self.dispatch_shader(
            SHADER_ADD_PER_COL, Some("add_per_col"),
            &params, &[a, b], &out,
            super::dispatch_1d(total),
        );
        Ok(out)
    }

    /// Sum along row dim: out[c] = sum_r(src[r*cols + c]).
    /// Used for add_per_col backward (grad_b = sum over rows).
    pub(crate) fn sum_rows(
        &self,
        src: &GpuBuffer,
        rows: u32, cols: u32,
    ) -> Result<GpuBuffer> {
        ensure!(src.len == (rows * cols) as usize);
        let out = self.alloc(cols as usize);
        let params = SumRowsParams { rows, cols, _pad: [0; 2] };
        self.dispatch_shader(
            SHADER_SUM_ROWS, Some("sum_rows"),
            &params, &[src], &out,
            super::dispatch_1d(cols),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    #[test]
    fn test_concat_flat() {
        let result = dev().read(&dev().concat(&dev().upload(&[1.0, 2.0, 3.0]), &dev().upload(&[4.0, 5.0, 6.0]), 1, 3, 3).unwrap()).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_asymmetric() {
        // Different sized inner dims: a has 2 elements, b has 3 per outer block
        let a = dev().upload(&[1.0, 2.0]);
        let b = dev().upload(&[3.0, 4.0, 5.0]);
        let result = dev().read(&dev().concat(&a, &b, 1, 2, 3).unwrap()).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_concat_batched_channel_axis() {
        // batch=2, concat 1-channel and 2-channel tensors along C, spatial=2
        // a: [batch=2, c=1, spatial=2] = [10, 20, 30, 40]
        // b: [batch=2, c=2, spatial=2] = [1, 2, 3, 4, 5, 6, 7, 8]
        let a = dev().upload(&[10.0, 20.0, 30.0, 40.0]);
        let b = dev().upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // outer=batch=2, a_inner=1*2=2, b_inner=2*2=4
        let result = dev().read(&dev().concat(&a, &b, 2, 2, 4).unwrap()).unwrap();
        // batch 0: [10,20, 1,2,3,4], batch 1: [30,40, 5,6,7,8]
        assert_eq!(result, vec![10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_transpose_2d() {
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dev().read(&dev().transpose(&a, 1, 2, 3, 1).unwrap()).unwrap();
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_square() {
        // 3x3 -> 3x3 transpose
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = dev().read(&dev().transpose(&a, 1, 3, 3, 1).unwrap()).unwrap();
        assert_eq!(result, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_transpose_batched() {
        let a = dev().upload(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let result = dev().read(&dev().transpose(&a, 2, 2, 3, 1).unwrap()).unwrap();
        assert_eq!(result, vec![
            1.0, 4.0, 2.0, 5.0, 3.0, 6.0,
            7.0, 10.0, 8.0, 11.0, 9.0, 12.0,
        ]);
    }

    #[test]
    fn test_transpose_1x_n() {
        // 1xN transpose = Nx1 (column vector)
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = dev().read(&dev().transpose(&a, 1, 1, 5, 1).unwrap()).unwrap();
        // 5x1 is same flat data (no-op for 1-row)
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_transpose_roundtrip() {
        // Transpose twice = identity
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect(); // 4x5
        let t1 = dev().transpose(&dev().upload(&data), 1, 4, 5, 1).unwrap();
        let t2 = dev().transpose(&t1, 1, 5, 4, 1).unwrap();
        let result = dev().read(&t2).unwrap();
        assert_eq!(result, data);
    }
}
