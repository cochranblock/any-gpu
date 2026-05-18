// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// f640=concat, f641=transpose, f642=add_broadcast, f643=slice_per_block,
// f644=sum_inner, f645=add_per_col, f646=sum_rows.

use crate::device::{t500, t501};
use anyhow::{ensure, Result};

// --- Concat ---

/// t515 = ConcatParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t515 {
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

/// t516 = SliceParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t516 {
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

/// t517 = BroadcastAddParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t517 {
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

/// t518 = SumInnerParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t518 {
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

/// t519 = AddPerColParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t519 {
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

/// t520 = SumRowsParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t520 {
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

/// t521 = TransposeParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t521 {
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

impl t500 {
    /// f640 = concat. Two buffers along a given axis.
    /// `p2` = product of dims before concat axis. `p3` = a's size along concat axis * product of dims after.
    /// `p4` = same for b.
    pub fn f640(
        &self,
        p0: &t501, p1: &t501,
        p2: u32, p3: u32, p4: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p2 * p3) as usize);
        ensure!(p1.s507 == (p2 * p4) as usize);
        let v0 = p2 * (p3 + p4);
        let v1 = self.f503(v0 as usize);
        let v2 = t515 { n: v0, outer: p2, a_inner: p3, b_inner: p4 };
        self.f543(
            SHADER_CONCAT, Some("concat"),
            &v2, &[p0, p1], &v1,
            super::f540(v0),
        );
        Ok(v1)
    }

    /// f641 = transpose. Swap two dimensions of a tensor.
    /// Shape is [..., d0, d1, ...inner_dims]. `p1` = outer_size, `p4` = inner.
    pub fn f641(
        &self,
        p0: &t501,
        p1: u32, p2: u32, p3: u32, p4: u32,
    ) -> Result<t501> {
        let v0 = p1 * p2 * p3 * p4;
        ensure!(p0.s507 == v0 as usize);
        let v1 = self.f503(v0 as usize);
        let v2 = t521 {
            n: v0, d0: p2, d1: p3, inner: p4,
            outer_stride: p2 * p3 * p4, _pad: [0; 3],
        };
        self.f543(
            SHADER_TRANSPOSE, Some("transpose"),
            &v2, &[p0], &v1,
            super::f540(v0),
        );
        Ok(v1)
    }

    /// f643 = slice_per_block. Extract a contiguous slice from each outer block.
    /// Used for concat backward.
    pub(crate) fn f643(
        &self,
        p0: &t501,
        p1: u32, p2: u32, p3: u32, p4: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p1 * p4) as usize);
        ensure!(p3 + p2 <= p4);
        let v0 = p1 * p2;
        let v1 = self.f503(v0 as usize);
        let v2 = t516 {
            n: v0, outer: p1, slice_size: p2, slice_offset: p3, combined: p4, _pad: [0; 3],
        };
        self.f543(
            SHADER_SLICE, Some("slice"),
            &v2, &[p0], &v1,
            super::f540(v0),
        );
        Ok(v1)
    }

    /// f642 = add_broadcast. out[outer, inner] = a[outer, inner] + b[outer].
    pub fn f642(
        &self,
        p0: &t501, p1: &t501,
        p2: u32, p3: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p2 * p3) as usize);
        ensure!(p1.s507 == p2 as usize);
        let v0 = p2 * p3;
        let v1 = self.f503(v0 as usize);
        let v2 = t517 { n: v0, outer: p2, inner: p3, _pad: 0 };
        self.f543(
            SHADER_BROADCAST_ADD, Some("bcast_add"),
            &v2, &[p0, p1], &v1,
            super::f540(v0),
        );
        Ok(v1)
    }

    /// f644 = sum_inner. out[o] = sum_i(src[o, i]).
    pub(crate) fn f644(
        &self,
        p0: &t501,
        p1: u32, p2: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p1 * p2) as usize);
        let v0 = self.f503(p1 as usize);
        let v1 = t518 { outer: p1, inner: p2, _pad: [0; 2] };
        self.f543(
            SHADER_SUM_INNER, Some("sum_inner"),
            &v1, &[p0], &v0,
            super::f540(p1),
        );
        Ok(v0)
    }

    /// f645 = add_per_col. out[rows, cols] = a[rows, cols] + b[cols]. Linear bias.
    pub fn f645(
        &self,
        p0: &t501, p1: &t501,
        p2: u32, p3: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p2 * p3) as usize);
        ensure!(p1.s507 == p3 as usize);
        let v0 = p2 * p3;
        let v1 = self.f503(v0 as usize);
        let v2 = t519 { n: v0, rows: p2, cols: p3, _pad: 0 };
        self.f543(
            SHADER_ADD_PER_COL, Some("add_per_col"),
            &v2, &[p0, p1], &v1,
            super::f540(v0),
        );
        Ok(v1)
    }

    /// f646 = sum_rows. out[c] = sum_r(src[r*cols + c]).
    pub(crate) fn f646(
        &self,
        p0: &t501,
        p1: u32, p2: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p1 * p2) as usize);
        let v0 = self.f503(p2 as usize);
        let v1 = t520 { rows: p1, cols: p2, _pad: [0; 2] };
        self.f543(
            SHADER_SUM_ROWS, Some("sum_rows"),
            &v1, &[p0], &v0,
            super::f540(p2),
        );
        Ok(v0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    #[test]
    fn f640_flat() {
        let v0 = dev().f504(&dev().f640(&dev().f502(&[1.0, 2.0, 3.0]), &dev().f502(&[4.0, 5.0, 6.0]), 1, 3, 3).unwrap()).unwrap();
        assert_eq!(v0, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn f640_asymmetric() {
        let v0 = dev().f502(&[1.0, 2.0]);
        let v1 = dev().f502(&[3.0, 4.0, 5.0]);
        let v2 = dev().f504(&dev().f640(&v0, &v1, 1, 2, 3).unwrap()).unwrap();
        assert_eq!(v2, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn f640_batched_channel_axis() {
        let v0 = dev().f502(&[10.0, 20.0, 30.0, 40.0]);
        let v1 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let v2 = dev().f504(&dev().f640(&v0, &v1, 2, 2, 4).unwrap()).unwrap();
        assert_eq!(v2, vec![10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn f641_2d() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v1 = dev().f504(&dev().f641(&v0, 1, 2, 3, 1).unwrap()).unwrap();
        assert_eq!(v1, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn f641_square() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let v1 = dev().f504(&dev().f641(&v0, 1, 3, 3, 1).unwrap()).unwrap();
        assert_eq!(v1, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn f641_batched() {
        let v0 = dev().f502(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let v1 = dev().f504(&dev().f641(&v0, 2, 2, 3, 1).unwrap()).unwrap();
        assert_eq!(v1, vec![
            1.0, 4.0, 2.0, 5.0, 3.0, 6.0,
            7.0, 10.0, 8.0, 11.0, 9.0, 12.0,
        ]);
    }

    #[test]
    fn f641_1x_n() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = dev().f504(&dev().f641(&v0, 1, 1, 5, 1).unwrap()).unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn f641_roundtrip() {
        let v0: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let v1 = dev().f641(&dev().f502(&v0), 1, 4, 5, 1).unwrap();
        let v2 = dev().f641(&v1, 1, 5, 4, 1).unwrap();
        let v3 = dev().f504(&v2).unwrap();
        assert_eq!(v3, v0);
    }

    // --- f642 = broadcast_add: out[r, c] = a[r, c] + b[r] ---

    #[test]
    fn f642_known_values() {
        // a = [[1,2,3],[4,5,6]], b = [10, 20] (per-row bias).
        // out[0,*] += 10, out[1,*] += 20.
        let v0 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v1 = dev().f502(&[10.0f32, 20.0]);
        let v2 = dev().f504(&dev().f642(&v0, &v1, 2, 3).unwrap()).unwrap();
        assert_eq!(v2, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn f642_single_row() {
        let v0 = dev().f502(&[1.0f32, 2.0, 3.0]);
        let v1 = dev().f502(&[5.0f32]);
        let v2 = dev().f504(&dev().f642(&v0, &v1, 1, 3).unwrap()).unwrap();
        assert_eq!(v2, vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn f642_size_mismatch() {
        let v0 = dev().f502(&[1.0f32; 6]);
        let v1 = dev().f502(&[1.0f32; 3]);
        assert!(dev().f642(&v0, &v1, 2, 3).is_err());
    }

    // --- f645 = add_per_col: out[r, c] = a[r, c] + b[c] (linear bias) ---

    #[test]
    fn f645_known_values() {
        // a = [[1,2,3],[4,5,6]], b = [10,20,30] (per-col bias).
        // out[*,0] += 10, out[*,1] += 20, out[*,2] += 30.
        let v0 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v1 = dev().f502(&[10.0f32, 20.0, 30.0]);
        let v2 = dev().f504(&dev().f645(&v0, &v1, 2, 3).unwrap()).unwrap();
        assert_eq!(v2, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn f645_single_col() {
        let v0 = dev().f502(&[1.0f32, 2.0, 3.0]);
        let v1 = dev().f502(&[100.0f32]);
        let v2 = dev().f504(&dev().f645(&v0, &v1, 3, 1).unwrap()).unwrap();
        assert_eq!(v2, vec![101.0, 102.0, 103.0]);
    }

    #[test]
    fn f645_zero_bias() {
        let v0: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let v1 = dev().f502(&[0.0f32; 4]);
        let v2 = dev().f504(&dev().f645(&dev().f502(&v0), &v1, 2, 4).unwrap()).unwrap();
        assert_eq!(v2, v0);
    }
}
