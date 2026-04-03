// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Element-wise ops: add, mul, sub, scale, relu, sigmoid, swish, tanh.

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

const SHADER_ADD: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    out[idx] = a[idx] + b[idx];
}
";

const SHADER_SUB: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    out[idx] = a[idx] - b[idx];
}
";

const SHADER_MUL: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    out[idx] = a[idx] * b[idx];
}
";

const SHADER_RELU: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    out[idx] = max(a[idx], 0.0);
}
";

const SHADER_SIGMOID: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    out[idx] = 1.0 / (1.0 + exp(-a[idx]));
}
";

const SHADER_SWISH: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    let x = a[idx];
    out[idx] = x / (1.0 + exp(-x));
}
";

const SHADER_TANH: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    out[idx] = tanh(a[idx]);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleParams {
    n: u32,
    scale: f32,
    _pad: [u32; 2],
}

const SHADER_SCALE: &str = "
struct Params { n: u32, scale: f32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    out[idx] = a[idx] * params.scale;
}
";

impl GpuDevice {
    pub fn add(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        ensure!(a.len == b.len, "add: length mismatch ({} vs {})", a.len, b.len);
        self.binary_op(SHADER_ADD, a, b)
    }

    pub fn sub(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        ensure!(a.len == b.len, "sub: length mismatch ({} vs {})", a.len, b.len);
        self.binary_op(SHADER_SUB, a, b)
    }

    pub fn mul(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        ensure!(a.len == b.len, "mul: length mismatch ({} vs {})", a.len, b.len);
        self.binary_op(SHADER_MUL, a, b)
    }

    pub fn relu(&self, a: &GpuBuffer) -> Result<GpuBuffer> {
        self.unary_op(SHADER_RELU, a)
    }

    pub fn sigmoid(&self, a: &GpuBuffer) -> Result<GpuBuffer> {
        self.unary_op(SHADER_SIGMOID, a)
    }

    pub fn swish(&self, a: &GpuBuffer) -> Result<GpuBuffer> {
        self.unary_op(SHADER_SWISH, a)
    }

    pub fn tanh_act(&self, a: &GpuBuffer) -> Result<GpuBuffer> {
        self.unary_op(SHADER_TANH, a)
    }

    pub fn scale(&self, a: &GpuBuffer, s: f32) -> Result<GpuBuffer> {
        let out = self.alloc(a.len);
        let params = ScaleParams { n: a.len as u32, scale: s, _pad: [0; 2] };
        self.dispatch_shader(SHADER_SCALE, None, &params, &[a], &out, super::dispatch_1d(a.len as u32));
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
    fn test_add() {
        let a = DEV.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = DEV.upload(&[10.0, 20.0, 30.0, 40.0]);
        let c = DEV.add(&a, &b).unwrap();
        let result = DEV.read(&c).unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_sub() {
        let a = DEV.upload(&[10.0, 20.0, 30.0]);
        let b = DEV.upload(&[1.0, 2.0, 3.0]);
        let result = DEV.read(&DEV.sub(&a, &b).unwrap()).unwrap();
        assert_eq!(result, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul() {
        let a = DEV.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = DEV.upload(&[10.0, 20.0, 30.0, 40.0]);
        let c = DEV.mul(&a, &b).unwrap();
        let result = DEV.read(&c).unwrap();
        assert_eq!(result, vec![10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn test_relu() {
        let a = DEV.upload(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = DEV.read(&DEV.relu(&a).unwrap()).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let a = DEV.upload(&[0.0, 1.0, -1.0, 10.0, -10.0]);
        let result = DEV.read(&DEV.sigmoid(&a).unwrap()).unwrap();
        assert_approx(&result, &[0.5, 0.7311, 0.2689, 0.99995, 0.00005], 1e-3);
    }

    #[test]
    fn test_swish() {
        let a = DEV.upload(&[0.0, 1.0, -1.0, 2.0]);
        let result = DEV.read(&DEV.swish(&a).unwrap()).unwrap();
        // swish(x) = x * sigmoid(x)
        assert_approx(&result, &[0.0, 0.7311, -0.2689, 1.7616], 1e-3);
    }

    #[test]
    fn test_tanh() {
        let a = DEV.upload(&[0.0, 1.0, -1.0, 3.0]);
        let result = DEV.read(&DEV.tanh_act(&a).unwrap()).unwrap();
        assert_approx(&result, &[0.0, 0.7616, -0.7616, 0.9951], 1e-3);
    }

    #[test]
    fn test_scale() {
        let a = DEV.upload(&[1.0, 2.0, 3.0, 4.0]);
        let result = DEV.read(&DEV.scale(&a, 0.5).unwrap()).unwrap();
        assert_eq!(result, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_matmul_2x2() {
        let a = DEV.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = DEV.upload(&[5.0, 6.0, 7.0, 8.0]);
        let c = DEV.matmul(&a, &b, 2, 2, 2).unwrap();
        let result = DEV.read(&c).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
