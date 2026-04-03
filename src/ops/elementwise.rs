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
    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    // CPU references for cross-validation
    fn cpu_sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
    fn cpu_swish(x: f32) -> f32 { x * cpu_sigmoid(x) }

    #[test]
    fn test_add() {
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = dev().upload(&[10.0, 20.0, 30.0, 40.0]);
        let result = dev().read(&dev().add(&a, &b).unwrap()).unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_add_odd_size() {
        // 13 elements — not aligned to workgroup size 256
        let a_data: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..13).map(|i| i as f32 * 10.0).collect();
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a + b).collect();
        let result = dev().read(&dev().add(&dev().upload(&a_data), &dev().upload(&b_data)).unwrap()).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_single_element() {
        let result = dev().read(&dev().add(&dev().upload(&[42.0]), &dev().upload(&[-42.0])).unwrap()).unwrap();
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn test_sub() {
        let a = dev().upload(&[10.0, 20.0, 30.0]);
        let b = dev().upload(&[1.0, 2.0, 3.0]);
        let result = dev().read(&dev().sub(&a, &b).unwrap()).unwrap();
        assert_eq!(result, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul() {
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = dev().upload(&[10.0, 20.0, 30.0, 40.0]);
        let result = dev().read(&dev().mul(&a, &b).unwrap()).unwrap();
        assert_eq!(result, vec![10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn test_mul_zeros() {
        let a = dev().upload(&[1.0, 2.0, 3.0]);
        let b = dev().upload(&[0.0, 0.0, 0.0]);
        let result = dev().read(&dev().mul(&a, &b).unwrap()).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu() {
        let a = dev().upload(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = dev().read(&dev().relu(&a).unwrap()).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_all_negative() {
        let result = dev().read(&dev().relu(&dev().upload(&[-100.0, -0.001, -1e-10])).unwrap()).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sigmoid_vs_cpu() {
        let data: Vec<f32> = vec![-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0];
        let expected: Vec<f32> = data.iter().map(|&x| cpu_sigmoid(x)).collect();
        let result = dev().read(&dev().sigmoid(&dev().upload(&data)).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-4);
    }

    #[test]
    fn test_swish_vs_cpu() {
        let data: Vec<f32> = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expected: Vec<f32> = data.iter().map(|&x| cpu_swish(x)).collect();
        let result = dev().read(&dev().swish(&dev().upload(&data)).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-4);
    }

    #[test]
    fn test_tanh_vs_cpu() {
        let data: Vec<f32> = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        let expected: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
        let result = dev().read(&dev().tanh_act(&dev().upload(&data)).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-4);
    }

    #[test]
    fn test_scale() {
        let result = dev().read(&dev().scale(&dev().upload(&[1.0, 2.0, 3.0, 4.0]), 0.5).unwrap()).unwrap();
        assert_eq!(result, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_scale_zero() {
        let result = dev().read(&dev().scale(&dev().upload(&[99.0, -99.0]), 0.0).unwrap()).unwrap();
        assert_eq!(result, vec![0.0, 0.0]);
    }

    #[test]
    fn test_scale_negative() {
        let result = dev().read(&dev().scale(&dev().upload(&[1.0, -2.0, 3.0]), -2.0).unwrap()).unwrap();
        assert_eq!(result, vec![-2.0, 4.0, -6.0]);
    }
}
