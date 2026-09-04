// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Element-wise ops: f550..f558 forward, f559..f562 backward, f563 = gelu.

use crate::device::{t500, t501};
use anyhow::{Result, ensure};

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

// GELU — tanh approximation as used by GPT-2 / BERT / ViT and the `approximate="tanh"`
// path in PyTorch. sqrt(2/pi) ~= 0.7978845608. The exact erf form is not exposed in WGSL.
const SHADER_GELU: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    let x = a[idx];
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    out[idx] = 0.5 * x * (1.0 + tanh(inner));
}
";

/// t512 = ScaleParams. Uniform for f553 (scale).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t512 {
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

// --- Backward shaders ---

const SHADER_RELU_BACKWARD: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    out[idx] = select(0.0, grad_out[idx], input[idx] > 0.0);
}
";

const SHADER_SIGMOID_BACKWARD: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read> sig_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    let s = sig_out[idx];
    out[idx] = grad_out[idx] * s * (1.0 - s);
}
";

const SHADER_SWISH_BACKWARD: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    let x = input[idx];
    let s = 1.0 / (1.0 + exp(-x));
    out[idx] = grad_out[idx] * (s + x * s * (1.0 - s));
}
";

const SHADER_TANH_BACKWARD: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read> tanh_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= params.n { return; }
    let t = tanh_out[idx];
    out[idx] = grad_out[idx] * (1.0 - t * t);
}
";

impl t500 {
    /// f550 = add. Element-wise add of two equally sized buffers.
    pub fn f550(&self, p0: &t501, p1: &t501) -> Result<t501> {
        ensure!(
            p0.s507 == p1.s507,
            "add: length mismatch ({} vs {})",
            p0.s507,
            p1.s507
        );
        self.f542(SHADER_ADD, p0, p1)
    }

    /// f551 = sub. Element-wise subtraction.
    pub fn f551(&self, p0: &t501, p1: &t501) -> Result<t501> {
        ensure!(
            p0.s507 == p1.s507,
            "sub: length mismatch ({} vs {})",
            p0.s507,
            p1.s507
        );
        self.f542(SHADER_SUB, p0, p1)
    }

    /// f552 = mul. Element-wise multiply.
    pub fn f552(&self, p0: &t501, p1: &t501) -> Result<t501> {
        ensure!(
            p0.s507 == p1.s507,
            "mul: length mismatch ({} vs {})",
            p0.s507,
            p1.s507
        );
        self.f542(SHADER_MUL, p0, p1)
    }

    /// f554 = relu.
    pub fn f554(&self, p0: &t501) -> Result<t501> {
        self.f541(SHADER_RELU, p0)
    }

    /// f555 = sigmoid.
    pub fn f555(&self, p0: &t501) -> Result<t501> {
        self.f541(SHADER_SIGMOID, p0)
    }

    /// f556 = swish (a.k.a. SiLU): x * sigmoid(x).
    pub fn f556(&self, p0: &t501) -> Result<t501> {
        self.f541(SHADER_SWISH, p0)
    }

    /// f557 = tanh_act.
    pub fn f557(&self, p0: &t501) -> Result<t501> {
        self.f541(SHADER_TANH, p0)
    }

    /// f563 = gelu (tanh approximation). 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))).
    /// Used by GPT-2, BERT, ViT, and `approximate="tanh"` in PyTorch.
    pub fn f563(&self, p0: &t501) -> Result<t501> {
        self.f541(SHADER_GELU, p0)
    }

    /// f553 = scale. out = a * scalar.
    pub fn f553(&self, p0: &t501, p1: f32) -> Result<t501> {
        let v0 = self.f503(p0.s507);
        let v1 = t512 {
            n: p0.s507 as u32,
            scale: p1,
            _pad: [0; 2],
        };
        self.f543(
            SHADER_SCALE,
            None,
            &v1,
            &[p0],
            &v0,
            super::f540(p0.s507 as u32),
        );
        Ok(v0)
    }

    // --- Backward shaders for autograd ---

    /// f559 = relu_backward. grad_a = grad_out * (input > 0).
    pub fn f559(&self, p0: &t501, p1: &t501) -> Result<t501> {
        ensure!(p0.s507 == p1.s507);
        self.f542(SHADER_RELU_BACKWARD, p0, p1)
    }

    /// f560 = sigmoid_backward. grad_a = grad_out * output * (1 - output).
    pub fn f560(&self, p0: &t501, p1: &t501) -> Result<t501> {
        ensure!(p0.s507 == p1.s507);
        self.f542(SHADER_SIGMOID_BACKWARD, p0, p1)
    }

    /// f561 = swish_backward. grad_a = grad_out * (sig(x) + x * sig(x) * (1 - sig(x))).
    pub fn f561(&self, p0: &t501, p1: &t501) -> Result<t501> {
        ensure!(p0.s507 == p1.s507);
        self.f542(SHADER_SWISH_BACKWARD, p0, p1)
    }

    /// f562 = tanh_backward. grad_a = grad_out * (1 - output^2).
    pub fn f562(&self, p0: &t501, p1: &t501) -> Result<t501> {
        ensure!(p0.s507 == p1.s507);
        self.f542(SHADER_TANH_BACKWARD, p0, p1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;
    fn dev() -> &'static t500 {
        &crate::ops::TEST_DEV
    }

    // CPU references for cross-validation
    fn cpu_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn cpu_swish(x: f32) -> f32 {
        x * cpu_sigmoid(x)
    }

    #[test]
    fn f550_basic() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0]);
        let v1 = dev().f502(&[10.0, 20.0, 30.0, 40.0]);
        let v2 = dev().f504(&dev().f550(&v0, &v1).unwrap()).unwrap();
        assert_eq!(v2, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn f550_odd_size() {
        // 13 elements — not aligned to workgroup size 256
        let v0: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let v1: Vec<f32> = (0..13).map(|i| i as f32 * 10.0).collect();
        let v2: Vec<f32> = v0.iter().zip(&v1).map(|(a, b)| a + b).collect();
        let v3 = dev()
            .f504(&dev().f550(&dev().f502(&v0), &dev().f502(&v1)).unwrap())
            .unwrap();
        assert_eq!(v3, v2);
    }

    #[test]
    fn f550_single_element() {
        let v0 = dev()
            .f504(
                &dev()
                    .f550(&dev().f502(&[42.0]), &dev().f502(&[-42.0]))
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(v0, vec![0.0]);
    }

    #[test]
    fn f551_basic() {
        let v0 = dev().f502(&[10.0, 20.0, 30.0]);
        let v1 = dev().f502(&[1.0, 2.0, 3.0]);
        let v2 = dev().f504(&dev().f551(&v0, &v1).unwrap()).unwrap();
        assert_eq!(v2, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn f552_basic() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0]);
        let v1 = dev().f502(&[10.0, 20.0, 30.0, 40.0]);
        let v2 = dev().f504(&dev().f552(&v0, &v1).unwrap()).unwrap();
        assert_eq!(v2, vec![10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn f552_zeros() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0]);
        let v1 = dev().f502(&[0.0, 0.0, 0.0]);
        let v2 = dev().f504(&dev().f552(&v0, &v1).unwrap()).unwrap();
        assert_eq!(v2, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn f554_basic() {
        let v0 = dev().f502(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let v1 = dev().f504(&dev().f554(&v0).unwrap()).unwrap();
        assert_eq!(v1, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn f554_all_negative() {
        let v0 = dev()
            .f504(&dev().f554(&dev().f502(&[-100.0, -0.001, -1e-10])).unwrap())
            .unwrap();
        assert_eq!(v0, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn f555_vs_cpu() {
        let v0: Vec<f32> = vec![-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0];
        let v1: Vec<f32> = v0.iter().map(|&x| cpu_sigmoid(x)).collect();
        let v2 = dev().f504(&dev().f555(&dev().f502(&v0)).unwrap()).unwrap();
        f544(&v2, &v1, 1e-4);
    }

    #[test]
    fn f556_vs_cpu() {
        let v0: Vec<f32> = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let v1: Vec<f32> = v0.iter().map(|&x| cpu_swish(x)).collect();
        let v2 = dev().f504(&dev().f556(&dev().f502(&v0)).unwrap()).unwrap();
        f544(&v2, &v1, 1e-4);
    }

    #[test]
    fn f557_vs_cpu() {
        let v0: Vec<f32> = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        let v1: Vec<f32> = v0.iter().map(|&x| x.tanh()).collect();
        let v2 = dev().f504(&dev().f557(&dev().f502(&v0)).unwrap()).unwrap();
        f544(&v2, &v1, 1e-4);
    }

    #[test]
    fn f555_known_reference_values() {
        // Hardcoded sigmoid values from the math definition 1/(1+e^-x), computed
        // at f64 precision externally. Independent of our shader AND the cpu_sigmoid
        // helper (which shares the shader's formula). A bug in either expression
        // would diverge from these reference values.
        let v0: Vec<f32> = vec![-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0];
        let v1: Vec<f32> = vec![
            4.5397868e-5, // sigmoid(-10)
            0.11920292,   // sigmoid(-2)
            0.26894142,   // sigmoid(-1)
            0.5,          // sigmoid(0)
            0.73105858,   // sigmoid(1)
            0.88079708,   // sigmoid(2)
            0.9999546,    // sigmoid(10)
        ];
        let v2 = dev().f504(&dev().f555(&dev().f502(&v0)).unwrap()).unwrap();
        f544(&v2, &v1, 1e-5);
    }

    #[test]
    fn f556_known_reference_values() {
        // Hardcoded SiLU (= x * sigmoid(x)) values at f64 precision. Independent
        // of the cpu_swish helper which is the same formula as the shader.
        let v0: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let v1: Vec<f32> = vec![
            -0.23840584, // silu(-2) = -2 * sigmoid(-2)
            -0.26894142, // silu(-1)
            0.0,         // silu(0)
            0.73105858,  // silu(1)
            1.76159416,  // silu(2)
        ];
        let v2 = dev().f504(&dev().f556(&dev().f502(&v0)).unwrap()).unwrap();
        f544(&v2, &v1, 1e-5);
    }

    #[test]
    fn f563_vs_pytorch_reference() {
        // Reference values from torch.nn.functional.gelu(x, approximate="tanh").
        // Independent of our shader — if the shader and these diverge, one is wrong.
        let v0: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let v1: Vec<f32> = vec![
            -0.04540231,
            -0.15880801,
            -0.15428598,
            0.0,
            0.34571400,
            0.84119199,
            1.95459769,
        ];
        let v2 = dev().f504(&dev().f563(&dev().f502(&v0)).unwrap()).unwrap();
        f544(&v2, &v1, 1e-4);
    }

    #[test]
    fn f563_zero_is_zero() {
        let v0 = dev()
            .f504(&dev().f563(&dev().f502(&[0.0])).unwrap())
            .unwrap();
        assert!(v0[0].abs() < 1e-6);
    }

    #[test]
    fn f563_large_positive_passes_through() {
        let v0 = vec![10.0f32, 100.0];
        let v1 = dev().f504(&dev().f563(&dev().f502(&v0)).unwrap()).unwrap();
        assert!((v1[0] - 10.0).abs() < 1e-3);
        assert!((v1[1] - 100.0).abs() < 1e-2);
    }

    #[test]
    fn f563_large_negative_kills_signal() {
        let v0 = vec![-10.0f32, -100.0];
        let v1 = dev().f504(&dev().f563(&dev().f502(&v0)).unwrap()).unwrap();
        assert!(v1[0].abs() < 1e-3);
        assert!(v1[1].abs() < 1e-3);
    }

    #[test]
    fn f553_basic() {
        let v0 = dev()
            .f504(&dev().f553(&dev().f502(&[1.0, 2.0, 3.0, 4.0]), 0.5).unwrap())
            .unwrap();
        assert_eq!(v0, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn f553_zero() {
        let v0 = dev()
            .f504(&dev().f553(&dev().f502(&[99.0, -99.0]), 0.0).unwrap())
            .unwrap();
        assert_eq!(v0, vec![0.0, 0.0]);
    }

    #[test]
    fn f553_negative() {
        let v0 = dev()
            .f504(&dev().f553(&dev().f502(&[1.0, -2.0, 3.0]), -2.0).unwrap())
            .unwrap();
        assert_eq!(v0, vec![-2.0, 4.0, -6.0]);
    }

    // --- Error path tests ---

    #[test]
    fn f550_length_mismatch() {
        let v0 = dev().f502(&[1.0, 2.0]);
        let v1 = dev().f502(&[1.0, 2.0, 3.0]);
        assert!(dev().f550(&v0, &v1).is_err());
    }

    #[test]
    fn f551_length_mismatch() {
        let v0 = dev().f502(&[1.0]);
        let v1 = dev().f502(&[1.0, 2.0]);
        assert!(dev().f551(&v0, &v1).is_err());
    }

    #[test]
    fn f552_length_mismatch() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0]);
        let v1 = dev().f502(&[1.0]);
        assert!(dev().f552(&v0, &v1).is_err());
    }

    // --- CPU cross-validation for add/sub/mul ---

    #[test]
    fn f550_vs_cpu() {
        let v0: Vec<f32> = (0..100).map(|i| (i as f32) * 0.3 - 15.0).collect();
        let v1: Vec<f32> = (0..100).map(|i| (i as f32) * -0.2 + 10.0).collect();
        let v2: Vec<f32> = v0.iter().zip(&v1).map(|(x, y)| x + y).collect();
        let v3 = dev()
            .f504(&dev().f550(&dev().f502(&v0), &dev().f502(&v1)).unwrap())
            .unwrap();
        f544(&v3, &v2, 1e-5);
    }

    #[test]
    fn f551_vs_cpu() {
        let v0: Vec<f32> = (0..100).map(|i| (i as f32) * 0.7).collect();
        let v1: Vec<f32> = (0..100).map(|i| (i as f32) * 0.3).collect();
        let v2: Vec<f32> = v0.iter().zip(&v1).map(|(x, y)| x - y).collect();
        let v3 = dev()
            .f504(&dev().f551(&dev().f502(&v0), &dev().f502(&v1)).unwrap())
            .unwrap();
        f544(&v3, &v2, 1e-5);
    }

    #[test]
    fn f552_vs_cpu() {
        let v0: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let v1: Vec<f32> = (0..100).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let v2: Vec<f32> = v0.iter().zip(&v1).map(|(x, y)| x * y).collect();
        let v3 = dev()
            .f504(&dev().f552(&dev().f502(&v0), &dev().f502(&v1)).unwrap())
            .unwrap();
        f544(&v3, &v2, 1e-4);
    }

    // --- Backward shader direct tests ---

    #[test]
    fn f559_vs_cpu() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let v1 = dev().f502(&[-1.0, 0.5, 0.0, -0.1, 2.0]);
        let v2 = dev().f504(&dev().f559(&v0, &v1).unwrap()).unwrap();
        f544(&v2, &[0.0, 2.0, 0.0, 0.0, 5.0], 1e-5);
    }

    #[test]
    fn f560_vs_cpu() {
        let v0 = vec![0.5, 0.7311, 0.2689]; // sigmoid outputs
        let v1 = vec![1.0, 1.0, 1.0];
        let v2: Vec<f32> = v0.iter().zip(&v1).map(|(s, g)| g * s * (1.0 - s)).collect();
        let v3 = dev()
            .f504(&dev().f560(&dev().f502(&v1), &dev().f502(&v0)).unwrap())
            .unwrap();
        f544(&v3, &v2, 1e-3);
    }

    #[test]
    fn f561_vs_cpu() {
        let v0 = vec![0.0, 1.0, -1.0, 2.0];
        let v1 = vec![1.0, 1.0, 1.0, 1.0];
        let v2: Vec<f32> = v0
            .iter()
            .map(|&x| {
                let s = 1.0f32 / (1.0f32 + (-(x as f32)).exp());
                s + x * s * (1.0 - s)
            })
            .collect();
        let v3 = dev()
            .f504(&dev().f561(&dev().f502(&v1), &dev().f502(&v0)).unwrap())
            .unwrap();
        f544(&v3, &v2, 1e-3);
    }

    #[test]
    fn f562_vs_cpu() {
        let v0 = vec![0.0, 0.7616, -0.7616, 0.9951]; // tanh outputs
        let v1 = vec![1.0, 1.0, 1.0, 1.0];
        let v2: Vec<f32> = v0.iter().map(|&t| 1.0 - t * t).collect();
        let v3 = dev()
            .f504(&dev().f562(&dev().f502(&v1), &dev().f502(&v0)).unwrap())
            .unwrap();
        f544(&v3, &v2, 1e-3);
    }
}
