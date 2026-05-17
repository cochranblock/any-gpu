// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// AdamW optimizer. Single WGSL shader: weight update step.
// t507=AdamW, t508=AdamWParams (private). f720=AdamW::new, f721=AdamW::step.

use crate::device::{t500, t501};
use anyhow::{ensure, Result};

/// t508 = AdamWParams. WGSL uniform; field names map to shader struct.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t508 {
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    beta1_t: f32, // beta1^t (for bias correction)
    beta2_t: f32, // beta2^t
}

const SHADER_ADAMW: &str = "
struct P {
    n: u32, lr: f32, beta1: f32, beta2: f32,
    eps: f32, weight_decay: f32, beta1_t: f32, beta2_t: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.n { return; }

    let g = grad[idx];

    // Update biased first moment
    m[idx] = p.beta1 * m[idx] + (1.0 - p.beta1) * g;
    // Update biased second moment
    v[idx] = p.beta2 * v[idx] + (1.0 - p.beta2) * g * g;

    // Bias correction
    let m_hat = m[idx] / (1.0 - p.beta1_t);
    let v_hat = v[idx] / (1.0 - p.beta2_t);

    // Weight decay + Adam update
    param[idx] = param[idx] * (1.0 - p.lr * p.weight_decay) - p.lr * m_hat / (sqrt(v_hat) + p.eps);
}
";

/// t507 = AdamW. Optimizer state for a single parameter group.
pub struct t507 {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    step: u32,
    // Per-parameter state: (first moment, second moment)
    states: Vec<(t501, t501)>,
}

impl t507 {
    /// f720 = AdamW::new. Default hyperparams (beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01).
    pub fn f720(p0: f32) -> Self {
        Self {
            lr: p0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            states: Vec::new(),
        }
    }

    /// f721 = AdamW::step. Run one optimizer step. Updates params in-place using
    /// their gradients. `p1` and `p2` must be the same length, with matching buffer sizes.
    pub fn f721(&mut self, p0: &t500, p1: &mut [t501], p2: &[t501]) -> Result<()> {
        ensure!(p1.len() == p2.len(), "params/grads length mismatch");

        self.step += 1;
        let v0 = self.beta1.powi(self.step as i32);
        let v1 = self.beta2.powi(self.step as i32);

        // Lazy init state buffers
        while self.states.len() < p1.len() {
            let v2 = p1[self.states.len()].s507;
            let v3 = p0.f502(&vec![0.0f32; v2]);
            let v4 = p0.f502(&vec![0.0f32; v2]);
            self.states.push((v3, v4));
        }

        for (v5, (v6, v7)) in p1.iter().zip(p2.iter()).enumerate() {
            ensure!(v6.s507 == v7.s507, "param/grad size mismatch at index {v5}");
            let v8 = v6.s507 as u32;
            let (v9, v10) = &self.states[v5];

            let v11 = t508 {
                n: v8,
                lr: self.lr,
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.eps,
                weight_decay: self.weight_decay,
                beta1_t: v0,
                beta2_t: v1,
            };

            // AdamW needs read_write on param, m, v. Use raw dispatch with cached pipeline.
            let v12 = p0.f506(&v11);
            let v13 = p0.f507(SHADER_ADAMW, Some("adamw"));
            let v14 = p0.s500.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &v13.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: v12.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: v6.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: v7.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: v9.s505.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: v10.s505.as_entire_binding() },
                ],
            });
            let (v15, v16, v17) = crate::ops::f540(v8);
            let mut v18 = p0.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut v19 = v18.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("adamw"),
                    timestamp_writes: None,
                });
                v19.set_pipeline(&v13);
                v19.set_bind_group(0, &v14, &[]);
                v19.dispatch_workgroups(v15, v16, v17);
            }
            p0.s501.submit(Some(v18.finish()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;

    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    #[test]
    fn f721_basic() {
        let mut v0 = dev().f502(&[1.0, 2.0, 3.0]);
        let v1 = dev().f502(&[0.1, 0.2, 0.3]);

        let mut v2 = t507::f720(0.01);
        v2.weight_decay = 0.0;
        v2.f721(dev(), std::slice::from_mut(&mut v0), std::slice::from_ref(&v1)).unwrap();

        let v3 = dev().f504(&v0).unwrap();
        assert!(v3[0] < 1.0, "param[0] should decrease, got {}", v3[0]);
        assert!(v3[1] < 2.0, "param[1] should decrease, got {}", v3[1]);
        assert!(v3[2] < 3.0, "param[2] should decrease, got {}", v3[2]);
    }

    #[test]
    fn f721_multiple_steps() {
        let mut v0 = dev().f502(&[10.0]);
        let v1 = dev().f502(&[1.0]);

        let mut v2 = t507::f720(0.1);
        v2.weight_decay = 0.0;

        for _ in 0..10 {
            v2.f721(dev(), std::slice::from_mut(&mut v0), std::slice::from_ref(&v1)).unwrap();
        }

        let v3 = dev().f504(&v0).unwrap();
        assert!(v3[0] < 10.0, "after 10 steps param should decrease, got {}", v3[0]);
    }

    #[test]
    fn f721_weight_decay() {
        let mut v0 = dev().f502(&[10.0]);
        let v1 = dev().f502(&[0.0]);

        let mut v2 = t507::f720(0.01);
        v2.weight_decay = 0.1;

        v2.f721(dev(), std::slice::from_mut(&mut v0), std::slice::from_ref(&v1)).unwrap();

        let v3 = dev().f504(&v0).unwrap();
        f544(&v3, &[9.99], 1e-3);
    }

    #[test]
    fn f721_params_grads_length_mismatch() {
        let v0 = dev().f502(&[1.0]);
        let v1 = dev().f502(&[2.0]);
        let v2 = dev().f502(&[0.1]);
        let mut v3 = t507::f720(0.01);
        assert!(v3.f721(dev(), &mut [v0, v1], &[v2]).is_err());
    }

    #[test]
    fn f721_param_grad_size_mismatch() {
        let mut v0 = dev().f502(&[1.0, 2.0, 3.0]);
        let v1 = dev().f502(&[0.1, 0.2]);
        let mut v2 = t507::f720(0.01);
        assert!(v2.f721(dev(), std::slice::from_mut(&mut v0), std::slice::from_ref(&v1)).is_err());
    }

    #[test]
    fn f721_negative_gradient() {
        let mut v0 = dev().f502(&[5.0]);
        let v1 = dev().f502(&[-1.0]);
        let mut v2 = t507::f720(0.01);
        v2.weight_decay = 0.0;
        v2.f721(dev(), std::slice::from_mut(&mut v0), std::slice::from_ref(&v1)).unwrap();
        let v3 = dev().f504(&v0).unwrap();
        assert!(v3[0] > 5.0, "negative grad should increase param, got {}", v3[0]);
    }

    #[test]
    fn f721_lr_zero() {
        let mut v0 = dev().f502(&[10.0]);
        let v1 = dev().f502(&[100.0]);
        let mut v2 = t507::f720(0.0);
        v2.weight_decay = 0.0;
        v2.f721(dev(), std::slice::from_mut(&mut v0), std::slice::from_ref(&v1)).unwrap();
        let v3 = dev().f504(&v0).unwrap();
        f544(&v3, &[10.0], 1e-5);
    }

    #[test]
    fn f720_defaults() {
        let v0 = t507::f720(0.001);
        assert_eq!(v0.lr, 0.001);
        assert_eq!(v0.beta1, 0.9);
        assert_eq!(v0.beta2, 0.999);
        assert_eq!(v0.eps, 1e-8);
        assert_eq!(v0.weight_decay, 0.01);
    }
}
