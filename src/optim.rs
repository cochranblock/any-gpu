// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// AdamW optimizer. Single WGSL shader: weight update step.

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamWParams {
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

/// AdamW optimizer state for a single parameter group.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    step: u32,
    // Per-parameter state: (first moment, second moment)
    states: Vec<(GpuBuffer, GpuBuffer)>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            states: Vec::new(),
        }
    }

    /// Run one optimizer step. Updates params in-place using their gradients.
    /// `params` and `grads` must be the same length, with matching buffer sizes.
    pub fn step(&mut self, dev: &GpuDevice, params: &mut [GpuBuffer], grads: &[GpuBuffer]) -> Result<()> {
        ensure!(params.len() == grads.len(), "params/grads length mismatch");

        self.step += 1;
        let beta1_t = self.beta1.powi(self.step as i32);
        let beta2_t = self.beta2.powi(self.step as i32);

        // Lazy init state buffers
        while self.states.len() < params.len() {
            let n = params[self.states.len()].len;
            let m = dev.upload(&vec![0.0f32; n]);
            let v = dev.upload(&vec![0.0f32; n]);
            self.states.push((m, v));
        }

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            ensure!(param.len == grad.len, "param/grad size mismatch at index {i}");
            let n = param.len as u32;
            let (m, v) = &self.states[i];

            let p = AdamWParams {
                n,
                lr: self.lr,
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.eps,
                weight_decay: self.weight_decay,
                beta1_t,
                beta2_t,
            };

            // AdamW needs read_write on param, m, v. Use raw dispatch.
            let params_buf = dev.upload_uniform(&p);
            let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("adamw"),
                source: wgpu::ShaderSource::Wgsl(SHADER_ADAMW.into()),
            });
            let pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adamw"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: param.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: grad.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: m.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: v.buffer.as_entire_binding() },
                ],
            });
            let (wx, wy, wz) = crate::ops::dispatch_1d(n);
            let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("adamw"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wx, wy, wz);
            }
            dev.queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::assert_approx;

    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    #[test]
    fn test_adamw_basic() {
        let mut param = dev().upload(&[1.0, 2.0, 3.0]);
        let grad = dev().upload(&[0.1, 0.2, 0.3]);

        let mut opt = AdamW::new(0.01);
        opt.weight_decay = 0.0; // disable weight decay for predictable test
        opt.step(dev(), std::slice::from_mut(&mut param), std::slice::from_ref(&grad)).unwrap();

        // After 1 step with no weight decay:
        // m = 0.1 * grad, v = 0.001 * grad^2
        // m_hat = m / (1 - 0.9) = grad, v_hat = v / (1 - 0.999) = grad^2
        // update = lr * grad / (sqrt(grad^2) + eps) ≈ lr * sign(grad)
        // For positive grads: param -= 0.01 (approximately)
        let result = dev().read(&param).unwrap();
        // params should decrease since gradients are positive
        assert!(result[0] < 1.0, "param[0] should decrease, got {}", result[0]);
        assert!(result[1] < 2.0, "param[1] should decrease, got {}", result[1]);
        assert!(result[2] < 3.0, "param[2] should decrease, got {}", result[2]);
    }

    #[test]
    fn test_adamw_multiple_steps() {
        let mut param = dev().upload(&[10.0]);
        let grad = dev().upload(&[1.0]); // constant gradient pushing param down

        let mut opt = AdamW::new(0.1);
        opt.weight_decay = 0.0;

        for _ in 0..10 {
            opt.step(dev(), std::slice::from_mut(&mut param), std::slice::from_ref(&grad)).unwrap();
        }

        let result = dev().read(&param).unwrap();
        // After 10 steps with lr=0.1 and constant positive grad, param should decrease
        assert!(result[0] < 10.0, "after 10 steps param should decrease, got {}", result[0]);
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut param = dev().upload(&[10.0]);
        let grad = dev().upload(&[0.0]); // zero gradient, only weight decay

        let mut opt = AdamW::new(0.01);
        opt.weight_decay = 0.1;

        opt.step(dev(), std::slice::from_mut(&mut param), std::slice::from_ref(&grad)).unwrap();

        let result = dev().read(&param).unwrap();
        // With zero grad, only weight decay: param *= (1 - lr * wd) = 10 * (1 - 0.001) = 9.99
        assert_approx(&result, &[9.99], 1e-3);
    }
}
