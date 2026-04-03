// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Softmax and scaled dot-product attention.

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

// --- Softmax (two-pass) ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    rows: u32,
    cols: u32,
    _pad: [u32; 2],
}

// Pass 1: one thread per row. Compute max and sum(exp(x - max)).
const SHADER_SOFTMAX_STATS: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= p.rows { return; }

    let base = row * p.cols;
    var mx: f32 = input[base];
    for (var i: u32 = 1u; i < p.cols; i++) {
        mx = max(mx, input[base + i]);
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < p.cols; i++) {
        sum += exp(input[base + i] - mx);
    }
    stats[row * 2u] = mx;
    stats[row * 2u + 1u] = sum;
}
";

// Pass 2: one thread per element. exp(x - max) / sum.
const SHADER_SOFTMAX_APPLY: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.rows * p.cols { return; }

    let row = idx / p.cols;
    let mx = stats[row * 2u];
    let sum = stats[row * 2u + 1u];
    out[idx] = exp(input[idx] - mx) / sum;
}
";

// --- MSE Loss ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    n: u32,
    _pad: [u32; 3],
}

const SHADER_MSE_SUM: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pred: array<f32>;
@group(0) @binding(2) var<storage, read> tgt: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < p.n; i++) {
        let d = pred[i] - tgt[i];
        sum += d * d;
    }
    out[0] = sum / f32(p.n);
}
";

impl GpuDevice {
    /// Softmax along the last dimension. Input shape: [rows, cols].
    pub fn softmax(&self, input: &GpuBuffer, rows: u32, cols: u32) -> Result<GpuBuffer> {
        ensure!(input.len == (rows * cols) as usize);

        let params = SoftmaxParams { rows, cols, _pad: [0; 2] };

        // Pass 1: per-row max and sum
        let stats = self.alloc((rows * 2) as usize);
        self.dispatch_shader(
            SHADER_SOFTMAX_STATS, Some("softmax_stats"),
            &params, &[input], &stats,
            super::dispatch_1d(rows),
        );

        // Pass 2: normalize
        let total = rows * cols;
        let out = self.alloc(total as usize);

        let params_buf = self.upload_uniform(&params);
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax_apply"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOFTMAX_APPLY.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_apply"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: stats.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.buffer.as_entire_binding() },
            ],
        });
        let (wx, wy, wz) = super::dispatch_1d(total);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_apply"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wx, wy, wz);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(out)
    }

    /// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
    /// Q,K,V: [batch_heads, seq_len, d_k]. Returns [batch_heads, seq_len, d_k].
    pub fn scaled_dot_product_attention(
        &self,
        q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
        batch_heads: u32, seq_len: u32, d_k: u32,
    ) -> Result<GpuBuffer> {
        // 1. K^T: [batch_heads, d_k, seq_len]
        let kt = self.transpose(k, batch_heads, seq_len, d_k, 1)?;

        // 2. scores = Q @ K^T: [batch_heads, seq_len, seq_len]
        let scores = self.batch_matmul(q, &kt, batch_heads, seq_len, seq_len, d_k)?;

        // 3. Scale by 1/sqrt(d_k)
        let scale = 1.0 / (d_k as f32).sqrt();
        let scaled = self.scale(&scores, scale)?;

        // 4. Softmax over last dim (each row of seq_len)
        let attn = self.softmax(&scaled, batch_heads * seq_len, seq_len)?;

        // 5. attn @ V: [batch_heads, seq_len, d_k]
        self.batch_matmul(&attn, v, batch_heads, seq_len, d_k, seq_len)
    }

    /// MSE loss: mean((pred - target)^2). Returns a 1-element buffer.
    pub fn mse_loss(&self, pred: &GpuBuffer, target: &GpuBuffer) -> Result<GpuBuffer> {
        ensure!(pred.len == target.len, "mse: length mismatch");
        let out = self.alloc(1);
        let params = ReduceParams { n: pred.len as u32, _pad: [0; 3] };
        self.dispatch_shader(
            SHADER_MSE_SUM, Some("mse"),
            &params, &[pred, target], &out,
            (1, 1, 1),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::assert_approx;

    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    #[test]
    fn test_softmax() {
        let input = dev().upload(&[1.0, 2.0, 3.0]);
        let out = dev().softmax(&input, 1, 3).unwrap();
        let result = dev().read(&out).unwrap();
        assert_approx(&result, &[0.0900, 0.2447, 0.6652], 1e-3);
        // Sum should be 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn test_softmax_multirow() {
        let input = dev().upload(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let out = dev().softmax(&input, 2, 3).unwrap();
        let result = dev().read(&out).unwrap();
        // Row 0: standard softmax
        assert_approx(&result[0..3], &[0.0900, 0.2447, 0.6652], 1e-3);
        // Row 1: uniform
        assert_approx(&result[3..6], &[0.3333, 0.3333, 0.3333], 1e-3);
    }

    #[test]
    fn test_attention_identity() {
        // 1 head, seq=2, d_k=2. Q=K=V=identity-like.
        // With Q=K, scores should be high on diagonal -> attention ~ identity -> output ~ V
        let qkv = dev().upload(&[
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let out = dev().scaled_dot_product_attention(&qkv, &qkv, &qkv, 1, 2, 2).unwrap();
        let result = dev().read(&out).unwrap();
        assert_eq!(result.len(), 4);
        // With orthogonal Q=K, attention should heavily weight diagonal -> output close to V
        // score matrix after softmax: [[e^(1/sqrt2)/(e^(1/sqrt2)+e^0), ...], ...]
        // Just check output shape and reasonable values
        assert!(result[0] > 0.5, "expected first elem > 0.5, got {}", result[0]);
    }

    #[test]
    fn test_mse_loss() {
        let pred = dev().upload(&[1.0, 2.0, 3.0]);
        let target = dev().upload(&[1.5, 2.5, 3.5]);
        let loss = dev().mse_loss(&pred, &target).unwrap();
        let result = dev().read(&loss).unwrap();
        // MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.75/3 = 0.25
        assert_approx(&result, &[0.25], 1e-5);
    }

    #[test]
    fn test_mse_loss_zero() {
        let a = dev().upload(&[1.0, 2.0, 3.0]);
        let b = dev().upload(&[1.0, 2.0, 3.0]);
        let result = dev().read(&dev().mse_loss(&a, &b).unwrap()).unwrap();
        assert_approx(&result, &[0.0], 1e-6);
    }
}
