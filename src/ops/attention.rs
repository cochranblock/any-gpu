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

    // CPU reference softmax
    fn cpu_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        for r in 0..rows {
            let row = &input[r * cols..(r + 1) * cols];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
            for c in 0..cols {
                out[r * cols + c] = (row[c] - mx).exp() / sum;
            }
        }
        out
    }

    // CPU reference attention
    fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, dk: usize) -> Vec<f32> {
        let scale = 1.0 / (dk as f32).sqrt();
        // scores = Q @ K^T * scale: [seq, seq]
        let mut scores = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let mut s = 0.0;
                for d in 0..dk { s += q[i * dk + d] * k[j * dk + d]; }
                scores[i * seq + j] = s * scale;
            }
        }
        let attn = cpu_softmax(&scores, seq, seq);
        // out = attn @ V: [seq, dk]
        let mut out = vec![0.0f32; seq * dk];
        for i in 0..seq {
            for d in 0..dk {
                let mut s = 0.0;
                for j in 0..seq { s += attn[i * seq + j] * v[j * dk + d]; }
                out[i * dk + d] = s;
            }
        }
        out
    }

    #[test]
    fn test_softmax_vs_cpu() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 5.0, 5.0, 5.0];
        let expected = cpu_softmax(&data, 3, 3);
        let result = dev().read(&dev().softmax(&dev().upload(&data), 3, 3).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-4);
        // Verify each row sums to 1.0
        for r in 0..3 {
            let sum: f32 = result[r*3..(r+1)*3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {r} sum = {sum}");
        }
    }

    #[test]
    fn test_softmax_large_values() {
        // Numerical stability: large values should not overflow
        let data = vec![1000.0, 1001.0, 1002.0];
        let expected = cpu_softmax(&data, 1, 3);
        let result = dev().read(&dev().softmax(&dev().upload(&data), 1, 3).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-4);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
    }

    #[test]
    fn test_softmax_single_element() {
        let result = dev().read(&dev().softmax(&dev().upload(&[42.0]), 1, 1).unwrap()).unwrap();
        assert_approx(&result, &[1.0], 1e-5);
    }

    #[test]
    fn test_attention_vs_cpu() {
        // 1 head, seq=3, d_k=4 — fully verified against CPU reference
        let q: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let k: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let v: Vec<f32> = (0..12).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let expected = cpu_attention(&q, &k, &v, 3, 4);
        let result = dev().read(&dev().scaled_dot_product_attention(
            &dev().upload(&q), &dev().upload(&k), &dev().upload(&v), 1, 3, 4
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-3);
    }

    #[test]
    fn test_attention_uniform_qk() {
        // When Q and K are identical uniform vectors, attention is uniform -> output = mean(V) per position
        let q = vec![1.0, 1.0, 1.0, 1.0]; // seq=2, dk=2, both rows identical
        let k = q.clone();
        let v = vec![0.0, 10.0, 20.0, 30.0]; // seq=2, dk=2
        let expected = cpu_attention(&q, &k, &v, 2, 2);
        let result = dev().read(&dev().scaled_dot_product_attention(
            &dev().upload(&q), &dev().upload(&k), &dev().upload(&v), 1, 2, 2
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-3);
    }

    #[test]
    fn test_mse_loss() {
        let pred = dev().upload(&[1.0, 2.0, 3.0]);
        let target = dev().upload(&[1.5, 2.5, 3.5]);
        let result = dev().read(&dev().mse_loss(&pred, &target).unwrap()).unwrap();
        // MSE = ((0.5)^2 * 3) / 3 = 0.25
        assert_approx(&result, &[0.25], 1e-5);
    }

    #[test]
    fn test_mse_loss_zero() {
        let a = dev().upload(&[1.0, 2.0, 3.0]);
        let result = dev().read(&dev().mse_loss(&a, &a).unwrap()).unwrap();
        assert_approx(&result, &[0.0], 1e-6);
    }

    #[test]
    fn test_mse_loss_known_value() {
        let pred = dev().upload(&[0.0, 0.0, 0.0]);
        let target = dev().upload(&[1.0, 2.0, 3.0]);
        let result = dev().read(&dev().mse_loss(&pred, &target).unwrap()).unwrap();
        assert_approx(&result, &[14.0 / 3.0], 1e-5);
    }

    #[test]
    fn test_softmax_size_mismatch() {
        let input = dev().upload(&[1.0, 2.0, 3.0]); // 3 elements
        assert!(dev().softmax(&input, 2, 3).is_err()); // expects 6
    }

    #[test]
    fn test_mse_loss_length_mismatch() {
        let pred = dev().upload(&[1.0, 2.0]);
        let target = dev().upload(&[1.0, 2.0, 3.0]);
        assert!(dev().mse_loss(&pred, &target).is_err());
    }

    #[test]
    fn test_mse_loss_single_element() {
        let result = dev().read(&dev().mse_loss(&dev().upload(&[5.0]), &dev().upload(&[3.0])).unwrap()).unwrap();
        assert_approx(&result, &[4.0], 1e-5); // (5-3)^2 / 1 = 4
    }

    #[test]
    fn test_mse_loss_negative_values() {
        let pred = dev().upload(&[-1.0, -2.0]);
        let target = dev().upload(&[1.0, 2.0]);
        let result = dev().read(&dev().mse_loss(&pred, &target).unwrap()).unwrap();
        // ((-1-1)^2 + (-2-2)^2) / 2 = (4 + 16) / 2 = 10
        assert_approx(&result, &[10.0], 1e-5);
    }
}
