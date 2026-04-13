// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Group normalization (two-pass: compute stats, then normalize+affine).

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GNStatsParams {
    batch: u32,
    channels: u32,
    spatial: u32,
    groups: u32,
    channels_per_group: u32,
    eps: f32,
    _pad: [u32; 2],
}

// Pass 1: one thread per (batch, group). Computes mean and variance.
const SHADER_GN_STATS: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.batch * p.groups;
    if idx >= total { return; }

    let g = idx % p.groups;
    let n = idx / p.groups;

    let count = p.cpg * p.spatial;
    var sum: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cpg; c++) {
        let ch = g * p.cpg + c;
        let base = n * (p.channels * p.spatial) + ch * p.spatial;
        for (var s: u32 = 0u; s < p.spatial; s++) {
            let v = input[base + s];
            sum += v;
            sum_sq += v * v;
        }
    }
    let mean = sum / f32(count);
    let variance = sum_sq / f32(count) - mean * mean;
    stats[idx * 2u] = mean;
    stats[idx * 2u + 1u] = 1.0 / sqrt(variance + p.eps);
}
";

// Pass 2: one thread per element. Normalize and apply affine.
const SHADER_GN_NORM: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.channels * p.spatial;
    if idx >= total { return; }

    let s = idx % p.spatial;
    let ch = (idx / p.spatial) % p.channels;
    let n = idx / (p.spatial * p.channels);
    let g = ch / p.cpg;

    let stat_idx = n * p.groups + g;
    let mean = stats[stat_idx * 2u];
    let inv_std = stats[stat_idx * 2u + 1u];

    out[idx] = (input[idx] - mean) * inv_std * gamma[ch] + beta[ch];
}
";

// --- GroupNorm backward ---
// Pass A: recompute stats (mean, inv_std) per (n, g) — reuse SHADER_GN_STATS.
// Pass B: compute per-(n, g) reduction sums for d_input:
//         s1 = sum over group of (grad_out * gamma)
//         s2 = sum over group of (grad_out * gamma * x_hat)
// Pass C: compute d_input per element using stats and sums.
// Pass D: compute d_gamma and d_beta per channel by reduction over batch and spatial.

const SHADER_GN_BACK_SUMS: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> sums: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.batch * p.groups;
    if idx >= total { return; }
    let g = idx % p.groups;
    let n = idx / p.groups;

    let mean = stats[idx * 2u];
    let inv_std = stats[idx * 2u + 1u];

    var s1: f32 = 0.0;
    var s2: f32 = 0.0;
    for (var c: u32 = 0u; c < p.cpg; c++) {
        let ch = g * p.cpg + c;
        let base = n * (p.channels * p.spatial) + ch * p.spatial;
        for (var s: u32 = 0u; s < p.spatial; s++) {
            let dx_hat = grad_out[base + s] * gamma[ch];
            let x_hat = (input[base + s] - mean) * inv_std;
            s1 += dx_hat;
            s2 += dx_hat * x_hat;
        }
    }
    sums[idx * 2u] = s1;
    sums[idx * 2u + 1u] = s2;
}
";

const SHADER_GN_BACK_DINPUT: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read> sums: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_input: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.channels * p.spatial;
    if idx >= total { return; }

    let s_idx = idx % p.spatial;
    let ch = (idx / p.spatial) % p.channels;
    let n = idx / (p.spatial * p.channels);
    let g = ch / p.cpg;

    let stat_idx = n * p.groups + g;
    let mean = stats[stat_idx * 2u];
    let inv_std = stats[stat_idx * 2u + 1u];
    let x_hat = (input[idx] - mean) * inv_std;
    let dx_hat = grad_out[idx] * gamma[ch];

    let s1 = sums[stat_idx * 2u];
    let s2 = sums[stat_idx * 2u + 1u];
    let count = f32(p.cpg * p.spatial);
    d_input[idx] = inv_std * (dx_hat - (s1 + x_hat * s2) / count);
}
";

const SHADER_GN_BACK_DAFFINE: &str = "
struct P {
    batch: u32, channels: u32, spatial: u32, groups: u32,
    cpg: u32, eps: f32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_gamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_beta: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if ch >= p.channels { return; }
    let g = ch / p.cpg;

    var dg: f32 = 0.0;
    var db: f32 = 0.0;
    for (var n: u32 = 0u; n < p.batch; n++) {
        let stat_idx = n * p.groups + g;
        let mean = stats[stat_idx * 2u];
        let inv_std = stats[stat_idx * 2u + 1u];
        let base = n * (p.channels * p.spatial) + ch * p.spatial;
        for (var s: u32 = 0u; s < p.spatial; s++) {
            let x_hat = (input[base + s] - mean) * inv_std;
            dg += grad_out[base + s] * x_hat;
            db += grad_out[base + s];
        }
    }
    d_gamma[ch] = dg;
    d_beta[ch] = db;
}
";

impl GpuDevice {
    /// Group normalization: input[N,C,*spatial] with C/groups groups.
    /// gamma[C] and beta[C] are learnable affine params.
    pub fn group_norm(
        &self,
        input: &GpuBuffer,
        gamma: &GpuBuffer,
        beta: &GpuBuffer,
        batch: u32, channels: u32, spatial: u32, groups: u32,
        eps: f32,
    ) -> Result<GpuBuffer> {
        ensure!(input.len == (batch * channels * spatial) as usize);
        ensure!(gamma.len == channels as usize);
        ensure!(beta.len == channels as usize);
        ensure!(channels % groups == 0, "channels must be divisible by groups");

        let cpg = channels / groups;
        let params = GNStatsParams { batch, channels, spatial, groups, channels_per_group: cpg, eps, _pad: [0; 2] };

        // Pass 1: compute per-group mean and inv_std
        let stats = self.alloc((batch * groups * 2) as usize);
        self.dispatch_shader(
            SHADER_GN_STATS, Some("gn_stats"),
            &params, &[input], &stats,
            super::dispatch_1d(batch * groups),
        );

        // Pass 2: normalize + affine
        let total = batch * channels * spatial;
        let out = self.alloc(total as usize);

        // For pass 2 we need 5 storage bindings + params. Use raw dispatch.
        let params_buf = self.upload_uniform(&params);
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gn_norm"),
            source: wgpu::ShaderSource::Wgsl(SHADER_GN_NORM.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gn_norm"),
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
                wgpu::BindGroupEntry { binding: 3, resource: gamma.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: beta.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out.buffer.as_entire_binding() },
            ],
        });
        let (wx, wy, wz) = super::dispatch_1d(total);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gn_norm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wx, wy, wz);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(out)
    }

    /// Backward pass for group_norm. Returns (d_input, d_gamma, d_beta).
    /// Recomputes stats from input internally to avoid storing them on the tape.
    pub fn group_norm_backward(
        &self,
        grad_out: &GpuBuffer,
        input: &GpuBuffer,
        gamma: &GpuBuffer,
        batch: u32, channels: u32, spatial: u32, groups: u32, eps: f32,
    ) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer)> {
        ensure!(input.len == (batch * channels * spatial) as usize);
        ensure!(grad_out.len == (batch * channels * spatial) as usize);
        ensure!(gamma.len == channels as usize);
        ensure!(channels % groups == 0);

        let cpg = channels / groups;
        let params = GNStatsParams {
            batch, channels, spatial, groups,
            channels_per_group: cpg, eps, _pad: [0; 2],
        };

        // Pass A: recompute stats per (n, g) (reuse forward stats shader).
        let stats = self.alloc((batch * groups * 2) as usize);
        self.dispatch_shader(
            SHADER_GN_STATS, Some("gn_back_stats"),
            &params, &[input], &stats,
            super::dispatch_1d(batch * groups),
        );

        // Pass B: compute reduction sums (s1, s2) per (n, g). 5 storage bindings → raw dispatch.
        let sums = self.alloc((batch * groups * 2) as usize);
        {
            let params_buf = self.upload_uniform(&params);
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gn_back_sums"),
                source: wgpu::ShaderSource::Wgsl(SHADER_GN_BACK_SUMS.into()),
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gn_back_sums"),
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
                    wgpu::BindGroupEntry { binding: 2, resource: grad_out.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: gamma.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: stats.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: sums.buffer.as_entire_binding() },
                ],
            });
            let (wx, wy, wz) = super::dispatch_1d(batch * groups);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gn_back_sums"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wx, wy, wz);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        // Pass C: compute d_input per element. 6 storage bindings → raw dispatch.
        let total = batch * channels * spatial;
        let d_input = self.alloc(total as usize);
        {
            let params_buf = self.upload_uniform(&params);
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gn_back_dinput"),
                source: wgpu::ShaderSource::Wgsl(SHADER_GN_BACK_DINPUT.into()),
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gn_back_dinput"),
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
                    wgpu::BindGroupEntry { binding: 2, resource: grad_out.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: gamma.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: stats.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: sums.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: d_input.buffer.as_entire_binding() },
                ],
            });
            let (wx, wy, wz) = super::dispatch_1d(total);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gn_back_dinput"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wx, wy, wz);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        // Pass D: compute d_gamma and d_beta per channel. 5 storage bindings → raw dispatch.
        let d_gamma = self.alloc(channels as usize);
        let d_beta = self.alloc(channels as usize);
        {
            let params_buf = self.upload_uniform(&params);
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gn_back_daffine"),
                source: wgpu::ShaderSource::Wgsl(SHADER_GN_BACK_DAFFINE.into()),
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gn_back_daffine"),
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
                    wgpu::BindGroupEntry { binding: 2, resource: grad_out.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: stats.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: d_gamma.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: d_beta.buffer.as_entire_binding() },
                ],
            });
            let (wx, wy, wz) = super::dispatch_1d(channels);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gn_back_daffine"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wx, wy, wz);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        Ok((d_input, d_gamma, d_beta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::assert_approx;
    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    // CPU reference group_norm
    fn cpu_group_norm(
        input: &[f32], gamma: &[f32], beta: &[f32],
        batch: usize, channels: usize, spatial: usize, groups: usize, eps: f32,
    ) -> Vec<f32> {
        let cpg = channels / groups;
        let mut out = vec![0.0f32; input.len()];
        for n in 0..batch {
            for g in 0..groups {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                let count = (cpg * spatial) as f32;
                for c in 0..cpg {
                    let ch = g * cpg + c;
                    for s in 0..spatial {
                        let v = input[n * channels * spatial + ch * spatial + s];
                        sum += v;
                        sum_sq += v * v;
                    }
                }
                let mean = sum / count;
                let var = sum_sq / count - mean * mean;
                let inv_std = 1.0 / (var + eps).sqrt();
                for c in 0..cpg {
                    let ch = g * cpg + c;
                    for s in 0..spatial {
                        let idx = n * channels * spatial + ch * spatial + s;
                        out[idx] = (input[idx] - mean) * inv_std * gamma[ch] + beta[ch];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_group_norm_per_channel() {
        // groups=channels: each channel is its own group
        let input = dev().upload(&[1.0, 3.0, 2.0, 4.0]);
        let gamma = dev().upload(&[1.0, 1.0]);
        let beta = dev().upload(&[0.0, 0.0]);
        let result = dev().read(&dev().group_norm(&input, &gamma, &beta, 1, 2, 2, 2, 1e-5).unwrap()).unwrap();
        assert_approx(&result, &[-1.0, 1.0, -1.0, 1.0], 1e-3);
    }

    #[test]
    fn test_group_norm_single_group() {
        // groups=1: all channels normalized together
        // 1 batch, 4 channels, 1 spatial -> normalize all 4 values as one group
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = cpu_group_norm(&data, &gamma, &beta, 1, 4, 1, 1, 1e-5);
        let result = dev().read(&dev().group_norm(
            &dev().upload(&data), &dev().upload(&gamma), &dev().upload(&beta),
            1, 4, 1, 1, 1e-5
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-3);
    }

    #[test]
    fn test_group_norm_with_affine() {
        let input = dev().upload(&[1.0, 3.0, 2.0, 4.0]);
        let gamma = dev().upload(&[2.0, 0.5]);
        let beta = dev().upload(&[1.0, -1.0]);
        let result = dev().read(&dev().group_norm(&input, &gamma, &beta, 1, 2, 2, 2, 1e-5).unwrap()).unwrap();
        assert_approx(&result, &[-1.0, 3.0, -1.5, -0.5], 1e-3);
    }

    #[test]
    fn test_group_norm_batched_vs_cpu() {
        // batch=2, channels=4, spatial=3, groups=2
        let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let gamma = vec![1.0, 2.0, 0.5, 1.5];
        let beta = vec![0.0, 1.0, -1.0, 0.5];
        let expected = cpu_group_norm(&data, &gamma, &beta, 2, 4, 3, 2, 1e-5);
        let result = dev().read(&dev().group_norm(
            &dev().upload(&data), &dev().upload(&gamma), &dev().upload(&beta),
            2, 4, 3, 2, 1e-5
        ).unwrap()).unwrap();
        assert_approx(&result, &expected, 1e-3);
    }

    #[test]
    fn test_group_norm_constant_input() {
        // All same values -> normalized to 0 (var=0, eps prevents div by zero)
        let data = vec![5.0; 8]; // 1 batch, 2 channels, 4 spatial, 2 groups
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let result = dev().read(&dev().group_norm(
            &dev().upload(&data), &dev().upload(&gamma), &dev().upload(&beta),
            1, 2, 4, 2, 1e-5
        ).unwrap()).unwrap();
        assert_approx(&result, &[0.0; 8], 1e-3);
    }

    #[test]
    fn test_group_norm_channels_not_divisible() {
        let data = vec![1.0; 5]; // 5 channels, groups=2 -> not divisible
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        assert!(dev().group_norm(&dev().upload(&data), &dev().upload(&gamma), &dev().upload(&beta), 1, 5, 1, 2, 1e-5).is_err());
    }

    #[test]
    fn test_group_norm_input_size_mismatch() {
        let data = vec![1.0; 10]; // 10 elements but batch*channels*spatial = 1*4*4 = 16
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        assert!(dev().group_norm(&dev().upload(&data), &dev().upload(&gamma), &dev().upload(&beta), 1, 4, 4, 2, 1e-5).is_err());
    }
}
