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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::assert_approx;
    use std::sync::LazyLock;

    static DEV: LazyLock<GpuDevice> = LazyLock::new(|| GpuDevice::gpu().expect("need a GPU"));

    #[test]
    fn test_group_norm_basic() {
        // batch=1, channels=2, spatial=2, groups=2 (each channel is its own group)
        // Input: channel 0 = [1, 3], channel 1 = [2, 4]
        let input = DEV.upload(&[1.0, 3.0, 2.0, 4.0]);
        let gamma = DEV.upload(&[1.0, 1.0]);
        let beta = DEV.upload(&[0.0, 0.0]);
        let out = DEV.group_norm(&input, &gamma, &beta, 1, 2, 2, 2, 1e-5).unwrap();
        let result = DEV.read(&out).unwrap();
        // Channel 0: mean=2, var=1 -> [-1, 1]
        // Channel 1: mean=3, var=1 -> [-1, 1]
        assert_approx(&result, &[-1.0, 1.0, -1.0, 1.0], 1e-3);
    }

    #[test]
    fn test_group_norm_with_affine() {
        let input = DEV.upload(&[1.0, 3.0, 2.0, 4.0]);
        let gamma = DEV.upload(&[2.0, 0.5]);
        let beta = DEV.upload(&[1.0, -1.0]);
        let out = DEV.group_norm(&input, &gamma, &beta, 1, 2, 2, 2, 1e-5).unwrap();
        let result = DEV.read(&out).unwrap();
        // Ch0: norm=[-1,1] * 2 + 1 = [-1, 3]
        // Ch1: norm=[-1,1] * 0.5 + (-1) = [-1.5, -0.5]
        assert_approx(&result, &[-1.0, 3.0, -1.5, -0.5], 1e-3);
    }
}
