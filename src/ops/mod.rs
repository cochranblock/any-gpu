// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6

mod elementwise;
mod conv;
mod norm;
mod tensor_ops;
mod upsample;
mod attention;

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::Result;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ElemParams {
    pub n: u32,
    pub _pad: [u32; 3],
}

/// Dispatch helper: handles >65535 workgroups by spilling into gid.y.
pub(crate) fn dispatch_1d(total: u32) -> (u32, u32, u32) {
    let wgs = total.div_ceil(256);
    if wgs <= 65535 {
        (wgs, 1, 1)
    } else {
        (65535, wgs.div_ceil(65535), 1)
    }
}

impl GpuDevice {
    /// Run a unary element-wise shader (one input, one output).
    pub(crate) fn unary_op(&self, shader_src: &str, a: &GpuBuffer) -> Result<GpuBuffer> {
        let out = self.alloc(a.len);
        let params_buf = self.upload_uniform(&ElemParams {
            n: a.len as u32,
            _pad: [0; 3],
        });

        let pipeline = self.pipeline(shader_src, None);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out.buffer.as_entire_binding() },
            ],
        });

        let (wx, wy, wz) = dispatch_1d(a.len as u32);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wx, wy, wz);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(out)
    }

    /// Run a binary element-wise shader (two inputs, one output).
    pub(crate) fn binary_op(&self, shader_src: &str, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        let out = self.alloc(a.len);
        let params_buf = self.upload_uniform(&ElemParams {
            n: a.len as u32,
            _pad: [0; 3],
        });

        let pipeline = self.pipeline(shader_src, None);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.buffer.as_entire_binding() },
            ],
        });

        let (wx, wy, wz) = dispatch_1d(a.len as u32);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wx, wy, wz);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(out)
    }

    /// Generic dispatch: custom params struct, arbitrary bindings, custom workgroup counts.
    pub(crate) fn dispatch_shader<P: bytemuck::Pod>(
        &self,
        shader_src: &str,
        label: Option<&str>,
        params: &P,
        storage_bufs: &[&GpuBuffer],
        out: &GpuBuffer,
        workgroups: (u32, u32, u32),
    ) {
        let params_buf = self.upload_uniform(params);

        let pipeline = self.pipeline(shader_src, label);

        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
        ];
        for (i, buf) in storage_bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: buf.buffer.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: (storage_bufs.len() + 1) as u32,
            resource: out.buffer.as_entire_binding(),
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
pub(crate) static TEST_DEV: std::sync::LazyLock<crate::GpuDevice> =
    std::sync::LazyLock::new(|| crate::GpuDevice::gpu().expect("need a GPU"));

#[cfg(test)]
pub(crate) fn assert_approx(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch: got {} want {}", got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < tol,
            "index {i}: got {g}, want {w} (diff {})",
            (g - w).abs()
        );
    }
}
