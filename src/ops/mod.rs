// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7

mod elementwise;
mod conv;
mod norm;
mod tensor_ops;
mod upsample;
mod attention;
pub(crate) mod transformer;

use crate::device::{t500, t501};
use anyhow::Result;

/// t511 = ElemParams. Single-uint uniform used by unary/binary element-wise shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct t511 {
    pub n: u32,
    pub _pad: [u32; 3],
}

/// f540 = dispatch_1d. Returns the workgroup-count triple for a flat dispatch
/// of `p0` elements (workgroup size 256). Spills into gid.y when `p0/256 > 65535`.
pub(crate) fn f540(p0: u32) -> (u32, u32, u32) {
    let v0 = p0.div_ceil(256);
    if v0 <= 65535 {
        (v0, 1, 1)
    } else {
        (65535, v0.div_ceil(65535), 1)
    }
}

impl t500 {
    /// f541 = unary_op. Run a unary element-wise shader (one input, one output).
    pub(crate) fn f541(&self, p0: &str, p1: &t501) -> Result<t501> {
        let v0 = self.f503(p1.s507);
        let v1 = self.f506(&t511 {
            n: p1.s507 as u32,
            _pad: [0; 3],
        });

        let v2 = self.f507(p0, None);

        let v3 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &v2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p1.s505.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v0.s505.as_entire_binding() },
            ],
        });

        let (v4, v5, v6) = f540(p1.s507 as u32);
        let mut v7 = self.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut v8 = v7.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            v8.set_pipeline(&v2);
            v8.set_bind_group(0, &v3, &[]);
            v8.dispatch_workgroups(v4, v5, v6);
        }
        self.s501.submit(Some(v7.finish()));
        Ok(v0)
    }

    /// f542 = binary_op. Run a binary element-wise shader (two inputs, one output).
    pub(crate) fn f542(&self, p0: &str, p1: &t501, p2: &t501) -> Result<t501> {
        let v0 = self.f503(p1.s507);
        let v1 = self.f506(&t511 {
            n: p1.s507 as u32,
            _pad: [0; 3],
        });

        let v2 = self.f507(p0, None);

        let v3 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &v2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p1.s505.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2.s505.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v0.s505.as_entire_binding() },
            ],
        });

        let (v4, v5, v6) = f540(p1.s507 as u32);
        let mut v7 = self.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut v8 = v7.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            v8.set_pipeline(&v2);
            v8.set_bind_group(0, &v3, &[]);
            v8.dispatch_workgroups(v4, v5, v6);
        }
        self.s501.submit(Some(v7.finish()));
        Ok(v0)
    }

    /// f543 = dispatch_shader. Generic dispatch: custom params struct, arbitrary
    /// bindings, custom workgroup counts.
    pub(crate) fn f543<P: bytemuck::Pod>(
        &self,
        p0: &str,
        p1: Option<&str>,
        p2: &P,
        p3: &[&t501],
        p4: &t501,
        p5: (u32, u32, u32),
    ) {
        let v0 = self.f506(p2);

        let v1 = self.f507(p0, p1);

        let mut v2 = vec![
            wgpu::BindGroupEntry { binding: 0, resource: v0.as_entire_binding() },
        ];
        for (v3, v4) in p3.iter().enumerate() {
            v2.push(wgpu::BindGroupEntry {
                binding: (v3 + 1) as u32,
                resource: v4.s505.as_entire_binding(),
            });
        }
        v2.push(wgpu::BindGroupEntry {
            binding: (p3.len() + 1) as u32,
            resource: p4.s505.as_entire_binding(),
        });

        let v5 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &v1.get_bind_group_layout(0),
            entries: &v2,
        });

        let mut v6 = self.s508.lock().unwrap();
        if let Some(ref mut batch) = *v6 {
            // Batch mode: record into the active encoder, defer submission.
            {
                let mut v7 = batch.enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: p1, timestamp_writes: None,
                });
                v7.set_pipeline(&v1);
                v7.set_bind_group(0, &v5, &[]);
                v7.dispatch_workgroups(p5.0, p5.1, p5.2);
            }
            batch.keep.push(v5);
        } else {
            // Eager mode: own encoder, submit immediately.
            drop(v6);
            let mut v7 = self.s500.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut v8 = v7.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: p1, timestamp_writes: None,
                });
                v8.set_pipeline(&v1);
                v8.set_bind_group(0, &v5, &[]);
                v8.dispatch_workgroups(p5.0, p5.1, p5.2);
            }
            self.s501.submit(Some(v7.finish()));
        }
    }
}

#[cfg(test)]
pub(crate) static TEST_DEV: std::sync::LazyLock<crate::t500> =
    std::sync::LazyLock::new(|| crate::t500::f500().expect("need a GPU"));

/// f544 = assert_approx. Element-wise tolerance check for f32 vectors. Test-only.
#[cfg(test)]
pub(crate) fn f544(p0: &[f32], p1: &[f32], p2: f32) {
    assert_eq!(p0.len(), p1.len(), "length mismatch: got {} want {}", p0.len(), p1.len());
    for (v0, (v1, v2)) in p0.iter().zip(p1).enumerate() {
        assert!(
            (v1 - v2).abs() < p2,
            "index {v0}: got {v1}, want {v2} (diff {})",
            (v1 - v2).abs()
        );
    }
}
