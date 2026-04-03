// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

// --- WGSL Shaders ---
// All elementwise ops pass element count via uniform instead of arrayLength().
// arrayLength() generates OpArrayLength SPIR-V which crashes some AMD RADV drivers.

const SHADER_ADD: &str = "
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.n { return; }
    out[gid.x] = a[gid.x] + b[gid.x];
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
    if gid.x >= params.n { return; }
    out[gid.x] = a[gid.x] * b[gid.x];
}
";

const SHADER_MATMUL: &str = "
struct Dims { m: u32, n: u32, k: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if row >= dims.m || col >= dims.n { return; }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        sum = sum + a[row * dims.k + i] * b[i * dims.n + col];
    }
    out[row * dims.n + col] = sum;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ElemParams {
    n: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulDims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

impl GpuDevice {
    /// Element-wise add: out[i] = a[i] + b[i]
    pub fn add(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        ensure!(a.len == b.len, "add: buffer length mismatch ({} vs {})", a.len, b.len);
        self.elementwise_op(SHADER_ADD, a, b)
    }

    /// Element-wise mul: out[i] = a[i] * b[i]
    pub fn mul(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        ensure!(a.len == b.len, "mul: buffer length mismatch ({} vs {})", a.len, b.len);
        self.elementwise_op(SHADER_MUL, a, b)
    }

    /// Matrix multiply: A(m,k) x B(k,n) = C(m,n). Row-major layout.
    pub fn matmul(
        &self,
        a: &GpuBuffer,
        b: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer> {
        ensure!(a.len == (m * k) as usize, "matmul: A has {} elems, expected {}", a.len, m * k);
        ensure!(b.len == (k * n) as usize, "matmul: B has {} elems, expected {}", b.len, k * n);

        let out = self.alloc((m * n) as usize);
        let dims_buf = self.upload_uniform(&MatmulDims { m, n, k, _pad: 0 });

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(SHADER_MATMUL.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
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
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(m.div_ceil(16), n.div_ceil(16), 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(out)
    }

    fn elementwise_op(&self, shader_src: &str, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        let out = self.alloc(a.len);
        let params_buf = self.upload_uniform(&ElemParams {
            n: a.len as u32,
            _pad: [0; 3],
        });

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
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
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((a.len as u32).div_ceil(256), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let dev = GpuDevice::gpu().expect("need a GPU");
        let a = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = dev.upload(&[10.0, 20.0, 30.0, 40.0]);
        let c = dev.add(&a, &b).unwrap();
        let result = dev.read(&c).unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_mul() {
        let dev = GpuDevice::gpu().expect("need a GPU");
        let a = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = dev.upload(&[10.0, 20.0, 30.0, 40.0]);
        let c = dev.mul(&a, &b).unwrap();
        let result = dev.read(&c).unwrap();
        assert_eq!(result, vec![10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn test_matmul_2x2() {
        let dev = GpuDevice::gpu().expect("need a GPU");
        // [1 2] x [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        let a = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = dev.upload(&[5.0, 6.0, 7.0, 8.0]);
        let c = dev.matmul(&a, &b, 2, 2, 2).unwrap();
        let result = dev.read(&c).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
