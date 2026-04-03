// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

/// GPU device handle. wgpu picks the right backend — Vulkan, Metal, DX12.
/// One codepath, every vendor.
pub struct GpuDevice {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
}

/// GPU-resident f32 buffer with element count metadata.
pub struct GpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) size: u64,
    pub len: usize,
}

impl GpuDevice {
    /// Discover the best GPU and initialize it. wgpu auto-selects the backend:
    /// Vulkan on Linux (AMD/NVIDIA/Intel), Metal on macOS, DX12 on Windows.
    pub fn gpu() -> Result<Self> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Skip enumerate_adapters — can crash on Linux when probing GL/other backends.
        // Just request the best adapter directly.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("no GPU found")?;

        let info = adapter.get_info();
        eprintln!("  any-gpu: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        // Use the adapter's actual limits — not Limits::default() which can
        // request capabilities the driver doesn't support (SIGSEGV on RADV/RDNA1).
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("any-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("failed to create GPU device")?;

        Ok(Self {
            device,
            queue,
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
        })
    }

    /// Upload f32 slice to GPU. Returns a storage buffer usable in compute shaders.
    pub fn upload(&self, data: &[f32]) -> GpuBuffer {
        let bytes = bytemuck::cast_slice(data);
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        GpuBuffer {
            size: bytes.len() as u64,
            len: data.len(),
            buffer,
        }
    }

    /// Allocate an empty GPU buffer for `n` f32 elements.
    pub fn alloc(&self, n: usize) -> GpuBuffer {
        let size = (n * std::mem::size_of::<f32>()) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GpuBuffer {
            size,
            len: n,
            buffer,
        }
    }

    /// Read GPU buffer back to CPU as f32 vec.
    pub fn read(&self, buf: &GpuBuffer) -> Result<Vec<f32>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buf.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&buf.buffer, 0, &staging, 0, buf.size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .context("channel closed")?
            .context("buffer map failed")?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Create a small uniform buffer from a bytemuck-able struct.
    pub(crate) fn upload_uniform<T: bytemuck::Pod>(&self, data: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }
}
