// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

/// GPU device handle. Holds wgpu device + queue for compute dispatch.
pub struct VulkanDevice {
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

impl VulkanDevice {
    /// Discover the best Vulkan-capable GPU and initialize it.
    /// Prefers discrete GPUs over integrated.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Result<Self> {
        // Prefer Vulkan, fall back to any available backend (Metal on macOS, DX12 on Windows)
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
        if adapters.is_empty() {
            anyhow::bail!("no Vulkan-capable GPU found");
        }

        // Print discovered GPUs
        for (i, adapter) in adapters.iter().enumerate() {
            let info = adapter.get_info();
            eprintln!(
                "  [{}] {} ({:?}, {:?})",
                i, info.name, info.device_type, info.backend
            );
        }

        // Prefer discrete GPU, fall back to whatever is available
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("failed to find a suitable GPU adapter")?;

        let info = adapter.get_info();
        eprintln!("  selected: {} ({:?})", info.name, info.device_type);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("candle-vulkan"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
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
            label: Some("staging"),
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
