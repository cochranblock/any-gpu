// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher, DefaultHasher};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// GPU device handle. wgpu picks the right backend — Vulkan, Metal, DX12.
/// One codepath, every vendor.
pub struct GpuDevice {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
    /// Compiled pipeline cache. Key = hash of WGSL source. Eliminates per-dispatch recompilation.
    pipeline_cache: Mutex<HashMap<u64, Arc<wgpu::ComputePipeline>>>,
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
            pipeline_cache: Mutex::new(HashMap::new()),
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

    /// Get or create a compiled compute pipeline for the given WGSL source.
    /// First call compiles; subsequent calls return the cached Arc. Thread-safe.
    pub(crate) fn pipeline(&self, shader_src: &str, label: Option<&str>) -> Arc<wgpu::ComputePipeline> {
        let mut h = DefaultHasher::new();
        shader_src.hash(&mut h);
        let key = h.finish();

        let mut cache = self.pipeline_cache.lock().unwrap();
        if let Some(p) = cache.get(&key) {
            return Arc::clone(p);
        }

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        cache.insert(key, Arc::clone(&pipeline));
        pipeline
    }

    /// Number of pipelines currently in the cache. For testing only.
    #[cfg(test)]
    pub(crate) fn pipeline_cache_len(&self) -> usize {
        self.pipeline_cache.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    #[test]
    fn test_gpu_init() {
        let d = dev();
        assert!(!d.adapter_name.is_empty(), "adapter_name should be populated");
        assert!(!d.backend.is_empty(), "backend should be populated");
    }

    #[test]
    fn test_upload_read_roundtrip() {
        let data = vec![1.0f32, 2.5, -3.7, 0.0, f32::MIN_POSITIVE, 999.999];
        let buf = dev().upload(&data);
        assert_eq!(buf.len, data.len());
        let result = dev().read(&buf).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_upload_odd_length() {
        // 13 elements — not aligned to any power of 2
        let data: Vec<f32> = (0..13).map(|i| i as f32 * 0.1).collect();
        let buf = dev().upload(&data);
        assert_eq!(buf.len, 13);
        let result = dev().read(&buf).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_upload_single_element() {
        let buf = dev().upload(&[42.0]);
        assert_eq!(dev().read(&buf).unwrap(), vec![42.0]);
    }

    #[test]
    fn test_alloc_size() {
        let buf = dev().alloc(100);
        assert_eq!(buf.len, 100);
        assert_eq!(buf.size, 400); // 100 * 4 bytes
    }

    #[test]
    fn test_alloc_buffers_independent() {
        // Two allocations should not share data
        let a = dev().upload(&[1.0, 2.0, 3.0]);
        let b = dev().upload(&[10.0, 20.0, 30.0]);
        assert_eq!(dev().read(&a).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(dev().read(&b).unwrap(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_pipeline_cache_same_shader_returns_same_arc() {
        // Two calls with identical shader source must return the same compiled pipeline.
        const SRC: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= p.n { return; }
    out[gid.x] = a[gid.x];
}";
        let p1 = dev().pipeline(SRC, None);
        let p2 = dev().pipeline(SRC, None);
        assert!(Arc::ptr_eq(&p1, &p2), "same shader src must return the same Arc");
    }

    #[test]
    fn test_pipeline_cache_different_shaders_different_arcs() {
        const SRC_A: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= p.n { return; }
    out[gid.x] = a[gid.x] + 1.0;
}";
        const SRC_B: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= p.n { return; }
    out[gid.x] = a[gid.x] + 2.0;
}";
        let pa = dev().pipeline(SRC_A, None);
        let pb = dev().pipeline(SRC_B, None);
        assert!(!Arc::ptr_eq(&pa, &pb), "different shaders must produce different pipeline entries");
    }

    #[test]
    fn test_pipeline_cache_grows_then_stabilizes() {
        // Cache starts with some entries from prior tests. Calling the same shader
        // N times must not grow the cache past the first insertion.
        const SRC: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= p.n { return; }
    out[gid.x] = a[gid.x] * 3.0;
}";
        // Warm the cache for this shader.
        dev().pipeline(SRC, None);
        let len_after_first = dev().pipeline_cache_len();
        // Call 9 more times — cache must not grow.
        for _ in 0..9 {
            dev().pipeline(SRC, None);
        }
        assert_eq!(dev().pipeline_cache_len(), len_after_first,
            "repeated calls with same shader must not grow the cache");
    }

    #[test]
    fn test_pipeline_cache_correctness_after_caching() {
        // Verify that an op produces correct results on the 2nd+ call (uses cached pipeline).
        let a = dev().upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = dev().upload(&[10.0, 20.0, 30.0, 40.0]);
        // Run add twice — second call hits pipeline cache.
        let r1 = dev().add(&a, &b).unwrap();
        let r2 = dev().add(&a, &b).unwrap();
        let v1 = dev().read(&r1).unwrap();
        let v2 = dev().read(&r2).unwrap();
        assert_eq!(v1, v2);
        assert_eq!(v1, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_read_preserves_precision() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.001 + 0.0001).collect();
        let buf = dev().upload(&data);
        let result = dev().read(&buf).unwrap();
        for (i, (g, e)) in result.iter().zip(data.iter()).enumerate() {
            assert!((g - e).abs() < 1e-7, "index {i}: got {g}, expected {e}");
        }
    }
}
