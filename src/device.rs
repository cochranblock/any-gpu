// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Token-Optimized Code Representation per docs/compression_map.md.
// t500 = GpuDevice. t501 = GpuBuffer. f500..f508 device methods.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher, DefaultHasher};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// t500 = GpuDevice. wgpu picks the right backend — Vulkan, Metal, DX12.
/// One codepath, every vendor.
pub struct t500 {
    /// s500 = device. The wgpu logical device used to dispatch shaders.
    pub(crate) s500: wgpu::Device,
    /// s501 = queue. Submits encoded command buffers to the GPU.
    pub(crate) s501: wgpu::Queue,
    /// s502 = adapter_name. Human-readable adapter (e.g. "AMD Radeon RX 5700 XT").
    pub s502: String,
    /// s503 = backend. Backend tag ("Vulkan", "Metal", "Dx12", ...).
    pub s503: String,
    /// s504 = pipeline_cache. Compiled pipeline cache. Key = hash of WGSL source.
    /// Eliminates per-dispatch recompilation.
    s504: Mutex<HashMap<u64, Arc<wgpu::ComputePipeline>>>,
}

/// t501 = GpuBuffer. GPU-resident f32 buffer with element-count metadata.
pub struct t501 {
    /// s505 = buffer. The underlying wgpu storage buffer.
    pub(crate) s505: wgpu::Buffer,
    /// s506 = size. Byte length of the buffer.
    pub(crate) s506: u64,
    /// s507 = len. Number of f32 elements.
    pub s507: usize,
}

impl t500 {
    /// f500 = gpu. Discover the best GPU and initialize it. wgpu auto-selects
    /// the backend: Vulkan on Linux (AMD/NVIDIA/Intel), Metal on macOS, DX12 on
    /// Windows. Desktop-only blocking entry point. Browser callers must use f501.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn f500() -> Result<Self> {
        pollster::block_on(Self::f501())
    }

    /// f501 = gpu_async. Async device init — works on desktop and in the browser.
    /// In a browser, wgpu picks WebGPU; on native it picks Vulkan/Metal/DX12.
    /// Falls back to a software adapter (SwiftShader on web, llvmpipe on
    /// native) when no real GPU is available — important for headless CI
    /// runs and for users on machines without modern GPU drivers.
    pub async fn f501() -> Result<Self> {
        let v0 = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // First try: prefer real high-performance hardware.
        let v1 = match v0
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(a) => a,
            None => {
                // Second try: any low-power adapter.
                match v0
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                {
                    Some(a) => a,
                    None => v0
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            // Last resort: explicit software fallback.
                            power_preference: wgpu::PowerPreference::LowPower,
                            compatible_surface: None,
                            force_fallback_adapter: true,
                        })
                        .await
                        .context("no GPU found (tried hardware, low-power, and software fallback)")?,
                }
            }
        };

        let v2 = v1.get_info();
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!("  any-gpu: {} ({:?}, {:?})", v2.name, v2.device_type, v2.backend);

        // Use the adapter's actual limits — not Limits::default() which can
        // request capabilities the driver doesn't support (SIGSEGV on RADV/RDNA1).
        let (v3, v4) = v1
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("any-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: v1.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("failed to create GPU device")?;

        Ok(Self {
            s500: v3,
            s501: v4,
            s502: v2.name.clone(),
            s503: format!("{:?}", v2.backend),
            s504: Mutex::new(HashMap::new()),
        })
    }

    /// f502 = upload. Upload f32 slice to GPU. Returns a storage buffer usable
    /// in compute shaders.
    pub fn f502(&self, p0: &[f32]) -> t501 {
        let v0 = bytemuck::cast_slice(p0);
        let v1 = self.s500.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: v0,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        t501 {
            s506: v0.len() as u64,
            s507: p0.len(),
            s505: v1,
        }
    }

    /// f503 = alloc. Allocate an empty GPU buffer for `p0` f32 elements.
    pub fn f503(&self, p0: usize) -> t501 {
        let v0 = (p0 * std::mem::size_of::<f32>()) as u64;
        let v1 = self.s500.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: v0,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        t501 {
            s506: v0,
            s507: p0,
            s505: v1,
        }
    }

    /// f504 = read. Read GPU buffer back to CPU as f32 vec. Desktop-only blocking
    /// version; browser callers must use f505.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn f504(&self, p0: &t501) -> Result<Vec<f32>> {
        let v0 = self.s500.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: p0.s506,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut v1 = self
            .s500
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        v1.copy_buffer_to_buffer(&p0.s505, 0, &v0, 0, p0.s506);
        self.s501.submit(Some(v1.finish()));

        let v2 = v0.slice(..);
        let (v3, v4) = std::sync::mpsc::channel();
        v2.map_async(wgpu::MapMode::Read, move |result| {
            let _ = v3.send(result);
        });
        self.s500.poll(wgpu::Maintain::Wait);
        v4.recv()
            .context("channel closed")?
            .context("buffer map failed")?;

        let v5 = v2.get_mapped_range();
        let v6: Vec<f32> = bytemuck::cast_slice(&v5).to_vec();
        drop(v5);
        v0.unmap();

        Ok(v6)
    }

    /// f505 = read_async. Async readback — works on desktop and in the browser.
    /// On native we still pump the device with `Maintain::Wait` so the future
    /// resolves without an external runtime. On wasm the browser's GPU runtime
    /// drives the mapping completion via the JS event loop.
    pub async fn f505(&self, p0: &t501) -> Result<Vec<f32>> {
        let v0 = self.s500.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: p0.s506,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut v1 = self
            .s500
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        v1.copy_buffer_to_buffer(&p0.s505, 0, &v0, 0, p0.s506);
        self.s501.submit(Some(v1.finish()));

        let v2 = v0.slice(..);
        let (v3, v4) = futures_channel::oneshot::channel();
        v2.map_async(wgpu::MapMode::Read, move |result| {
            let _ = v3.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.s500.poll(wgpu::Maintain::Wait);

        v4.await
            .ok()
            .context("channel closed")?
            .context("buffer map failed")?;

        let v5 = v2.get_mapped_range();
        let v6: Vec<f32> = bytemuck::cast_slice(&v5).to_vec();
        drop(v5);
        v0.unmap();

        Ok(v6)
    }

    /// f506 = upload_uniform. Create a small uniform buffer from a bytemuck-able struct.
    pub(crate) fn f506<T: bytemuck::Pod>(&self, p0: &T) -> wgpu::Buffer {
        self.s500
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(p0),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// f507 = pipeline. Get or create a compiled compute pipeline for the given
    /// WGSL source. First call compiles; subsequent calls return the cached Arc.
    /// Thread-safe.
    pub(crate) fn f507(&self, p0: &str, p1: Option<&str>) -> Arc<wgpu::ComputePipeline> {
        let mut v0 = DefaultHasher::new();
        p0.hash(&mut v0);
        let v1 = v0.finish();

        let mut v2 = self.s504.lock().unwrap();
        if let Some(v3) = v2.get(&v1) {
            return Arc::clone(v3);
        }

        let v4 = self.s500.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: p1,
            source: wgpu::ShaderSource::Wgsl(p0.into()),
        });
        let v5 = Arc::new(self.s500.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: p1,
            layout: None,
            module: &v4,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        v2.insert(v1, Arc::clone(&v5));
        v5
    }

    /// f508 = pipeline_cache_len. Number of pipelines currently in the cache. For testing only.
    #[cfg(test)]
    pub(crate) fn f508(&self) -> usize {
        self.s504.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    #[test]
    fn f500_init() {
        let v0 = dev();
        assert!(!v0.s502.is_empty(), "adapter_name should be populated");
        assert!(!v0.s503.is_empty(), "backend should be populated");
    }

    #[test]
    fn f502_read_roundtrip() {
        let v0 = vec![1.0f32, 2.5, -3.7, 0.0, f32::MIN_POSITIVE, 999.999];
        let v1 = dev().f502(&v0);
        assert_eq!(v1.s507, v0.len());
        let v2 = dev().f504(&v1).unwrap();
        assert_eq!(v2, v0);
    }

    #[test]
    fn f502_odd_length() {
        // 13 elements — not aligned to any power of 2
        let v0: Vec<f32> = (0..13).map(|i| i as f32 * 0.1).collect();
        let v1 = dev().f502(&v0);
        assert_eq!(v1.s507, 13);
        let v2 = dev().f504(&v1).unwrap();
        assert_eq!(v2, v0);
    }

    #[test]
    fn f502_single_element() {
        let v0 = dev().f502(&[42.0]);
        assert_eq!(dev().f504(&v0).unwrap(), vec![42.0]);
    }

    #[test]
    fn f503_size() {
        let v0 = dev().f503(100);
        assert_eq!(v0.s507, 100);
        assert_eq!(v0.s506, 400); // 100 * 4 bytes
    }

    #[test]
    fn f503_buffers_independent() {
        // Two allocations should not share data
        let v0 = dev().f502(&[1.0, 2.0, 3.0]);
        let v1 = dev().f502(&[10.0, 20.0, 30.0]);
        assert_eq!(dev().f504(&v0).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(dev().f504(&v1).unwrap(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn f507_same_shader_returns_same_arc() {
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
        let v0 = dev().f507(SRC, None);
        let v1 = dev().f507(SRC, None);
        assert!(Arc::ptr_eq(&v0, &v1), "same shader src must return the same Arc");
    }

    #[test]
    fn f507_different_shaders_different_arcs() {
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
        let v0 = dev().f507(SRC_A, None);
        let v1 = dev().f507(SRC_B, None);
        assert!(!Arc::ptr_eq(&v0, &v1), "different shaders must produce different pipeline entries");
    }

    #[test]
    fn f507_cache_grows_then_stabilizes() {
        // Same source string must return the same Arc on every call — proves the
        // pipeline is cached and not recompiled. Arc::ptr_eq is robust to parallel
        // tests adding other shaders to the shared cache concurrently.
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
        let v0 = dev().f507(SRC, None);
        let v1 = dev().f507(SRC, None);
        assert!(Arc::ptr_eq(&v0, &v1),
            "same shader source must return the same cached Arc<ComputePipeline>");
    }

    #[test]
    fn f507_correctness_after_caching() {
        // Verify that an op produces correct results on the 2nd+ call (uses cached pipeline).
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0]);
        let v1 = dev().f502(&[10.0, 20.0, 30.0, 40.0]);
        // Run add twice — second call hits pipeline cache.
        let v2 = dev().f550(&v0, &v1).unwrap();
        let v3 = dev().f550(&v0, &v1).unwrap();
        let v4 = dev().f504(&v2).unwrap();
        let v5 = dev().f504(&v3).unwrap();
        assert_eq!(v4, v5);
        assert_eq!(v4, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn f504_preserves_precision() {
        let v0: Vec<f32> = (0..100).map(|i| (i as f32) * 0.001 + 0.0001).collect();
        let v1 = dev().f502(&v0);
        let v2 = dev().f504(&v1).unwrap();
        for (v3, (v4, v5)) in v2.iter().zip(v0.iter()).enumerate() {
            assert!((v4 - v5).abs() < 1e-7, "index {v3}: got {v4}, expected {v5}");
        }
    }
}
