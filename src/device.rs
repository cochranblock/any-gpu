// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Token-Optimized Code Representation per docs/compression_map.md.
// t500 = GpuDevice. t501 = GpuBuffer. t540 = GpuBufferF16. f500..f508, f771..f772.

use anyhow::{ensure, Context, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher, DefaultHasher};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// t549 = GpuBatch. Recording guard returned by f509 (begin).
/// All ops dispatched while this is live record into a single CommandEncoder.
/// Call f511 (sync) or f510 (execute) to submit; dropping auto-submits.
pub struct t549<'a>(&'a t500);

/// Internal state for batch recording. Lives inside t500.s508.
pub(crate) struct t549State {
    pub(crate) enc: wgpu::CommandEncoder,
    /// BindGroups keep their referenced uniform buffers alive until submission.
    pub(crate) keep: Vec<wgpu::BindGroup>,
}

impl<'a> t549<'a> {
    /// f510 = execute. Submit all recorded ops to the GPU without waiting.
    pub fn f510(self) {
        let dev = self.0;
        std::mem::forget(self);
        if let Some(st) = dev.s508.lock().unwrap().take() {
            dev.s501.submit(Some(st.enc.finish()));
        }
    }

    /// f511 = sync. Submit all recorded ops and block until GPU completes.
    pub fn f511(self) {
        let dev = self.0;
        std::mem::forget(self);
        if let Some(st) = dev.s508.lock().unwrap().take() {
            dev.s501.submit(Some(st.enc.finish()));
            dev.s500.poll(wgpu::Maintain::Wait);
        }
    }
}

impl Drop for t549<'_> {
    fn drop(&mut self) {
        if let Some(st) = self.0.s508.lock().unwrap().take() {
            self.0.s501.submit(Some(st.enc.finish()));
        }
    }
}

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
    /// s521 = has_f16. True when the adapter supports wgpu::Features::SHADER_F16.
    /// RDNA1+ (RX 5700 XT, RADV/Vulkan) exposes this. Required by f771 and f772.
    pub s521: bool,
    /// s509 = has_subgroup. True when the adapter supports wgpu::Features::SUBGROUP.
    /// Enables subgroupMax/subgroupAdd in WGSL shaders for O(1) intra-wavefront reductions.
    /// RDNA1+ via VK_EXT_subgroup_size_control (Vulkan 1.3 on RADV).
    pub s509: bool,
    /// s508 = batch_state. Active batch encoder, or None in eager mode.
    pub(crate) s508: Mutex<Option<t549State>>,
}

/// t501 = GpuBuffer. GPU-resident f32 buffer with element-count metadata.
#[derive(Clone)]
pub struct t501 {
    /// s505 = buffer. The underlying wgpu storage buffer.
    pub(crate) s505: wgpu::Buffer,
    /// s506 = size. Byte length of the buffer.
    pub(crate) s506: u64,
    /// s507 = len. Number of f32 elements.
    pub s507: usize,
}

/// t540 = GpuBufferF16. GPU-resident f16 buffer; element size is 2 bytes.
/// Requires device.s521 (SHADER_F16) to use in shaders. Expand to f32 via f772.
#[derive(Clone)]
pub struct t540 {
    /// s522 = buf. The underlying wgpu storage buffer (f16 bits packed as raw bytes).
    pub(crate) s522: wgpu::Buffer,
    /// s523 = size. Byte length (= s524 * 2).
    pub(crate) s523: u64,
    /// s524 = len. Number of f16 elements.
    pub s524: usize,
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
        // Opt into SHADER_F16 if the adapter supports it (RDNA1+ / RADV does).
        let v5 = v1.features();
        let has_f16 = v5.contains(wgpu::Features::SHADER_F16);
        let has_sg = v5.contains(wgpu::Features::SUBGROUP);
        let mut v6 = wgpu::Features::empty();
        if has_f16 { v6 |= wgpu::Features::SHADER_F16; }
        if has_sg  { v6 |= wgpu::Features::SUBGROUP; }

        let (v3, v4) = v1
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("any-gpu"),
                    required_features: v6,
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
            s521: has_f16,
            s509: has_sg,
            s508: Mutex::new(None),
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

    /// f509 = begin. Enter batch recording mode. All subsequent op dispatches
    /// record into a single CommandEncoder instead of submitting immediately.
    /// Call f511 (sync) or f510 (execute) on the returned t549 to flush.
    /// Dropping t549 without calling either auto-submits without polling.
    pub fn f509(&self) -> t549<'_> {
        let enc = self.s500.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("batch") });
        *self.s508.lock().unwrap() = Some(t549State { enc, keep: Vec::new() });
        t549(self)
    }

    /// f771 = upload_f16. Upload raw f16 bits (&[u16]) into a VRAM storage buffer.
    /// Two f16 elements are packed per u32 (lower-then-upper 16 bits); f772 unpacks via
    /// unpack2x16float. Odd-length slices are padded with 0x0000 (f16 +0.0).
    pub fn f771(&self, p0: &[u16]) -> t540 {
        // Pad to even length so every u32 in the buffer is fully initialised.
        let mut v0 = p0.to_vec();
        if v0.len() % 2 != 0 { v0.push(0); }
        let v1: &[u8] = bytemuck::cast_slice(&v0);
        let v2 = self.s500.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: v1,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        t540 { s522: v2, s523: v1.len() as u64, s524: p0.len() }
    }

    /// alloc_f16. Allocate a zero-filled f16 VRAM buffer for `p0` f16 elements.
    /// Buffer size = ceil(p0/2)*4 bytes (4-byte aligned, last u32 zero-padded for odd p0).
    pub(crate) fn alloc_f16(&self, p0: usize) -> t540 {
        let pairs = (p0 + 1) / 2;
        let v0 = (pairs * 4) as u64;
        let v1 = self.s500.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: v0,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-fill so the padding u16 (odd p0) is 0x0000 = f16(+0.0).
        let mut v2 = self.s500.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        v2.clear_buffer(&v1, 0, None);
        self.s501.submit(Some(v2.finish()));
        t540 { s522: v1, s523: v0, s524: p0 }
    }

    /// f772 = f16_to_f32. GPU kernel: expand a packed-f16 buffer (t540) into a new f32
    /// buffer (t501). One thread per u32 pair; uses unpack2x16float for the expansion.
    /// No special device features required — unpack2x16float is standard WGSL.
    /// Note: wgpu/Naga does not yet support `enable f16;` (WGSL extension #4384), so we
    /// use the packing/unpacking built-ins instead.
    pub fn f772(&self, p0: &t540) -> Result<t501> {
        // One thread per f16 pair (u32). Pairs index into src; base = pair*2 indexes dst.
        const SHADER: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let base = idx * 2u;
    if base >= p.n { return; }
    let v = unpack2x16float(src[idx]);
    dst[base] = v.x;
    if base + 1u < p.n { dst[base + 1u] = v.y; }
}";
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct F16P { n: u32, _p: [u32; 3] }

        let n = p0.s524 as u32;
        let n_pairs = (n + 1) / 2;
        let v0 = self.f503(p0.s524);
        let v1 = self.f506(&F16P { n, _p: [0; 3] });
        let v2 = self.f507(SHADER, Some("f16_to_f32"));
        let v3 = self.s500.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &v2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p0.s522.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v0.s505.as_entire_binding() },
            ],
        });
        let wg = n_pairs.div_ceil(256);
        let (wg_x, wg_y) = if wg <= 65535 { (wg, 1) } else { (65535, wg.div_ceil(65535)) };
        let mut v4 = self.s500.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut v5 = v4.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None });
            v5.set_pipeline(&v2);
            v5.set_bind_group(0, &v3, &[]);
            v5.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.s501.submit(Some(v4.finish()));
        Ok(v0)
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
        // s521 is bool — just verify it's accessible (true on RDNA1/Vulkan).
        let _ = v0.s521;
    }

    // f771+f772: upload f16 bits, expand to f32 via unpack2x16float kernel, verify values.
    #[test]
    fn f771_f772_roundtrip() {
        let f32_vals: [f32; 6] = [1.0, -1.0, 0.5, 2.0, 0.0, -0.25];
        let f16_bits: Vec<u16> = f32_vals.iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let buf = dev().f771(&f16_bits);
        assert_eq!(buf.s524, 6);
        assert_eq!(buf.s523, 12); // 6 elements: ceil(6/2)*4 = 12 bytes
        let expanded = dev().f772(&buf).unwrap();
        let back = dev().f504(&expanded).unwrap();
        assert_eq!(back.len(), 6);
        for (&a, &b) in f32_vals.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01, "f16 round-trip: {a} vs {b}");
        }
    }

    // f771+f772: odd element count — padding must not bleed into the result.
    #[test]
    fn f771_f772_odd_count() {
        let f32_vals: [f32; 5] = [1.0, -1.0, 0.5, 2.0, 0.0];
        let f16_bits: Vec<u16> = f32_vals.iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let buf = dev().f771(&f16_bits);
        assert_eq!(buf.s524, 5);
        assert_eq!(buf.s523, 12); // ceil(5/2)*4 = 12 bytes (one u32 has padding)
        let expanded = dev().f772(&buf).unwrap();
        let back = dev().f504(&expanded).unwrap();
        assert_eq!(back.len(), 5); // padding element must NOT appear in output
        for (&a, &b) in f32_vals.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01, "odd-n f16: {a} vs {b}");
        }
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
