# any-gpu

Tensor engine that runs on every GPU. AMD, NVIDIA, Intel, Apple. One codebase, one shader language, zero vendor lock-in.

wgpu picks the backend — Vulkan on Linux, Metal on macOS, DX12 on Windows. You write WGSL compute shaders once. They run everywhere.

## Why this exists

CUDA only runs on NVIDIA. Metal only runs on Apple. If you have an AMD RX 5700 XT or an Intel Arc, your options for GPU-accelerated ML in Rust are: nothing. any-gpu fills that gap.

If you have an NVIDIA GPU and want peak performance, use CUDA. If you're on macOS and want peak performance, use Metal Performance Shaders. They're faster — we measured it, the numbers are below, and we're not hiding them.

Use any-gpu when:
- Your GPU is AMD or Intel (where CUDA can't run)
- You need one binary that works on every machine
- You refuse to vendor-lock your compute pipeline
- You're building for a heterogeneous fleet (NVIDIA in the cloud, AMD on workstations, Apple on laptops)

## Current state (Sprint 1)

Three tensor ops as WGSL compute shaders:

- **matmul** — naive matrix multiply (16x16 workgroups)
- **add** — element-wise addition
- **mul** — element-wise multiplication

```rust
use any_gpu::GpuDevice;

let dev = GpuDevice::gpu()?;

let a = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
let b = dev.upload(&[5.0, 6.0, 7.0, 8.0]);
let c = dev.matmul(&a, &b, 2, 2, 2)?;

let result = dev.read(&c)?;
// [19.0, 22.0, 43.0, 50.0]
```

## Tested Hardware

All results from `cargo test --release` on 2026-04-02. 54 tests per GPU, each cross-validated against a CPU reference implementation.

| GPU | Vendor | VRAM | Driver | OS | Tests | Matmul 512x512 (ms) |
|-----|--------|------|--------|----|-------|----------------------|
| AMD Radeon RX 5700 XT | AMD (RADV NAVI10) | 8 GB | Mesa 25.0.7 | Debian 13, kernel 6.12.73 | 54/54 pass | 5.67 |
| NVIDIA GeForce RTX 3070 Laptop | NVIDIA | 8 GB | 550.163.01 | Debian 13, kernel 6.12.73 | 54/54 pass | 3.03 |
| NVIDIA GeForce RTX 3050 Ti Laptop | NVIDIA | 4 GB | 550.163.01 | Debian 13, kernel 6.12.73 | 54/54 pass | 5.61 |
| Apple M4 | Apple (Metal) | Unified | macOS 25.3.0 | macOS Tahoe | 54/54 pass | 3.36 |

**Reproduce any claim:**

```bash
# Run all 54 correctness tests (every op verified against CPU reference)
cargo test --release

# Run matmul benchmark (produces the ms and GFLOPS numbers above)
cargo run --release --example bench

# On AMD RADV, force Vulkan backend:
WGPU_BACKEND=vulkan cargo test --release
```

Test results verified at commit [`801c4de`](https://github.com/cochranblock/any-gpu/commit/801c4de). Benchmark numbers from commit [`56976a7`](https://github.com/cochranblock/any-gpu/commit/56976a7).

### Known issues

- **AMD RADV/RDNA1**: concurrent `wgpu::Instance` creation segfaults. Fixed by sharing a single `GpuDevice` via `LazyLock` ([`e124fbb`](https://github.com/cochranblock/any-gpu/commit/e124fbb)). Individual ops work fine.
- **Intel Iris Xe**: untested in isolation (wgpu prefers discrete NVIDIA when both are present).

## Benchmarks

Matmul performance across all tested GPUs. GPU numbers are compute-only (data already on device).

### 512x512 matmul

| GPU | CPU (ms) | GPU compute (ms) | GPU GFLOPS | Speedup |
|-----|----------|-------------------|------------|---------|
| AMD RX 5700 XT (Vulkan) | 180.74 | 5.67 | 47.35 | **31.9x** |
| NVIDIA RTX 3070 Laptop (Vulkan) | 103.97 | 3.03 | 88.59 | **34.3x** |
| NVIDIA RTX 3050 Ti Laptop (Vulkan) | 97.28 | 5.61 | 47.81 | **17.3x** |
| Apple M4 (Metal via wgpu) | 87.32 | 3.36 | 79.88 | **26.0x** |

### Full matrix — all sizes, all GPUs

#### AMD Radeon RX 5700 XT (Vulkan, RADV NAVI10)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|-----------------|-------------------|------------|------------|---------|
| 64x64 | 0.17 | 2.59 | 1.24 | 3.12 | 0.42 | 0.1x |
| 128x128 | 2.22 | 0.84 | 0.68 | 1.89 | 6.13 | 3.2x |
| 256x256 | 16.09 | 1.66 | 1.48 | 2.09 | 22.66 | 10.9x |
| 512x512 | 180.74 | 7.79 | 5.67 | 1.49 | 47.35 | 31.9x |
| 1024x1024 | 5641.56 | 39.99 | 31.22 | 0.38 | 68.78 | **180.7x** |

#### NVIDIA GeForce RTX 3070 Laptop (Vulkan)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|-----------------|-------------------|------------|------------|---------|
| 64x64 | 0.15 | 22.98 | 2.01 | 3.55 | 0.26 | 0.1x |
| 128x128 | 1.16 | 1.70 | 1.57 | 3.61 | 2.68 | 0.7x |
| 256x256 | 8.80 | 1.61 | 1.72 | 3.81 | 19.52 | 5.1x |
| 512x512 | 107.35 | 4.47 | 3.03 | 2.50 | 88.59 | 35.4x |
| 1024x1024 | 2142.91 | 19.69 | 14.25 | 1.00 | 150.71 | **150.4x** |

#### NVIDIA GeForce RTX 3050 Ti Laptop (Vulkan)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|-----------------|-------------------|------------|------------|---------|
| 64x64 | 0.16 | 23.21 | 1.87 | 3.28 | 0.28 | 0.1x |
| 128x128 | 1.05 | 1.35 | 1.50 | 3.98 | 2.79 | 0.7x |
| 256x256 | 8.28 | 1.34 | 1.37 | 4.05 | 24.53 | 6.1x |
| 512x512 | 97.28 | 5.92 | 5.61 | 2.76 | 47.81 | 17.3x |
| 1024x1024 | 2071.17 | 32.67 | 34.20 | 1.04 | 62.79 | **60.6x** |

#### Apple M4 (Metal via wgpu)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|-----------------|-------------------|------------|------------|---------|
| 64x64 | 0.11 | 4.53 | 2.02 | 4.62 | 0.26 | 0.1x |
| 128x128 | 1.97 | 2.04 | 2.12 | 2.13 | 1.98 | 0.9x |
| 256x256 | 17.35 | 2.18 | 1.95 | 1.93 | 17.20 | 8.9x |
| 512x512 | 87.32 | 3.79 | 3.36 | 3.07 | 79.88 | 26.0x |
| 1024x1024 | 773.88 | 23.01 | 17.55 | 2.77 | 122.37 | **44.1x** |

### Honest comparison: any-gpu vs CUDA and Metal

We benchmarked candle (v0.10.2) with cuBLAS on CUDA and Metal Performance Shaders on the same hardware. CUDA and MPS are faster. Here are the numbers.

#### NVIDIA RTX 3070 Laptop — Vulkan vs CUDA

| Size | any-gpu Vulkan (ms) | candle CUDA (ms) | CUDA faster by |
|------|---------------------|-------------------|----------------|
| 128x128 | 1.57 | 0.07 | 22x |
| 256x256 | 1.72 | 0.20 | 9x |
| 512x512 | 3.03 | 0.75 | 4x |
| 1024x1024 | 14.25 | 2.80 | 5x |

#### NVIDIA RTX 3050 Ti Laptop — Vulkan vs CUDA

| Size | any-gpu Vulkan (ms) | candle CUDA (ms) | CUDA faster by |
|------|---------------------|-------------------|----------------|
| 128x128 | 1.50 | 0.03 | 50x |
| 256x256 | 1.37 | 0.07 | 20x |
| 512x512 | 5.61 | 0.33 | 17x |
| 1024x1024 | 34.20 | 1.43 | 24x |

#### Apple M4 — wgpu Metal vs candle MPS

| Size | any-gpu Metal (ms) | candle MPS (ms) | MPS faster by |
|------|---------------------|-----------------|---------------|
| 128x128 | 2.12 | 0.36 | 6x |
| 256x256 | 1.95 | 0.31 | 6x |
| 512x512 | 3.36 | 0.47 | 7x |
| 1024x1024 | 17.55 | 1.94 | 9x |

#### What to make of this

CUDA and MPS are faster because cuBLAS and Metal Performance Shaders use tiled matmul with shared memory, register blocking, and vendor-tuned kernels. any-gpu uses a naive triple-loop WGSL shader.

**That's not the point.** The point is:

- The AMD RX 5700 XT has zero CUDA support and zero MPS support. any-gpu is the only option that gives it GPU compute for ML in Rust.
- Intel Arc and Iris Xe — same story.
- One `cargo build` produces a binary that runs on all four GPUs above. No feature flags, no conditional compilation, no vendor SDKs.

The performance gap closes with better shaders, not more backends:

1. **Tiled matmul** with workgroup shared memory — expected 5-10x gain
2. **Subgroup operations** for warp-level reduction
3. **Pipeline caching** to eliminate per-dispatch compilation cost

The RTX 3070 is 4x behind CUDA at 512x512. Tiling alone should close most of that. The goal isn't to beat cuBLAS — it's to be fast enough that vendor lock-in isn't worth it.

### How these numbers were produced

```bash
# any-gpu benchmarks (all GPUs):
cargo run --release --example bench

# CUDA comparison (requires --features cuda and NVIDIA GPU):
cargo run --release --example candle_bench --features cuda

# Metal comparison (requires --features metal and macOS):
cargo run --release --example candle_bench --features metal
```

Benchmark numbers from commit [`56976a7`](https://github.com/cochranblock/any-gpu/commit/56976a7). CUDA/Metal comparison from commit [`d6ab4ec`](https://github.com/cochranblock/any-gpu/commit/d6ab4ec).

### Notes

- CPU is single-threaded naive matmul (triple nested loop). Not BLAS.
- "GPU total" includes upload + compute + readback. "GPU compute" is dispatch + readback with data already resident.
- First GPU call pays pipeline compilation cost (~1-30ms depending on driver).
- candle CUDA numbers use cuBLAS (averaged over 20-100 iterations with warmup). any-gpu numbers are single-run.
- Max numerical error across all GPUs: 0.000023 (f32 accumulation, expected).

## Architecture

Two layers. The tensor API is the product. The GPU backend is plumbing.

**Layer 1: Tensor API** — backend-agnostic transforms. `Tensor::matmul`, `Tensor::conv2d`, `Tensor::relu`. User code never touches GPU backends. Write once, runs anywhere.

**Layer 2: Backend router** — a `match` statement, not a framework. Compile-time feature flags pick the backend: `features = ["metal"]` on Mac, `features = ["cuda"]` on NVIDIA, Vulkan is the default/universal fallback. The router adds <100ns overhead vs calling the backend directly. Every abstraction pays rent or gets evicted.

```rust
let device = Device::auto();        // picks fastest available backend
let t = Tensor::new(data, &device); // user never knows if it's CUDA, Metal, or Vulkan
let out = t.conv2d(&weight, stride, padding)?;
```

Vulkan is the Rosetta Stone — it makes AMD and Intel GPUs possible. CUDA is faster on NVIDIA. Metal is faster on Apple. The router picks the winner. The user gets the fastest path without caring which one.

## Ops (Sprint 2 — shipped)

54 tests, all passing on AMD RX 5700 XT (Vulkan) and Apple M4 (Metal). Every op cross-validated against a CPU reference.

| Category | Ops |
|----------|-----|
| Elementwise | add, sub, mul, scale, relu, sigmoid, swish/silu, tanh |
| Convolution | conv2d, conv_transpose2d, batch_matmul, matmul |
| Normalization | group_norm (two-pass) |
| Tensor manipulation | concat, transpose |
| Attention | softmax (two-pass), scaled_dot_product_attention |
| Spatial | upsample_nearest2d |
| Loss | mse_loss |

All shaders use uniform params (no `arrayLength()` — crashes RADV). All ops handle >65535 workgroups via 2D dispatch.

## Roadmap

### Sprint 3: Tiled matmul + Tensor type

- **Tiled matmul** with workgroup shared memory — the single biggest perf win, expected 5-10x
- **Tensor type** with shape tracking, strides, and views — Copy struct, one pointer + size
- **Pipeline caching** — eliminate per-dispatch shader compilation

### Sprint 4: Autograd + Training

- **Autograd** — reverse-mode autodiff, backward pass for all ops
- **AdamW optimizer**
- **Training loop** as a function call, not a framework

### Sprint 5: Stratagems (training pipelines)

Pre-built training pipelines. Like air strikes — call in what you need, it drops in ready to go.

```
any-gpu train mnist --epochs 10
any-gpu train diffusion --data ./sprites --size 32
any-gpu train classifier --data ./labeled/ --classes 10
any-gpu bench
any-gpu info
```

Each stratagem is a function, not a framework. User provides data, any-gpu handles model architecture, training loop, checkpointing, loss curves. One command, one binary.

### Vision: The Rosetta Stone that learns your hardware

Self-optimizing routing layer:

1. **Auto-benchmark on first run.** Microbenchmarks per op type at various sizes. Real dispatch, real numbers.
2. **Bake a subatomic routing model.** Nanobyte `.weights` file capturing your hardware's exact performance profile. Same architecture as kova's pyramid.
3. **Route by measurement, not vendor name.** 512x512 matmul might go to discrete GPU while 64x64 add stays on integrated.
4. **Hot-swap on hardware changes.** Re-benchmark, retrain, patch the memory map. Like a firmware update.
5. **Multi-GPU dispatch.** Split work across devices by measured throughput.

## Build

```
cargo add any-gpu
```

Or clone and run the benchmark:

```
cargo run --release --example bench
```

## Test

```
cargo test
```

54 correctness tests. Every op verified against a CPU reference implementation. Requires a GPU (any backend).

## License

[Unlicense](UNLICENSE) — [cochranblock.org](https://cochranblock.org)

---

Part of [The Cochran Block](https://cochranblock.org) — see also [kova](https://github.com/cochranblock/kova), [pixel-forge](https://github.com/cochranblock/pixel-forge), [tmuxisfree](https://github.com/cochranblock/tmuxisfree)
