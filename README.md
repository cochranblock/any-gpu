# any-gpu

Bare metal tensor engine. Runs on any GPU — AMD, NVIDIA, Intel, Apple. One codebase, zero vendor lock-in.

wgpu under the hood. Vulkan on Linux, Metal on macOS, DX12 on Windows. You write WGSL compute shaders once, they run everywhere.

## Current state (Sprint 1)

Three tensor ops as WGSL compute shaders:

- **matmul** — naive matrix multiply (16x16 workgroups)
- **add** — element-wise addition
- **mul** — element-wise multiplication

Device discovery picks the best GPU automatically. Upload f32 data, dispatch compute, read results back.

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

All results from `cargo test --release` on 2026-04-02.

| GPU | Vendor | VRAM | Driver | OS | Tests | Matmul 512x512 (ms) |
|-----|--------|------|--------|----|-------|----------------------|
| AMD Radeon RX 5700 XT | AMD (RADV NAVI10) | 8 GB | Mesa 25.0.7 | Debian 13, kernel 6.12.73 | 3/3 pass | 5.67 |
| NVIDIA GeForce RTX 3070 Laptop | NVIDIA | 8 GB | 550.163.01 | Debian 13, kernel 6.12.73 | 3/3 pass | 3.15 |
| NVIDIA GeForce RTX 3050 Ti Laptop | NVIDIA | 4 GB | 550.163.01 | Debian 13, kernel 6.12.73 | 3/3 pass | 4.92 |
| Apple M4 | Apple (Metal) | Unified | macOS 25.3.0 | macOS Tahoe | 3/3 pass | 4.91 |

### Known issues

- **AMD RADV/RDNA1**: concurrent `wgpu::Instance` creation segfaults. Fixed by sharing a single `GpuDevice` via `LazyLock` (e124fbb). Individual ops work fine.
- **Intel Iris Xe on gd**: wgpu selects NVIDIA 3050 Ti (Vulkan) over Intel iGPU. Intel backend untested in isolation.

## Benchmarks

Matmul performance across all tested GPUs. Single run, includes pipeline creation overhead on first call. GPU numbers are compute-only (data already on device).

### 512x512 matmul — the money shot

| GPU | CPU (ms) | GPU compute (ms) | GPU GFLOPS | Speedup |
|-----|----------|-------------------|------------|---------|
| AMD RX 5700 XT (Vulkan) | 180.74 | 5.67 | 47.35 | **31.9x** |
| NVIDIA RTX 3070 Laptop (Vulkan) | 103.97 | 3.15 | 85.12 | **33.0x** |
| NVIDIA RTX 3050 Ti Laptop (Vulkan) | 106.81 | 4.92 | 54.58 | **21.7x** |
| Apple M4 (Metal) | 87.61 | 4.91 | 54.71 | **17.9x** |

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
| 64x64 | 0.15 | 29.25 | 1.99 | 3.45 | 0.26 | 0.1x |
| 128x128 | 1.14 | 1.59 | 1.48 | 3.67 | 2.83 | 0.8x |
| 256x256 | 8.84 | 1.97 | 1.74 | 3.80 | 19.27 | 5.1x |
| 512x512 | 103.97 | 4.49 | 3.15 | 2.58 | 85.12 | 33.0x |
| 1024x1024 | 2133.27 | 19.59 | 14.52 | 1.01 | 147.95 | **147.0x** |

#### NVIDIA GeForce RTX 3050 Ti Laptop (Vulkan)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|-----------------|-------------------|------------|------------|---------|
| 64x64 | 0.14 | 29.70 | 1.76 | 3.64 | 0.30 | 0.1x |
| 128x128 | 1.07 | 1.50 | 1.40 | 3.92 | 2.99 | 0.8x |
| 256x256 | 8.32 | 9.77 | 1.82 | 4.03 | 18.43 | 4.6x |
| 512x512 | 106.81 | 5.53 | 4.92 | 2.51 | 54.58 | 21.7x |
| 1024x1024 | 2076.37 | 33.61 | 32.79 | 1.03 | 65.50 | **63.3x** |

#### Apple M4 (Metal)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|-----------------|-------------------|------------|------------|---------|
| 64x64 | 0.11 | 4.11 | 2.23 | 4.76 | 0.23 | 0.0x |
| 128x128 | 2.28 | 2.16 | 2.00 | 1.84 | 2.10 | 1.1x |
| 256x256 | 19.92 | 2.12 | 1.85 | 1.68 | 18.14 | 10.8x |
| 512x512 | 87.61 | 5.43 | 4.91 | 3.06 | 54.71 | 17.9x |
| 1024x1024 | 769.38 | 24.37 | 15.88 | 2.79 | 135.22 | **48.4x** |

### Notes

- CPU is single-threaded naive matmul (triple nested loop). Not BLAS. The point is to show GPU dispatch overhead vs compute wins.
- "GPU total" includes upload + compute + readback. "GPU compute" is dispatch + readback with data already resident.
- First GPU call in each process pays pipeline compilation cost (~1-30ms depending on driver).
- Naive WGSL shader, no tiling, no shared memory. Real performance will be 5-10x better with tiled matmul.
- RTX 3070 hits 148 GFLOPS at 1024x1024 — that's ~1.5% of its theoretical peak (20 TFLOPS). Tiling will close this gap.
- Max numerical error across all GPUs: 0.000021 (f32 accumulation, expected).

## What this will become

A full tensor framework for ML training. The roadmap:

- **Tensor type** with shape tracking and autograd
- **Full op set** — sub, div, exp, log, sqrt, relu, sigmoid, tanh, softmax, conv2d, transpose, reduce_sum/mean, layer_norm, batch_norm, embedding, cross_entropy, mse
- **Autograd** — reverse-mode autodiff, backward pass, gradient accumulation
- **Optimizer** — SGD, Adam
- **Layer types** — Linear, Conv2d, LayerNorm, BatchNorm
- **Training loop** as a function call, not a framework
- **Tiled matmul** with shared memory for real performance

Target: replace CUDA dependency entirely. Train models on any GPU from any vendor.

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

Three tests: add, mul, matmul (2x2). Requires a GPU (any backend).

## License

Unlicense. Public domain.
