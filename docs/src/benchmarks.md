# Benchmarks

Reproduce: `cargo run --release --example bench`

Numbers from commit [`56976a7`](https://github.com/cochranblock/any-gpu/commit/56976a7) (matmul + conv2d). CUDA/Metal comparison from commit [`d6ab4ec`](https://github.com/cochranblock/any-gpu/commit/d6ab4ec).

## Matmul 512x512 — All GPUs

| GPU | GPU compute (ms) | GFLOPS | Speedup vs CPU |
|-----|------------------|--------|----------------|
| NVIDIA RTX 3070 (Vulkan) | 3.03 | 88.59 | 35.4x |
| Apple M4 (Metal) | 3.36 | 79.88 | 26.0x |
| NVIDIA RTX 3050 Ti (Vulkan) | 5.61 | 47.81 | 17.3x |
| AMD RX 5700 XT (Vulkan) | 5.67 | 47.35 | 31.9x |

## Matmul 1024x1024 — Peak Throughput

| GPU | GPU compute (ms) | GFLOPS | Speedup vs CPU |
|-----|------------------|--------|----------------|
| NVIDIA RTX 3070 | 14.25 | 150.71 | 150.4x |
| Apple M4 | 17.55 | 122.37 | 44.1x |
| AMD RX 5700 XT | 31.22 | 68.78 | 180.7x |
| NVIDIA RTX 3050 Ti | 34.20 | 62.79 | 60.6x |

## Full Matrix — All Sizes, All GPUs

### AMD Radeon RX 5700 XT (Vulkan, RADV NAVI10)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|----------------|------------------|------------|------------|---------|
| 64x64 | 0.17 | 2.59 | 1.24 | 3.12 | 0.42 | 0.1x |
| 128x128 | 2.22 | 0.84 | 0.68 | 1.89 | 6.13 | 3.2x |
| 256x256 | 16.09 | 1.66 | 1.48 | 2.09 | 22.66 | 10.9x |
| 512x512 | 180.74 | 7.79 | 5.67 | 1.49 | 47.35 | 31.9x |
| 1024x1024 | 5641.56 | 39.99 | 31.22 | 0.38 | 68.78 | **180.7x** |

### NVIDIA GeForce RTX 3070 Laptop (Vulkan)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|----------------|------------------|------------|------------|---------|
| 64x64 | 0.15 | 22.98 | 2.01 | 3.55 | 0.26 | 0.1x |
| 128x128 | 1.16 | 1.70 | 1.57 | 3.61 | 2.68 | 0.7x |
| 256x256 | 8.80 | 1.61 | 1.72 | 3.81 | 19.52 | 5.1x |
| 512x512 | 107.35 | 4.47 | 3.03 | 2.50 | 88.59 | 35.4x |
| 1024x1024 | 2142.91 | 19.69 | 14.25 | 1.00 | 150.71 | **150.4x** |

### NVIDIA GeForce RTX 3050 Ti Laptop (Vulkan)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|----------------|------------------|------------|------------|---------|
| 64x64 | 0.16 | 23.21 | 1.87 | 3.28 | 0.28 | 0.1x |
| 128x128 | 1.05 | 1.35 | 1.50 | 3.98 | 2.79 | 0.7x |
| 256x256 | 8.28 | 1.34 | 1.37 | 4.05 | 24.53 | 6.1x |
| 512x512 | 97.28 | 5.92 | 5.61 | 2.76 | 47.81 | 17.3x |
| 1024x1024 | 2071.17 | 32.67 | 34.20 | 1.04 | 62.79 | **60.6x** |

### Apple M4 (Metal via wgpu)

| Size | CPU (ms) | GPU total (ms) | GPU compute (ms) | CPU GFLOPS | GPU GFLOPS | Speedup |
|------|----------|----------------|------------------|------------|------------|---------|
| 64x64 | 0.11 | 4.53 | 2.02 | 4.62 | 0.26 | 0.1x |
| 128x128 | 1.97 | 2.04 | 2.12 | 2.13 | 1.98 | 0.9x |
| 256x256 | 17.35 | 2.18 | 1.95 | 1.93 | 17.20 | 8.9x |
| 512x512 | 87.32 | 3.79 | 3.36 | 3.07 | 79.88 | 26.0x |
| 1024x1024 | 773.88 | 23.01 | 17.55 | 2.77 | 122.37 | **44.1x** |

## Conv2d — UNet Layers on AMD RX 5700 XT

10-iteration average, compute + readback.

| Layer | Shape | Time (ms) | GFLOPS |
|-------|-------|-----------|--------|
| Input (3->64) | 3x32x32 -> 64x32x32, k=3 | 1.08 | 3.28 |
| Down (64->128) | 64x16x16 -> 128x16x16, k=3 | 1.30 | 29.10 |
| Bottleneck (128->256) | 128x8x8 -> 256x8x8, k=3 | 1.47 | 25.61 |
| Up (256->128) | 256x8x8 -> 128x8x8, k=3 | 1.80 | 21.01 |
| Decoder (128->64) | 128x16x16 -> 64x16x16, k=3 | 1.24 | 30.52 |
| Output (64->3) | 64x32x32 -> 3x32x32, k=3 | 0.97 | 3.64 |

Full UNet forward pass for 32x32 sprites: **~7.9ms** (127 forward passes/second).

## Honest Comparison: any-gpu vs CUDA and Metal

CUDA and MPS are faster. Here are the numbers.

### NVIDIA RTX 3070 Laptop — Vulkan vs CUDA

| Size | any-gpu Vulkan (ms) | candle CUDA (ms) | CUDA faster by |
|------|---------------------|------------------|----------------|
| 128x128 | 1.57 | 0.07 | 22x |
| 256x256 | 1.72 | 0.20 | 9x |
| 512x512 | 3.03 | 0.75 | 4x |
| 1024x1024 | 14.25 | 2.80 | 5x |

### NVIDIA RTX 3050 Ti Laptop — Vulkan vs CUDA

| Size | any-gpu Vulkan (ms) | candle CUDA (ms) | CUDA faster by |
|------|---------------------|------------------|----------------|
| 128x128 | 1.50 | 0.03 | 50x |
| 256x256 | 1.37 | 0.07 | 20x |
| 512x512 | 5.61 | 0.33 | 17x |
| 1024x1024 | 34.20 | 1.43 | 24x |

### Apple M4 — wgpu Metal vs candle MPS

| Size | any-gpu Metal (ms) | candle MPS (ms) | MPS faster by |
|------|--------------------|-----------------|---------------|
| 128x128 | 2.12 | 0.36 | 6x |
| 256x256 | 1.95 | 0.31 | 6x |
| 512x512 | 3.36 | 0.47 | 7x |
| 1024x1024 | 17.55 | 1.94 | 9x |

### What to Make of This

CUDA and MPS use tiled matmul with shared memory, register blocking, and vendor-tuned kernels. any-gpu uses a tiled 16x16 shared-memory WGSL shader.

The point is not performance parity with cuBLAS:

- The AMD RX 5700 XT has zero CUDA support and zero MPS support. any-gpu is the only option that gives it GPU compute for ML in Rust.
- Intel Arc and Iris Xe — same story.
- One `cargo build` produces a binary that runs on all four GPUs above. No feature flags, no vendor SDKs.

The performance gap closes with better shaders. The RTX 3070 is 4x behind CUDA at 512x512. Tiling alone should close most of that.

## Measurement Notes

- CPU is single-threaded naive matmul (triple nested loop). Not BLAS.
- "GPU total" includes upload + compute + readback. "GPU compute" is dispatch + readback with data already resident.
- First GPU call pays pipeline compilation cost (~1–30ms depending on driver).
- candle CUDA numbers use cuBLAS (averaged over 20–100 iterations with warmup). any-gpu numbers are single-run.
- Max numerical error across all GPUs: 0.000023 (f32 accumulation, expected).
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
