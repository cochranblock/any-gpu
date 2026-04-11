<!-- Unlicense — cochranblock.org -->

# Proof of Artifacts

*Concrete evidence that this project works, ships, and is real.*

> This is not a demo repo. This is a GPU compute engine tested on real hardware across 3 nodes. The artifacts below prove it.

## Architecture

```
GpuDevice::gpu()
    │
    ▼
wgpu (auto-selects backend)
    │
    ├── Vulkan (Linux: AMD, NVIDIA, Intel)
    ├── Metal (macOS: Apple Silicon)
    └── DX12 (Windows)
    │
    ▼
27 WGSL compute shaders
    │
    ├── elementwise: add, sub, mul, scale, relu, sigmoid, swish, tanh
    ├── backward: relu_bw, sigmoid_bw, swish_bw, tanh_bw
    ├── conv: tiled matmul (16x16 shared mem), batch_matmul, conv2d, conv_transpose2d
    ├── conv grad: conv2d_grad_weight, conv2d_grad_bias
    ├── norm: group_norm (two-pass: stats → normalize)
    ├── attention: softmax (two-pass: max/sum → exp/div), sdpa
    ├── tensor: concat, transpose
    ├── spatial: upsample_nearest2d
    ├── loss: mse_loss
    └── optim: adamw (in-place, momentum + velocity + weight decay)
    │
    ▼
Autograd (reverse-mode autodiff)
    │
    ├── Flat tape with enum ops (no trait objects)
    ├── 13 differentiable ops (add, sub, mul, scale, relu, sigmoid, swish, tanh, matmul, mse_loss, conv2d)
    ├── Backward pass: topological sort, accumulate grads via GPU shaders
    └── train_step(): forward + backward + AdamW in one call
    │
    ▼
NanoSign (model integrity)
    │
    └── NSIG + BLAKE3 hash (36 bytes) on every .weights file
```

## Build Output

| Metric | Value |
|--------|-------|
| Lines of Rust | 4,393 across 13 source files (+601 in examples) |
| Public ops | 19 GPU forward ops + 7 backward ops + 7 NanoSign functions |
| Modules | device, ops (6 submodules), tensor, autograd, optim, train, nanosign |
| WGSL shaders | 27 (20 forward + 4 activation backward + 2 conv2d grad + 1 adamw) |
| Tests | 145 (54 GPU ops + 17 autograd + 11 device + 17 tensor + 13 nanosign + 8 optim + 1 train + 24 elementwise backward) |
| Bench binary (release) | 1.5 MB (opt-z, LTO, strip, panic=abort) |
| Train binary (release) | 1.5 MB |
| Dependencies | 5 (wgpu, bytemuck, anyhow, pollster, blake3) |
| Model signing | NanoSign v1 — NSIG + BLAKE3 (36 bytes per file) |
| Pipeline caching | Compile once, reuse Arc\<ComputePipeline\> via source hash |

## Hardware Verification

Tested on 2026-04-02 at commit [`f3319fb`](https://github.com/cochranblock/any-gpu/commit/f3319fb).

| Node | GPU | Driver | OS | Tests | Result |
|------|-----|--------|----|-------|--------|
| bt | AMD Radeon RX 5700 XT (RADV NAVI10) | Mesa 25.0.7 | Debian 13, kernel 6.12.73 | 54/54 | **pass** |
| lf | NVIDIA GeForce RTX 3070 Laptop | 550.163.01 | Debian 13, kernel 6.12.73 | 54/54 | **pass** |
| gd | NVIDIA GeForce RTX 3050 Ti Laptop | 550.163.01 | Debian 13, kernel 6.12.73 | 54/54 | **pass** |
| local | Apple M4 | macOS 25.3.0 | macOS Tahoe | 54/54 | **pass** |

**Reproduce:**

```bash
cargo test --release                           # local
WGPU_BACKEND=vulkan cargo test --release       # force Vulkan on AMD
ssh bt 'cd ~/any-gpu && git pull && cargo test --release'  # remote
```

## Benchmark Proof

### Matmul 512x512 — all GPUs

From commit [`56976a7`](https://github.com/cochranblock/any-gpu/commit/56976a7). Reproduce: `cargo run --release --example bench`

| GPU | GPU compute (ms) | GFLOPS | Speedup vs CPU |
|-----|-------------------|--------|----------------|
| NVIDIA RTX 3070 (Vulkan) | 3.03 | 88.59 | 35.4x |
| Apple M4 (Metal) | 3.36 | 79.88 | 26.0x |
| NVIDIA RTX 3050 Ti (Vulkan) | 5.61 | 47.81 | 17.3x |
| AMD RX 5700 XT (Vulkan) | 5.67 | 47.35 | 31.9x |

### Matmul 1024x1024 — peak throughput

| GPU | GPU compute (ms) | GFLOPS | Speedup vs CPU |
|-----|-------------------|--------|----------------|
| NVIDIA RTX 3070 | 14.25 | 150.71 | 150.4x |
| Apple M4 | 17.55 | 122.37 | 44.1x |
| AMD RX 5700 XT | 31.22 | 68.78 | 180.7x |
| NVIDIA RTX 3050 Ti | 34.20 | 62.79 | 60.6x |

### Conv2d — UNet layers on AMD RX 5700 XT

From commit [`56976a7`](https://github.com/cochranblock/any-gpu/commit/56976a7). 10-iteration average, compute + readback.

| Layer | Shape | Time (ms) | GFLOPS |
|-------|-------|-----------|--------|
| Input (3->64) | 3x32x32 -> 64x32x32, k=3 | 1.08 | 3.28 |
| Down (64->128) | 64x16x16 -> 128x16x16, k=3 | 1.30 | 29.10 |
| Bottleneck (128->256) | 128x8x8 -> 256x8x8, k=3 | 1.47 | 25.61 |
| Up (256->128) | 256x8x8 -> 128x8x8, k=3 | 1.80 | 21.01 |
| Decoder (128->64) | 128x16x16 -> 64x16x16, k=3 | 1.24 | 30.52 |
| Output (64->3) | 64x32x32 -> 3x32x32, k=3 | 0.97 | 3.64 |

Full UNet forward pass for 32x32 sprites: **~7.9ms** (127 forward passes/second).

### Honest comparison: any-gpu vs CUDA/Metal

From commit [`d6ab4ec`](https://github.com/cochranblock/any-gpu/commit/d6ab4ec). Reproduce: `cargo run --release --example candle_bench --features cuda`

| GPU | Size | any-gpu (ms) | candle CUDA/MPS (ms) | Vendor faster by |
|-----|------|--------------|----------------------|------------------|
| RTX 3070 | 512x512 | 3.03 | 0.75 (CUDA) | 4x |
| RTX 3070 | 1024x1024 | 14.25 | 2.80 (CUDA) | 5x |
| RTX 3050 Ti | 512x512 | 5.61 | 0.33 (CUDA) | 17x |
| Apple M4 | 512x512 | 3.36 | 0.47 (MPS) | 7x |

CUDA and MPS use tiled matmul with vendor-tuned kernels. any-gpu uses a naive WGSL shader. The gap closes with tiled matmul (roadmap). The point: any-gpu runs the same shader on AMD, NVIDIA, Intel, and Apple. CUDA can't run on AMD. MPS can't run on NVIDIA.

## P23: Triple Lens

All any-gpu work is evaluated through the Triple Lens quality gate:

| Lens | Question | Evidence |
|------|----------|----------|
| Technical | Does it compile, pass tests, run on real hardware? | 145/145 tests, 4 GPUs, 3 nodes (bt/lf/gd + local). Full autograd + training loop. |
| Product | Does it solve a real problem? | AMD/Intel GPU compute for ML in Rust — nobody else does this. Now trains models, not just inference. |
| Honest | Are the claims verifiable? | Every benchmark has a reproduce command. Every GPU claim links to a commit. CUDA comparison shows where we lose. Backward shaders have numeric gradient tests. |

## Named Techniques

| Technique | What | Where |
|-----------|------|-------|
| Flat Tape Autograd | Enum ops, no trait objects, reverse topo sort | `src/autograd.rs` |
| Inline Shape | Max 6 dims on the stack, no heap for shape metadata | `src/tensor.rs` |
| Tiled Matmul | 16x16 shared memory tiles, 256-thread workgroups | `src/ops/conv.rs` |
| Pipeline Caching | Hash shader source → `Arc<ComputePipeline>`, compile once | `src/device.rs` |
| Two-Pass Reduction | Softmax (max/sum → exp/div), GroupNorm (stats → normalize) | `src/ops/attention.rs`, `src/ops/norm.rs` |
| NanoSign | NSIG + BLAKE3 (36 bytes) — sign on save, verify on load | `src/nanosign.rs` |
| Single-Shader AdamW | Momentum, velocity, weight decay, bias correction in one dispatch | `src/optim.rs` |
| Conv2d Backward | grad_weight shader + grad_bias reduction + grad_input via conv_transpose2d | `src/ops/conv.rs` |

## What's Not Here (Yet)

- ~~Tensor type with shape tracking~~ — **shipped** (commit `dd55772`)
- ~~Autograd / backward pass~~ — **shipped** (commit `5137d40`, 7 backward shaders)
- ~~Tiled matmul with shared memory~~ — **shipped** (commit `0ca243d`)
- Backend router (CUDA/Metal/Vulkan dispatch) (planned)
- Stratagems CLI — `any-gpu train`, `any-gpu bench`, `any-gpu info` (planned)
- Starter nanobyte — first model trained and shipped with any-gpu (planned)
- Multi-node distributed training via C2 (planned)

---

Part of [The Cochran Block](https://cochranblock.org) — see also [kova](https://github.com/cochranblock/kova), [pixel-forge](https://github.com/cochranblock/pixel-forge), [tmuxisfree](https://github.com/cochranblock/tmuxisfree)
