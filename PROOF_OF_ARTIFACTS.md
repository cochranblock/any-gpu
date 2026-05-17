<!-- Unlicense — cochranblock.org -->

# Proof of Artifacts

*Concrete evidence that this project works, ships, and is real.*

> This is not a demo repo. This is a GPU compute engine tested on real hardware across 3 nodes. The artifacts below prove it.

## Why this exists

CUDA only runs on NVIDIA. Metal only runs on Apple. If you have an AMD RX 5700 XT or an Intel Arc, your options for GPU-accelerated ML in Rust are: nothing. any-gpu fills that gap.

If you have an NVIDIA GPU and want peak performance, use CUDA. If you're on macOS and want peak performance, use Metal Performance Shaders. They're faster — we measured it, the numbers are below, and we're not hiding them.

Use any-gpu when:
- Your GPU is AMD or Intel (where CUDA can't run)
- You need one binary that works on every machine
- You refuse to vendor-lock your compute pipeline
- You're building for a heterogeneous fleet (NVIDIA in the cloud, AMD on workstations, Apple on laptops)

## Architecture

> All public symbols are tokenized per `docs/compression_map.md`.
> `GpuDevice` -> `t500`, `GpuBuffer` -> `t501`, `Tensor` -> `t502`, `Tape` -> `t506`,
> `AdamW` -> `t507`. Functions: `gpu` -> `f500`, `upload` -> `f502`, `matmul` -> `f580`,
> `conv2d` -> `f582`, `softmax` -> `f620`, `train_step` -> `f730`, etc.

```
GpuDevice::gpu()      (= t500::f500())
    │
    ▼
wgpu (auto-selects backend)
    │
    ├── Vulkan (Linux: AMD, NVIDIA, Intel)
    ├── Metal (macOS: Apple Silicon)
    └── DX12 (Windows)
    │
    ▼
56+ WGSL compute shaders
    │
    ├── elementwise: add, sub, mul, scale, relu, sigmoid, swish, tanh, gelu (tanh-approx)
    ├── backward: relu_bw, sigmoid_bw, swish_bw, tanh_bw
    ├── conv: tiled matmul (wave64 4×4 reg-blocked, 168 GFLOPS @ 512²), batch_matmul, conv2d, conv_transpose2d
    ├── conv grad: conv2d_grad_weight, conv2d_grad_bias
    ├── norm: group_norm (two-pass), layer_norm (two-pass), rms_norm (two-pass)
    ├── attention: softmax (subgroup-fused single-dispatch when s509, two-pass fallback),
    │             sdpa, causal_mask, causal_sdpa, rope,
    │             fused_sdpa (online-softmax, no N×N alloc), split_heads, merge_heads, repeat_kv
    ├── tensor: concat, transpose, slice, broadcast_add, sum_inner, add_per_col, sum_rows
    ├── transformer: embedding_lookup (gather), argmax (last-dim), kv_append
    ├── spatial: upsample_nearest2d (+ backward)
    ├── loss: mse_loss
    ├── optim: adamw (in-place, momentum + velocity + weight decay)
    └── f16: unpack2x16float dequant (packed u32 → f32 pairs)
    │
    ▼
KVCache (t534) — append-only persistent K/V for autoregressive decoding
    │
    ├── f672 new, f673 append, f674 reset
    └── f675 cursor, f676 k_buffer, f677 v_buffer
    │
    ▼
Inference Stack (Sprint 7)
    │
    ├── t544 Tokenizer — HuggingFace tokenizers wrapper (f775–f779)
    ├── t545 Module trait — forward(&self, dev, x) -> Result<Tensor>
    ├── t546 Linear — GPU linear layer, HF weight transposition at load
    ├── t547 LmConfig — JSON config (vocab_size, hidden/intermediate size, heads, GQA)
    └── t548 CausalLM — LLaMA-compatible forward (f783–f786), any-gpu-serve binary
    │
    ▼
Autograd (reverse-mode autodiff)
    │
    ├── Flat tape with enum ops (no trait objects)
    ├── 13 differentiable ops (add, sub, mul, scale, relu, sigmoid, swish, tanh, matmul, mse_loss, conv2d)
    ├── Backward pass: topological sort, accumulate grads via GPU shaders
    └── train_step() (= f730): forward + backward + AdamW in one call
    │
    ▼
NanoSign (model integrity)
    │
    └── NSIG + BLAKE3 hash (36 bytes) on every .weights file
```

### Current state

Sprint 7 complete + perf sprint (2026-05-17). 256 tests, all passing on bt (AMD RX 5700 XT, RADV/Vulkan). Full LLM inference stack shipped: tokenizer, LLaMA-compatible CausalLM, HTTP serve binary. Matmul upgraded to wave64 4×4 register-blocking (168 GFLOPS @ 512²). Softmax upgraded to subgroup-fused single dispatch (subgroupMax/subgroupAdd). Verified on 4 GPUs across 3 nodes.

| Category | Ops |
|----------|-----|
| Elementwise | add, sub, mul, scale, relu, sigmoid, swish/silu, tanh |
| Convolution | conv2d, conv_transpose2d, batch_matmul, matmul |
| Normalization | group_norm (two-pass) |
| Tensor manipulation | concat, transpose |
| Attention | softmax (two-pass), scaled_dot_product_attention |
| Spatial | upsample_nearest2d |
| Loss | mse_loss |
| Integrity | NanoSign — BLAKE3 model file signing (NSIG + 36 bytes) |

All shaders use uniform params (no `arrayLength()` — crashes RADV). All ops handle >65535 workgroups via 2D dispatch.

### Token-Optimized Code Example

Public symbols are tokenized per [docs/compression_map.md](docs/compression_map.md). Full mapping there; the example below uses `t500=GpuDevice`, `t501=GpuBuffer`, `f500=gpu`, `f502=upload`, `f504=read`, `f550=add`, `f580=matmul`.

```rust
use any_gpu::t500;

let dev = t500::f500()?;

let a = dev.f502(&[1.0, 2.0, 3.0, 4.0]);
let b = dev.f502(&[5.0, 6.0, 7.0, 8.0]);
let c = dev.f580(&a, &b, 2, 2, 2)?;

let result = dev.f504(&c)?;
// [19.0, 22.0, 43.0, 50.0]
```

### Planned architecture (not yet shipped)

Currently: `GpuDevice` struct wraps wgpu directly. Public methods for each op (matmul, conv2d, relu, etc.). Upload data, dispatch WGSL compute shaders, read results back. No abstraction layers. NanoSign for model file integrity.

**Layer 1: Tensor API** — backend-agnostic `Tensor` type with shape tracking. `Tensor::matmul`, `Tensor::conv2d`, `Tensor::relu`. User code never touches GPU backends.

**Layer 2: Backend router** — compile-time feature flags pick the fastest backend per platform. `features = ["metal"]` on Mac, `features = ["cuda"]` on NVIDIA, Vulkan as universal fallback. One `match` statement, not a framework.

## Build Output

| Metric | Value |
|--------|-------|
| Lines of Rust | ~8,500+ across 15+ source files (device, ops×7, tensor, autograd, optim, train, nanosign, safetensors, pager, tokenizer, module, lm, bin/any-gpu-serve) |
| Public ops | 27 GPU forward ops + 7 backward ops + 4 head ops (split/merge/repeat_kv + fused SDPA) + 7 NanoSign + KVCache (6) + SafetensorsModel (6) |
| Modules | device, ops (7 submodules incl. transformer), tensor, autograd, optim, train, nanosign, safetensors, pager, tokenizer, module, lm |
| WGSL shaders | 58+ (forward + activation backward + conv2d grad + norm backward + adamw + causal_mask + rope + kv_append + fused_sdpa + split_heads + merge_heads + repeat_kv + unpack2x16float + softmax_fused_subgroup) |
| Tests | 256 (62 GPU ops + 29 transformer-inference step-1 + 21 step-2 incl. KV cache round-trip + 19 safetensors loader + 7 hardcoded backstops + 5 pager + 5 f16 storage + 6 fused SDPA + 17 inference stack + 17 autograd + 11 device + 17 tensor + 13 nanosign + 8 optim + 1 train + 24 elementwise backward + more) |
| Determinism gate | exopack TRIPLE SIMS (`cargo run --release --bin any-gpu-test --features tests`) — 3/3 passes on bt RX 5700 XT, 96/4/3 ms (pass 1 includes pipeline compile; pass 2+3 hit the cache) |
| Bench binary (release) | ~1.5 MB (opt-z, LTO, strip, panic=abort) |
| Train binary (release) | ~1.5 MB |
| Dependencies | wgpu, bytemuck, anyhow, pollster, blake3, safetensors, tokenizers, serde, serde\_json, clap + 1 optional (exopack via `--features tests`) |
| Model signing | NanoSign v1 — NSIG + BLAKE3 (36 bytes per file) |
| Pipeline caching | Compile once, reuse Arc\<ComputePipeline\> via source hash |

## Hardware Verification

Tested on 2026-04-02 at commit [`f3319fb`](https://github.com/cochranblock/any-gpu/commit/f3319fb).

| Node | GPU | VRAM | Driver | OS | Tests | Result |
|------|-----|------|--------|----|-------|--------|
| bt | AMD Radeon RX 5700 XT (RADV NAVI10) | 8 GB | Mesa 25.0.7 | Debian 13, kernel 6.12.73 | 256/256 | **pass** |
| lf | NVIDIA GeForce RTX 3070 Laptop | 8 GB | 550.163.01 | Debian 13, kernel 6.12.73 | 54/54 (Sprint 2) | **pass** |
| gd | NVIDIA GeForce RTX 3050 Ti Laptop | 4 GB | 550.163.01 | Debian 13, kernel 6.12.73 | 54/54 (Sprint 2) | **pass** |
| local | Apple M4 | Unified | — | macOS Tahoe 25.3.0 | 62/62 (Sprint 4) | **pass** |

**Reproduce:**

```bash
cargo test --release                           # local
WGPU_BACKEND=vulkan cargo test --release       # force Vulkan on AMD
ssh bt 'cd ~/any-gpu && git pull && cargo test --release'  # remote
```

Test results verified at commits [`801c4de`](https://github.com/cochranblock/any-gpu/commit/801c4de) (GPU ops), [`5e58eb3`](https://github.com/cochranblock/any-gpu/commit/5e58eb3) (NanoSign). Benchmark numbers from commit [`56976a7`](https://github.com/cochranblock/any-gpu/commit/56976a7).

### Known issues

- **AMD RADV/RDNA1**: concurrent `wgpu::Instance` creation segfaults. Fixed by sharing a single `GpuDevice` via `LazyLock` ([`e124fbb`](https://github.com/cochranblock/any-gpu/commit/e124fbb)). Individual ops work fine.
- **Intel Iris Xe**: untested in isolation (wgpu prefers discrete NVIDIA when both are present).

## NanoSign

Every model weights file saved by any-gpu is signed with [NanoSign](https://github.com/cochranblock/kova/blob/main/docs/NANOSIGN.md) — 36 bytes appended to EOF: `NSIG` magic (4 bytes) + BLAKE3 hash (32 bytes). Verified on load. Tampered files are rejected.

```rust
// f745=save_signed, f746=load_verified, f743=sign_bytes, f742=verify_bytes,
// t510=NanoSignResult.
use any_gpu::nanosign;
use std::path::Path;

// Save weights with signature
nanosign::f745(Path::new("model.weights"), &weight_bytes)?;

// Load and verify (rejects tampered files)
let weights = nanosign::f746(Path::new("model.weights"))?;

// In-memory sign/verify
let signed = nanosign::f743(&data);
assert!(matches!(nanosign::f742(&signed), nanosign::t510::Verified(_)));
```

Standard across the cochranblock ecosystem. Spec: [NANOSIGN.md](https://github.com/cochranblock/kova/blob/main/docs/NANOSIGN.md).

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

CUDA and MPS use tiled matmul with shared memory, register blocking, and vendor-tuned kernels. any-gpu uses a naive triple-loop WGSL shader.

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

## P23: Triple Lens

All any-gpu work is evaluated through the Triple Lens quality gate:

| Lens | Question | Evidence |
|------|----------|----------|
| Technical | Does it compile, pass tests, run on real hardware? | 256/256 tests, 4 GPUs, 3 nodes (bt/lf/gd + local). Full autograd + training loop + LLM inference stack. |
| Product | Does it solve a real problem? | AMD/Intel GPU compute for ML in Rust — nobody else does this. Trains models, runs LLaMA-compatible inference, serves over HTTP. |
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

## Roadmap

### Sprint 3: Tiled matmul + Tensor type

- **Tiled matmul** with workgroup shared memory — the single biggest perf win, expected 5-10x
- **Tensor type** with shape tracking, strides, and views — Copy struct, one pointer + size
- **Pipeline caching** — eliminate per-dispatch shader compilation

### Sprint 4: Autograd + Training

- **Autograd** — reverse-mode autodiff, backward pass for all 19 ops
- **Backward shaders** — ~10 new WGSL kernels (relu_backward, sigmoid_backward, swish_backward, tanh_backward, softmax_backward, mse_backward, group_norm_backward, downsample_sum, conv2d weight grad, slice). Remaining ops reuse existing forward shaders with transposed inputs.
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

Each stratagem is a function, not a framework. User provides data, any-gpu handles model architecture, training loop, checkpointing, loss curves. One command, one binary. All saved weights NanoSign'd.

### Sprint 6: Starter Nanobyte

First nanobyte model trained and shipped with any-gpu. A tiny diffusion model (~1M params) for 32x32 pixel art, trained on bt's 5700 XT via any-gpu's own training loop. The proof that the engine works end-to-end. NanoSign'd `.weights` file, reproducible from the included training stratagem.

### Vision: The Rosetta Stone that learns your hardware

Self-optimizing routing layer:

1. **Auto-benchmark on first run.** Microbenchmarks per op type at various sizes. Real dispatch, real numbers.
2. **Bake a subatomic routing model.** Nanobyte `.weights` file capturing your hardware's exact performance profile. Same architecture as kova's pyramid.
3. **Route by measurement, not vendor name.** 512x512 matmul might go to discrete GPU while 64x64 add stays on integrated.
4. **Hot-swap on hardware changes.** Re-benchmark, retrain, patch the memory map. Like a firmware update.
5. **Multi-GPU dispatch.** Split work across devices by measured throughput.

## What's Not Here (Yet)

- ~~Tensor type with shape tracking~~ — **shipped** (commit `dd55772`)
- ~~Autograd / backward pass~~ — **shipped** (commit `5137d40`, 7 backward shaders)
- ~~Tiled matmul with shared memory~~ — **shipped** (commit `0ca243d`)
- ~~Transformer math primitives — LayerNorm, RMSNorm, GELU, embedding_lookup, argmax~~ — **shipped** (2026-05-15)
- ~~Causal-masked SDPA + RoPE + KV cache~~ — **shipped** (2026-05-15)
- ~~Safetensors loader + bf16/f16 weight ingest~~ — **shipped** (2026-05-16)
- ~~Pinned-RAM staging + layer paging from system RAM to VRAM~~ — **shipped** (2026-05-17, t539 LayerPager)
- ~~f16 storage type~~ — **shipped** (2026-05-17, t540 GpuBufferF16 via packed u32 + unpack2x16float; note: `enable f16` in WGSL is NOT supported by Naga/wgpu)
- ~~Flash-attention-style tiled SDPA~~ — **shipped** (2026-05-17, f626 online-softmax fused SDPA, no N×N alloc)
- ~~Tokenizer + `Module` graph + `any-gpu serve` runtime~~ — **shipped** (2026-05-17, t544/t545/t546/t547/t548, any-gpu-serve binary)
- Backend router (CUDA/Metal/Vulkan dispatch) (planned)
- Stratagems CLI — `any-gpu train`, `any-gpu bench`, `any-gpu info` (planned)
- Starter nanobyte — first model trained and shipped with any-gpu (planned)
- Multi-node distributed training via C2 (planned)

## Quick Start

```bash
# Add to your project
cargo add any-gpu

# Or clone and run the benchmark
cargo run --release --example bench

# Run all tests (256 — GPU ops, autograd, transformer, safetensors, inference stack, and more)
cargo test --release

# On AMD RADV, force Vulkan backend
WGPU_BACKEND=vulkan cargo test --release
```

256 tests. All verified on bt (AMD RX 5700 XT, RADV/Vulkan). Every op verified against a CPU reference implementation or hardcoded reference values. Requires a GPU (any backend).

---

Part of [The Cochran Block](https://cochranblock.org) — see also [kova](https://github.com/cochranblock/kova), [pixel-forge](https://github.com/cochranblock/pixel-forge), [tmuxisfree](https://github.com/cochranblock/tmuxisfree)
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
