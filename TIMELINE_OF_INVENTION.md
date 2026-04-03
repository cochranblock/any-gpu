<!-- Unlicense — cochranblock.org -->

# Timeline of Invention

*Dated, commit-level record of what was built, when, and why. Proves human-piloted AI development — not generated spaghetti.*

> Every entry below maps to real commits. Run `git log --oneline` to verify.

## How to Read This Document

Each entry follows this format:
- **Date**: When the work shipped
- **What**: Concrete deliverable
- **Why**: Business or technical reason
- **AI Role**: What the AI did vs. what the human directed

---

## Entries

### 2026-04-03 — NanoSign Integration + Full Doc Update

**What:** Added NanoSign module — BLAKE3 model file signing (NSIG + 36 bytes appended to EOF). Every model weights file saved by any-gpu is signed on write, verified on load. Tampered files rejected. 8 tests: sign/verify roundtrip, tamper detection, unsigned detection, strip, empty payload, file I/O roundtrip. Updated all docs: README (62 tests, NanoSign section, P23 Triple Lens, autograd roadmap with backward shader inventory, starter nanobyte sprint, accurate test counts), TIMELINE, PROOF_OF_ARTIFACTS.
**Commits:** [`5e58eb3`](https://github.com/cochranblock/any-gpu/commit/5e58eb3) (NanoSign), current commit (docs update)
**AI Role:** AI implemented NanoSign module from kova's NANOSIGN.md spec. Human directed integration (sign on save, verify on load) and the doc update scope.

### 2026-04-02 — CPU-Validated Test Suite

**What:** Replaced 27 smoke tests with 54 correctness tests. Every op cross-validated against a CPU reference implementation. Added edge cases: odd sizes (13 elements, misaligned to workgroup 256), 1x1 tensors, zero/negative inputs, non-square matmul (17x13x11), 5x5 kernels, multi-channel batch=2 conv2d, constant-input group_norm, softmax numerical stability with large values, transpose roundtrip identity.
**Commit:** [`801c4de`](https://github.com/cochranblock/any-gpu/commit/801c4de)
**Verified:** 54/54 pass on bt (AMD RX 5700 XT), lf (NVIDIA RTX 3070), gd (NVIDIA RTX 3050 Ti), local (Apple M4). All at commit `f3319fb`.
**AI Role:** AI wrote CPU reference functions and test cases. Human directed the audit (which tests were garbage, what edge cases were missing).

### 2026-04-02 — 15 Diffusion Training Ops

**What:** Implemented 15 new WGSL compute shader ops for UNet-based diffusion model training: relu, sigmoid, swish/silu, tanh, conv2d, conv_transpose2d, batch_matmul, group_norm (two-pass), concat, transpose, upsample_nearest2d, softmax (two-pass), scaled_dot_product_attention, scale, sub, mse_loss. Restructured ops.rs into ops/ module directory. Added dispatch_1d helper for >65535 workgroup handling, unary_op/binary_op/dispatch_shader shared helpers, single shared test device via LazyLock.
**Commit:** [`8aa9fc1`](https://github.com/cochranblock/any-gpu/commit/8aa9fc1) (ops), [`56976a7`](https://github.com/cochranblock/any-gpu/commit/56976a7) (RADV fix)
**Conv2d benchmark on AMD RX 5700 XT:** Full UNet forward pass for 32x32 sprites ~7.9ms (127 fps).
**AI Role:** AI designed shader architecture, wrote all WGSL shaders and Rust dispatch code. Human directed op priority order (diffusion training deps) and target hardware (bt's 5700 XT for pixel-forge/Anvil).

### 2026-04-02 — CUDA/Metal Comparison Benchmarks

**What:** Created candle_bench example for head-to-head matmul comparison on same hardware. candle v0.10.2 with cuBLAS (CUDA) and Metal Performance Shaders. Measured on RTX 3070 (CUDA 4-22x faster), RTX 3050 Ti (CUDA 17-50x faster), Apple M4 (MPS 6-9x faster). Documented honestly in README.
**Commit:** [`d6ab4ec`](https://github.com/cochranblock/any-gpu/commit/d6ab4ec)
**AI Role:** AI built comparison benchmark, ran on all nodes, compiled tables. Human directed the comparison methodology and "be honest" positioning.

### 2026-04-02 — AMD RADV Segfault Fix

**What:** Three fixes for SIGSEGV on AMD RADV/RDNA1 (Navi 10): (1) `adapter.limits()` instead of `Limits::default()` — stopped requesting capabilities the driver can't provide. (2) Removed `enumerate_adapters()` — crashed on Linux when probing GL backends. (3) Replaced `arrayLength()` with uniform params in add/mul shaders — `OpArrayLength` SPIR-V crashes some RADV drivers. Later: (4) Shared single GpuDevice via LazyLock — concurrent adapter requests crash RADV.
**Commits:** [`35c75ef`](https://github.com/cochranblock/any-gpu/commit/35c75ef) (initial fixes), [`e124fbb`](https://github.com/cochranblock/any-gpu/commit/e124fbb) (LazyLock), [`56976a7`](https://github.com/cochranblock/any-gpu/commit/56976a7) (cross-module shared device)
**Diagnosed by:** Running individual tests on bt via SSH, isolating that all 3 tests pass alone but crash together. `--test-threads=1` confirmed concurrent init was the trigger.
**AI Role:** AI diagnosed via SSH test isolation, identified RADV-specific triggers, implemented fixes. Human directed test strategy (run on bt, check each test individually).

### 2026-04-02 — 4-GPU Benchmark Matrix

**What:** Ran matmul benchmarks (64x64 through 1024x1024) on 4 GPUs: AMD RX 5700 XT (Vulkan/RADV), NVIDIA RTX 3070 Laptop (Vulkan), NVIDIA RTX 3050 Ti Laptop (Vulkan), Apple M4 (Metal). Results: 5700 XT hits 69 GFLOPS at 1024x1024 (181x speedup vs CPU). RTX 3070 hits 151 GFLOPS (150x). M4 hits 122 GFLOPS (44x). All with naive (untiled) WGSL shader.
**Commit:** [`1a93e7f`](https://github.com/cochranblock/any-gpu/commit/1a93e7f)
**AI Role:** AI ran benchmarks on all nodes via SSH in parallel, compiled tables. Human directed the benchmark matrix and "include all sizes" approach.

### 2026-04-02 — Sprint 1: wgpu Compute Backend

**What:** Initial implementation: GpuDevice struct wrapping wgpu, GPU auto-discovery via `request_adapter(HighPerformance)`, upload/alloc/read buffer management, 3 WGSL compute shaders (matmul with 16x16 workgroups, elementwise add, elementwise mul), bench example with CPU vs GPU comparison.
**Commit:** [`e1a6d96`](https://github.com/cochranblock/any-gpu/commit/e1a6d96)
**AI Role:** AI wrote all initial code. Human directed the API surface (GpuDevice::gpu(), upload/read pattern) and target (wgpu for vendor-agnostic compute).

---

Part of [The Cochran Block](https://cochranblock.org) — see also [kova](https://github.com/cochranblock/kova), [pixel-forge](https://github.com/cochranblock/pixel-forge)
