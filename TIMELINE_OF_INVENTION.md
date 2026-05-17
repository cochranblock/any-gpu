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

### 2026-05-16 — Self-Licking Test Audit + Hardcoded-Reference Backstops

**What:** Full audit of every test that compares GPU output against a `cpu_*` helper, looking for cases where the CPU "reference" implements the *same formula* as the shader (a self-licking-ice-cream cone — both could share a bug undetectably). Found 5 ops where the CPU helper and shader use literally identical math: sigmoid (f555), silu (f556), rms_norm (f603), softmax (f620), SDPA (f621). For each, added a new test with hardcoded reference values computed at f64 precision from the math definition or from PyTorch / IEEE specs — **independent of any code we wrote**, so a typo or wrong-formula bug in the shared expression now surfaces. 7 new tests: `f555_known_reference_values` (sigmoid table from `1/(1+e^-x)` at -10, -2, -1, 0, 1, 2, 10), `f556_known_reference_values` (SiLU table), `f603_known_analytical_values` (RMSNorm of [3,4] → analytical `[3,4]/sqrt(12.5)`), `f603_known_with_per_col_weight` (RMSNorm of all-ones → identity = weight), `f620_known_uniform_input` (softmax([0,0,0]) = [1/3, 1/3, 1/3] analytical), `f620_known_binary_input` (softmax([0,1]) matches sigmoid relation), `f621_known_uniform_attention` (Q=K=0 → uniform attention → mean(V) per row, hand-derived). Existing `cpu_*` helpers retained — they're still valuable for edge-case coverage at scale, just no longer the only source of truth. **221/221 tests pass, TRIPLE SIMS gate still 3/3 on the 5700 XT.**

Also reconfirmed that ops which already had hardcoded backstops — gelu (PyTorch), layer_norm (`f602_unit_affine` analytical `[-1,1,-1,1]`), group_norm (`test_group_norm_per_channel`), matmul (`f580_2x2` matching the README example `[19,22,43,50]`), conv2d (`f582_1x1_kernel` per-channel mixing + `f582_stride2` stride sampling), causal mask (hardcoded triangular bit patterns), causal SDPA (`f623_first_row_only_sees_first_kv` invariant: row 0 must equal V[0] under mask), RoPE (cos(1)/sin(1) hardcoded), KV cache (slot layout), embedding/argmax (hand-derived), bf16/f16 (IEEE spec tables), mse_loss (hardcoded 0.25, 14/3), tanh (uses Rust stdlib's `f32::tanh()` which is an independent implementation) — and noted these in audit but did not modify.
**Commits:** pending
**Verified:** 221/221 pass on bt (AMD RX 5700 XT, RADV/Vulkan, Mesa 25.0.7); TRIPLE SIMS 3/3 still green.
**AI Role:** AI performed the audit, identified the 5 self-licking ops, computed independent reference values from math definitions / PyTorch / IEEE specs, wrote and validated the 7 new tests. Human requested the audit framed as "no self-licking ice cream cones for any-gpu" — matching the kova cursor plan's existing discipline.

### 2026-05-16 — Safetensors Loader (Sprint 7, step 3)

**What:** First step at letting any-gpu load real models off disk/RAM. New `src/safetensors.rs` module exposing `t538 = SafetensorsModel` — a parsed safetensors file with weights resident in CPU RAM, dequantized to f32 once at load time. Methods: `f760` load-from-path (NanoSign-aware: wraps `nanosign::f746`, so signed cochranblock files are integrity-verified and HuggingFace-format unsigned files pass through unchanged), `f761` from-bytes (in-memory path), `f762` names, `f763` shape, `f764` data (CPU f32 slice), `f765` upload (RAM → GPU as t501). Free functions: `f766 = bf16_to_f32` (one-line bit shift since bf16 IS the top half of f32) and `f767 = f16_to_f32` (full IEEE 754 binary16 decode handling normals, denormals, zeros, infinities, and NaNs). Dependency added: `safetensors = "0.4"` (official HuggingFace crate, chosen over hand-rolling JSON header parsing). Dev-dep added: `tempfile = "3"` for the on-disk round-trip test. **19 new tests, 214 total.** Tests cover: bf16/f16 known-value tables (hand-derived from the IEEE spec), f32/bf16/f16 in-memory round-trip via the safetensors crate's serializer, multi-tensor file with different shapes, unsupported-dtype error path, GPU upload round-trip, on-disk unsigned load, on-disk NanoSign-signed load, on-disk tampered-signature rejection. Per the project's safetensors-only stance (saved to memory as `feedback-safetensors-only`), the loader explicitly rejects every dtype except F32/BF16/F16 with a message naming the dtype.
**Commits:** pending
**Verified:** 214/214 pass on bt (AMD RX 5700 XT, RADV/Vulkan, Mesa 25.0.7). Example `decode_step2` still builds clean against the public API.
**AI Role:** AI wrote the loader, dequant helpers, NanoSign integration, and tests. Human directed the safetensors-only constraint, the "RAM-first / GPU on demand" design (rather than the obvious "load everything to VRAM at load time"), and the prohibition on GGUF / PyTorch .bin / ONNX even as fallbacks.

### 2026-05-15 — Causal SDPA + RoPE + KV Cache (Sprint 7, step 2)

**What:** Three new GPU primitives plus a stateful KV cache type — together this is the autoregressive-decoding backbone. (1) `f624 = apply_causal_mask` — in-place WGSL shader that sets `scores[bh, i, j] = -1e30` for `j > i + (kv_seq_len - q_seq_len)`. Supports asymmetric `q_seq_len < kv_seq_len` so the same op handles prefill (standard triangular), partial-prefill (offset triangular), and decode (no mask). (2) `f623 = scaled_dot_product_attention_causal` — wraps `transpose → batch_matmul → scale → mask → softmax → batch_matmul` with the asymmetric Q/KV seq-len shape, the exact shape a transformer needs for both prefill and 1-token decode. (3) `f625 = rope` — rotary position embeddings on `[batch_heads, seq_len, head_dim]` with adjacent-pair rotation (Llama/Mistral/Qwen/Gemma convention), parameterized by `start_pos` (for KV-cache decode steps) and `base` (10000 default). One WGSL shader, one thread per output scalar. (4) `t534 = KVCache` — append-only persistent K and V buffers with cursor tracking. `f672 = new`, `f673 = append` (strided-write WGSL shader), `f674 = reset`, `f675 = cursor`, `f676/f677 = k_buffer/v_buffer` accessors. 3 new WGSL shaders, 21 new tests (195 total). RoPE validated against hand-derived cos(1)/sin(1) reference values plus an orthogonality-invariant check; causal mask validated against hand-derived triangular bit patterns; KV cache verified end-to-end with prefill→decode→causal-SDPA round-trip against a CPU reference.
**Commits:** pending
**Verified:** 195/195 pass on bt (AMD RX 5700 XT, RADV/Vulkan, Mesa 25.0.7).
**AI Role:** AI wrote shaders, dispatch wiring, KVCache type, CPU references, and tests. Human directed the priority order (Sprint 7 ladder), the asymmetric q/kv seq-len design so one op handles both prefill and decode, and the safetensors-only stance for the next step (model loading).

### 2026-05-15 — Transformer Inference Primitives (Sprint 7, step 1)

**What:** Five new forward ops needed for LLM inference on any-gpu — the math gap between "general tensor engine" and "can run a hand-coded transformer forward pass." LayerNorm (two-pass: per-row mean/inv_std → normalize + affine), RMSNorm (two-pass: per-row inv_rms → scale + per-col weight; Llama/Mistral/Qwen/Gemma standard), GELU (tanh approximation as used by GPT-2/BERT/ViT and PyTorch `approximate="tanh"`), embedding_lookup (gather rows from `[vocab, d_model]` weights by token id; ids passed as f32 since vocab ≤ 2^24), argmax along last dim (LM-head sampler, tie-break to lowest index). 7 new WGSL shaders, 29 new tests (174 total). New file `src/ops/transformer.rs`. GELU validated against hard-coded PyTorch reference values (gelu(0)=0, gelu(1)≈0.84119, gelu(2)≈1.95460) instead of a CPU re-implementation of the same formula — explicit avoidance of the self-licking-ice-cream pattern called out in `~/.cursor/plans/triple_sims_all_work_no_self_licking_ice_cream.plan.md`.
**Commits:** pending
**Verified:** 174/174 pass on bt (AMD RX 5700 XT, RADV/Vulkan, Mesa 25.0.7).
**AI Role:** AI wrote shaders, dispatch wiring, CPU references, and tests. Human directed the priority order (Sprint 7 ladder: math ops → KV cache → safetensors loader → RAM staging → f16 → flash-attention → server) and called out the self-licking concern that drove the PyTorch-reference GELU test.

### 2026-04-09 — Autograd, Training Loop, Pipeline Caching

**What:** Built the full training stack in one sprint. Tiled matmul (16x16 shared memory tiles for matmul and batch_matmul). Tensor type — shaped view over GpuBuffer, max 6 dims inline on the stack, no heap allocation. Reverse-mode autograd with flat tape and enum ops (666 lines, 13 differentiable ops: add, sub, mul, scale, relu, sigmoid, swish, tanh, matmul, mse_loss, conv2d, plus backward for all). 7 new WGSL backward shaders: relu_backward, sigmoid_backward, swish_backward, tanh_backward, conv2d_grad_weight, conv2d_grad_bias, plus grad_input via conv_transpose2d. AdamW optimizer — single WGSL shader, in-place weight update with momentum, velocity, weight decay, bias correction. Training loop: forward + backward + optimizer in one `train_step()` call. Pipeline caching: compile each WGSL shader once, reuse `Arc<ComputePipeline>` on every dispatch via `HashMap<u64, Arc<ComputePipeline>>`. Added 83 new tests (62→145): device roundtrips, tensor shape ops, autograd backward correctness, optimizer convergence, conv2d gradient numerics. Total: 4,393 lines of Rust across 13 source files, 27 WGSL compute shaders, 145 tests all passing on Apple M4.
**Commits:** [`0ca243d`](https://github.com/cochranblock/any-gpu/commit/0ca243d) (tiled matmul), [`dd55772`](https://github.com/cochranblock/any-gpu/commit/dd55772) (Tensor), [`5137d40`](https://github.com/cochranblock/any-gpu/commit/5137d40) (autograd), [`c09b0ee`](https://github.com/cochranblock/any-gpu/commit/c09b0ee) (AdamW), [`9511a61`](https://github.com/cochranblock/any-gpu/commit/9511a61) (train_step), [`9f0b567`](https://github.com/cochranblock/any-gpu/commit/9f0b567) (conv2d backward), [`6d93866`](https://github.com/cochranblock/any-gpu/commit/6d93866) (tests), [`a905cd1`](https://github.com/cochranblock/any-gpu/commit/a905cd1) (pipeline caching)
**AI Role:** AI wrote all shaders, autograd tape, optimizer, and training loop code. Human directed the architecture (flat tape not trait-object graph, enum ops, inline dims), the training API (single train_step function), and pipeline caching strategy (hash shader source, return Arc).

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
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
