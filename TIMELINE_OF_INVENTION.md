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

### 2026-05-19 — Sampler Suite, Transformer Backward Shaders, top_k Perf Fix

**What:** Three distinct deliverables shipped in one sprint.

**Nanobyte DDPM diffusion model (`01fc203`):** `examples/nanobyte.rs` — NanoUNet-class DDPM with 3-level encoder/decoder (`CHANNELS=[32,64,64]`, `TIME_DIM=64`, 2 ResBlocks per level, ~1.09M params). GPU-resident via `f734`. Trained on pixel-forge 32×32 RGBA sprites. CLI flags for data, steps, batch, lr, save/resume. PNG sample output. ~2 steps/s on RX 5700 XT.

**A3 Sampler suite (`ee6603a`):** `src/ops/sampler.rs` — four autoregressive sampling ops completing the LLM decode path. `f787 top_k_mask`: per-row top-k gate (k ≤ 128). `f788 top_p_mask`: nucleus cutoff via descending sort + cumsum. `f789 sample_multinomial`: PCG32 inverse-CDF — deterministic by (seed, step, row), TRIPLE SIMS safe. `f790 rep_penalty`: HF convention (pos ÷ penalty, neg × penalty). Temperature via `f553(logits, 1/temp)` upstream. 14 new tests, 309→323.

**B1–B6 Transformer backward shaders (`ee6603a`):** Completed autograd coverage for the full transformer. `f791 layer_norm_backward` (3-pass). `f792 rms_norm_backward` (3-pass). `f793 embedding_backward` (f32 CAS scatter-add via `atomicCompareExchangeWeak` on bitcast u32). `f794 softmax_backward` (dot=Σ(grad·p) per row, then p·(grad−dot)). `f796 rope_backward` (conjugate rotation: negate sin — RoPE is orthogonal). Six new `t504 Op` variants + tape wrappers `f712–f717` + dispatch in `f702`. SDPA backward handled by composing through existing matmul + softmax backward on tape. 8 new tests with finite-difference validation, 323→331.

**P8 top_k_mask 26.6× (`32e5c05`):** Old shader: 255 of 256 threads idle while thread 0 did a serial O(V×k) = 32000×50 = 1.6M-op heap scan. New shader: WG=64 (one wave64). Phase 1 — 64 threads stride-64 scan, each maintains a private min-heap of k values in registers (128 VGPRs). Phase 2 — write 64×128=8192 candidates to 32KB LDS, thread 0 merges. Phase 3 — 64 threads apply mask. b=1 v=32k k=50: 106ms → 4.0ms. Full decode pipeline: 113ms → 11.8ms.

**B7 embedding_backward (`32e5c05`):** `f793` called `clear_buffer(vocab×d_model)` on a freshly allocated buffer — a redundant 512MB GPU write on every backward pass. Root cause: wgpu guarantees zero-init on `create_buffer` per the WebGPU spec. One-line fix removed it. ops_bench corrected to vocab=512 (8MB readback vs 512MB) so PCIe transfer no longer dominates the measured time.
**Commits:** [`01fc203`](https://github.com/cochranblock/any-gpu/commit/01fc203), [`ee6603a`](https://github.com/cochranblock/any-gpu/commit/ee6603a), [`56255b9`](https://github.com/cochranblock/any-gpu/commit/56255b9), [`32e5c05`](https://github.com/cochranblock/any-gpu/commit/32e5c05)
**Verified:** TRIPLE SIMS 3/3 PASS on bt (AMD RX 5700 XT, RADV NAVI10). 331/331 tests pass.
**AI Role:** AI wrote all shaders, Rust wrappers, Op variants, backward dispatch, and benchmark functions. Human directed the sampler pipeline design (PCG for determinism, HF rep-penalty convention), the backward scope (which ops need dedicated backward vs composing through primitives), and the perf investigation cadence (benchmark first, fix what the numbers reveal).

### 2026-05-18 — P7: Two-Pass GEMV for M=1 Decode

**What:** For the single-token decode step (M=1), matrix-vector multiply was using the general GEMM shader. New two-pass GEMV shader (`93321779`): pass 1 — each thread accumulates a partial dot product over a stripe of the K dimension into a local VGPR accumulator; pass 2 — one thread per output row reduces the per-thread partials via a small LDS reduction. Result: decode-path matmul avoids the 32×32 tile machinery entirely, removing unnecessary LDS traffic and thread-idle time for the M=1 case. Part of the ongoing RDNA1 decode-path perf sprint.
**Commit:** [`93321779`](https://github.com/cochranblock/any-gpu/commit/93321779)
**Verified:** 292/292 pass on bt (AMD RX 5700 XT, RADV NAVI10, Mesa 25.0.7).
**AI Role:** AI wrote the two-pass GEMV shader and dispatch wiring. Human directed the decode-path focus and the two-pass accumulation design.

### 2026-05-18 — P6: Wave64 Fused SDPA for Decode

**What:** For the decode (M=1) attention step, the existing fused SDPA shader (`f626`) used one workgroup per output token. New P6 variant (`69c1ed02`): one wavefront of 64 threads per Q row, each thread handles a slice of the K/V sequence and does its dot products with `subgroupAdd` — no LDS needed. Eliminates the inter-thread-group barrier overhead for single-row attention. Combined with the P5 VGPR accumulator, the full decode path now avoids both LDS allocation and multi-kernel submission for the attention block. Backlog updated to mark P5a/P5b/P6 done and P7 queued (`5fc1894a`, `f3fb023b`).
**Commits:** [`69c1ed02`](https://github.com/cochranblock/any-gpu/commit/69c1ed02), [`f3fb023b`](https://github.com/cochranblock/any-gpu/commit/f3fb023b)
**Verified:** 292/292 pass on bt.
**AI Role:** AI wrote the wave64 SDPA variant. Human directed the single-wavefront decode optimization and the subgroupAdd dot-product design.

### 2026-05-18 — P5: SDPA VGPR Accumulator + LayerNorm 2-Barrier Fuse + ops_bench

**What:** Two perf improvements and a benchmark harness (`0c204d86`):

**SDPA VGPR accumulator:** The online-softmax fused SDPA shader (`f626`) previously maintained running max/sum in workgroup LDS. Changed to VGPR-resident accumulation per thread — eliminates the LDS read/write on each K tile step, replacing it with register arithmetic. Significant reduction in LDS bandwidth pressure for long-context decode.

**LayerNorm 2-barrier fuse:** LayerNorm dispatch previously used N barriers across the two-pass (mean then variance) reduction. Fused to 2 barriers total by computing mean and partial variance in the same first pass, then doing a single second-pass correction. Eliminates N-2 barrier synchronizations per layer.

**ops_bench:** New benchmark binary measuring end-to-end dispatch time for key ops at representative sizes (512², 1024², attention sequence lengths). Wired to the BACKLOG P5b item (bench reference, `#26`).
**Commit:** [`0c204d86`](https://github.com/cochranblock/any-gpu/commit/0c204d86)
**Verified:** 292/292 pass on bt.
**AI Role:** AI wrote the VGPR accumulator variant, barrier-fused LayerNorm, and ops_bench binary. Human directed the VGPR-over-LDS strategy and the 2-barrier target.

### 2026-05-17 — Full Op Coverage: 256→292 Tests

**What:** Two coverage rounds closed the gap between "every public op compiles" and "every public op has at least one dedicated test":

**Round 1 (262→275, `edb09113`):** Added tests for `f628` (merge_heads), `f642`, `f645`, `f661`, `f580` tiled matmul, `f621` multi-batch SDPA.

**Round 2 (275→292, `1fe65f76`):** Added tests for `f643`, `f644`, `f646`, 2D dispatch, depthwise conv, `f584` stride-2 conv, `f583` multichannel, `f601` backward pass, `f625` multi-position RoPE. Every public and `pub(crate)` op now has at least one dedicated test.
**Commits:** [`edb09113`](https://github.com/cochranblock/any-gpu/commit/edb09113), [`1fe65f76`](https://github.com/cochranblock/any-gpu/commit/1fe65f76)
**Verified:** 292/292 pass on bt (AMD RX 5700 XT, RADV NAVI10, Mesa 25.0.7).
**AI Role:** AI wrote all new tests. Human directed the "no op without a test" discipline and identified which ops were missing coverage.

### 2026-05-17 — P4: batch_matmul 4×4/wave64 Register Blocking

**What:** Applied the same 4×4 register blocking and wave64 workgroup design from the P1 GEMM matmul to the batched variant (`f641` / `batch_matmul`). Replaces the previous 2×2-blocked batch GEMM with `@workgroup_size(64)` = one wave64 per tile, 16 FMAs per thread per k-step, 32×32 LDS tiles. Test count 260→262 (two new batch-shape tests). This completes the register-blocking series for both unbatched and batched GEMM on RDNA1.
**Commit:** [`8ea2b50a`](https://github.com/cochranblock/any-gpu/commit/8ea2b50a)
**Verified:** 262/262 pass on bt.
**AI Role:** AI ported the wave64 4×4 design to the batch variant. Human directed "same optimization for batch_matmul."

### 2026-05-17 — Matmul 4×4/wave64 + subgroup-fused softmax

**What:** Two GPU performance optimizations that required understanding RDNA1 microarchitecture, not just shader mechanics.

**Matmul (ops/conv.rs — SHADER_MATMUL replaced):** Previous shader used a `@workgroup_size(16, 16)` = 256-thread workgroup (4 wavefronts of 64) with 2×2 register blocking. Each `workgroupBarrier` coordinated across all 4 wavefronts — roughly 100+ cycles per barrier. New shader uses `@workgroup_size(64)` = exactly one wave64 on RDNA1. A `workgroupBarrier` inside a single wavefront degrades to a free intra-wavefront fence. Register blocking upgraded from 2×2 (4 outputs/thread, 2×2 FMA outer product) to 4×4 (16 outputs/thread, 4 A-values × 4 B-values = 16 FMAs per k-step). The 64-thread thread layout is 8×8 within the workgroup; each thread owns output rows `[tr*4..(tr+1)*4]` and cols `[tc*4..(tc+1)*4]`. Tile load: 64 threads × 16 elements = 1024 = 32×32 tile. LDS: 2 × 32×32 × 4B = 8 KB/workgroup (fits 8 per RDNA1 CU at 64 KB LDS). Result: 512² +42% (118 → 168 GFLOPS), 1024² +24% (118 → 146 GFLOPS).

**Subgroup-fused softmax (ops/attention.rs — SHADER_SOFTMAX_FUSED added):** Previous `f620` dispatched two kernels: (1) per-row stats pass (one thread per row, O(cols) sequential loop) then (2) per-element normalize pass. New fused path (selected when `dev.s509 == true`) dispatches one kernel: one 256-thread workgroup per row. Each thread handles `ceil(cols/256)` elements locally, then `subgroupMax` and `subgroupAdd` replace the multi-round LDS tree reduction. On RDNA1 (64-wide wavefronts), 4 subgroups exist within the 256-thread workgroup. Cross-subgroup aggregation uses 8 LDS slots (`wg_scratch`) plus a final `subgroupMax`/`subgroupAdd` over threads 0–3. Eliminates the intermediate stats buffer allocation and inter-kernel submission roundtrip. Overflow-safe: rows > 65535 use `wg.x + wg.y * 65535u` row encoding.

**Device (device.rs):** `s509: bool` added — true when `wgpu::Features::SUBGROUP` is available (confirmed on AMD RX 5700 XT / RADV via `VK_EXT_subgroup_size_control`). `f501` now optionally requests SUBGROUP alongside SHADER_F16.

**Commits:** [`6216f68`](https://github.com/cochranblock/any-gpu/commit/6216f68)
**Verified:** 256/256 pass on bt (AMD RX 5700 XT, RADV NAVI10, Mesa 25.0.7). `dev.s509 = true`. Benchmark: 512² 168 GFLOPS, 1024² 146 GFLOPS.
**AI Role:** AI wrote both shaders and the device feature detection. Human directed "go for the hardest thing" — which AI interpreted as eliminating barrier overhead (wave64 workgroup for matmul) and eliminating multi-dispatch overhead (subgroup reductions for softmax).

### 2026-05-17 — Sprint 7 complete: fused SDPA + tokenizer + LLM inference stack + serve runtime

**What:** Two final Sprint 7 steps shipped together. S7.6: `f626 = scaled_dot_product_attention_fused` — online-softmax fused causal SDPA that never materializes the full N×N score matrix. For a sequence of length N, standard SDPA allocates O(N²) VRAM; f626 streams through K/V in tiles and keeps only running max/sum state — O(N) VRAM. This is what allows long-context inference within the bt node's 8 GB VRAM limit. New WGSL shader, 6 new tests (233→239). S7.7: three new head-manipulation ops — `f627 = split_heads` ([seq, n*hd] → [n, seq, hd]), `f628 = merge_heads` ([n, seq, hd] → [seq, n*hd]), `f629 = repeat_kv` (GQA key/value expansion [n_kv, kv_seq, hd] → [n, kv_seq, hd] by tiling). New `t544 = Tokenizer` (HuggingFace tokenizers crate wrapper, `f775–f779`: load, from_bytes, encode, decode, vocab_size, eos_id). New `t545 = Module` trait + `t546 = Linear` (GPU linear layer — `f780` takes pre-transposed weight, `f781` loads HF [out,in] and transposes at load time to [in,out]). New `t547 = LmConfig` (JSON config parser, `f782`: vocab_size, hidden_size, num_attention_heads, num_key_value_heads, etc.) + `t548 = CausalLM` (LLaMA-compatible forward pass: embedding → N×(RMSNorm+QKV+RoPE+KVCache+SDPA+residual+MLP) → RMSNorm → LM head; `f783` load, `f784` prefill, `f785` decode_one, `f786` generate). New `any-gpu-serve` binary: HTTP server, `POST /generate`, `GET /health`. New deps: `tokenizers = "0.21"`, `serde = "1"`, `serde_json = "1"`, `clap = "4"`. 17 new tests (239→256). All 256 pass on bt.
**Commits:** [`ae8a688`](https://github.com/cochranblock/any-gpu/commit/ae8a688) (S7.6 fused SDPA), [`522a63a`](https://github.com/cochranblock/any-gpu/commit/522a63a) (S7.7 tokenizer + inference stack + serve)
**Verified:** 256/256 pass on bt (AMD RX 5700 XT, RADV/Vulkan, Mesa 25.0.7).
**AI Role:** AI wrote all shaders, dispatch wiring, type definitions, and tests. Human directed the online-softmax memory budget constraint (fit long contexts in 8 GB VRAM), the GQA repeat_kv design, the HF weight transposition convention, and the serve API surface.

### 2026-05-17 — S7.4: LayerPager + S7.5: GpuBufferF16 (233 tests)

**What:** Two hardware-budget steps. S7.4: `t539 = LayerPager` — pinned-RAM staging buffer (`MAP_WRITE | COPY_SRC`) for streaming model weights from system RAM to VRAM one layer at a time. Methods: `f768` new (allocate staging buffer of configurable byte capacity, default 256 MB), `f769` upload (f32 slice → VRAM via staging write + copy), `f770` page_layer (named tensors from SafetensorsModel → HashMap<String, GpuBuffer>), `f773` upload_f16_raw, `f774` page_layer_f16. Purpose: allow models larger than VRAM capacity (8 GB on bt) by streaming layers on demand rather than loading everything at once. 5 new tests (228→233). S7.5: `t540 = GpuBufferF16` — packed f16 storage: 2 f16 elements per u32 (wgpu requires 4-byte alignment). `f771 = GpuDevice::upload_f16` takes &[u16] (raw f16 bits) and packs 2 per u32. `f772 = GpuDevice::f16_to_f32` dispatches a WGSL kernel using `unpack2x16float` to dequantize to a full f32 GpuBuffer. Note: `enable f16;` in WGSL is NOT supported by Naga/wgpu ([gfx-rs/wgpu#4384](https://github.com/gfx-rs/wgpu/issues/4384)) — using `unpack2x16float` instead. This means 2× storage bandwidth for f16 weights loaded from safetensors.
**Commits:** pending (shipped, not yet logged with individual hashes)
**Verified:** 233/233 pass on bt (AMD RX 5700 XT, RADV/Vulkan, Mesa 25.0.7).
**AI Role:** AI wrote the LayerPager type, upload pipeline, and GpuBufferF16 packed storage. Human directed the pinned-staging design (pre-allocate one staging buffer, reuse per layer), the 256 MB default, and the packed-u32 workaround for Naga's lack of f16 support.

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
