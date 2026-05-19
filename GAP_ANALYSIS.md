<!-- Unlicense — cochranblock.org -->

# any-gpu Gap Analysis

*A snapshot of what any-gpu can and cannot do as of 2026-05-19, against the stated goal "models hosted on RAM, any-gpu as the interface so the AMD RX 5700 XT can be used for transformer inference."*

> Method: enumerate every capability needed to ship a usable cross-vendor LLM inference engine, mark each as **shipped** / **partial** / **missing**, and estimate effort to close. Compared to the stated user goal, not to a feature-complete reference like vLLM.

## Snapshot at writing time

| Metric | Value |
|---|---|
| Lines of Rust | 11,028 across src/ |
| WGSL shaders | 55 |
| Tests | 309 (single lib binary, all passing) |
| TRIPLE SIMS gate | **3/3 PASS** (verified 2026-05-19) |
| Public ops on t500 | 39 forward + 7 backward |
| Public types | t500, t501, t502, t503, t506, t507, t511, t534, t538, t539, t540, t544, t545, t546, t547, t548, t550 |
| Verified hardware | AMD RX 5700 XT (RADV/Vulkan), NVIDIA RTX 3070 + 3050 Ti (Vulkan), Apple M4 (Metal) |
| Sprint 7 ladder | **All 7 steps shipped** (S7.1–S7.7) |
| Perf sprint | **P1–P7 complete** |
| Backlog | #1–#21, P1–P7, #9–#12 done; #13, #14, #22–#25 open |

## Gap inventory

### A. Inference runtime gaps

| # | Gap | Status | Notes |
|---|---|---|---|
| A1 | Tokenizer (BPE / SentencePiece) | ✅ **shipped** S7.7 | `t544 Tokenizer`, f775–f779. HF `tokenizers` crate. BPE + SentencePiece. Detokenize via f778/f779. |
| A2 | `Module` graph / `TransformerBlock` / `Linear` / `MultiHeadAttention` | 🟡 **partial** S7.7 | `t545 Module` trait, `t546 Linear`, `t548 CausalLM` (LLaMA-style MHA+GQA). Custom architectures still require hand-wiring. No generic residual / FFN block abstraction beyond LLaMA layout. |
| A3 | Sampler beyond argmax (top-k / top-p / temperature / repetition penalty) | ❌ **missing** | `f671` argmax only. **This is the single remaining blocker** between current state and a usable chat session. Three new WGSL shaders needed (topk reduction, cumulative-sum for topp, multinomial). |
| A4 | Stop-token handling / chat template formatting | ❌ **missing** | Glue layer on top of A1 tokenizer. Required for chat-format models (Llama-2-chat, Qwen-chat). Easy to add after A3. |
| A5 | Pinned-RAM staging + layer paging | ✅ **shipped** S7.4 | `t539 LayerPager`, f768–f770. MAP_WRITE staging buffer, chunked upload. `DEFAULT_STAGE_BYTES = 512 MiB`. Eviction policy: caller-managed (page one layer at a time). |
| A6 | f16 storage type | ✅ **shipped** S7.5 | `t540 GpuBufferF16`, f771–f774. `unpack2x16float` pack/unpack (Naga `enable f16;` not yet supported — tracked gfx-rs/wgpu#4384). 2 f16/u32. |
| A7 | Flash-attention tiled SDPA | ✅ **shipped** P5a/P6 | `f626 SHADER_FUSED_SDPA_W64`: one wave64 per q_row, online softmax, VGPR accumulator. Decode 1q/512kv: 1.1 ms (38× vs naive). Prefill 512q/512kv: 8 ms (17× vs naive). |
| A8 | `any-gpu serve` runtime / CLI | ✅ **shipped** S7.7 | `any-gpu-serve` binary: HTTP server, `POST /generate`, `GET /health`. Clean shutdown. |
| A9 | Stratagems CLI (`info`, `bench`, `train`) | ✅ **shipped** #11 | `any-gpu info` (adapter/backend/subgroup/f16), `any-gpu bench` (LLaMA-7B-scale shapes, GPU-fenced), `any-gpu train subatomic` (3 GPU-trained classifiers). |
| A10 | Streaming output (token-by-token SSE) | 🟡 **partial** | Serve binary exists. SSE framing not yet implemented — `/generate` returns full response, not streamed tokens. |
| A11 | Batched inference (multiple prompts in flight) | ❌ **missing** | No dynamic batching scheduler. Each request serialized. Unscoped. |

**Verdict for section A:** Eight of eleven items shipped. Only A3 (sampler) blocks the headline user goal. A4, A10, A11 are follow-ons.

### B. Op / math gaps

| # | Gap | Status | Blocks |
|---|---|---|---|
| B1 | Backward for LayerNorm (f602) | ❌ **missing** | Training BERT/GPT-2 |
| B2 | Backward for RMSNorm (f603) | ❌ **missing** | Training Llama/Mistral |
| B3 | Backward for embedding lookup (f670) | ❌ **missing** | Training any model with learnable embeddings |
| B4 | Backward for softmax (f620) | ❌ **missing** | Training attention |
| B5 | Backward for causal SDPA (f623/f626) | ❌ **missing** | Training transformers end-to-end |
| B6 | Backward for RoPE (f625) | ❌ **missing** | Training Llama-class models |
| B7 | SwiGLU fused op | 🟡 **composable** | `f692 swish` + `f690 mul` + `f694 matmul` cover it. No fused shader (saves one intermediate buffer). |
| B8 | Grouped Query Attention (GQA) — Q more heads than K/V | ✅ **shipped** S7.7 | `f629 repeat_kv` + `f627 split_heads` + `f628 merge_heads`. Llama-3, Mistral, Qwen2 supported. |
| B9 | Multi-Query Attention (MQA) — single K/V head | ✅ **shipped** S7.7 | Special case of B8 (n_kv_heads=1). f629 handles it. |
| B10 | Sliding-window / local attention | ❌ **missing** | Mistral, Phi |
| B11 | Quantized weights on GPU (Q4 / Q8 / AWQ / GPTQ) | ❌ **missing** | Fitting 13B+ in 8 GB VRAM. Largest single piece of new shader work in the inventory. |
| B12 | bf16/f16 compute (not just storage dequant) | 🟡 **partial** | f16 storage works (t540); compute-in-f16 shaders not yet. f32 accumulator retained. |
| B13 | conv3d | ❌ **missing** | 3D vision transformers (out of scope for LLM goal) |
| B14 | Sparse / longformer attention | ❌ **missing** | Out of scope |
| B15 | INT8 matmul / weight-only quant | ❌ **missing** | Speed boost on int8-capable hardware |
| B16 | Layer-wise dropout / attention dropout | ❌ **missing** | Training regularization |

**Verdict for section B:** Inference for vanilla MHA and GQA models is complete. Llama-2-7B / Mistral-7B / Llama-3 all work. Real 13B+ hosting requires **B11 quantization** (the largest remaining item). Full transformer training requires **B1–B6 backward shaders** (each is one shader + one Op variant).

### C. Performance gaps

| # | Gap | Status | Notes |
|---|---|---|---|
| C1 | Subgroup ops for softmax / RMSNorm reductions | ✅ **shipped** P2/P3 | `SHADER_SOFTMAX_FUSED` (P2), `SHADER_RMS_NORM_FUSED` + `SHADER_LN_NORM_FUSED` (P3). Gate: s509 (has_subgroup). |
| C2 | Register-blocked / wave64 tiled matmul | ✅ **shipped** P1/P4 | Wave64 + 4×4 register blocking: 512² +42%, 1024² +24%. Batch_matmul same upgrade (P4). |
| C3 | Fused RoPE+Q/K split | ❌ **open** | Currently 3 ops: split_heads (f627) → rope (f625) → no re-stack needed. One fused shader saves 2 memory round-trips per attention layer. |
| C4 | KV cache append fused for K and V | ❌ **open** | Currently 2 dispatches (one per key, one per value). One shader with 2 output bindings halves dispatch overhead. |
| C5 | RoPE cos/sin cache (avoid per-thread `pow`) | ❌ **open** | RoPE recomputes cos/sin per thread per call. Precomputed LUT per seq-position would help at large batch. |
| C6 | Flash-attention (eliminating N×N score matrix) | ✅ **shipped** P5a/P6 | f626 SDPA_W64 keeps acc_p in VGPRs; no materialized score buffer. O(B×H×S×D) peak extra memory. |
| C7 | Pipeline cache eviction policy | ❌ **open** | `Mutex<HashMap<u64, Arc<ComputePipeline>>>` is unbounded. No LRU eviction. Memory hygiene, not a perf win. |
| C8 | Vendor-specific matrix accelerators | ❌ **future** | RDNA1 lacks matrix accelerators. RDNA3+, RTX 30+, M-series have them. wgpu doesn't yet expose WMMA/WGMMA equivalents. |

**Verdict for section C:** The headline perf items (subgroup reductions, tiled matmul, flash-attention, GEMV two-pass) are all shipped. Remaining C items are micro-optimizations (C3–C5) and housekeeping (C7).

### D. Developer / API gaps

| # | Gap | Status |
|---|---|---|
| D1 | Tensor type (t502) wired to most ops | 🟡 **partial** — f531–f535 ship in #10; high-arity ops (conv2d 14 params) still tape-level only |
| D2 | Autograd for Sprint 7 ops | ❌ **missing** — same as B1–B6 |
| D3 | Model serialization back to safetensors | ❌ **missing** — inverse of f761; no `save_to_safetensors` helper |
| D4 | NanoSign `save_signed` for safetensors | 🟡 **partial** — f760 reads signed; no signed-write helper |
| D5 | Shape inference / broadcasting in autograd | ❌ **missing** |
| D6 | CPU fallback for ops | ❌ **missing** |
| D7 | Tensor `Debug` / pretty-print | ❌ **missing** |
| D8 | Typed u32 buffer for token IDs | ❌ **missing** — embedding_lookup (f670) currently uses f32-cast-to-int |
| D9 | Multi-GPU dispatch (backlog #24) | ❌ **missing** |
| D10 | Auto-benchmark routing model (backlog #22) | ❌ **missing** |

### E. Ecosystem / distribution gaps

| # | Gap | Status |
|---|---|---|
| E1 | Published on crates.io (backlog #25) | ❌ **not yet** |
| E2 | Python bindings (PyO3) | ❌ **missing** — out of scope for current owner goals |
| E3 | docs.rs documentation | 🟡 **autogenerated only** — not curated |
| E4 | Quick-start tutorial: "load a model, generate text" | ❌ **missing** |
| E5 | Reference benchmark vs llama.cpp / candle | ❌ **missing** |
| E6 | CI workflow (GitHub Actions) | ❌ **unknown** — not seen in repo |
| E7 | CHANGELOG.md | ❌ **missing** |
| E8 | CONTRIBUTING.md | ❌ **missing** |
| E9 | Test-fleet script (bt/lf/gd/M4 — backlog #13) | ❌ **missing** |
| E10 | End-to-end inference example (chat, not just bench) | ❌ **missing** — `nanobyte.rs` is training; no generation walkthrough |

### F. Hardware verification staleness

| # | Gap | Status |
|---|---|---|
| F1 | Re-verify lf (RTX 3070) post Sprint 7 + P1–P7 | ❌ **stale** — last verified pre-Sprint 7 |
| F2 | Re-verify gd (RTX 3050 Ti) post Sprint 7 + P1–P7 | ❌ **stale** |
| F3 | Re-verify Apple M4 post Sprint 7 | ❌ **stale** |
| F4 | Test Intel Iris Xe in isolation (backlog #14) | ❌ **never** |
| F5 | Verify GQA path (f629) on non-RDNA1 hardware | ❌ **never** |

## Critical-path summary (as of 2026-05-19)

The previously six-item critical path has shrunk to **two items** for the stated user goal:

1. **A3 sampler suite** — top-k / top-p / temperature. This is the **only remaining blocker** between today and Mike typing a prompt and getting a coherent response. Everything else (tokenizer, forward pass, KV cache, pager, f16, serve binary) has shipped.

2. **B11 quantization** — to break the 7B ceiling and reach 13B+ class on 8 GB VRAM. This is the *next* hard wall after A3 ships — not the current one.

For Jess's training use case, the critical path remains **B1–B6 backward shaders** (six shaders, each following the existing activation-backward pattern).

## What we should *not* prioritize now

- B14 sparse attention — out of scope.
- B13 conv3d — out of scope for LLM goal.
- D9 multi-GPU — the user has one 5700 XT.
- E2 Python bindings — Rust-first stack.
- A11 batching — follow-on to A3; not needed for single-user chat.

## Open questions for the next sprint

1. **Sampler route**: one combined `SHADER_SAMPLE` with configurable temperature/top-k/top-p uniforms, or three separate shaders (topk, topp, temperature)? Single shader favors pipeline-cache reuse; three shaders are simpler to test individually.
2. **Quantization timeline**: ship A3 first (quick), then decide whether to tackle B11 (AWQ/GPTQ in safetensors) or stay at bf16/f16 with paging. With paging, 7B class at bf16 is reachable. AWQ/GPTQ unlocks 13B+.
3. **Chat template**: implement Llama-2-chat / Qwen-chat templates inline, or defer to caller? Inline is the friendlier default.
4. **test-fleet**: every-push sweep (backlog #13), or sprint-boundary only?

<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
