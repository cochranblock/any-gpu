<!-- Unlicense — cochranblock.org -->

# any-gpu Breadth-Parallel Build Plan

*How to skip the vertical-slice "MVP first, then scale" path and ship a full-featured cross-vendor LLM inference engine via parallel work, without producing a refactor disaster.*

> Method: enumerate every missing piece needed for a full-featured product (per `GAP_ANALYSIS.md`), identify which can run truly independently versus which couple at the contract boundary, write the contract specs, then describe the parallel dispatch + integration gate.

## The strategic premise

A vertical-slice sprint builds one paper-thin end-to-end model first, then widens. Each merge produces a usable artifact, but breadth comes slowly. A breadth-parallel sprint builds every missing piece simultaneously against pinned contracts, then integrates at the end. Calendar wins are real only if the contracts hold under integration — otherwise you pay both the parallel cost and the refactor cost.

The contracts are the entire risk reduction. **They are written first, by you, before any parallel work begins.** Everything else is execution.

## Phase 1 — pin the contracts (sequential, single agent)

Three interfaces. Once these are committed to a `feat/sprint-7-contracts` branch and merged, every parallel agent below can build against them without coordinating with the others.

### Contract C1 — `t545` (Module forward trait)

The shape that every layer (Linear, MultiHeadAttention, TransformerBlock, Embedding, LayerNorm-as-module, RMSNorm-as-module) implements.

Design constraints:
- `forward` must accept the device by reference (every op routes through `&t500`).
- Must accept input as `t501` (or `t502` once D1 lands — choice has cascading effect; see "Open questions" below).
- Must accept `&mut t534` (KV cache) so attention layers can append during decode.
- Must return `Result<t501, anyhow::Error>` so dispatch errors propagate.
- Weight binding from `t538::f761` happens at construction time, not at forward time — keeps the hot path allocation-free.
- Layers store their own buffers; the trait does not prescribe storage layout.

Spec to commit:

```rust
/// t545 = Module forward trait. Every transformer layer implements this.
pub trait t545 {
    /// f<N> = forward. Input -> output for one layer.
    /// p0 = device. p1 = input. p2 = optional KV cache (None for non-attention layers).
    /// p3 = position offset into the KV cache (0 for prefill, current cursor for decode).
    fn forward(
        &self,
        p0: &t500,
        p1: &t501,
        p2: Option<&mut t534>,
        p3: u32,
    ) -> Result<t501>;
}
```

Why this shape:
- Single-method trait keeps the dyn-vs-generic decision deferrable (start with generic `impl t545 for Linear`, monomorphize per layer type; if dynamic dispatch is needed later, blanket `impl t545 for Box<dyn t545>`).
- Optional KV cache is cleaner than two methods (forward vs forward_with_cache).
- Position offset is explicit so RoPE inside the MultiHeadAttention block knows whether this is prefill (start_pos = 0) or decode (start_pos = cache cursor).

### Contract C2 — `t546` (typed tensor block)

The shape that every weight block (loaded from safetensors) and every intermediate activation uses.

Design constraints:
- Tracks dtype so quantization (B11), f16 storage (A6), and dequant-on-the-fly all compose.
- Tracks shape so debugging and shape-mismatch errors are caught at the type boundary.
- Carries an optional `scales` buffer for quantized formats (AWQ/GPTQ both have per-group scales).
- Carries an optional `zeros` buffer for asymmetric quant.
- Backward compatibility with raw t501 via a `From<t501>` impl that defaults to f32 dtype.

Spec to commit:

```rust
/// t546 = TensorBlock. Typed view over a t501 with dtype + shape + (optional) quant metadata.
pub struct t546 {
    pub(crate) s<N>: t501,        // primary data (always present)
    pub(crate) s<N+1>: Dtype,     // F32, BF16, F16, Q4_AWQ, Q4_GPTQ, Q8_0, INT8, U32, ...
    pub(crate) s<N+2>: Vec<u32>,  // shape (inline, max 6 dims to match t502)
    pub(crate) s<N+3>: u8,        // ndim
    pub(crate) s<N+4>: Option<t501>, // scales (Q-formats)
    pub(crate) s<N+5>: Option<t501>, // zeros (asymmetric Q)
    pub(crate) s<N+6>: u32,       // group_size (Q-formats, e.g., 128 for AWQ)
}

pub enum Dtype {
    F32, BF16, F16, Q4_AWQ, Q4_GPTQ, Q8_0, INT8, U32,
}
```

Why this shape:
- Single struct rather than enum-of-dtype-specific-structs keeps the dispatcher monomorphic.
- Optional scales/zeros mean F32/BF16/F16 blocks pay no overhead; quant blocks carry their metadata.
- Shape inline (no Vec heap allocation per tensor) matches the existing t502 pattern.

### Contract C3 — `t547` (Sampler trait)

The shape that every token-selection strategy implements.

Design constraints:
- Stateful (temperature, top-k, top-p, repetition penalty all need state).
- Deterministic when seeded (TRIPLE SIMS requirement).
- Returns one token id per call (caller drives the loop).
- Reads logits from a t501 in CPU-readable form via `f504`, OR runs entirely on GPU and returns a u32 (latter is faster for batched).

Spec to commit:

```rust
/// t547 = Sampler trait. Pick a token from a logits distribution.
pub trait t547 {
    /// f<N> = sample. p0 = device. p1 = logits [vocab_size]. p2 = previous token ids (for repetition penalty).
    /// Returns the chosen token id.
    fn sample(&mut self, p0: &t500, p1: &t501, p2: &[u32]) -> Result<u32>;
}
```

Why this shape:
- `&mut self` because samplers carry state (RNG counter, repetition-penalty history window).
- Previous token ids as `&[u32]` not as a t501 because repetition penalty is cheap on CPU; if it ever needs GPU, change the signature.
- One method, no separate "warmup" — samplers initialize in their constructor.

### What Phase 1 produces

A single PR that commits `t545`, `t546`, `t547` as empty trait/struct definitions with doc comments, plus an integration test that confirms the trait is `dyn`-safe (if dynamic dispatch is wanted later) and that `t546::from(t501)` works for the f32 default case. Compiles. No behavior change to any existing test.

## Phase 2 — parallel tracks against the pinned contracts

Each track is a self-contained branch that builds against C1/C2/C3 and a slice of the existing crate. The TRIPLE SIMS gate (the `any-gpu-test` binary) is the merge criterion for each — extended to exercise that track's surface.

Tracks are listed in dependency-graph order so the parallel dispatch is obvious. Tracks at the same depth can run concurrently.

### Depth 0 — no dependencies, can launch immediately after Phase 1

#### T1 — Tokenizer integration

Touches: a new module `src/tokenizer.rs` (or `src/tokenizers.rs`), depends on the HuggingFace `tokenizers` crate as a runtime dep. Defines:

- `t<N>` (Tokenizer): wraps `tokenizers::Tokenizer`.
- `f<N>` (load_from_file): reads `tokenizer.json` from a path.
- `f<N>` (encode): `&str -> Vec<u32>`.
- `f<N>` (decode): `&[u32] -> String`.
- `f<N>` (vocab_size): `-> u32`.
- Stop-token handling: per-tokenizer EOS / BOS / pad token ids surfaced.
- Chat template formatting (A4): glue layer that formats `[(role, content)]` into the model's expected prompt string. Initial implementation targets the Llama-2-chat template and the ChatML template (Qwen, openchat).

Test coverage: round-trip "Hello world" through encode → decode; vocab_size matches model config; chat template produces byte-identical output to the model's reference. TRIPLE SIMS extension: encode/decode is pure CPU, so the test asserts byte-equality across three invocations.

Hardness reasons (not duration): tokenizer compatibility quirks vary by model. Llama uses SentencePiece; GPT-2 uses byte-level BPE with leading-space handling; Qwen uses tiktoken. The HuggingFace `tokenizers` crate handles all three but its `Tokenizer` type has subtle behavior around special-token decoding that depends on serialized configuration. Hand-rolling any of these is dramatically harder than borrowing.

#### T3 — Sampler suite

Touches: a new module `src/sampler.rs`. Implements `t547` for:

- `t<N>` GreedySampler: wraps `f671` (existing argmax).
- `t<N>` TemperatureSampler: divides logits by temperature, calls multinomial draw.
- `t<N>` TopKSampler: zeroes all but top-k logits, normalizes, multinomial draw.
- `t<N>` TopPSampler: sorts logits descending, cumulative sum, keep prefix where cumsum < p, multinomial.
- `t<N>` RepetitionPenaltySampler: composable wrapper that applies `penalty^count` to repeated token ids before passing to inner sampler.

Three new WGSL shaders:
- `topk_select`: partial sort returning top-k indices + values (workgroup-shared, then global merge).
- `cumsum_exclusive`: prefix sum over sorted logits (Hillis-Steele scan).
- `multinomial_draw`: given a CDF and a uniform random number, returns the first index where CDF >= random.

Determinism: each sampler carries a counter-based RNG (xoshiro or PCG seeded by the user); TRIPLE SIMS runs three times with the same seed and asserts identical token sequences.

Hardness reasons: top-p sampling requires a sort over the full vocab (50k–200k entries). A full sort per token is expensive; the standard trick is to sort once, then mask in subsequent steps — but cache invalidation when the distribution shifts complicates this. Initial implementation uses a full sort; optimization is a follow-on.

#### T8 — Sprint 7 op autograd

Touches: `src/autograd.rs`, plus new backward shaders in `src/ops/`. New Op enum variants for LayerNorm, RMSNorm, Embedding, Softmax, SDPA, RoPE. Six new shaders (B1–B6 per `GAP_ANALYSIS.md`). Each backward saves any needed forward intermediates onto the tape (e.g., softmax saves its output; SDPA saves the attention probabilities post-softmax).

Test coverage: numeric gradient check per op against a small input. Each is verified by perturbing one input element by epsilon, forward-running, measuring the loss delta, and confirming the analytic backward agrees within `1e-3` (the existing pattern for activation backwards in elementwise.rs).

Hardness reasons: embedding_backward is the trickiest because multiple tokens can index the same row, requiring atomic adds in WGSL. wgpu exposes `@group(0) @binding(N) var<storage, read_write>` plus `atomicAdd` on i32 — the float-atomic-add workaround is the compare-exchange loop, which RDNA1 supports but is slower than a dedicated atomic float (which Vulkan 1.2 has, wgpu 24 does not yet expose). Either accept the compare-exchange path, or accept that embedding_backward is f32-on-i32-bit-pattern atomicAdd, which works for monotone updates but breaks under truly concurrent grad accumulation. The latter is fine for SGD but not for AdamW. This is the only nontrivial backward.

#### T4 — GQA support in causal SDPA

Touches: `src/ops/attention.rs` — modifies `f623` to accept separate `n_q_heads` and `n_kv_heads`. The score-computation step (currently `f581` batch_matmul) needs to broadcast each KV head across multiple Q heads. Cleanest path: a new uniform parameter `kv_head_repeat = n_q_heads / n_kv_heads` and an updated kernel that adjusts the K/V indexing accordingly.

Test coverage: f623 with `n_q_heads = n_kv_heads` (existing case, must still pass), `n_kv_heads = 1` (MQA case, B9), `n_q_heads = 4 * n_kv_heads` (Llama-2-70B case). Hardcoded reference values computed at f64 from the GQA definition.

Hardness reasons: GQA is mathematically a straightforward broadcast over the head dimension. The hardness is that `f623` currently calls `f641` (transpose) and `f581` (batch_matmul), both of which assume one head per batch slot. Decision: either modify f581 to accept a "broadcast factor" parameter (touches every f581 caller), or insert an explicit K/V broadcast op before the batch_matmul (extra memory bandwidth, simpler shader change). Second option is preferred for the first GQA landing; first option is a follow-on optimization.

### Depth 1 — depends on Depth 0

#### T2 — Module graph implementation

Touches: a new module `src/modules.rs` (or `src/nn.rs`). Implements `t545` for:

- `t<N>` Linear: weight matmul + optional bias add.
- `t<N>` LayerNormModule: stores gamma + beta, calls f602.
- `t<N>` RMSNormModule: stores weight, calls f603.
- `t<N>` EmbeddingModule: stores embedding table, calls f670.
- `t<N>` MultiHeadAttention: stores Q/K/V/output projections + a `RoPE` config, splits heads, applies RoPE (f625), runs causal SDPA (f623, now with GQA via T4), reassembles.
- `t<N>` SwiGLU FFN: stores w1, w2, w3, computes `(silu(w1·x) ⊙ (w3·x)) · w2`.
- `t<N>` TransformerBlock: composes LN/RMSNorm → MultiHeadAttention → residual → LN/RMSNorm → SwiGLU FFN → residual.
- `t<N>` Transformer: composes Embedding → [TransformerBlock × n_layers] → final LayerNorm → output projection (often tied to embedding weights).

Weight binding: each module's constructor takes a `&t538` plus a prefix string ("model.layers.0.") and pulls the named tensors out, uploading them to GPU via `t538::f765`. Mismatched names error at construction, not at forward.

Test coverage:
- Per-module: forward against hand-derived hardcoded outputs for very small shapes.
- Composition: TransformerBlock forward against a CPU reference written using existing `cpu_layer_norm` + `cpu_attention` + manual SwiGLU. The CPU side is a fresh implementation (not copy of the shader), so the cross-check is real.
- End-to-end: Transformer forward against the same CPU composition.

Hardness reasons: weight name mapping. HuggingFace safetensors files use varied naming schemes — `model.layers.0.self_attn.q_proj.weight` for Llama, `transformer.h.0.attn.c_attn.weight` for GPT-2, `model.transformer.layers.0.self_attention.query_key_value.weight` for fused Q/K/V layouts. Initial implementation targets one naming scheme (Llama-style) and errors on others; a name-mapping table is a follow-on.

#### T6 — f16 storage type

Touches: requests `wgpu::Features::SHADER_F16` in `t500::f500()`; new dtype variants on `t546` (already specced in C2); pipeline cache key extended to include dtype; every shader that reads weights gets an f16 variant. Initial scope: only the matmul (f580, f581) and the embedding gather (f670) get f16 variants — these dominate the model size. LayerNorm/RMSNorm/RoPE/softmax/SDPA stay f32 because they're small relative to matmul.

A fallback path is required for hardware without `SHADER_F16` (some integrated GPUs and older drivers). The fallback unpacks two f16 values from a u32 in the shader: `vec2<f32>(unpack2x16float(packed))`. This is portable WGSL but slightly slower than native f16.

Test coverage: load a known f16 safetensors tensor, perform matmul against an f32 vector, compare against the f32-only path within `5e-3` (the f16 rounding error budget for moderate matrix sizes).

Hardness reasons: numerical stability across the f32 vs f16 path comparison is fiddly. The accumulator stays f32 even when inputs are f16, which is the standard mixed-precision pattern, but small loss of precision at the bias-add step can drift results. Test tolerance has to be set per-shape and per-input-distribution; arbitrary tolerance hides bugs.

#### T5 — Pinned RAM staging + layer paging

Touches: a new module `src/paging.rs`. Allocates a pinned host buffer via `wgpu::BufferDescriptor` with `usage: COPY_SRC | MAP_WRITE`. Provides:

- `t<N>` LayerPager: holds a `t538` (CPU-resident weights) and tracks which layer's weights are currently in VRAM.
- `f<N>` page_in_layer: copies a named layer's weights from CPU staging buffer into the GPU-resident layer buffer. Uses `queue.write_buffer` for the copy.
- `f<N>` prefetch_layer: starts the copy for the next layer asynchronously while the current layer is computing (double-buffering).

Initial scope: synchronous page-in only (no prefetch). Prefetch is a follow-on once we have benchmarks showing the copy-vs-compute overlap matters.

Test coverage: load a tiny 2-layer transformer (T2's Transformer type), allocate VRAM for only 1 layer's weights, page through both layers during forward, compare against a baseline that holds both layers in VRAM. Outputs must be identical to bit precision.

Hardness reasons: wgpu's async submission model isn't fully exposed for double-buffered prefetch. The crate has `queue.submit` which returns immediately but actual completion happens later; the timing semantics for "copy completes before compute starts" rely on submission order within a queue. Validated through Vulkan synchronization primitives that wgpu inserts automatically — investigation needed to confirm RDNA1 driver respects them. Initial sync-only implementation sidesteps this; prefetch is the optimization.

#### T7 — `any-gpu serve` runtime / CLI

Touches: a new binary `src/bin/any-gpu.rs` with subcommands (`serve`, `chat`, `bench`, `info`). Depends on `tiny_http = "0.12"` (no async runtime, no transitive deps to speak of) for the HTTP server, and `argh` for argument parsing. The serve subcommand exposes a single endpoint `/v1/chat/completions` (OpenAI-compatible request format).

Wiring: takes a `--model PATH` flag pointing at a safetensors file; loads via `t538::f760`; constructs a `Transformer` via T2; constructs a sampler via T3 (configurable per-request via `temperature`, `top_p`, `top_k` JSON fields); runs the autoregressive loop with KV cache; streams tokens via SSE.

Test coverage: integration test that POSTs a fake prompt at `localhost:<port>/v1/chat/completions` and asserts the response is valid SSE with at least one `data:` line ending in a JSON-encoded chunk. The actual model output isn't asserted (depends on the weights); only the protocol shape.

Hardness reasons: SSE framing is finicky (newline handling, the `data: ` prefix, the terminating `\n\n`). OpenAI-compatible streaming uses a specific format that vLLM and llama.cpp both follow; the test fixture comes from one of those reference outputs.

This track depends on T1 (tokenizer), T2 (Module graph), T3 (sampler), and T5 (paging — needed for any non-trivial model size). T7 cannot land until all four are merged.

### Tracks deferred to a follow-on sprint

- **B11 — quantization on GPU (AWQ, GPTQ, Q4_K)**: the largest single piece of new shader work. Each quant format has a distinct unpack pattern; AWQ uses group-wise asymmetric scales; GPTQ uses different group layout; Q4_K (llama.cpp-style, not in scope per the safetensors-only mandate) has its own. Initial sprint targets bf16/f16 weights only — that's 7B-class on the 5700 XT with paging. Quantization unlocks 13B+ and is the natural sprint-after-this-sprint.
- **A11 — batched inference (continuous batching)**: requires per-request KV cache slot management in a shared `t534`, attention mask handling for padded batches (extends f624), and a scheduler. Out of scope for the first serve binary; defer until throughput measurement justifies it.
- **D9 — multi-GPU**: out of scope, single-device for the headline persona.
- **C6 — flash attention**: large fused kernel work, independent of all other tracks. Can be a parallel track in this sprint if the worktree budget allows, but increases the integration-gate risk. Recommend deferring unless seq_len > 2048 is required for the target model. Llama-2-7B is fine at the current SDPA up to seq_len ≈ 1024 on 8 GB VRAM.

## Phase 3 — integration gate

A single integration agent (or you) merges the parallel branches in this strict order, running `cargo test --release --lib` after each:

1. **Contracts (Phase 1)** — already on `main`.
2. **T8 (Sprint 7 backwards)** — pure additive, no impact on existing surface.
3. **T4 (GQA)** — modifies `f623` signature. Existing `f623` callsites need updating (the in-crate tests + `examples/decode_step2.rs`). Trivial wins by inserting `1` for `n_q_heads = n_kv_heads`.
4. **T6 (f16 storage)** — touches `t501` indirectly via `t546` (per C2). Existing tests stay f32 by default.
5. **T1 (tokenizer)** — pure additive, depends on nothing else above.
6. **T3 (sampler)** — pure additive.
7. **T2 (Module graph)** — depends on T4 (GQA) for MultiHeadAttention.
8. **T5 (paging)** — depends on T6 (f16 storage) for the dtype-aware copy.
9. **T7 (serve)** — depends on T1, T2, T3, T5 all merged.

After merge of T7, the integration gate runs:

- `cargo test --release --lib` — every prior test plus the new ones from each track. Number must be **strictly greater** than the prior 221, and **zero failures**.
- `cargo run --release --bin any-gpu-test --features tests` — TRIPLE SIMS must stay 3/3. The smoke test inside the gate is extended to exercise: load a tiny safetensors file → tokenize a prompt → run forward → sample → decode. Three back-to-back runs with the same seed must produce byte-identical output strings.
- A new example, `examples/decode_llama_tiny.rs`, hand-writes a 2-layer GPT-2-style transformer using the new `t545` Module trait, pulls weights from a fixture safetensors file (built in the test setup), tokenizes a prompt, generates 10 tokens, and asserts the output starts with a known prefix. **This example is the only gate that proves the abstractions compose into a deliverable.**

If `decode_llama_tiny.rs` fails or doesn't exist, the abstractions did not survive integration. Refactor pass goes here: the contracts in Phase 1 are revised, the affected tracks are rebuilt against the new contracts, the gate re-runs.

## Risks unique to the breadth-parallel path

| Risk | Mitigation |
|---|---|
| Contract C1 (Module trait) shape wrong → every Module rewrites | Phase 1 spec includes the `decode_llama_tiny.rs` skeleton (commented-out, no Module impl yet); confirm trait signature lets the skeleton compile *before* dispatching parallel work. |
| Contract C2 (TensorBlock) dtype enum misses a needed variant → quant-format track can't land | Initial enum covers F32/BF16/F16 only. Quant variants added in the follow-on sprint when scoped. |
| T4 GQA changes `f623` signature → every f623 caller breaks | Existing callers (tests + `decode_step2`) are updated as part of T4's PR. Integration merge order puts T4 before T2 to surface this early. |
| T8 embedding_backward atomicAdd-on-float racy → grads incorrect under concurrent updates | Initial implementation uses compare-exchange loop (correct but slower). Test exercises a single-batch case; concurrent training is gated on a follow-on. |
| T5 paging double-buffering doesn't actually overlap → no perf win, only correctness | Initial implementation is synchronous page-in only. Test confirms correctness. Async prefetch is a follow-on after measuring. |
| T6 f16 fallback path produces drift > tolerance → bf16-only models become unreliable | Test tolerance set per-shape, per-input. CI runs both `SHADER_F16`-on and `SHADER_F16`-off configurations (gated by feature). |
| T7 serve uses `tiny_http` blocking I/O → can't handle concurrent requests | Initial scope is single-request-at-a-time. Concurrent serving is a follow-on (depends on A11 batched inference). |
| Agents in parallel worktrees produce conflicting test names | Compression map (`docs/compression_map.md`) extended in Phase 1 to reserve number ranges per track: T1 uses f<800-819>, T2 uses f<820-879>, T3 uses f<880-899>, T4 reuses existing f623, T5 uses f<900-919>, T6 uses f<920-939>, T7 uses f<940-959>, T8 uses f<960-999>. No two agents touch the same number. |

## Decisions you (the user) need to make before Phase 1

These are not implementation choices — they're scope choices that change the contracts.

1. **Quantization in this sprint or after?** Including AWQ/GPTQ extends C2 with quant-aware variants and pulls in B11 shader work. **Default: defer to follow-on sprint.** First sprint targets bf16/f16 only; ceiling is 7B-class on the 5700 XT.
2. **Module trait monomorphic or dyn-dispatched?** Monomorphic is faster and matches kova/cochranblock style; dyn-dispatched is more flexible for runtime-loaded model configs. **Default: monomorphic with a `Box<dyn t545>` blanket impl available if needed later.**
3. **Tokenizer crate or hand-roll?** Per A1 in `GAP_ANALYSIS.md`, hand-rolling a correct BPE for three families (Llama/GPT-2/Qwen) is a separate undertaking. **Default: borrow the HuggingFace `tokenizers` crate; the supply-chain cost is one well-maintained dep.**
4. **English-name aliases?** Currently disallowed by your "no English aliases" directive. Reconfirming for this sprint: do parallel tracks ship in tokenized form only, or is there a `--features english-api` for downstream consumers? **Default: tokenized only, per your prior direction.**
5. **Test-fleet sweep after integration?** The README claims "verified on 4 GPUs"; that claim is currently stale for Sprint 7 ops. After the breadth-parallel sprint lands, should the integration gate include SSH'ing tests to lf (RTX 3070), gd (RTX 3050 Ti), and the M4? **Default: yes, gated as a separate CI step that must pass before tagging a release.**

## What this plan deliberately does not promise

- A specific calendar date.
- Per-track cost in time units.
- A guarantee that the integration gate passes on the first attempt — the contracts in Phase 1 are designed to minimize refactor cost, not eliminate it.
- That every model architecture works out of the box. Initial scope is GPT-2-style and Llama-2-style dense-attention models; non-standard architectures (MoE, Mamba, vision-language hybrids) are out of scope.

The integration gate (`decode_llama_tiny.rs` running end-to-end on the 5700 XT, TRIPLE SIMS 3/3, ≥221 lib tests passing) is the only definition of "done" for this sprint.

<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
