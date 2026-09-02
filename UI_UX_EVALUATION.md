<!-- Unlicense — cochranblock.org -->

# any-gpu UI/UX Evaluation

*Honest critique of what it feels like to use any-gpu as a downstream developer. "UI" here = the public Rust API surface + README + error messages + examples.*

> Method: read the README, scan the public surface (`pub` items), trace what a real user would type to do common tasks, and list the friction points. Compared against PyTorch / Candle / wgpu / llama.cpp as ergonomic baselines.

## Scoring rubric

- **🟢 strong** — at parity with or better than the comparison ecosystem.
- **🟡 friction** — works, but the user pays a tax.
- **🔴 blocking** — the user is likely to give up or get it wrong.

## What's good (🟢 strong)

| Area | Why |
|---|---|
| Cross-vendor abstraction | One `cargo build` produces a binary that runs on AMD, NVIDIA, Intel, Apple. No `--features cuda` flags at the call site. |
| Init story | `t500::f500()` returns a ready device. wgpu auto-picks the backend. Zero config. |
| Test discipline | TRIPLE SIMS gate, hardcoded reference backstops, audited 2026-05-16 for no self-licking — exemplary for a small crate. |
| Error messages with context | `anyhow::Context` is used for most fallible paths; messages name the operation and the offending values. |
| Hardware honesty in README | Tables show speedup AND the head-to-head loss vs CUDA / MPS. Reproduction commands for every claim. Refreshing. |
| RADV/RDNA1 footguns documented | The `arrayLength()` ban, the LazyLock-shared-instance fix, the `adapter.limits()` workaround — all in code comments + README. |
| Pipeline caching is transparent | First call compiles, subsequent calls hit the `Arc<ComputePipeline>` map. Users don't think about it. Measurable: TRIPLE SIMS pass-1 96 ms → pass-2 4 ms. |
| 6 runtime deps | Minimal supply chain. Reads + audits in a sitting. |

## Friction points (🟡)

### F1. Tokenized public names without an IDE-friendly alias

```rust
let v3 = dev.f623(&q, &k, &v, 1, 4, 4, 4)?;
```

A reader has to consult `docs/compression_map.md` to know `f623 = scaled_dot_product_attention_causal`. Doc comments help when you hover, but the function-listing autocomplete shows `f500..f767` — sequential numerals, no semantic clustering. Compare:

```rust
// Candle
let out = candle_nn::ops::softmax(&scores, D::Minus1)?;
// any-gpu
let v0 = dev.f620(&v_scores, p_rows, p_cols)?;
```

The any-gpu form is shorter at the call site but **opaque on first read**. Mitigation: every fN has a `/// fN = human_name` doc comment. Real fix: an optional English-aliases feature flag (`pub use`). Today: there is no such flag.

### F2. Positional integer arguments for shape-bearing ops

```rust
dev.f623(&q, &k, &v, 1, 4, 4, 4)?;
//                   ^batch_heads, q_seq_len, kv_seq_len, d_k
```

Four integer positions is the danger zone — easy to swap `q_seq_len` and `kv_seq_len`, especially for the asymmetric decode case where they're different. A builder or named-args pattern would catch this at compile time:

```rust
dev.attention_causal()
    .q(&q).k(&k).v(&v)
    .batch_heads(1).q_seq(4).kv_seq(4).d_k(4)
    .run()?;
```

`f582` (conv2d) takes **14 positional args** (`p3..p13`). That's a typing-error magnet. Same builder pattern would help.

### F3. `t501` (GpuBuffer) is a flat f32 vector — shape lives in the caller's head

`t501` only tracks `len` (s507). The user mentally remembers `[batch, heads, seq, d_k]`. A wrong-shape buffer compiles fine and crashes at dispatch with a runtime `ensure!` panic.

`t502` (Tensor) **does** track shape but most ops take `t501`. Backlog #10 ("Wire Tensor to ops") is the planned fix. Until then, every Sprint 7 demo includes manual `dev.f502(...)` calls that detach the shape.

### F4. No `Module` graph

To wire one transformer block (LayerNorm → Q/K/V projection → RoPE → causal SDPA → output projection → residual → FFN), the user writes ~30 calls. PyTorch / Candle let you compose `Module` types:

```rust
// PyTorch
let block = TransformerBlock::new(d_model, n_heads, ffn_dim);
let out = block.forward(x);
// any-gpu (today)
let h_norm = dev.f602(&x, &w_ln, &b_ln, rows, cols, eps)?;
let q = dev.f580(&h_norm, &w_q, ...)?;
let k = dev.f580(&h_norm, &w_k, ...)?;
let v = dev.f580(&h_norm, &w_v, ...)?;
let q_rot = dev.f625(&q, ...)?;
let k_rot = dev.f625(&k, ...)?;
let attn = dev.f623(&q_rot, &k_rot, &v, ...)?;
// ... 20+ more lines
```

This is the **single biggest UX gap** for the stated goal. See [GAP_ANALYSIS.md](GAP_ANALYSIS.md) A2.

### F5. Hand-managed GPU buffer lifetimes

No automatic eviction. Every intermediate `t501` lives until dropped. For a long forward pass this can fragment VRAM. PyTorch hides this with an arena/caching allocator; wgpu / any-gpu doesn't. For now, the user manually scopes buffers, e.g.:

```rust
let attn = {
    let q_rot = dev.f625(&q, ...)?;
    let k_rot = dev.f625(&k, ...)?;
    dev.f623(&q_rot, &k_rot, &v, ...)?
};  // q_rot, k_rot dropped here
```

### F6. Token IDs stored as f32

```rust
let ids_f32: Vec<f32> = ids_u32.iter().map(|&i| i as f32).collect();
let ids_buf = dev.f502(&ids_f32);
dev.f670(&ids_buf, &weights, n_ids, vocab, d_model)?;
```

Works correctly for `vocab ≤ 2^24 = 16.7M` (every published tokenizer). But the leaky abstraction surfaces in error messages and tests. A typed `GpuBufferU32` would fix it; not on the backlog yet.

### F7. No `Debug` / pretty-print for t501 / t502

```rust
println!("{tensor:?}");  // doesn't compile
let v = dev.f504(&tensor)?;
println!("{v:?}");  // works, but flat Vec<f32> ignoring shape
```

A `Debug` impl that prints shape + a head-and-tail slice would help debugging.

### F8. Error messages partially use tokens, partially use names

The norm ensure! says `"rms_norm: input size mismatch"` (good — human name). The KV cache ensure! says `"KVCache::append: cursor {} + new {} exceeds max_seq {}"` (good). But some internal helpers fall back to terse text. Consistency would help. Adding the offending tensor's `s507` (len) and the expected shape would help even more.

### F9. The README opens with brand boilerplate, not a quickstart

The first ~25 lines of `README.md` are the cochranblock brand header + license badges + "Why this exists" + bullet list. The first code example is at ~line 50. A reader looking for "how do I use this" has to scroll. Compare crates.io top-100 READMEs: most show a 5-line install + use snippet in the first viewport.

### F10. Single matmul example, no full-model walkthrough

The README's only Rust example is `dev.f580(&a, &b, 2, 2, 2)?` returning `[19, 22, 43, 50]`. There's no "load a model and generate text" example. The reader has to infer the full stack from the BACKLOG / TIMELINE / PROOF_OF_ARTIFACTS docs.

### F11. Cargo features aren't documented in Cargo.toml comments

```toml
[features]
default = []
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
candle-bench = ["candle-core"]
tests = ["dep:exopack"]
```

What does each gate? `cuda` and `metal` only affect `candle_bench.rs` (it's a comparison benchmark using candle, not any-gpu). `tests` gates the exopack TRIPLE SIMS binary. A reader can't tell without grep. Inline comments would help.

### F12. The `any-gpu-test` binary uses `f60` (kova-compressed) name

```rust
let v0 = pollster::block_on(exopack::triple_sims::f60(|| async { run_once() }));
```

This is because exopack 0.2.1 only exposes the compressed name. The 0.3 line adds `run`. The comment in `src/bin/any-gpu-test.rs` flags this — minor friction, but real if a downstream user copies this pattern.

## Blocking issues (🔴)

### B1. No path from text to model output

The user **cannot** do this today:

```rust
let model = t538::f760(Path::new("llama.safetensors"))?;
let tokens = model.tokenize("Hello")?;   // tokenizer doesn't exist
let out = model.generate(tokens, 50)?;   // model.generate doesn't exist
let text = model.detokenize(out)?;       // detokenizer doesn't exist
println!("{text}");
```

Every line above is missing. Only `f760` exists. The README's claim that any-gpu is "the interface for hosting transformer models on RAM" is currently **not deliverable** at the API level — the foundation ops are there, but the connective tissue (tokenizer, module graph, sampler, run-loop) isn't.

### B2. No way to load Llama-2-7B even with the safetensors loader

Llama-2-7B uses **Grouped Query Attention** (Q has 32 heads, K/V has 32 heads → 32, but 70B has GQA 64-Q / 8-KV). any-gpu's `f623` assumes Q/K/V share the same `batch_heads`. A user trying to load a Llama-2-70B checkpoint and run inference hits this wall after step 3.

For Llama-2-7B (no GQA), the wall is instead at "manual layer wiring without a Module graph" — solvable, but ~200 lines of boilerplate per inference.

### B3. The published API uses tokens; the README example uses tokens; the on-disk docs explain tokens

For a first-time visitor to the repo (e.g., a user evaluating "should I use any-gpu?"), the entire surface is in compressed form. There's no "English face" of the crate. The kova compression discipline is consistent (good) but means the discovery experience is opaque (less good). PyTorch and Candle don't have this barrier.

Mitigation: the doc comments include the human name (`/// f580 = matmul`). But you have to be in an editor with hover-doc to see them. The README's example block doesn't show what `f580` is.

## Comparison table

| Concern | PyTorch | Candle | llama.cpp | any-gpu |
|---|---|---|---|---|
| Cross-vendor (AMD + NV + Apple + Intel) | requires ROCm/CUDA/MPS per backend | partial (cuda/metal features) | Vulkan/Metal/CUDA backends | ✅ one binary, wgpu |
| Public API readability | excellent | good | good (C) | 🟡 tokenized |
| Module graph | excellent (`nn.Module`) | good | structured | ❌ none |
| Tokenizer included | external (`tokenizers`) | external | built-in | ❌ none |
| Sampler | excellent | good | excellent | 🟡 argmax only |
| Serve / CLI | external (`torchserve`) | external | `llama-server` | ❌ none |
| Quantization | external (`bitsandbytes`, AWQ) | partial | excellent (GGUF Q-class) | ❌ none |
| Distinct binary per backend | yes | yes | yes | ✅ one |
| Determinism gate | none baked in | none | none | ✅ exopack TRIPLE SIMS |
| Self-licking-free tests | varies | varies | varies | ✅ audited 2026-05-16 |

## Concrete UX improvements

Ordered by structural impact on the codebase, not by guessed cost.

| # | Change | What it touches | Pain reduction |
|---|---|---|---|
| U6 | Wire Tensor (t502) to all ops (backlog #10) | every public op signature; every test in src/ops/*.rs; the README; all four example files. The Tensor type already exists — this is the cascading rename + signature change. | The single biggest readability fix: every op stops taking raw t501 + manual shape ints. |
| U2 | Add a `decode_tiny_gpt2.rs` example that hand-wires a 2-layer transformer end-to-end (no tokenizer, just integer-token input → integer-token output) | composes f670 embedding → f602 LN → f580 Q/K/V projection → f625 RoPE → f623 causal SDPA → f580 out-proj → f602 LN → SwiGLU FFN → repeat. Forces the missing Module abstraction to surface naturally. | Demonstrates the actual user goal end-to-end inside the crate's own examples; pulls the docs together. |
| U5 | Builder pattern for f582 (conv2d — 14 args) and f623 (causal SDPA — 7 args) | new associated types per op with named setters; backward-compat by leaving the positional version. | Typing-error reduction on the highest-arity ops. |
| U3 | Replace the README's bare matmul example with a 30-line example that loads a tiny safetensors file and runs one attention layer | README only; pulls from U2 once that exists. | Gives newcomers the headline path on first scroll. |
| U4 | Add `Debug` for t501 + t502 with shape-aware pretty-print | manual impl on t501 (reads back a slice via f504); on t502 (uses the shape it already tracks). | Painless debugging in `dbg!` / `println!`. |
| U7 | Optional `english-names` feature flag that re-exports `pub use t500::f500 as gpu_init` etc. | new `english-api` feature in Cargo.toml; a `pub use` block in lib.rs gated on that feature. **Currently disallowed by your "no English aliases" direction** — flagged here only because the discoverability cost is real. Re-confirm scope. | Makes the crate evaluable without the compression map. |
| U9 | Typed `GpuBufferU32` for token IDs | new t<N> in device.rs; embedding_lookup (f670) updated to consume it; cascading callers. | Eliminates the f32-as-int leaky abstraction. |
| U1 | Add per-feature comments in Cargo.toml describing what `cuda`, `metal`, `candle-bench`, `tests` gate | Cargo.toml only. | First-impression fix for new contributors. |
| U8 | Add a top-of-README "Quickstart in 5 lines" block above the brand boilerplate | README only. | First-impression fix; the brand block currently buries the code. |

**Structural-impact ranking:** U6 (Tensor wiring) and U2 (end-to-end example) sit at the top because they cascade: most other items become smaller (or disappear) once those two land. U1, U4, U8 are leaf changes — touch one file each, easy to land in parallel with the bigger work.

## Verdict

**Engine: 🟢 well-engineered.** The shader code, test discipline, and cross-vendor abstraction are genuinely strong.

**Front door: 🔴 opaque to newcomers.** The combination of (a) tokenized public API, (b) library-only with no service binary, (c) no Module graph, (d) no end-to-end model-running example, means a developer arriving via crates.io has no on-ramp.

**Stated goal alignment: 🟡 half-built.** "Models hosted on RAM, any-gpu as the interface" is *true* for primitives (load + GPU upload + attention + RoPE + KV cache) but *false* for the actual user-facing flow (text → tokens → model → tokens → text). See [USER_STORY_ANALYSIS.md](USER_STORY_ANALYSIS.md) for per-persona breakdown.

<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
