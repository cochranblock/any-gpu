<!-- Unlicense — cochranblock.org -->

# any-gpu User Story Analysis

*Honest assessment of which user personas any-gpu serves today, which it almost-serves, and which it misses — measured against the README's stated promises and the project owner's spoken intent.*

> Method: identify the personas the README implies and the headline goal the owner has stated, write the canonical user story for each, walk through it step-by-step using only the public API, and mark where the path breaks.

**Last updated: 2026-05-19 — post Sprint 7 + perf sprint (P1–P7) + backlog #9–#12.**

## Stated promises (the contract)

From `README.md`:
- "Tensor engine that runs on every GPU. AMD, NVIDIA, Intel, Apple."
- "Use any-gpu when your GPU is AMD or Intel (where CUDA can't run)."
- "You refuse to vendor-lock your compute pipeline."
- "You're building for a heterogeneous fleet (NVIDIA in the cloud, AMD on workstations, Apple on laptops)."

From the owner (spoken intent):
- "What is missing from this any-gpu for the transforms and having the models hosted on ram and any-gpu being the interface so I can use my 5700 XT OC mech edition can be used."
- Safetensors only — no GGUF, no PyTorch .bin, no ONNX.
- Kova tokenization — no English aliases.
- TRIPLE SIMS gate.
- No self-licking-ice-cream tests.

## Personas

### P1. **Mike** — AMD GPU owner, wants to run an LLM locally

> "I have an RX 5700 XT and 46 GB of RAM. I downloaded Llama-2-7B's safetensors. I want to chat with it locally without CUDA, without paying a cloud bill, on the hardware I already own."

**This is the headline persona — the project owner himself, and the audience the README is targeting first.**

#### Mike's intended path

1. `cargo add any-gpu` or clone the repo.
2. Find or write a `chat`-style example.
3. Point it at a safetensors file.
4. Type a prompt; receive tokens.

#### Mike's actual path today (2026-05-19)

| Step | Status | What happens |
|---|---|---|
| 1. Install | 🟡 | Not on crates.io yet (backlog #25), but clone-and-build works. `cargo build --bin any-gpu-serve` produces a binary. |
| 2. Find serving entry point | 🟢 | `any-gpu-serve` binary ships with S7.7. HTTP server on `POST /generate`, `GET /health`. |
| 3. Tokenize his prompt | 🟢 | `t544 Tokenizer` ships in S7.7 via `tokenizers` crate (f775–f779). BPE and SentencePiece-backed HF tokenizer files load directly. Detokenize via f778/f779. |
| 4. Load model weights | 🟢 | `t538::f760` loads safetensors from disk. `f770 page_layer` pages named tensors into VRAM via staging buffer. f16 weights upload via `f774 page_layer_f16`. |
| 5. Construct transformer forward pass | 🟢 | `t548 CausalLM` ships in S7.7 — LLaMA-style MHA+GQA forward via f783–f786. Per-layer: RMSNorm (f603) → Q/K/V project → RoPE (f625) → causal SDPA (f626 fused wave64) → output projection → residual → SwiGLU FFN. GQA handled by f629 repeat_kv. |
| 6. KV cache + autoregressive decode loop | 🟢 | `t534` KV cache (f672–f677), full autoregressive loop is in t548. Decode via f626 (wave64 SDPA, 1.1 ms for 1q/512kv on RX 5700 XT). |
| 7. Sample tokens | 🔴 | Only `f671` (argmax / greedy) exists. No temperature, top-k, top-p, repetition penalty. Greedy LLMs degenerate — the single remaining blocker for a usable interactive session. |
| 8. Detokenize → text | 🟢 | f778 (decode Vec\<u32\>) and f779 (decode single token) now ship. |
| 9. Llama-2-7B specifically | 🟢 | Dense MHA (32Q/32KV) works via f626. GQA models (Llama-3, Mistral) work via f629 repeat_kv. Llama-2-7B at bf16 / f16 = 14 GB (doesn't fit in 8 GB VRAM bare). |
| 10. Fit in 8 GB VRAM | 🟡 | t539 pager + f769 chunked upload let Mike stream layers. f774 page_layer_f16 halves storage. Llama-2-7B at f16 = 14 GB — still needs quantization (B11) or persistent-layer-in-RAM paging (A5 shipped but per-layer VRAM eviction policy is naive). Practical ceiling: 7B class at f16 with layers paged one at a time. |

**Verdict for Mike:** **One blocker remains.** The tokenizer, forward pass, KV cache, decode loop, serve binary, and detokenizer have all shipped. The single gap between today and a working chat: **A3 sampler suite** (top-k / top-p / temperature). Without it, greedy decode produces degenerate repetition. With it, Mike can serve Llama-2-7B.

Mike's reaction reading the README today: "the primitives are real, the tokenizer is real, the serve binary is real. I just need a real sampler."

### P2. **Jess** — ML researcher exploring cross-platform training

> "I want to prototype a small transformer variant. I want training to work on whatever GPU is in my laptop (Apple M4) and the lab's NVIDIA box without rewriting my code. I want to differentiate through everything."

#### Jess's actual path (2026-05-19)

| Step | Status | What happens |
|---|---|---|
| 1. Define model | 🟡 | `t545 Module` trait, `t546 Linear`, `t548 CausalLM` ship in S7.7. These cover a specific LLaMA-style forward. A custom transformer variant still requires hand-wiring ops; the trait is there but not batteries-included for arbitrary architectures. |
| 2. Forward pass | 🟢 | All op primitives exist. |
| 3. Backward pass | 🔴 | Autograd covers: add, sub, mul, scale, relu, sigmoid, swish, tanh, matmul, mse_loss, conv2d, conv_transpose2d, add_broadcast, add_per_col, group_norm, concat, upsample_nearest2d. **Still missing:** layer_norm_backward, rms_norm_backward, embedding_backward, softmax_backward, sdpa_backward, rope_backward. Every Sprint 7 transformer op is forward-only. Jess cannot train a transformer end-to-end. |
| 4. Optimizer step | 🟢 | AdamW (f720, f721). GPU-resident via f731–f734 (GpuParams, shipped in #9). |
| 5. Cross-platform | 🟢 | wgpu handles this. RX 5700 XT + NVIDIA Vulkan + M4 Metal all verified. |
| 6. Save trained weights | 🟡 | NanoSign (f745). No `save_to_safetensors` helper (D3 still open). |

**Verdict for Jess:** **Half-served.** She can train UNet-style models (conv2d, group_norm, swish — `nanobyte.rs` is a working example), MLPs, convnets. She cannot train transformers (autograd gap: B1–B6). Cross-platform training works.

### P3. **Dev** — backend engineer at a startup deploying to a heterogeneous cloud

> "We have NVIDIA A100s in cloud, AMD MI250s on prem, M4 Mac Studios for offline analysis. We want one binary, one config, one ops story. We want a service. We want batched requests."

#### Dev's actual path (2026-05-19)

| Step | Status | What happens |
|---|---|---|
| 1. One binary across hardware | 🟢 | `cargo build` produces one binary. wgpu picks the backend. |
| 2. Service / HTTP | 🟢 | `any-gpu-serve` ships in S7.7: HTTP server, `POST /generate`, `GET /health`. |
| 3. Batched requests | 🔴 | No batching primitive. Each request processes one prompt. No dynamic-batching scheduler. |
| 4. Observability / metrics | 🔴 | No metrics export. No Prometheus, no logging hooks. |
| 5. Hot-reload of weights | 🔴 | Not exposed in the serve API. |
| 6. Multi-GPU | 🔴 | Single device (backlog #24). |

**Verdict for Dev:** **Closer, but not production-ready.** The serve binary exists and works. Batching, observability, and multi-GPU remain unscoped. The current serve binary is a single-threaded request handler — viable for a personal/demo server, not for fleet deployment.

### P4. **Sam** — graduate student learning GPU compute

> "I want to read a small Rust + WGSL codebase that shows me how to write compute shaders correctly. I'm trying to figure out matmul tiling, softmax, attention. I want code I can copy and adapt."

#### Sam's actual path (2026-05-19)

| Step | Status | What happens |
|---|---|---|
| 1. Browse shaders | 🟢 | 55 WGSL shaders inline in `src/ops/*.rs`. Comments explain RDNA1 footguns (wave64, VGPR pressure, LDS layout). |
| 2. Understand dispatch wiring | 🟡 | Tokenization (`f543 = dispatch_shader`) means Sam consults `docs/compression_map.md`. After 5 minutes, it clicks. |
| 3. Find advanced examples | 🟢 | `examples/nanobyte.rs` (DDPM diffusion model — conv2d+GroupNorm+swish+upsample), `examples/ops_bench.rs` (comprehensive timing at LLaMA-7B shapes). |
| 4. Adapt a shader | 🟢 | The `dispatch_shader` + uniform-params pattern is reusable. |
| 5. Validate correctness | 🟢 | The test harness pattern (hardcoded reference + numeric grad check) is exemplary. |
| 6. Run on his GPU | 🟢 | wgpu auto-selects. Works. |

**Verdict for Sam:** **Well-served, with a 5-minute on-ramp tax** for the tokenization. The wave64 SDPA, two-pass GEMV, and subgroup-fused RMSNorm/LayerNorm are all fully commented reference implementations.

### P5. **Pat** — open-source contributor evaluating "should I help with this"

> "I'm a Rust dev who cares about cross-vendor GPU compute. Is this codebase well-run? Are the abstractions sound? Is the test discipline real?"

#### Pat's actual path (2026-05-19)

| Step | Status | What happens |
|---|---|---|
| 1. Read README | 🟢 | Honest about CUDA/MPS gap, head-to-head numbers, reproduction commands. |
| 2. Read TIMELINE_OF_INVENTION | 🟢 | Dated commit-level record with AI/human attribution. |
| 3. Read tests | 🟢 | 309 tests. TRIPLE SIMS gate (3/3 pass verified 2026-05-19). No mock-the-world tests. |
| 4. Check open issues | 🟡 | BACKLOG.md has ~8 open items with clear depends-on graph. No CONTRIBUTING.md, no public CI. |
| 5. Try test suite locally | 🟢 | `cargo test --release` works. `cargo run --release --bin any-gpu-test --features tests` runs TRIPLE SIMS gate. |
| 6. Understand architecture | 🟡 | compression_map.md explains tokenization. PROOF_OF_ARTIFACTS has the wire diagram. |

**Verdict for Pat:** **Well-served.** Test discipline is strong. Backlog is clear. The single contributor barrier is CONTRIBUTING.md (backlog E8).

## Aggregate scorecard (2026-05-19)

| Persona | Served today? | What's still missing | Priority |
|---|---|---|---|
| P1. Mike — AMD owner running LLMs | **Almost** — one blocker | A3 sampler (top-k/top-p/temperature) | **headline** |
| P2. Jess — researcher training transformers | **Partial** | B1–B6 transformer backward shaders | secondary |
| P3. Dev — fleet deployment | **Partial** | A11 batching + D9 multi-GPU + metrics | tertiary |
| P4. Sam — educational reader | **Yes, with friction** | compression_map on-ramp + tutorial walkthrough | quick win |
| P5. Pat — contributor evaluator | **Yes** | CONTRIBUTING.md (E8) | quick win |

## What the project promises vs. what it delivers (today)

| README promise | Delivered today? | Gap |
|---|---|---|
| "Tensor engine that runs on every GPU" | ✅ For the full inference op set | Training: transformer backward shaders still missing |
| "AMD, NVIDIA, Intel, Apple" | ✅ Three of four verified | Intel Iris Xe untested in isolation (backlog #14) |
| "One codebase, zero vendor lock-in" | ✅ | true |
| "GPU-accelerated ML in Rust" | ✅ For training CNNs/diffusion, inference for transformers | Transformer training: no backward shaders |
| (Owner) "Models hosted on RAM, any-gpu as interface" | 🟢 Near-complete | Pager + CausalLM + serve binary all ship. Sampler is the final piece. |
| (Owner) "Use my 5700 XT" | 🟢 Verified | Runs on 5700 XT. Llama-2-7B at f16 needs per-layer paging; argmax-only until sampler ships. |

## Recommendations ranked by structural impact

### Tier 1 — close the headline gap for Mike (one item)

1. **A3 sampler suite** (top-k / top-p / temperature / multinomial / repetition penalty): three new WGSL shaders. After this ships, Mike can type a prompt and get non-degenerate tokens back. This is the single highest-ROI item in the backlog.

### Tier 2 — close training gap for Jess

2. **B1–B6 backward shaders** (layer_norm, rms_norm, embedding scatter-add, softmax, SDPA, RoPE conjugate): each is one WGSL shader + one Op variant on the tape. The pattern is identical to the four existing activation backward shaders. Together they unlock full transformer fine-tuning.

### Tier 3 — quick infrastructure wins

3. **CONTRIBUTING.md + GitHub Actions CI** (E8, E6): low cost, high signal for contributors (Pat).
4. **`save_to_safetensors` serializer** (D3): round-trips the weight lifecycle for training use cases.

### Deferred

- **B11 Quantization** (Q4 / AWQ / GPTQ): largest single piece of new shader work. Defer until Mike has a working bf16 chat session (sampler first). With paging, 7B class at bf16 is reachable without quantization.
- **D9 Multi-GPU** (backlog #24): persona P3 needs it; P1 and P2 do not. Defer.
- **E2 Python bindings**: Rust-first stack. Defer.

<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
