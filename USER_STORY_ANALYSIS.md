<!-- Unlicense — cochranblock.org -->

# any-gpu — Definitive User Story Analysis

*15-phase analysis grounded in source code. Every finding links to a real file, function, or shader.*

**Updated: 2026-05-26** — post Sprint 7 + P1–P8 + B1–B7 + backlog #27 (batched inference). 341 tests.

---

## Phase 1 — Reconnaissance Summary

**Repo:** `/home/mcochran/any-gpu` · `v0.7.1` · Unlicense · `github.com/cochranblock/any-gpu`

**Binaries:**
- `any-gpu` — Stratagems CLI: `info`, `bench`, `train subatomic`
- `any-gpu-serve` — HTTP inference server with continuous batching
- `any-gpu-test` — TRIPLE SIMS gate (requires `--features tests`)

**Source modules:** `device`, `ops` (7 submodules), `tensor`, `autograd`, `nanosign`, `safetensors`, `pager`, `optim`, `train`, `tokenizer`, `module`, `lm`

**WGSL shaders (56+):** matmul (wave64 4×4 blocking), batch_matmul, conv2d, conv_transpose2d, rms_norm (subgroup-fused), layer_norm (subgroup-fused), softmax (subgroup-fused), fused SDPA (wave64 VGPR accumulator), GEMV two-pass, embedding_lookup, argmax, KV append, batch KV pool, RoPE, RoPE backward, top_k_mask (wave64 3-phase), top_p_mask, sample_multinomial (PCG32), rep_penalty, all activation backward shaders

**Key types (compression map `t5xx`):**
`t500=GpuDevice`, `t501=GpuBuffer`, `t502=Tensor`, `t506=Tape`, `t507=AdamW`, `t510=NanoSignResult`, `t534=KVCache`, `t538=SafetensorsModel`, `t539=LayerPager`, `t540=GpuBufferF16`, `t544=Tokenizer`, `t547=LmConfig`, `t548=CausalLM`, `t551=BatchKvPool`, `t556=DecodeSlot`

**Test count:** 341 passing. Hardware verified: AMD RX 5700 XT (RADV/Vulkan), NVIDIA RTX 3070 + 3050 Ti (Vulkan), Apple M4 (Metal).

---

## Phase 2 — Ecosystem Map

```
any-gpu
  ├── kova          (augment engine / C2 mesh — uses any-gpu for pixel-forge training)
  ├── pixel-forge   (sprite diffusion model — trains on any-gpu via nanobyte.rs pattern)
  ├── approuter     (HTTP routing / ingress — can front any-gpu-serve)
  ├── exopack       (TRIPLE SIMS testing framework — optional dep, "tests" feature)
  ├── cochranblock  (org website / brand)
  ├── whobelooking  (observability — potential consumer of any-gpu metrics)
  └── cochranblock-mail (alerting — potential consumer of inference events)
```

**IRONHIVE cluster nodes (from `~/.claude/CLAUDE.md`):**
- `bt` — 12 cores, 46 GB RAM, RX 5700 XT (primary any-gpu node)
- `lf` — RTX 3070 (secondary, Vulkan)
- `gd` — RTX 3050 Ti + Intel Iris Xe (tertiary, Vulkan; Iris Xe untested in isolation)
- Mac Mini (M4, Metal — main dev machine)

**IRONHIVE shared queue:** `~/.claude/SHARED_QUEUE.md` — 4 agents coordinate via handoff CLI. any-gpu builds and test runs are dispatched through this queue.

**Key external dependencies (`Cargo.toml`):**
- `wgpu = "24"` — GPU backend (Vulkan/Metal/DX12)
- `safetensors = "0.4"` — weight loading
- `tokenizers = "0.21"` — BPE/SentencePiece via HuggingFace crate
- `blake3 = "1"` — NanoSign hash
- `half = "2"` — f16 pack/unpack via `bytemuck`
- `exopack = "0.2"` — optional TRIPLE SIMS gate

---

## Phase 3 — Persona Identification

### P1 · Mike · AMD GPU Owner / Primary User
**Context:** Owns an RX 5700 XT 8 GB VRAM, 46 GB RAM. Has safetensors model files. Wants local LLM inference without CUDA, without cloud costs. This is the project owner.
**Technical level:** Systems-oriented, comfortable with Rust and CLI tools.
**Primary goal:** Type a prompt, receive coherent non-degenerate tokens, on hardware he already owns.

### P2 · Jess · ML Researcher
**Context:** PhD student or industry researcher. Has Apple M4 laptop and access to a lab NVIDIA box. Wants cross-platform training without rewriting code per vendor.
**Technical level:** Expert in ML algorithms, intermediate in Rust, understands autograd.
**Primary goal:** Prototype novel transformer variants and train them end-to-end.

### P3 · Dev · Backend Infrastructure Engineer
**Context:** At a startup deploying AI to production. Has NVIDIA in cloud, AMD on-prem, Apple M-series for offline. Needs one binary, one API, fleet scalability.
**Technical level:** Expert in distributed systems, intermediate in ML.
**Primary goal:** Serve inference requests at scale with SLA guarantees.

### P4 · Sam · Graduate Student / GPU Compute Learner
**Context:** CS PhD student studying GPU architecture and ML systems. Wants to read and understand real WGSL and wgpu code, not toy examples.
**Technical level:** Strong theory, learning GPU implementation.
**Primary goal:** Read correct, commented, real-world shaders; adapt patterns for research.

### P5 · Pat · Open-Source Rust Contributor
**Context:** Rust ecosystem developer who cares about cross-vendor GPU compute. Evaluating whether to contribute time to this project.
**Technical level:** Expert in Rust, intermediate GPU knowledge.
**Primary goal:** Determine if the codebase is well-run and worth contributing to.

### P6 · Kai · IRONHIVE Cluster Administrator
**Context:** Manages the bt/lf/gd/Mac Mini cluster. Responsible for keeping nodes in sync, running test gates, distributing builds. Operates via IRONHIVE SHARED_QUEUE.
**Technical level:** Expert in DevOps and cluster management, intermediate in ML.
**Primary goal:** Every push to any-gpu passes on all hardware nodes without manual intervention.

### P7 · Chris · Federal Contractor / Defense Procurement Officer
**Context:** At a DoD contractor or DARPA program office. Evaluating any-gpu for an edge AI deployment on tactical hardware. The Cochran Block holds CAGE 1CQ66 and SDVOSB certification.
**Technical level:** Light technical, strong compliance/procurement background.
**Primary goal:** Confirm SDVOSB eligibility, Unlicense compatibility with government use, supply chain provenance.

### P8 · River · ML Security Engineer
**Context:** Red team / security researcher at an AI company or defense contractor. Evaluates integrity of model loading, sampling pipeline, and HTTP serving for adversarial robustness.
**Technical level:** Expert in security and ML, strong in GPU compute.
**Primary goal:** Find exploitable gaps in model integrity, serving API, and RNG security.

### P9 · Alex · Angel Investor / Startup Founder
**Context:** Evaluating the Cochran Block's portfolio for investment or partnership. Comparing any-gpu against llama.cpp, candle, ggml, and Triton Inference Server.
**Technical level:** Light technical, strong in product and market analysis.
**Primary goal:** Determine if any-gpu is a credible foundation for a commercial AI product.

### P10 · Morgan · Adversarial Threat Actor
**Context:** Nation-state APT, disgruntled insider, or sophisticated script kiddie. Targeting model IP, user data, or service availability.
**Technical level:** Expert in offensive security.
**Primary goal:** Extract model weights, corrupt generation, or deny service.

---

## Phase 4 — User Story Enumeration

### P1 · Mike (AMD GPU Owner)

**M-01** As Mike, I want to run `any-gpu-serve` with a Llama-2-7B safetensors file so that I get coherent token generation without CUDA.

**M-02** As Mike, I want temperature scaling (via `f553`) applied before top-k so that my outputs feel human rather than robotically repetitive.

**M-03** As Mike, I want top-k sampling (`f787`, k≤128) so that obviously wrong tokens are masked before I sample.

**M-04** As Mike, I want top-p nucleus sampling (`f788`) so that I can control diversity independently of vocabulary size.

**M-05** As Mike, I want repetition penalty (`f790`, HF convention) so that the model doesn't loop on the same phrase.

**M-06** As Mike, I want to configure `max_new_tokens` per request in the POST /generate body so that I can control cost per call.

**M-07** As Mike, I want GET /health to return `{"status":"ok","device":"...","active_slots":N}` so that I can monitor the server with a simple uptime check.

**M-08** As Mike, I want the server to handle `max_batch` concurrent requests via continuous batching (`t551 BatchKvPool`) so that I don't waste GPU time between single-request gaps.

**M-09** As Mike, I want EOS token handling (via `tok.f779()`) so that generation terminates cleanly without truncating at `max_new_tokens` for short answers.

**M-10** As Mike, I want `t539 LayerPager` with 512 MiB staging (`DEFAULT_STAGE_BYTES`) so that I can stream a 7B model through 8 GB VRAM one layer at a time.

**M-11** As Mike, I want f16 weight support (`f774 page_layer_f16`) so that a 7B model's 14 GB bf16 file gets halved to 7 GB on the staging path.

**M-12** As Mike, I want GQA model support (`f629 repeat_kv`) so that Llama-3-8B and Mistral-7B work without code changes.

**M-13** As Mike, I want `NanoSign` verification (`f741`, `t510::Verified`) to log a clear message at load time so I know the weights haven't been tampered.

**M-14** As Mike, I want `any-gpu info` to print my adapter name, backend, subgroup capability, and f16 flag so that I can confirm Vulkan is selected on my RX 5700 XT.

**M-15** As Mike, I want `any-gpu bench` to run LLaMA-7B-scale shapes (512×4096 prefill, 1×4096 decode) and print GFLOPS/GB-s so that I can compare my card to published numbers.

**M-16** As Mike, I want a 30-second read timeout on incoming connections (`stream.set_read_timeout`) so that stalled clients don't hold a slot forever.

**M-17** As Mike, I want requests that arrive when all `max_batch` slots are full to queue in `VecDeque<PendingReq>` so that they wait rather than fail.

**M-18** As Mike, I want completed slots to be compacted via `f807 migrate_slot` so that the dense slot invariant holds and new requests fill freed positions without fragmentation.

**M-19** As Mike, I want the KV cache pre-allocated to `max_seq` tokens at startup so that I never allocate during a decode step.

**M-20** As Mike, I want `WGPU_BACKEND=vulkan` to force Vulkan on AMD and bypass any DX12/OpenGL fallback so that I stay on RADV.

**M-21** As Mike, I want the serve binary to print device name and backend at startup (`eprintln!("[any-gpu-serve] device: {} ({})", dev.s502, dev.s503)`) so that I immediately know which GPU was picked.

**M-22** As Mike, I want greedy decoding (`f671 argmax`) to remain available as a sampling mode so that I can reproduce deterministic outputs for debugging.

**M-23** As Mike, I want `any-gpu train subatomic` to train three GPU-accelerated classifiers (slop_detector, code_vs_english, lang_detector) so that I can verify training works end-to-end in under 2 minutes.

**M-24** As Mike, I want the safetensors loader (`t538::f761`) to load a weight file entirely in RAM before uploading to VRAM so that I can verify the file before touching the GPU.

**M-25** As Mike, I want the `--max_seq` CLI override on `any-gpu-serve` so that I can reduce KV cache VRAM for shorter-context workloads on 8 GB.

**M-26** As Mike, I want `fused_sdpa` (`f626`, wave64, VGPR accumulator) to handle the decode step in ~1.1 ms for 1q/512kv so that 7B inference stays interactive.

**M-27** As Mike, I want the serve loop to sleep 1 ms when both `pending` and `active` are empty (instead of busy-polling) so that it doesn't peg one CPU core while idle.

---

### P2 · Jess (ML Researcher)

**J-01** As Jess, I want `f791 layer_norm_backward` (3-pass: stats, grad_input, grad_affine) so that I can differentiate through LayerNorm in a BERT-style architecture.

**J-02** As Jess, I want `f792 rms_norm_backward` (3-pass) so that I can fine-tune a Llama/Mistral checkpoint end-to-end.

**J-03** As Jess, I want `f793 embedding_backward` (f32 CAS scatter-add via `atomicCompareExchangeWeak`) so that gradient flows through the embedding table when I train from a vocabulary.

**J-04** As Jess, I want `f794 softmax_backward` (dot per row then `p*(grad-dot)`) so that I can backprop through attention weights.

**J-05** As Jess, I want `f796 rope_backward` (conjugate rotation with negated sin) so that gradients pass correctly through positional encodings.

**J-06** As Jess, I want the tape (`t506`) to record transformer ops (`f712–f717`) with enum Op variants so that `backward()` dispatches to the correct shader without heap allocations.

**J-07** As Jess, I want `train_step_gpu` (`f734`) to do forward + backward + AdamW in one call so that I don't manage pipeline stages manually.

**J-08** As Jess, I want `GpuParams` (`t550`) to hold weights GPU-resident across steps so that I avoid a CPU round-trip every iteration.

**J-09** As Jess, I want numeric gradient checks (finite-difference vs backprop) following the pattern in `f584_stride2_numeric` so that I can verify any new backward shader is correct.

**J-10** As Jess, I want `f745 save_signed` to persist trained weights as `<path> + NSIG + BLAKE3` so that checkpoints carry integrity metadata.

**J-11** As Jess, I want the autograd tape to handle the full `t548 CausalLM` forward pass (embedding → attention → MLP) so that I can fine-tune on a new task without reimplementing the forward.

**J-12** As Jess, I want `t544 Tokenizer` to encode/decode strings so that I can feed text directly to the model without pre-processing.

**J-13** As Jess, I want `AdamW` (`t507`) to accept `lr`, `beta1`, `beta2`, `eps`, and `weight_decay` so that I can sweep hyperparameters without recompiling.

**J-14** As Jess, I want the test suite to run on Apple M4 via Metal so that I can verify identical forward-pass outputs on my laptop before running on the lab NVIDIA box.

**J-15** As Jess, I want `f626 fused_sdpa` to produce numerically identical results to `f623 causal_sdpa` within 1e-3 so that I can switch flash-attention on/off as a correctness check.

**J-16** As Jess, I want the training stratagem pattern (`any-gpu train <name>`) so that I can package a reproducible experiment as a named stratagem without external scripts.

**J-17** As Jess, I want `examples/nanobyte.rs` (NanoUNet DDPM, ~1.09M params) as a worked example of GPU-resident training on pixel-forge sprites so that I can adapt it for a ViT.

**J-18** As Jess, I want a `save_to_safetensors` function (gap D3) so that I can export trained weights for use with HuggingFace tools.

**J-19** As Jess, I want `B12 bf16 compute` (gap) so that I can run float-16 accumulation on hardware that supports it and measure the speedup.

**J-20** As Jess, I want shape inference / broadcasting in autograd (gap D5) so that I don't manually track tensor dimensions through ops.

**J-21** As Jess, I want tensor `Debug` / pretty-print (gap D7) so that I can inspect intermediate activations without writing to a temp file.

**J-22** As Jess, I want layer-wise dropout / attention dropout (gap B16) so that I can regularize during fine-tuning.

**J-23** As Jess, I want SDPA backward (gap B5) so that attention weights participate in the gradient graph.

**J-24** As Jess, I want a `CHANGELOG.md` (gap E7) so that I know what changed between versions when I update the dependency.

**J-25** As Jess, I want the TRIPLE SIMS gate (`any-gpu-test --features tests`) to pass on every node after I land new backward shaders so that I don't silently break inference.

**J-26** As Jess, I want `f630 fused_sdpa_batch_decode` to work correctly in batch mode so that fine-tuning with batch size > 1 uses the efficient batch path.

---

### P3 · Dev (Fleet Infrastructure Engineer)

**D-01** As Dev, I want `any-gpu-serve` to parse `{"prompt":"...","max_new_tokens":128}` from POST /generate and return `{"output":"..."}` so that I can integrate it behind any HTTP gateway.

**D-02** As Dev, I want continuous batching (`t551 BatchKvPool`, `f788b prefill_slot`, `f789b batch_decode_step`) so that I can saturate the GPU across concurrent users instead of waiting for each to complete.

**D-03** As Dev, I want `max_batch` to be a config.json field (`t547.max_batch`) so that I can tune it per deployment without recompiling.

**D-04** As Dev, I want `GET /health` to include `active_slots` so that my load balancer can drain a node before a planned maintenance.

**D-05** As Dev, I want the serve binary to log `[serve] prefilling slot N, M tokens` and `[serve] retired slot N` to stderr so that I can correlate per-request latency with cluster logs.

**D-06** As Dev, I want the TCP listener to be non-blocking (`listener.set_nonblocking(true)`) so that the serve loop doesn't block on accept and can still drain completed decode steps.

**D-07** As Dev, I want parse errors (bad JSON, unknown endpoint) to return HTTP 500 with `{"error":"..."}` so that clients get actionable error messages.

**D-08** As Dev, I want the model and tokenizer loaded from paths supplied via `--model`, `--config`, and `--tokenizer` CLI flags so that I can deploy different model versions by changing the launch command.

**D-09** As Dev, I want the serve binary to bind on `0.0.0.0:<port>` (default 8080) so that it's reachable from other machines in the fleet.

**D-10** As Dev, I want connection timeouts (`Duration::from_secs(30)`) on incoming streams so that stalled clients release their slot without operator intervention.

**D-11** As Dev, I want completed slots to be reclaimed via swap-remove + f807 migration so that VRAM usage stays constant regardless of request completion order.

**D-12** As Dev, I want the single-binary `cargo build --release` output to work on AMD, NVIDIA, and Apple without per-platform builds so that my CI pipeline has one artifact.

**D-13** As Dev, I want `f631 rope_batch_decode` so that positional encodings are correct across all concurrently decoding batch slots.

**D-14** As Dev, I want SSE (server-sent events) / streaming output (gap A10) so that the client UI can display tokens as they arrive instead of waiting for the full response.

**D-15** As Dev, I want observability hooks (gap) — at minimum token/s and TTFT metrics per request — so that I can set and monitor SLOs.

**D-16** As Dev, I want configurable per-request token limits enforced at the serve layer so that a single runaway request can't starve all batch slots.

**D-17** As Dev, I want hot-reload of model weights (gap) so that I can update a deployment without restarting the serve process.

**D-18** As Dev, I want multi-GPU dispatch (gap backlog #28) so that I can tensor-parallel a 30B+ model across two cards on the same host.

**D-19** As Dev, I want a `--bind` flag so that I can specify the listen address for multi-network-interface hosts.

**D-20** As Dev, I want the server to handle requests that arrive during a long prefill without dropping them so that bursty traffic is absorbed into the pending queue.

**D-21** As Dev, I want the routing model (backlog #22) to auto-benchmark ops and pick the optimal device per workload so that I don't manually tune per-node.

**D-22** As Dev, I want `test-fleet.sh` (backlog #13) to SSH all three nodes and run `cargo test --release` so that a push regression surfaces on every hardware variant within minutes.

**D-23** As Dev, I want the serve binary to gracefully drain active slots on SIGTERM so that in-flight requests complete before shutdown.

**D-24** As Dev, I want quantized weight support (gap B11) so that I can fit a 13B model in 8 GB VRAM per node without sacrificing too much quality.

**D-25** As Dev, I want publish to crates.io (backlog #25) so that I can add `any-gpu` to my project's `Cargo.toml` without vendoring.

**D-26** As Dev, I want a Prometheus-compatible `/metrics` endpoint so that my Grafana dashboard shows inference throughput and latency percentiles.

---

### P4 · Sam (GPU Compute Learner)

**S-01** As Sam, I want to read `SHADER_FUSED_SDPA_W64` in `src/ops/attention.rs` with inline comments explaining the VGPR accumulator and wave64 design so that I understand why this is faster than a naive implementation.

**S-02** As Sam, I want `SHADER_MATMUL` (wave64, 4×4 register blocking, 32×32 LDS tile) in `src/ops/conv.rs` with comments explaining the 8 KB/workgroup LDS budget so that I can port the design to other GEMM variants.

**S-03** As Sam, I want `SHADER_SOFTMAX_FUSED` with comments on `subgroupMax`/`subgroupAdd` and the 4-subgroup cross-reduce so that I understand subgroup operations on RDNA1.

**S-04** As Sam, I want `SHADER_GEMV_P1`/`SHADER_GEMV_P2` with a comment explaining the K-stripe partitioning and why this yields 35% RDNA1 occupancy so that I understand the decode-path GEMV design.

**S-05** As Sam, I want `SHADER_TOP_K_MASK` with Phase 1/2/3 comments (private min-heap in VGPRs, LDS merge, mask apply) so that I understand how to do top-k on a GPU without sorting.

**S-06** As Sam, I want `SHADER_EMBED_BWD` with a comment on the `atomicCompareExchangeWeak` CAS loop for f32 scatter-add so that I understand why a plain atomic-add doesn't work for floats in WGSL.

**S-07** As Sam, I want `SHADER_KV_APPEND` with a comment explaining the `[bh, max_seq, head_dim]` cache layout so that I understand how KV caches are structured for multi-head attention.

**S-08** As Sam, I want `SHADER_ROPE_BWD` with a comment explaining that RoPE is an orthogonal rotation (RᵀR = I) so that I understand why negating sin inverts it.

**S-09** As Sam, I want `SHADER_BATCH_DECODE_APPEND` with a comment on the slot-major layout (`slot * nkv_h * max_seq * hd`) so that I understand continuous batching KV pool geometry.

**S-10** As Sam, I want `f543 dispatch_shader` in `src/ops/mod.rs` to document the uniform-params + storage-binding pattern so that I can add a new shader without reading all existing examples.

**S-11** As Sam, I want `examples/ops_bench.rs` to show me how to fence GPU commands with `f504` for accurate timing so that my benchmarks don't measure asynchronous submission.

**S-12** As Sam, I want `examples/nanobyte.rs` (DDPM diffusion, GPU-resident params, conv2d+GroupNorm+SwiGLU) to be a complete training example so that I can adapt it for a custom architecture.

**S-13** As Sam, I want `docs/compression_map.md` to map `t5xx` to human names so that I can navigate the source without memorizing the token table.

**S-14** As Sam, I want `PROOF_OF_ARTIFACTS.md` to list verified hardware and reproduce commands so that I know the claims are real before I spend a week on a project.

**S-15** As Sam, I want the numeric gradient check pattern (`f584_stride2_numeric` in `src/ops/conv.rs`) to be documented as the canonical way to validate a new backward shader so that I don't invent my own test approach.

**S-16** As Sam, I want `TIMELINE_OF_INVENTION.md` to explain the AI/human collaboration methodology so that I understand the project's development process.

**S-17** As Sam, I want to build the WASM32 target so that I can embed inference in a browser demo.

**S-18** As Sam, I want `NanoSign` (`f740–f747` in `src/nanosign.rs`) to be a standalone, readable module so that I can study BLAKE3-based file integrity as an independent example.

**S-19** As Sam, I want `src/autograd.rs` to show me how to implement reverse-mode autodiff with a flat tape and enum-dispatched ops so that I understand the tradeoffs vs trait-object-based approaches.

**S-20** As Sam, I want `fused_sdpa` cross-validated against `causal_sdpa` in tests so that I can see that a flash-attention equivalent produces the same numbers as the naive approach.

**S-21** As Sam, I want to run `cargo test --release` and get all 341 tests pass on my personal GPU (AMD RX 6700 or NVIDIA RTX 4060) so that I can verify the code works on hardware outside the three verified nodes.

**S-22** As Sam, I want to understand how 2D dispatch (`gid.y * 65535 * 256 + gid.x`) handles >65535 workgroups so that I know how to work around the WebGPU dispatch limit.

**S-23** As Sam, I want `f630 fused_sdpa_batch_decode` to be a separate, readable function with comments on how it differs from the single-request path so that I understand the batch-decode attention geometry.

**S-24** As Sam, I want to see a `CONTRIBUTING.md` (gap E8) with setup instructions so that I know how to run tests on a fresh machine.

**S-25** As Sam, I want `src/ops/sampler.rs` to include a comment explaining the PCG32 inverse-CDF sampling and why it's TRIPLE SIMS safe (deterministic given seed+step) so that I understand reproducibility in GPU sampling.

---

### P5 · Pat (Open-Source Contributor)

**PA-01** As Pat, I want to run `cargo test --release` and see 341 tests pass with zero warnings so that I can baseline the project state before I touch anything.

**PA-02** As Pat, I want to read `PROOF_OF_ARTIFACTS.md` and verify the benchmark numbers are honest (including "CUDA/MPS is faster — measured it") so that I know the project isn't inflating claims.

**PA-03** As Pat, I want `BACKLOG.md` to have a clear depends-on graph so that I can pick an open item without creating a dependency conflict.

**PA-04** As Pat, I want a `CONTRIBUTING.md` (gap E8) with instructions for adding a shader, adding a test, and running the TRIPLE SIMS gate so that I can onboard in under an hour.

**PA-05** As Pat, I want GitHub Actions CI (gap E6) to run `cargo test --release` and `cargo clippy` on push so that my PRs are validated before human review.

**PA-06** As Pat, I want the `dispatch_shader` helper (`f543`) documented in `docs/` so that I understand the one-line path to adding a new compute op.

**PA-07** As Pat, I want `#[must_use]` on `t510 NanoSignResult` (already shipped in backlog #19) so that I can confirm the compiler enforces that callers handle tamper results.

**PA-08** As Pat, I want an `ARCHITECTURE.md` (or the wire diagram in `PROOF_OF_ARTIFACTS.md` extracted to its own doc) so that I can understand the full module graph quickly.

**PA-09** As Pat, I want the compression-map convention explained in `CONTRIBUTING.md` (gap) so that I don't assign conflicting token numbers when adding new types.

**PA-10** As Pat, I want the pipeline cache (`Mutex<HashMap<u64, Arc<ComputePipeline>>>` in `device.rs`) to have a test verifying hit rate grows then stabilizes so that I can confirm cache eviction policy is sound.

**PA-11** As Pat, I want `cargo clippy -- -D warnings` to produce zero diagnostics so that I know my PR doesn't introduce new lint issues.

**PA-12** As Pat, I want the test harness for backward shaders to use finite-difference validation consistently so that I can verify any new backward shader I add follows the same correctness bar.

**PA-13** As Pat, I want an example of adding a new elementwise op (shader + Rust wrapper + test) documented so that the contribution path is a 30-minute exercise, not a 3-hour archaeology.

**PA-14** As Pat, I want `CHANGELOG.md` (gap E7) so that I can see what changed between releases when evaluating a version upgrade.

**PA-15** As Pat, I want a Windows/DX12 build documented in CI (gap) so that I know the DX12 backend isn't silently broken.

**PA-16** As Pat, I want `save_to_safetensors` (gap D3) so that the round-trip `load → train → save` is self-contained without reaching for another crate.

**PA-17** As Pat, I want the typed `u32` buffer for token IDs (gap D8) so that I can propose replacing the f32-as-int hack without it being a surprise to reviewers.

**PA-18** As Pat, I want pipeline cache LRU eviction (gap C7) so that long-running servers don't grow unbounded memory from cached pipelines.

**PA-19** As Pat, I want `f625 rope` and `f627 split_heads` fused into a single shader (gap C3) so that I can contribute the optimization and measure the 2-round-trip savings.

**PA-20** As Pat, I want `test-fleet.sh` (backlog #13) implemented so that I can run the multi-node regression suite locally via SSH.

**PA-21** As Pat, I want the README example commands to be tested in CI (not just documented) so that they don't drift.

**PA-22** As Pat, I want a published crate on crates.io (backlog #25) so that external users can add `any-gpu` as a dependency without vendoring.

**PA-23** As Pat, I want docs.rs documentation curated beyond autogeneration (gap E3) so that the public API is discoverable.

**PA-24** As Pat, I want the `#[allow(non_camel_case_types, non_snake_case)]` attributes in `lib.rs` to be limited to the tokenized types rather than applied globally so that non-tokenized identifiers still follow Rust conventions.

**PA-25** As Pat, I want an end-to-end chat example (gap E10) that shows `tokenize → prefill → decode → detokenize` in a runnable Rust binary so that new users have a working starting point.

---

### P6 · Kai (IRONHIVE Cluster Administrator)

**K-01** As Kai, I want `cargo run --release --bin any-gpu-test --features tests` to pass 3/3 TRIPLE SIMS checks (safetensors load → GPU upload → readback, RoPE determinism, KV cache prefill+decode+readback) on every node before I tag a release.

**K-02** As Kai, I want `test-fleet.sh` (backlog #13) to SSH bt, lf, and gd in parallel and run `cargo test --release`, aggregating pass/fail per node so that I can catch hardware-specific regressions in under 5 minutes.

**K-03** As Kai, I want `WGPU_BACKEND=vulkan cargo test --release` to force the Vulkan path on bt (RX 5700 XT) so that RADV is tested explicitly and not accidentally shadowed by another backend.

**K-04** As Kai, I want `any-gpu info` to print backend and feature flags so that after a Mesa/RADV driver update I can quickly verify subgroup and f16 flags are still active.

**K-05** As Kai, I want the target directory symlinked to `/mnt/data/targets/any-gpu` (already done) so that build artifacts don't fill the OS disk on bt.

**K-06** As Kai, I want `DEFAULT_STAGE_BYTES = 512 MiB` to be overridable via an env var or config so that on lf (less RAM than bt) I can reduce the staging buffer.

**K-07** As Kai, I want test results from each node logged to a timestamped file in `~/.claude/log-snapshots/` so that I can diff pass counts across driver updates.

**K-08** As Kai, I want gd's Intel Iris Xe path tested in isolation via `WGPU_BACKEND=vulkan` with Xe as primary adapter (backlog #14) so that we have verified hardware coverage on Intel.

**K-09** As Kai, I want the IRONHIVE `SHARED_QUEUE.md` to have an `any-gpu-test` task template so that the test node (bt) picks up test jobs automatically.

**K-10** As Kai, I want the `fleet.toml` in `~` to reference any-gpu so that kova's deploy scripts know to build it when deploying a cluster update.

**K-11** As Kai, I want a `--port` flag on `any-gpu-serve` (already ships) so that I can run multiple model versions on different ports on the same node.

**K-12** As Kai, I want `lm.cfg.max_batch` to be read from config.json so that each node's serve process is tuned to its VRAM without a separate binary per node.

**K-13** As Kai, I want `any-gpu bench` to dump output as JSON (gap) so that I can ingest benchmark results into a fleet-wide performance tracking system.

**K-14** As Kai, I want re-verification of lf (RTX 3070) post Sprint 7 + P1–P8 (gap F1) so that the hardware matrix in `PROOF_OF_ARTIFACTS.md` is current.

**K-15** As Kai, I want re-verification of gd (RTX 3050 Ti) post Sprint 7 + P1–P8 (gap F2) so that the hardware matrix is complete.

**K-16** As Kai, I want re-verification of Apple M4 post Sprint 7 (gap F3) so that the Mac Mini dev node is confirmed to pass the full test suite.

**K-17** As Kai, I want the handoff protocol (`~/.claude/collab/PROTOCOL.md`) to include an `any-gpu-serve:start` verb so that KOVA can spin up the inference server on bt via the handoff CLI.

**K-18** As Kai, I want the serve binary to log its PID to a file at startup so that I can send it SIGTERM cleanly from a management script.

**K-19** As Kai, I want staged rollouts: run the new binary on lf first (lower risk), verify with TRIPLE SIMS, then promote to bt so that production inference on bt is never disrupted by a bad build.

**K-20** As Kai, I want `cargo test` on a fresh `cargo clean` to complete in under 10 minutes on bt so that the CI loop is fast enough to not block developers.

**K-21** As Kai, I want large model weight files (7B safetensors ~14 GB) stored on the `/mnt/data` volume so that they don't fill the OS disk.

**K-22** As Kai, I want a health probe that fails during model loading (before the server is ready) so that the load balancer doesn't route traffic to a not-yet-ready instance.

**K-23** As Kai, I want the `any-gpu bench` output to include a "vs baseline" delta when a baseline file exists so that I can immediately see if a new build regressed.

**K-24** As Kai, I want the TRIPLE SIMS gate to check that RoPE output is bitwise identical across two runs (already implemented in `any-gpu-test`) so that PCG RNG seed behavior is deterministic across Mesa versions.

**K-25** As Kai, I want the IRONHIVE node spec (bt = 12 cores, 46 GB, RX 5700 XT) to be documented in the CLAUDE.md so that any node that picks up the test job knows the hardware context.

---

### P7 · Chris (Federal Procurement Officer)

**C-01** As Chris, I want to confirm CAGE code `1CQ66` is registered in SAM.gov so that the Cochran Block qualifies for sole-source and set-aside contracting under SDVOSB rules.

**C-02** As Chris, I want the README SDVOSB badge to link to a certification evidence page so that I have a citable document for my contracting officer.

**C-03** As Chris, I want the Unlicense dedication confirmed as compatible with Government Purpose Rights so that DoD can use, modify, and redistribute the software without license restrictions.

**C-04** As Chris, I want a Software Bill of Materials (SBOM) in CycloneDX or SPDX format so that my program office can run supply-chain vulnerability scans against the dependency tree.

**C-05** As Chris, I want `PROOF_OF_ARTIFACTS.md` to include a signed provenance attestation so that I can demonstrate to my Contracting Officer that the software was developed as claimed.

**C-06** As Chris, I want NanoSign (`t510`, NSIG + BLAKE3) described in a technical data package so that the model integrity protocol can be evaluated by the program security officer.

**C-07** As Chris, I want the `UEI = W7X3HAQL9CF9` and `EIN = 41-3835237` from the README footer to match SAM.gov records so that the vendor identity is confirmed before award.

**C-08** As Chris, I want `TIMELINE_OF_INVENTION.md` to serve as IP provenance documentation so that in a dispute over data rights, there is a dated commit-level record.

**C-09** As Chris, I want the project's dependencies (`wgpu`, `safetensors`, `tokenizers`, `blake3`) evaluated for export control (EAR/ITAR) classification so that we know if deployment to an allied nation requires a license.

**C-10** As Chris, I want an air-gapped deployment guide so that the software can run in a SCIF or disconnected tactical edge environment without internet access to crates.io.

**C-11** As Chris, I want FIPS 140-3 compliance evaluated for blake3 so that the NanoSign integrity protocol meets cryptographic standards for classified programs.

**C-12** As Chris, I want the software to run on government-furnished AMD GPU hardware (e.g., Radeon Pro W7900) so that I don't need to procure NVIDIA cards for GPU inference.

**C-13** As Chris, I want an assumed-breach threat model document (pattern: `ASSUMED_BREACH_THREAT_MODEL.md` exists in kova and pixel-forge neighbors) so that the security posture is documented for the ATO process.

**C-14** As Chris, I want multi-level security isolation evaluated (one inference process per classification level) so that classified and unclassified data don't share a GPU buffer.

**C-15** As Chris, I want `Unlicense` to be confirmed as compatible with the Defense Innovation Unit's open-source policy so that rapid fielding to a program of record is unobstructed.

**C-16** As Chris, I want performance benchmarks at realistic military edge hardware specs (4–8 GB VRAM, x86_64) so that I can size the hardware procurement for a deployed system.

**C-17** As Chris, I want a security assessment of the HTTP serve interface so that I know if it requires an ISSE-approved overlay before being networked.

**C-18** As Chris, I want the software verified on RHEL 9 or Ubuntu 22.04 LTS (DoD standard) so that I don't have to waiver a non-standard OS.

**C-19** As Chris, I want automated test results with date/time stamps from each hardware node so that I can include them in the program's artifact traceability matrix.

**C-20** As Chris, I want the license footer (`PUBLIC DOMAIN · UNLICENSE · RECEIPTS ATTACHED`) to explicitly state that no contributor retains copyright so that the government's unlimited rights are unambiguous.

**C-21** As Chris, I want a CMMC Level 2 gap assessment for the development environment so that the software supply chain doesn't introduce compliance risk.

**C-22** As Chris, I want a data flow diagram showing where model weights, prompts, and outputs reside (RAM, VRAM, disk) so that I can classify data handling under the program's data management plan.

**C-23** As Chris, I want an SBIR Phase I/II application template describing any-gpu's TRL and commercialization path so that I can sponsor funded development.

**C-24** As Chris, I want a published crates.io package (backlog #25) with a version locked to a specific commit so that I can pin a specific evaluated version in procurement.

**C-25** As Chris, I want long-term sustainment committed to by the vendor so that I can argue against technical obsolescence risk in my acquisition strategy.

---

### P8 · River (ML Security Engineer)

**R-01** As River, I want to confirm that `f742 verify_bytes` returns `t510::Failed` when I flip a single byte in a signed weight file so that NanoSign actually detects tampering.

**R-02** As River, I want to confirm that `#[must_use]` on `t510 NanoSignResult` causes a compiler warning if a caller ignores the return value of `f741`/`f742` so that silent tamper-ignore is rejected at compile time.

**R-03** As River, I want to test `parse_request` in `serve.rs` with a body that is 10× larger than `content_length` so that I confirm the server doesn't read past the declared length.

**R-04** As River, I want to send a `{"prompt": "...inject...", "max_new_tokens": 999999}` to `/generate` and confirm `max_new_tokens` is capped at a sane limit so that a single request can't monopolize a batch slot indefinitely.

**R-05** As River, I want to confirm that out-of-range token IDs in `SHADER_EMBED` are clamped to `vocab_size - 1` rather than causing out-of-bounds GPU memory access so that a malicious prompt can't corrupt VRAM.

**R-06** As River, I want to audit `SHADER_EMBED_BWD` for livelock in the CAS loop so that a degenerate gradient pattern (e.g., NaN propagation) can't cause the shader to spin forever.

**R-07** As River, I want to verify that KV cache overflow is guarded with an `ensure!` check (`cursor + new >= max_seq → error`) so that a request with `token_ids.len() > max_seq` doesn't write past the pre-allocated buffer.

**R-08** As River, I want to test `/generate` with a prompt containing Unicode null bytes, overlong UTF-8, and RTL control characters so that the tokenizer doesn't panic or corrupt heap.

**R-09** As River, I want to audit `sample_multinomial` (`f789`, PCG32 RNG) for predictability: given the same `seed` and `step`, the output is deterministic — which means if an attacker knows the seed, they can predict token sequences.

**R-10** As River, I want to confirm that GPU memory from completed slots is not readable by subsequent requests so that one user's KV cache contents can't leak to another user.

**R-11** As River, I want to confirm that the HTTP server doesn't reveal file system paths in error messages so that `parse error: No such file or directory: /home/mcochran/models/...` doesn't disclose the server's directory structure.

**R-12** As River, I want to test the pipeline cache for timing side channels: a cache miss requires shader compilation, which takes ~100 ms vs ~1 ms for a hit, potentially leaking which shader was last used.

**R-13** As River, I want to confirm that `f744 strip_bytes` correctly identifies the NSIG magic only at the tail (already tested by `f744_no_false_positive`) so that a payload with `NSIG` mid-file doesn't falsely strip data.

**R-14** As River, I want to send concurrent POST /generate requests at `max_batch + 1` concurrently so that the `pending` queue handles the overflow without dropping connections or corrupting active slots.

**R-15** As River, I want to audit the `f807 migrate_slot` path for TOCTOU races: is it possible for a concurrent decode step to read from a slot that is being migrated?

**R-16** As River, I want to confirm that `DEFAULT_STAGE_BYTES = 512 MiB` is a hard limit and that a malicious model file cannot cause the pager to allocate more than that in a single `f769` call.

**R-17** As River, I want to confirm that `serde_json::from_slice` in `parse_request` returns an error for nested JSON beyond a reasonable depth so that a deeply nested payload doesn't cause stack overflow.

**R-18** As River, I want to test that `f746 load_verified` rejects a file with `NanoSignResult::Failed` with a non-zero exit code so that a tampered model weight file is never loaded silently.

**R-19** As River, I want to confirm the blake3 hash in NanoSign covers the entire payload (not just a prefix) so that a length-extension attack pattern is not applicable.

**R-20** As River, I want to audit the tokenizer (`t544`, backed by `tokenizers` crate) for prompt injection: does the BPE encoder treat special tokens as literals or as control sequences?

**R-21** As River, I want to assess model extraction risk: can an attacker reconstruct weight values by querying `/generate` with crafted prompts and observing logits?

**R-22** As River, I want to confirm that the staging buffer (`MAP_WRITE|COPY_SRC` in pager) is not accessible after the upload completes so that weight values don't linger in host-visible memory.

**R-23** As River, I want to confirm that `listen.set_nonblocking(true)` combined with a `WouldBlock` break doesn't create a busy-poll that allows a DoS via connection flood.

**R-24** As River, I want to evaluate the `Unlicense` dedication: since it's public domain, there's no legal mechanism to enforce usage restrictions, which means a malicious fork could strip NanoSign and redistribute tampered weights under the same name.

**R-25** As River, I want to test behavior when `model.safetensors` is not found at the given path so that the server exits with a clear error rather than panicking with a stack trace.

---

### P9 · Alex (Startup Founder / Investor)

**A-01** As Alex, I want to see honest head-to-head benchmarks vs CUDA and Metal in `PROOF_OF_ARTIFACTS.md` so that I understand the real performance gap before I build a product pitch on this engine.

**A-02** As Alex, I want `Unlicense` (public domain) so that I can embed any-gpu in a commercial product without attribution requirements or license incompatibility with my investors' IP due diligence.

**A-03** As Alex, I want a published crates.io package (backlog #25) so that `cargo add any-gpu` works in a customer demo without patching or vendoring.

**A-04** As Alex, I want continuous batching (`t551`, `f789b`) already shipped so that I can claim our inference server handles real concurrent traffic, not just one request at a time.

**A-05** As Alex, I want `GET /health` with `active_slots` so that I can wire it to an ELB health check and claim production-grade observability on the demo slide.

**A-06** As Alex, I want the 26.6× top_k speedup (P8, 106 ms → 4 ms) documented in BACKLOG.md so that I have a concrete "orders of magnitude improvement" narrative for investors.

**A-07** As Alex, I want the hardware matrix (AMD RX 5700 XT, NVIDIA RTX 3070 + 3050 Ti, Apple M4) verified so that I can claim "runs on 4 GPU families" in a pitch.

**A-08** As Alex, I want the AMD GPU market angle (CUDA can't run on AMD outside of ROCm, ROCm has painful driver requirements) documented so that I can position against NVIDIA lock-in.

**A-09** As Alex, I want quantization (gap B11) on the roadmap so that I can include "13B+ model support coming" in the product roadmap without lying.

**A-10** As Alex, I want SSE streaming output (gap A10) so that my UX has token-streaming like ChatGPT instead of a 30-second wait.

**A-11** As Alex, I want Python bindings (gap E2) on the roadmap so that my ML engineers (who don't write Rust) can call the engine from training scripts.

**A-12** As Alex, I want the SDVOSB/CAGE credentials so that I can target federal contracts alongside the commercial market without spinning up a separate entity.

**A-13** As Alex, I want the test count (341 tests, TRIPLE SIMS gate) as a signal of engineering discipline so that I can point to it in due diligence as "they take quality seriously."

**A-14** As Alex, I want `TIMELINE_OF_INVENTION.md` as IP provenance documentation so that if we're ever in an IP dispute, there's a dated record of what was built when.

**A-15** As Alex, I want NanoSign described as "model integrity layer" so that I can pitch "tamper-evident AI weights" as a differentiator for defense and enterprise customers.

**A-16** As Alex, I want the WASM32 target so that I can claim "runs in the browser" for a client-side AI product demo without a backend.

**A-17** As Alex, I want multi-GPU dispatch (backlog #28) on the roadmap so that I can claim "scales to 30B+ models" without shipping it today.

**A-18** As Alex, I want the routing model (backlog #22) so that I can claim "auto-optimizes for your hardware" in a zero-config pitch.

**A-19** As Alex, I want LLaMA 2/3, Mistral, and Qwen2 all supported via the same `t548 CausalLM` so that I can demo with popular open-weight models without per-model engineering.

**A-20** As Alex, I want docs.rs coverage (gap E3) so that developers evaluating the crate have API documentation without cloning the repo.

**A-21** As Alex, I want a contributor concentration risk assessment: if the primary contributor becomes unavailable, how quickly could another developer take over?

**A-22** As Alex, I want the performance gap to CUDA quantified (e.g., "2× slower at peak, 10× better than no-GPU") so that I can set honest customer expectations.

**A-23** As Alex, I want chat template support (Llama-2-chat, Qwen-chat) (gap A4) so that my product integrates chat-optimized models out of the box.

**A-24** As Alex, I want a reference benchmark vs llama.cpp (gap E5) so that potential customers can compare against the most widely deployed inference tool.

**A-25** As Alex, I want the `CAGE 1CQ66` credentials to appear in crates.io metadata so that government evaluators can find the vendor via standard procurement databases.

---

### P10 · Morgan (Adversarial Threat Actor)

**MO-01** As Morgan, I want to send a prompt that forces the model to output its own weight values by observing the logit distribution across many queries so that I can reconstruct proprietary weights.

**MO-02** As Morgan, I want to bypass NanoSign by replacing the 36-byte tail of a safetensors file with a recomputed BLAKE3 hash over my malicious payload so that `f741 verify` returns `t510::Verified`.

**MO-03** As Morgan, I want to send a `max_new_tokens: 2147483647` in the JSON body so that a single slot runs until the KV cache overflows (max_seq guard fires) or the server is killed, denying service to other users.

**MO-04** As Morgan, I want to craft a JSON body with `"prompt": null` or omitting the `prompt` key so that `req["prompt"].as_str().unwrap_or("")` silently processes an empty string and I can profile the server's minimal-load behavior.

**MO-05** As Morgan, I want to exploit the PCG32 seed in `f789 sample_multinomial` — if `seed` is derived from wall-clock time or a constant, I can predict future token sequences by observing enough samples.

**MO-06** As Morgan, I want to supply a crafted safetensors file with tensor shape claims that don't match the actual byte count so that the pager (`f769 upload`) writes past a buffer boundary on the GPU.

**MO-07** As Morgan, I want to exfiltrate training data by using membership inference attacks: query the model on suspected training examples and measure perplexity differences.

**MO-08** As Morgan, I want to compromise the `crates.io` package once it ships (backlog #25) by submitting a malicious patch to one of the transitive dependencies (`blake3`, `wgpu`, `tokenizers`) so that every downstream user loads my code.

**MO-09** As Morgan, I want to cause VRAM OOM by sending a request with `token_ids.len()` close to `max_seq` concurrently from `max_batch` connections so that the batch pool's memory budget overflows.

**MO-10** As Morgan, I want to time the response of `/generate` for a fixed prompt across many requests to fingerprint which GPU is executing so that I can infer the hardware configuration.

**MO-11** As Morgan, I want to abuse the `UNSIGNED` path in `f746 load_verified` — an unsigned model silently loads with only a warning to stderr, which means I can distribute an unsigned malicious model and it will load.

**MO-12** As Morgan, I want to embed adversarial triggers in a safetensors weight file (without changing the hash) so that specific input patterns cause the model to output harmful content.

**MO-13** As Morgan, I want to attack the `f793 embedding_backward` CAS loop by triggering a gradient explosion (NaN in grad_out) so that the CAS loop spins forever, hanging the training GPU.

**MO-14** As Morgan, I want to test whether KV cache contents from slot N-1 are visible in slot N when a new request reuses a freed slot — the `f802 reset_slot` zeros only the cursor, not the buffer.

**MO-15** As Morgan, I want to craft a `Content-Length` header that is larger than the actual body so that `reader.read_exact(&mut body)` blocks waiting for bytes that never arrive, holding a slot open indefinitely.

**MO-16** As Morgan, I want to intercept the `MAP_WRITE|COPY_SRC` staging buffer in the pager before `f769 upload` completes so that I can read the model weights from host-visible memory.

**MO-17** As Morgan, I want to send a request with a prompt in a language the model wasn't trained on but the tokenizer accepts so that the model generates garbage tokens that can be used to probe vocabulary coverage.

**MO-18** As Morgan, I want to flood the `TcpListener` with half-open connections (SYN flood) so that the non-blocking accept loop in `serve.rs` is saturated and legitimate requests can't be accepted.

**MO-19** As Morgan, I want to corrupt the IRONHIVE handoff CLI payload for `any-gpu-serve:start` so that bt starts serving a malicious model instead of the intended one.

**MO-20** As Morgan, I want to verify that `swap_remove` + `f807 migrate_slot` in the serve loop doesn't introduce a race where two operations modify the same batch slot concurrently when requests complete in the same batch step.

**MO-21** As Morgan, I want to abuse the `--model` CLI flag to supply a path that reads from `/proc/mem` or a FIFO so that the server stalls or reads arbitrary system memory.

**MO-22** As Morgan, I want to use adversarial suffix attacks (appending optimized tokens to a prompt) to jailbreak the model's output through the sampling pipeline despite top-k masking.

**MO-23** As Morgan, I want to submit a crafted `tokenizer.json` that maps a high-frequency byte to the EOS token so that `tok.f779()` is triggered prematurely, causing every request to return empty output.

**MO-24** As Morgan, I want to observe the timing difference between NanoSign cache hits (model already hashed) and misses (first load) to determine when new weights were deployed.

**MO-25** As Morgan, I want to exfiltrate the entire model by issuing thousands of `/generate` requests with crafted prompts designed to reproduce specific weight values via the logit outputs.

---

## Phase 5 — Acceptance Criteria (Selected High-Priority Stories)

*Given/When/Then format. Full coverage for Must Have stories; representative coverage for others.*

---

### M-01: Load Llama-2-7B and generate coherent tokens

**AC1:** Given a valid `model.safetensors`, `config.json`, and `tokenizer.json` on disk, When I run `cargo run --release --bin any-gpu-serve -- --model model.safetensors --config config.json --tokenizer tokenizer.json`, Then the server prints `[any-gpu-serve] ready — listening on 0.0.0.0:8080` within 60 seconds and exits with code 0 if killed via SIGTERM.

**AC2:** Given the server is running, When I POST `{"prompt":"Hello, world!","max_new_tokens":32}` to `http://localhost:8080/generate`, Then the response is HTTP 200 with a JSON body containing `"output"` as a non-empty string, and the output does not consist entirely of repeated tokens.

**AC3:** Given the server is running with `max_batch=4`, When I send 4 concurrent POST /generate requests, Then all 4 complete with HTTP 200, no response is `{"error":"..."}`, and total wall time is less than 4× the single-request time.

---

### M-03: Top-k sampling produces non-degenerate text

**AC1:** Given logits with a clear top-k distribution, When `f787 top_k_mask` is called with k=50, Then exactly k logits are kept (set to 0 or `-inf` for the rest), and the returned buffer has the same length as the input.

**AC2:** Given k=50 and vocab=32000, When the wave64 top-k shader runs, Then the result matches a CPU reference implementation within floating-point tolerance for at least 100 distinct logit vectors.

**AC3:** Given k > vocab_size, When `f787` is called, Then all logits are preserved (no masking applied), and the output equals the input.

---

### M-08: Continuous batching handles max_batch concurrent requests

**AC1:** Given `max_batch=4` in `config.json`, When 6 requests arrive simultaneously, Then requests 1–4 are prefilled immediately, requests 5–6 enter the pending queue, and each completes with HTTP 200 after slot compaction frees space.

**AC2:** Given 4 active slots and slot 2 completes, When `f807 migrate_slot` moves the last slot (slot 3) into position 2, Then `active[2].slot.slot == 2` and subsequent batch decode steps proceed correctly.

**AC3:** Given the active queue is empty and no pending requests exist, When the serve loop iterates, Then it sleeps for 1 ms (no busy-poll) and CPU utilization stays below 1%.

---

### R-01: NanoSign tamper detection

**AC1:** Given a signed model file produced by `f743`, When I flip byte 0 and call `f742`, Then the result is `t510::Failed { expected: <original_hash>, actual: <new_hash> }`.

**AC2:** Given a file without NSIG marker, When `f742` is called, Then the result is `t510::Unsigned` and no panic occurs.

**AC3:** Given `f741` returns `t510::Failed`, When the result is ignored by the caller, Then the `#[must_use]` attribute causes a compiler warning at the call site.

---

### MO-03: Malicious max_new_tokens DoS

**AC1:** Given `max_new_tokens = 2147483647`, When this value is parsed from the POST body, Then it is clamped to a maximum configured value (e.g., 4096) before being stored in `PendingReq`.

**AC2:** Given the KV cache is full (`cursor >= max_seq`), When `f785 decode_one` is called, Then it returns `Err(anyhow!("decode_one: KV cache is full"))` and the slot is retired.

**AC3:** Given a request that would run forever, When the serve loop processes it, Then other slots continue to decode normally without starvation.

*(Note: AC1 is a gap — there is currently no `max_new_tokens` cap in the serve binary. See Phase 11.)*

---

### J-03: Embedding backward produces correct gradients

**AC1:** Given `vocab=3, d_model=2, n_ids=2, tokens=[1,0], grad_out=[[1,2],[3,4]]`, When `f793 embedding_backward` is called, Then `grad_weight[0]=[3,4], grad_weight[1]=[1,2], grad_weight[2]=[0,0]` within 1e-5.

**AC2:** Given a repeated token (`tokens=[0,1,0]`), When `f793` is called, Then scatter-add correctly accumulates `grad_weight[0] += row0 + row2`.

**AC3:** Given a freshly allocated output buffer, When `f793` executes, Then the buffer is zero-initialized by wgpu (per WebGPU spec) without an explicit `clear_buffer` call.

---

### K-01: TRIPLE SIMS gate passes on all nodes

**AC1:** Given the bt node (RX 5700 XT, RADV/Vulkan), When `cargo run --release --bin any-gpu-test --features tests` is run, Then all 3 checks (safetensors round-trip, RoPE determinism, KV cache prefill+decode) print PASS and exit 0.

**AC2:** Given two consecutive runs of the RoPE determinism check with the same input, When both outputs are compared, Then they are bitwise identical (not just within floating-point tolerance).

**AC3:** Given the test binary is built with `--release`, When it runs, Then it completes in under 60 seconds on bt.

---

## Phase 6 — Path Analysis

### P1 (Mike) Story Cluster: Local LLM Inference

**Happy Path:**
1. `git clone` → `cargo build --release --bin any-gpu-serve`
2. Download Llama-2-7B safetensors from HuggingFace
3. `cargo run --release --bin any-gpu-serve -- --model llama2.safetensors --config config.json --tokenizer tokenizer.json`
4. Server prints device, max_batch, port
5. `curl -X POST http://localhost:8080/generate -d '{"prompt":"Tell me about the moon","max_new_tokens":128}'`
6. JSON response arrives with coherent output within ~10 seconds for 7B at f16

**Sad Path (partial failure):**
- Model is bf16 (14 GB) and exceeds 8 GB VRAM → `f769 upload` chunks via `LayerPager` → each layer pages through 512 MiB staging → functional but slow (seconds per layer)
- Recovery: use `f774 page_layer_f16` instead; or reduce `max_seq` to trade KV cache VRAM for layer storage

**Catastrophic Path (total failure):**
- `t500::f500()` returns error ("No suitable adapter found") → server exits before binding
- This happens if: no GPU present, WGPU_BACKEND is set to a backend not supported on the machine, or Mesa RADV not installed
- Recovery procedure: `any-gpu info` to check backend; `WGPU_BACKEND=vulkan any-gpu info` to force Vulkan; verify `libvulkan.so` is installed; on Debian: `apt install libvulkan1 mesa-vulkan-drivers`

---

### P2 (Jess) Story Cluster: Transformer Training

**Happy Path:**
1. Define model using `t545 Module` trait + `t546 Linear` + ops
2. Record forward pass on `t506 Tape`
3. Call `backward()` → dispatches B1–B6 shaders
4. `AdamW.step()` via `f720/f721`
5. Save checkpoint with `f745 save_signed`

**Sad Path:**
- Training a novel architecture with ops not yet in the autograd graph (e.g., sliding window attention)
- Backward not registered → panic or incorrect gradient
- Recovery: register new Op variant in `t504`, add case in backward dispatcher `f702`, add numeric gradient check test

**Catastrophic Path:**
- NaN propagation through gradient: `f793 embedding_backward` CAS loop spins if `val` is NaN (new_val = NaN, CAS never matches)
- Recovery: add NaN guard before gradient is written to tape; clip gradients before backward

---

### P3 (Dev) Story Cluster: Production Fleet Serving

**Happy Path:**
1. Build single binary
2. Deploy to bt (AMD) and lf (NVIDIA) with same binary
3. Configure `max_batch=8` per node in `config.json`
4. Load balancer health checks `/health`
5. Requests arrive, queue if full, process via continuous batching
6. Completed slots compacted via `f807`

**Sad Path:**
- One node (gd, RTX 3050 Ti) has lower VRAM → `max_batch=2` causes lower throughput
- Recovery: node-specific `config.json` with lower `max_batch` and `max_seq`

**Catastrophic Path:**
- `TcpListener::accept` returns `WouldBlock` in a tight loop if no clients connect → CPU busy-poll
- Mitigation already in code: `if pending.is_empty() && active.is_empty() { thread::sleep(1ms) }`
- But if pending is non-empty with no GPU capacity, the loop could spin. Recovery: add a sleep when active==max_batch and pending is non-empty

---

### P10 (Morgan) Story Cluster: Attack

**Attack Path (NanoSign bypass):**
1. Download legitimate safetensors file
2. Modify one weight value
3. Recompute BLAKE3 over modified payload → get new 32-byte hash
4. Construct new 36-byte trailer: `NSIG` + new hash
5. Append to modified file
6. File now passes `f742` as `t510::Verified`

**Mitigation status:** This is the expected and correct behavior of NanoSign — it signs the content hash. The security model requires that the canonical hash be distributed out-of-band (e.g., the original model card). NanoSign detects unintentional corruption and accidental tamper; it does NOT defend against a determined attacker who controls the file AND the 36-byte trailer. Gap: no public key signature.

**Recovery:** Add Ed25519 signature layer on top of BLAKE3. The NSIG trailer would contain a public-key signature over the hash, not just the hash itself.

---

## Phase 7 — Edge Cases and Failure Modes

### P1 (Mike)

| # | Edge Case | Current behavior | Risk |
|---|-----------|-----------------|------|
| E1 | `token_ids.len() > max_seq` | `ensure!` in `f784 prefill` returns `Err` | Low — gracefully errors |
| E2 | VRAM OOM during layer upload | wgpu `create_buffer` panics or returns error | High — not gracefully recovered; server crashes |
| E3 | Model file not found at startup | `std::fs::read` returns Err, process exits | Medium — clean exit, no partial state |
| E4 | Tokenizer encodes empty string | `f776` returns empty `Vec<u32>`, `f784` hits `ensure!(!token_ids.is_empty())` | Low — error returned |
| E5 | `max_new_tokens=0` in request body | `slot.tokens_left = 0.saturating_sub(1) = 0`, request completes immediately after prefill | Low — correct behavior |
| E6 | Config.json has unknown fields | `serde_json` ignores unknown fields (permissive deserialization) | Low — but silently masks typos |
| E7 | Driver update changes wgpu backend selection | Different adapter selected, tests fail | Medium — `any-gpu info` output changes, needs verification |

### P2 (Jess)

| # | Edge Case | Risk |
|---|-----------|------|
| E1 | NaN in gradient causes CAS livelock in `f793` | High |
| E2 | Gradient checkpointing not supported — long sequences OOM | High |
| E3 | Tape grows unbounded if `backward()` is never called | Medium (memory leak) |
| E4 | Mixed-precision (f16 forward, f32 backward) not tested | Medium |
| E5 | Large batch size (>512) in matmul exceeds LDS budget on RDNA1 | Low — clamped by workgroup dispatch |

### P3 (Dev)

| # | Edge Case | Risk |
|---|-----------|------|
| E1 | `f807 migrate_slot` called while decode is in progress for the source slot | High (race condition) — single-threaded serve loop prevents this today but would be a bug if async |
| E2 | Clock skew between nodes causes token count divergence in multi-GPU (when shipped) | High |
| E3 | Disk full during model load causes partial `read` | Medium — `std::fs::read` fails cleanly |
| E4 | Concurrent connection close mid-generation | Low — `write_response` returns Err, ignored |
| E5 | `config.json` specifies `max_batch=0` | `cfg.max_batch.max(1)` clamps to 1 — handled |

### P8 (River)

| # | Edge Case | Risk |
|---|-----------|------|
| E1 | Oversized Content-Length causes `read_exact` to block indefinitely | High — missing timeout on body read |
| E2 | Zero-length body with `Content-Length: 0` → `serde_json::from_slice(&[])` returns Err | Low — returns 500 |
| E3 | NSIG magic appearing mid-file (not at tail) triggers false verify detection | Already tested `f744_no_false_positive` — Low |
| E4 | PCG32 seed collision across sessions if seeded from pid or time | Medium |
| E5 | KV buffer contents from previous slot visible after `f802 reset_slot` (cursor reset, data not zeroed) | High — cross-request data leakage |

---

## Phase 8 — Cross-System Journey Maps

### Journey 1: Mike types a prompt and receives a response

```
[Mike's browser/terminal]
    │  POST {"prompt":"..."} to any-gpu-serve:8080
    ▼
[any-gpu-serve (bt node)]
    │  parse_request → tok.f776 → PendingReq
    │  prefill_slot (f788b) → t548.f784 → kv_caches filled
    │  greedy_sample OR f787/f788/f789 pipeline
    │  batch_decode_step (f789b) × N tokens
    │  finish_slot → tok.f777 → {"output":"..."}
    ▼
[Mike's terminal]
    Receives complete JSON response
```

**Handoff points:**
1. TCP accept → HTTP parse (friction: no streaming)
2. Tokenizer encoding (HF `tokenizers` crate, needs `tokenizer.json` file)
3. SafetensorsModel → LayerPager → GPU VRAM (friction: 14 GB f16 model needs paging)
4. KV cache management (seamless once allocated)
5. Token detokenization (f777/f778)

**Cross-repo dependency:** none currently. Future: kova dispatching requests to `any-gpu-serve` via handoff protocol.

---

### Journey 2: Jess trains a transformer on Mac then verifies on bt

```
[Jess's Mac M4]
    │  Write model → use Module trait + Tape
    │  Forward pass → backward() → AdamW step
    │  Save checkpoint: f745 → model.weights + NSIG + BLAKE3
    ▼
[IRONHIVE SHARED_QUEUE.md]
    │  Jess pushes checkpoint to shared storage
    ▼
[bt node (RX 5700 XT)]
    │  Kai dispatches any-gpu-test via IRONHIVE queue
    │  any-gpu-test: f761 load → f769 upload → f504 readback
    │  NanoSign: f741 verify → t510::Verified
    │  TRIPLE SIMS 3/3 pass
    ▼
[IRONHIVE SHARED_QUEUE.md]
    │  Result posted: "TRIPLE SIMS PASS on bt"
    ▼
[Jess's terminal]
    Verified: checkpoint passes on real AMD hardware
```

**Handoff points:**
1. Mac training → checkpoint file (friction: `save_to_safetensors` not yet implemented — gap D3)
2. Checkpoint transfer to bt (friction: manual rsync today; kova automation planned)
3. TRIPLE SIMS gate (seamless — already implemented)

---

### Journey 3: pixel-forge uses any-gpu for sprite diffusion training

```
[pixel-forge (sprite data: 20,599 RGBA 32×32)]
    │  Load images → tensor conversion → GPU upload (f502)
    ▼
[any-gpu nanobyte.rs pattern]
    │  NanoUNet forward: conv2d + GroupNorm + swish + upsample_nearest2d
    │  Tape records → backward() → AdamW step
    │  GPU-resident params (t550, f734) across training steps
    ▼
[any-gpu nanosign]
    │  f745 save_signed → model.weights + NSIG + BLAKE3
    ▼
[pixel-forge inference]
    │  f746 load_verified → weights loaded with tamper check
    │  Sample loop: DDPM reverse diffusion
```

**Friction points:**
1. pixel-forge's data pipeline needs to produce f32 tensors matching any-gpu's expected layout — not yet standardized
2. `save_to_safetensors` (gap D3) would allow using `t538` for trained model export
3. No `any-gpu` dependency listed in pixel-forge's `Cargo.toml` yet — integration is currently example-level

---

## Phase 9 — Adversarial User Stories

*From Morgan's perspective (nation-state / sophisticated attacker). Mitigations assessed against current code.*

| # | Story | Mitigation Status |
|---|-------|-------------------|
| MO-01 | Model weight extraction via logit queries | **No mitigation.** Output logits are returned as `{"output":"..."}` (decoded text), not raw floats. Attacker must infer weights from text, which requires many queries and is probabilistic. Medium risk. |
| MO-02 | NanoSign bypass via recomputed trailer | **By design.** NanoSign is content-addressed, not signed with a private key. Anyone can recompute the hash. The security model requires out-of-band hash distribution. Gap: no public-key signature layer. |
| MO-03 | DoS via `max_new_tokens=MAX_INT` | **No mitigation.** The serve binary does not cap `max_new_tokens`. The KV cache overflow guard in `f785` (`ensure!(start_pos < max_seq)`) will eventually fire, but only after `max_seq` tokens are generated, which can take minutes. **Critical gap.** |
| MO-05 | PCG32 seed prediction | **Partial.** The seed is `(seed, step, row)` parameters. If `seed` is a constant or time-based, the sequence is predictable. Check `src/ops/sampler.rs` seed sourcing. |
| MO-06 | Crafted safetensors with wrong tensor shape | **Partial.** `f769 upload` calls `ensure!` on data length matches. The `safetensors` crate validates the format. But GPU buffer overflows are checked by wgpu (panics, not silently corrupts). |
| MO-09 | VRAM OOM via concurrent max_seq requests | **No mitigation.** The KV pool is pre-allocated at startup, so the pool itself can't overflow. But loading a new model while serving could cause OOM. |
| MO-11 | Unsigned model loads silently | **Partial mitigation.** `f746 load_verified` prints `eprintln!("nanosign: ... is unsigned")` to stderr but does not fail. An operator may not see this warning. Gap: configurable `--require-signature` flag. |
| MO-14 | KV cache data leakage between slots | **Potential gap.** `f802 reset_slot` zeroes the cursor only. GPU buffer content from the previous request is not cleared. Next request that uses the same slot index will write from cursor=0, overwriting old data, but old data beyond `cursor` is still resident in VRAM. Whether this is exploitable depends on the attacker's ability to read partial VRAM. |
| MO-15 | Content-Length larger than body → read_exact blocks | **Real vulnerability.** `reader.read_exact(&mut body)` will block waiting for bytes that never arrive. The 30-second `set_read_timeout` mitigates this (returns `Err` after 30 seconds), but during those 30 seconds the slot is held. |
| MO-19 | IRONHIVE handoff protocol compromise | **Out of scope for this repo.** Managed by `~/.claude/collab/PROTOCOL.md`. |

---

## Phase 10 — MoSCoW Prioritization

### Must Have (ship blockers for stated goal)

| ID | Story | Justification |
|----|-------|---------------|
| M-01 | Load model + run inference | Core product |
| M-02/03/04/05 | Sampler suite (temp/top-k/top-p/rep_penalty) | **Shipped (A3, backlog done)** |
| M-08 | Continuous batching | **Shipped (backlog #27)** |
| M-07 | Health endpoint | **Shipped** |
| M-26 | Fused SDPA decode (1.1 ms) | **Shipped (P6)** |
| J-01–J-06 | Transformer backward shaders | **Shipped (B1–B6)** |
| K-01 | TRIPLE SIMS gate | **Shipped** |
| MO-03 mitigation | Cap max_new_tokens per request | **Not yet implemented — blocker for production** |

### Should Have (high value, workaround exists)

| ID | Story | Workaround |
|----|-------|------------|
| D-14 | SSE streaming output | Full response returned; clients poll |
| D-15 | Observability / metrics | Stderr logs as proxy |
| PA-04 | CONTRIBUTING.md | README covers basics |
| PA-05 | GitHub Actions CI | Manual `cargo test` |
| J-18 | `save_to_safetensors` | NanoSign `save_signed` works for native format |
| M-25 | `--max_seq` CLI override | Edit config.json |
| K-02 | test-fleet.sh | Manual SSH per node |
| C-04 | SBOM | `cargo tree` gives partial view |
| R-03 | Enforce body read timeout | 30s connection timeout partially mitigates |

### Could Have (nice to have)

| ID | Story |
|----|-------|
| D-24 | Quantization (B11 Q4/AWQ) |
| D-18 | Multi-GPU dispatch |
| J-20 | Shape inference / broadcasting in autograd |
| PA-23 | docs.rs curation |
| A-11 | Python bindings (PyO3) |
| C-04 | CycloneDX SBOM |
| R-09 | PCG32 seed audit |
| M-17 | Pending queue with backpressure signaling |
| D-16 | Per-request token limits (rate limiting) |

### Will Not Have (explicitly deferred)

| ID | Story | Reason |
|----|-------|--------|
| — | GGUF / PyTorch .bin loading | Safetensors-only policy |
| — | Python bindings (near term) | Rust-first stack |
| — | Sparse/longformer attention | Out of scope for LLM goal |
| — | conv3d | Out of scope |
| — | FIPS 140-3 blake3 certification | Upstream responsibility |

---

## Phase 11 — Gap Analysis (grounded in source)

### Critical Gaps (affect current shipping functionality)

| Gap | Source evidence | Impact |
|-----|----------------|--------|
| **No max_new_tokens cap** | `serve.rs:37 max_new: req["max_new_tokens"].as_u64().unwrap_or(128) as usize` — no upper bound enforced | DoS: single request runs until KV cache full (max_seq tokens) |
| **KV buffer not zeroed on slot reset** | `transformer.rs: f802 sets s536[p1]=0, no GPU clear` | Data leakage between requests sharing a slot index |
| **Unsigned model loads silently** | `nanosign.rs:109 eprintln!(...)` — warning not a hard error | Malicious model loads without operator awareness |
| **No body read timeout** | `serve.rs:231 reader.read_exact(&mut body)` — only 30s connection timeout | 30s slot hold for oversized Content-Length attack |

### Functional Gaps (from GAP_ANALYSIS.md, updated)

| Gap | Status (2026-05-26) |
|-----|---------------------|
| A3 Sampler suite | ✅ Shipped (A3, backlog done) |
| A4 Chat templates | ❌ Missing — Llama-2-chat, Qwen-chat glue layer |
| A10 SSE streaming | ❌ Missing |
| A11 Batched inference | ✅ Shipped (backlog #27) |
| B1–B6 Transformer backward | ✅ Shipped |
| B11 Quantization (Q4/AWQ/GPTQ) | ❌ Missing — largest remaining single item |
| C3 Fused RoPE+split_heads | ❌ Open |
| C5 RoPE cos/sin LUT | ❌ Open |
| C7 Pipeline cache LRU | ❌ Open |
| D3 save_to_safetensors | ❌ Missing |
| D7 Tensor Debug/pretty-print | ❌ Missing |
| D8 Typed u32 token IDs | ❌ Missing |
| E1 crates.io publish | ❌ Not yet |
| E6 GitHub Actions CI | ❌ Unknown / not verified |
| E7 CHANGELOG.md | ❌ Missing |
| E8 CONTRIBUTING.md | ❌ Missing |
| E9 test-fleet.sh | ❌ Missing |
| E10 End-to-end chat example | ❌ Missing |
| F1 lf re-verification (post Sprint 7) | ❌ Stale |
| F2 gd re-verification | ❌ Stale |
| F3 M4 re-verification | ❌ Stale |

---

## Phase 12 — Success Metrics

### P1 (Mike) — 30/90/365 day KPIs

| Timeframe | KPI | Target |
|-----------|-----|--------|
| 30 days | Mike has generated 100+ responses with Llama-2-7B on his RX 5700 XT | Non-degenerate outputs; no crashes |
| 30 days | Decode latency | < 200 ms/token for 7B at f16 with sampler pipeline |
| 90 days | Chat template support | Llama-2-chat format works without caller glue |
| 90 days | 13B model accessible | B11 quantization or paging strategy |
| 365 days | Mike recommends any-gpu to another AMD GPU owner | Champion NPS > 0 |

### P2 (Jess) — Success

| Timeframe | KPI |
|-----------|-----|
| 30 days | Full transformer training loop runs on M4 and bt with identical loss curves |
| 90 days | Fine-tunes a 7B model on a custom dataset with LoRA-style approach |
| 365 days | Publishes a paper using any-gpu as the compute backend |

### P3 (Dev) — SLO

| Metric | Target |
|--------|--------|
| TTFT (time to first token) | < 5 seconds for 7B on single node |
| Throughput | > 4 tokens/s per active slot at batch=4 |
| Availability | 99.5% uptime (health check passes) |
| Error rate | < 0.1% HTTP 500 responses |

### P7 (Chris) — Procurement

| Milestone | Target |
|-----------|--------|
| CAGE verified in SAM.gov | Before first contract action |
| SBOM generated | Within 30 days of crates.io publish |
| ATO documentation started | After CI and assumed-breach model are present |

---

## Phase 13 — Dependency Analysis

```
[core capability: GPU tensor ops]
    │
    ├──► [inference: safetensors loader S7.3]
    │        │
    │        ├──► [model paging S7.4] ──► [serve binary S7.7]
    │        │                                    │
    │        └──► [f16 storage S7.5]              │
    │                                             ▼
    ├──► [transformer math S7.1]         [continuous batching #27]
    │        │                                    │
    │        └──► [causal SDPA S7.2]              │
    │                  │                          │
    │                  └──► [tokenizer S7.7] ─────┘
    │
    ├──► [sampler suite A3] ──► [chat templates A4 — MISSING]
    │
    ├──► [autograd tape]
    │        │
    │        └──► [B1–B6 backward shaders] ──► [transformer fine-tuning — ENABLED]
    │                                                │
    │                                                └──► [LoRA / PEFT — unscoped]
    │
    └──► [perf: flash SDPA P5/P6] ──► [quantization B11 — MISSING]
              │
              └──► [multi-GPU #28 — MISSING] ──► [routing model #22 — MISSING]
```

**Critical path to first usable chat session:** tensor ops → transformer math → causal SDPA → tokenizer → safetensors → pager → sampler → serve binary. **All shipped.**

**Critical path to 13B+ inference:** all above + quantization (B11). **B11 not started.**

**Critical path to transformer fine-tuning:** tensor ops + autograd + B1–B6 backward shaders. **All shipped.**

---

## Phase 14 — Risk Register

### Must Have Stories Risk Assessment

| Story | Risk 1 | Risk 2 | Risk 3 |
|-------|--------|--------|--------|
| M-01 (LLM inference) | wgpu API break in wgpu=25 upgrade | RADV Mesa regression causes WGPU_BACKEND=vulkan failure | Safetensors format change in future HF models |
| M-08 (Continuous batching) | Race in `swap_remove` + `f807` if ever made async | Config typo (`max_batch=0`) — mitigated by `.max(1)` | VRAM fragmentation across many short sessions |
| MO-03 mitigation (max_new_tokens cap) | None — straightforward 2-line fix | — | — |
| J-01–J-06 (Transformer backward) | NaN gradient propagation → CAS livelock (B6 pattern) | Gradient accumulation correctness across layers | Numeric divergence between GPU and CPU reference |
| D-24 (Quantization B11) | AWQ/GPTQ format diversity requires multiple loader paths | WGSL doesn't have integer dot-product intrinsics yet | Quantization error degrades model quality below useful threshold on 8-bit models |

---

## Phase 15 — Executive Synthesis

### Top 10 Highest-Priority Stories Across All Personas

| Rank | Story | Persona | Rationale |
|------|-------|---------|-----------|
| 1 | **Cap max_new_tokens in serve.rs** (MO-03 mitigation) | P10/P3 | 2-line fix, blocks DoS, production blocker |
| 2 | **KV buffer zero-on-reset** (MO-14 mitigation) | P10/P8 | Cross-request data leak; 1 GPU clear call per slot retirement |
| 3 | **CONTRIBUTING.md + GitHub Actions** (E8/E6) | P5/P4 | Unlocks external contributors; low cost, high signal |
| 4 | **End-to-end chat example** (E10) | P1/P4 | Single most-requested missing artifact; tokenize→prefill→decode→detokenize in one runnable file |
| 5 | **test-fleet.sh** (backlog #13) | P6 | Multi-node regression in < 5 min; critical for IRONHIVE quality gate |
| 6 | **crates.io publish** (backlog #25) | P5/P9/P3 | Unlocks `cargo add any-gpu`; broadens adoption |
| 7 | **SSE streaming output** (A10) | P3/P9/P1 | Token-streaming UX; next most important serve feature after batching |
| 8 | **save_to_safetensors** (D3) | P2/P5 | Completes the weight lifecycle; needed for fine-tuning workflows |
| 9 | **Chat template support** (A4) | P1/P9 | Llama-2-chat / Qwen-chat; needed for production chat UX |
| 10 | **Quantization B11** (Q4/AWQ/GPTQ) | P1/P3/P9 | Breaks the 7B ceiling; reaches 13B+ on 8 GB VRAM |

### Top 5 Gaps Representing the Biggest Risk

| Rank | Gap | Risk |
|------|-----|------|
| 1 | **No max_new_tokens cap** | Production DoS: one request holds a batch slot for max_seq tokens |
| 2 | **KV buffer not cleared on slot reset** | Cross-request data leakage between users sharing a batch pool |
| 3 | **No CI pipeline** | Regressions on lf/gd/M4 are not caught automatically; hardware matrix is stale since Sprint 7 |
| 4 | **No quantization** | Hard ceiling at 7B on 8 GB VRAM; 13B+ requires full VRAM even with paging |
| 5 | **NanoSign has no public-key layer** | A determined attacker can recompute the BLAKE3 trailer; model integrity depends on out-of-band hash distribution |

### Recommended Sprint Scope

**Sprint 8 (immediate — all items are small):**
1. `max_new_tokens` cap in `serve.rs` (1 line)
2. `f802` zero KV buffer on slot reset (1 GPU clear call)
3. `--require-signature` flag for `any-gpu-serve` (fail if model is unsigned)
4. `CONTRIBUTING.md` + GitHub Actions `cargo test --release` + `cargo clippy`
5. End-to-end chat example binary (`src/bin/chat.rs`)
6. `test-fleet.sh` (SSH + `cargo test --release` on bt/lf/gd)

**Sprint 9 (medium scope):**
1. SSE streaming output in serve binary
2. `save_to_safetensors` (D3)
3. Chat template support for Llama-2-chat / Qwen-chat (A4)
4. crates.io publish
5. Hardware re-verification: lf (F1), gd (F2), M4 (F3)
6. `any-gpu bench --json` output flag for fleet ingestion

### Key Architectural Decisions Implied by User Stories

1. **max_new_tokens must be a hard server-side limit, not just a client parameter.** The serve binary is the last line of defense; client-supplied values cannot be trusted.

2. **Slot retirement must clear GPU KV buffers, not just CPU cursors.** The current `f802 reset_slot` is correct for performance but incorrect for security isolation. A configurable `--clear-slots` flag would satisfy security-conscious deployments.

3. **NanoSign needs a public-key layer for production model distribution.** Ed25519 over the BLAKE3 hash (36 bytes → 100 bytes: NSIG + 32B pubkey hash + 64B Ed25519 signature) would close the attacker-recomputes-trailer gap.

4. **The continuous batching loop must bound the maximum time a single request can hold a slot.** A per-slot wall-clock deadline (checked at each decode step) would allow the serve binary to evict runaway requests.

5. **The compression map (tN/fN) is the right design for this context** (IRONHIVE multi-agent collaboration, token budget management) but requires CONTRIBUTING.md coverage for external contributors. The design should be preserved, not removed.

---

*End of analysis. 250 user stories across 10 personas. All findings grounded in source files at `/home/mcochran/any-gpu/src/`.*

<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
