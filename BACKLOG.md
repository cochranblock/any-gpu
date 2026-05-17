<!-- Unlicense — cochranblock.org -->
<!-- Self-reorganizes by recency and relevance. Most important at top. -->

# Backlog

| # | Tag | Item | Depends on |
|---|-----|------|------------|
| ~~1~~ | ~~[feature]~~ | ~~Tiled matmul~~ — shipped [`0ca243d`](https://github.com/cochranblock/any-gpu/commit/0ca243d). 1.7x gain at 1024x1024 (117 GFLOPS). Further gains need register blocking + 32x32 tiles. | done |
| ~~2~~ | ~~[feature]~~ | ~~Tensor type~~ — shipped. 6-dim inline shape, reshape (zero-copy), zeros, from_buf. 8 tests. Autograd fields (requires_grad, grad_fn) deferred to Sprint 4. | done |
| ~~3~~ | ~~[feature]~~ | ~~Autograd tape~~ — shipped. Flat tape, enum Op, reverse walk. backward() for add/sub/mul/scale/relu/sigmoid/swish/tanh/matmul/mse_loss. 5 tests. | done |
| ~~4~~ | ~~[feature]~~ | ~~Backward shaders~~ — shipped relu_backward, sigmoid_backward, swish_backward, tanh_backward (4 new WGSL kernels). Remaining: softmax, group_norm, conv2d. | partial |
| ~~5~~ | ~~[feature]~~ | ~~Backward for conv2d/conv_transpose2d~~ — shipped. grad_input via conv_transpose2d (no new shader), grad_weight via new WGSL kernel, grad_bias via reduction. Tested with numeric gradient checks. | done |
| ~~6~~ | ~~[feature]~~ | ~~AdamW optimizer~~ — shipped. Single WGSL shader, bias correction, weight decay. In-place param update. 3 tests. | done |
| ~~7~~ | ~~[feature]~~ | ~~Training loop~~ — shipped. train_step() = forward + backward + optimizer. Linear regression test trains y=2x+1 from scratch. | done |
| ~~8~~ | ~~[perf]~~ | ~~Pipeline caching~~ — shipped. `Mutex<HashMap<u64, Arc<ComputePipeline>>>` in `GpuDevice`, keyed by `DefaultHasher(shader_src)`. `unary_op`, `binary_op`, `dispatch_shader`, and `AdamW::step` all route through `dev.pipeline()`. 4 new cache tests (same-Arc assertion, different-shader assertion, no-growth assertion, correctness-after-cache). 145 tests pass. | done |
| ~~S7.1~~ | ~~[feature]~~ | ~~Transformer math primitives (Sprint 7 step 1)~~ — shipped 2026-05-15. f602 layer_norm, f603 rms_norm, f563 gelu, f670 embedding_lookup, f671 argmax. 7 new WGSL shaders, 29 tests, 174 total. | done |
| ~~S7.2~~ | ~~[feature]~~ | ~~Causal SDPA + RoPE + KV cache (Sprint 7 step 2)~~ — shipped 2026-05-15. f624 apply_causal_mask, f623 scaled_dot_product_attention_causal (asymmetric q_seq_len ≤ kv_seq_len), f625 rope (adjacent-pair, start_pos + base params). t534 KVCache type with f672 new, f673 append, f674 reset, f675 cursor, f676 k_buffer, f677 v_buffer. 3 new WGSL shaders, 21 tests, 195 total. | done |
| ~~S7.3~~ | ~~[feature]~~ | ~~Safetensors loader + bf16/f16 weight ingest (Sprint 7 step 3)~~ — shipped 2026-05-16. New `src/safetensors.rs` module: t538 SafetensorsModel, f760 load-from-path (NanoSign-aware), f761 from-bytes, f762 names, f763 shape, f764 data, f765 upload-to-GPU. Free fns f766 bf16_to_f32 + f767 f16_to_f32. Deps: safetensors 0.4 + tempfile 3 (dev). 19 tests, 214 total. | done |
| ~~S7.3.1~~ | ~~[test]~~ | ~~exopack TRIPLE SIMS gate for any-gpu~~ — shipped 2026-05-16. Added optional `exopack 0.2` dep + `tests` feature, `src/bin/any-gpu-test.rs` test binary that exercises safetensors load → GPU upload → readback, RoPE determinism check (two runs must be bitwise equal), causal SDPA with first-row-equals-V[0] assertion, KV cache prefill + decode + readback. Invocation: `cargo run --release --bin any-gpu-test --features tests`. 3/3 passes on bt (RX 5700 XT, RADV/Vulkan). | done |
| ~~S7.4~~ | ~~[feature]~~ | ~~Pinned-RAM staging + layer paging from system RAM to VRAM (Sprint 7 step 4)~~ — shipped 2026-05-16. New `src/pager.rs` module: t539 LayerPager (MAP_WRITE\|COPY_SRC staging buffer in host-visible system RAM). f768 new (stage_bytes param; `DEFAULT_STAGE_BYTES` = 512 MiB covers any 13B-class tensor), f769 upload (f32 slice → VRAM; auto-chunks tensors larger than staging window), f770 page_layer (named tensors from SafetensorsModel → HashMap<String, t501>). Also fixed pre-existing flaky cache-count test (`f507_cache_grows_then_stabilizes`) to use Arc::ptr_eq instead of total-count assertion. 7 tests, 228 total. | Safetensors loader (S7.3) |
| ~~S7.5~~ | ~~[feature]~~ | ~~f16 storage type (Sprint 7 step 5)~~ — shipped 2026-05-16. t540 = GpuBufferF16 (packed 2 f16/u32). f771 upload_f16 (&[u16] → t540; odd-length zero-padded). f772 f16_to_f32 GPU kernel using WGSL `unpack2x16float` — `enable f16;` skipped (not yet in Naga/wgpu, tracked at gfx-rs/wgpu#4384). f773 pager::upload_f16_raw (chunked via staging). f774 pager::page_layer_f16 (f32→f16 on CPU via `half` crate, then staged into VRAM). s521 has_f16 added to t500 (SHADER_F16 adapter flag). dep: `half = "2"`. 5 new tests, 228→233 total. | Safetensors loader (S7.3) |
| S7.6 | [perf] | Flash-attention-style tiled SDPA (Sprint 7 step 6) — long contexts. | Causal SDPA (S7.2) |
| S7.7 | [feature] | Tokenizer + `Module` graph + `any-gpu serve` runtime (Sprint 7 step 7) — end-user interface. | All above |
| 9 | [perf] | **GPU-resident params across train steps** — `train_step()` reads all params to CPU Vec then re-uploads every step (acknowledged in a comment). For 100K params: 400KB×2 PCIe per step = ~800MB/1000 steps of pointless transfers. Fix: `GpuParams` struct holds `GpuBuffer` weights persistently. Optimizer updates in-place. Params only touch CPU at explicit checkpoint. | Pipeline caching (#8) |
| 10 | [feature] | **Wire `Tensor` to ops + fix README** — `Tensor` type (src/tensor.rs, 15 tests, exported) connects to nothing. Users track shapes manually. Add `Tensor::matmul`, `::relu`, `::conv2d`, `::softmax`, `::mse_loss` wrapping GpuDevice ops. Also fix README: test count (62→141), Sprint 3/4 marked shipped, remove "Planned: Layer 1" (it's partially built). | — |
| 11 | [build] | Stratagems CLI with clap: `any-gpu train <stratagem>`, `any-gpu bench`, `any-gpu info` | Training loop (#7) |
| 12 | [feature] | Starter nanobyte: ~1M param diffusion model for 32x32 pixel art, trained on bt's 5700 XT | Training loop (#7), pixel-forge sprite data |
| 13 | [test] | Run 141 tests on bt/lf/gd after every push — add a `test-fleet.sh` script that SSHs all 3 nodes | — |
| 14 | [fix] | gd Intel Iris Xe untested — force `WGPU_BACKEND=vulkan` to skip NVIDIA, test Intel path in isolation | — |
| 15 | [test] | conv2d backward stride/dilation coverage — numeric gradient checks only cover stride=1, pad=0. Add stride=2+padding cases (common ResNet/UNet pattern, off-by-one in grad_weight shader would not be caught by current tests) | — |
| 16 | [test] | Large tensor test (>16M elements) to exercise the >65535 workgroup 2D dispatch path | — |
| 17 | [feature] | conv2d groups > 1 test — depthwise separable conv needed for efficient UNet, shader supports it but untested | — |
| 18 | [fix] | `opt-level = "z"` → `opt-level = 3` in release profile — size optimization is wrong for a perf library; hurts CPU-side loops and reference impls | — |
| 19 | [fix] | `NanoSignResult` not enforced by compiler — `load_verified()` returns an enum, caller can silently accept `Unsigned` variant. Consider returning `Result<Vec<u8>, NanoSignError>` to make unsigned files an explicit error path | — |
| 20 | [feature] | Subgroup operations for warp-level reduction in softmax and group_norm | — |
| 21 | [research] | Benchmark tiled matmul designs: 16x16 tile vs 32x32 tile, shared memory budget per GPU | — |
| 22 | [feature] | Auto-benchmark routing model: microbench per op/size on first run, bake nanobyte .weights | Starter nanobyte (#12), kova pyramid architecture |
| 23 | [docs] | Add any-gpu to kova's module table and node deploy scripts (bt needs any-gpu for pixel-forge training) | kova docs |
| 24 | [feature] | Multi-GPU dispatch: split work across discrete + integrated by measured throughput | Routing model (#22) |
| 25 | [build] | Publish to crates.io: `cargo publish` — needs license file check, README examples, docs.rs badges | Tensor ops (#10) |

---

Part of [The Cochran Block](https://cochranblock.org) — see also [kova](https://github.com/cochranblock/kova), [pixel-forge](https://github.com/cochranblock/pixel-forge)
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
