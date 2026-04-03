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
| 8 | [perf] | **Pipeline caching** — every dispatch calls `create_shader_module` + `create_compute_pipeline`. On NVIDIA: 1-5ms per dispatch. A 3-layer MLP training step = ~15 compilations = 15-75ms of dead overhead per step. Fix: `HashMap<u64, Arc<ComputePipeline>>` keyed by `hash(shader_src)` inside `GpuDevice` behind a `Mutex`. Zero API changes, 10-100x training speedup. | — |
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
