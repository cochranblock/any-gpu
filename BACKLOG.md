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
| 8 | [feature] | Pipeline caching — cache compiled shader pipelines per (shader_src, bind_group_layout) to eliminate per-dispatch compilation | — |
| 9 | [build] | Stratagems CLI with clap: `any-gpu train <stratagem>`, `any-gpu bench`, `any-gpu info` | Training loop (#7) |
| 10 | [feature] | Starter nanobyte: ~1M param diffusion model for 32x32 pixel art, trained on bt's 5700 XT | Training loop (#7), pixel-forge sprite data |
| 11 | [test] | Run 62 tests on bt/lf/gd after every push — add a `test-fleet.sh` script that SSHs all 3 nodes | — |
| 12 | [fix] | gd Intel Iris Xe untested — force `WGPU_BACKEND=vulkan` to skip NVIDIA, test Intel path in isolation | — |
| 13 | [feature] | conv2d groups > 1 test — depthwise separable conv needed for efficient UNet, shader supports it but untested | — |
| 14 | [test] | Large tensor test (>16M elements) to exercise the >65535 workgroup 2D dispatch path | — |
| 15 | [feature] | Subgroup operations for warp-level reduction in softmax and group_norm | — |
| 16 | [research] | Benchmark tiled matmul designs: 16x16 tile vs 32x32 tile, shared memory budget per GPU | — |
| 17 | [feature] | Auto-benchmark routing model: microbench per op/size on first run, bake nanobyte .weights | Starter nanobyte (#10), kova pyramid architecture |
| 18 | [docs] | Add any-gpu to kova's module table and node deploy scripts (bt needs any-gpu for pixel-forge training) | kova docs |
| 19 | [feature] | Multi-GPU dispatch: split work across discrete + integrated by measured throughput | Routing model (#17) |
| 20 | [build] | Publish to crates.io: `cargo publish` — needs license file check, README examples, docs.rs badges | Tiled matmul (#1), Tensor type (#2) |

---

Part of [The Cochran Block](https://cochranblock.org) — see also [kova](https://github.com/cochranblock/kova), [pixel-forge](https://github.com/cochranblock/pixel-forge)
