<!-- Unlicense — cochranblock.org -->
<!-- Self-reorganizes by recency and relevance. Most important at top. -->

# Backlog

| # | Tag | Item | Depends on |
|---|-----|------|------------|
| 1 | [feature] | Tiled matmul with workgroup shared memory — 5-10x perf gain, closes CUDA gap from 4x to <2x | — |
| 2 | [feature] | Tensor type with shape, strides, requires_grad, grad_fn — needed before autograd | — |
| 3 | [feature] | Autograd tape: record forward ops, reverse-mode backward pass | Tensor type (#2) |
| 4 | [feature] | Backward shaders: relu, sigmoid, swish, tanh, softmax, mse, group_norm (~10 new WGSL kernels) | Autograd tape (#3) |
| 5 | [feature] | Backward for conv2d/conv_transpose2d — weight gradient conv, input gradient via transpose conv | Autograd tape (#3) |
| 6 | [feature] | AdamW optimizer — operates on grad buffers, weight update shader | Backward shaders (#4) |
| 7 | [feature] | Training loop: forward + backward + optimizer step as a single function call | AdamW (#6) |
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
