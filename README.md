# any-gpu

Bare metal tensor engine. Runs on any GPU — AMD, NVIDIA, Intel, Apple. One codebase, zero vendor lock-in.

wgpu under the hood. Vulkan on Linux, Metal on macOS, DX12 on Windows. You write WGSL compute shaders once, they run everywhere.

## Current state (Sprint 1)

Three tensor ops as WGSL compute shaders:

- **matmul** — naive matrix multiply (16x16 workgroups)
- **add** — element-wise addition
- **mul** — element-wise multiplication

Device discovery picks the best GPU automatically. Upload f32 data, dispatch compute, read results back.

```rust
use any_gpu::GpuDevice;

let dev = GpuDevice::gpu()?;

let a = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
let b = dev.upload(&[5.0, 6.0, 7.0, 8.0]);
let c = dev.matmul(&a, &b, 2, 2, 2)?;

let result = dev.read(&c)?;
// [19.0, 22.0, 43.0, 50.0]
```

### Benchmarks (Apple M4, Metal)

| Size | CPU | GPU (compute) | Speedup |
|------|-----|---------------|---------|
| 256x256 | 9.6 ms | 1.5 ms | 6.6x |
| 512x512 | 82 ms | 4.2 ms | 19.5x |
| 1024x1024 | 806 ms | 17.3 ms | **46.6x** |

Naive shader, no tiling. The 5700 XT on Vulkan will be faster.

## What this will become

A full tensor framework for ML training. The roadmap:

- **Tensor type** with shape tracking and autograd
- **Full op set** — sub, div, exp, log, sqrt, relu, sigmoid, tanh, softmax, conv2d, transpose, reduce_sum/mean, layer_norm, batch_norm, embedding, cross_entropy, mse
- **Autograd** — reverse-mode autodiff, backward pass, gradient accumulation
- **Optimizer** — SGD, Adam
- **Layer types** — Linear, Conv2d, LayerNorm, BatchNorm
- **Training loop** as a function call, not a framework
- **Nanobyte-native storage** — mmap weight blobs, zero-copy tensors
- **Tiled matmul** with shared memory for real performance

Target: replace CUDA dependency entirely. Train models on any GPU from any vendor. Mac Mini via Metal, NVIDIA via Vulkan, AMD 5700 XT via Vulkan. One codepath.

## Build

```
cargo add any-gpu
```

Or clone and run the benchmark:

```
cargo run --release --example bench
```

## Test

```
cargo test
```

Three tests: add, mul, matmul (2x2). Requires a GPU (any backend).

## License

Unlicense. Public domain.
