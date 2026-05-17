# any-gpu

Tensor engine for every GPU. AMD, NVIDIA, Intel, Apple. One codebase, one shader language, zero vendor lock-in.

## What It Is

any-gpu is a GPU compute engine written in Rust using [wgpu](https://github.com/gfx-rs/wgpu) and WGSL. It runs on every modern GPU without vendor SDKs: Vulkan on Linux (AMD, NVIDIA, Intel), Metal on macOS (Apple Silicon), DX12 on Windows.

You write your compute once in WGSL. wgpu picks the backend. The binary works everywhere.

## Who It's For

- You have an AMD or Intel GPU and want GPU-accelerated ML in Rust. CUDA can't help you. Metal can't help you. any-gpu can.
- You're building a binary that ships to a heterogeneous fleet — NVIDIA in the cloud, AMD on workstations, Apple on laptops.
- You refuse to vendor-lock your compute pipeline.
- You want a training and inference engine with no runtime dependencies beyond a GPU driver.

## Honest Positioning

CUDA is faster on NVIDIA. Metal Performance Shaders are faster on Apple. We measured it, the benchmark numbers are in the [Benchmarks](benchmarks.md) chapter, and we're not hiding them.

The point is not to beat cuBLAS. The point is that an AMD RX 5700 XT has zero CUDA support and zero MPS support. any-gpu is the only option that gives it GPU compute for ML in Rust. Intel Arc and Iris Xe are in the same position.

Use any-gpu when:
- Your GPU is AMD or Intel (where CUDA can't run)
- You need one binary that works on every machine in your fleet
- You're experimenting with ML on whatever GPU you have

## Current State (Sprint 7 complete, 2026-05-17)

| Metric | Value |
|--------|-------|
| Tests passing | 256 (all on bt — AMD RX 5700 XT, RADV/Vulkan) |
| WGSL compute shaders | 56+ |
| Rust modules | device, ops (7 submodules), tensor, autograd, optim, train, nanosign, safetensors, pager, tokenizer, module, lm |
| Inference stack | t544 Tokenizer, t545 Module, t546 Linear, t547 LmConfig, t548 CausalLM |
| Serve binary | any-gpu-serve — HTTP server, POST /generate, GET /health |
| Hardware verified | AMD RX 5700 XT, NVIDIA RTX 3070, NVIDIA RTX 3050 Ti, Apple M4 |

The engine covers the full pipeline: upload data, run compute shaders, train with autograd + AdamW, load safetensors model weights, tokenize, run LLaMA-compatible inference, serve over HTTP.
