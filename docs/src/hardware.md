# Hardware

## Verified Hardware

All 256 tests pass on bt (AMD RX 5700 XT). Full hardware verification table:

| Node | GPU | VRAM | Driver | OS | Tests | Status |
|------|-----|------|--------|----|-------|--------|
| bt | AMD Radeon RX 5700 XT (RADV NAVI10) | 8 GB | Mesa 25.0.7 | Debian 13, kernel 6.12.73 | 256/256 | **pass** |
| lf | NVIDIA GeForce RTX 3070 Laptop | 8 GB | 550.163.01 | Debian 13, kernel 6.12.73 | pass (earlier sprint) | pass |
| gd | NVIDIA GeForce RTX 3050 Ti Laptop | 4 GB | 550.163.01 | Debian 13, kernel 6.12.73 | pass (earlier sprint) | pass |
| local | Apple M4 | Unified | — | macOS Tahoe 25.3.0 | pass (earlier sprint) | pass |

Reproduce on bt:

```bash
WGPU_BACKEND=vulkan cargo test --release
```

## RADV Quirks (AMD on Linux)

The AMD RADV Vulkan driver has several constraints that drove design decisions across the codebase:

### Shared LazyLock device

Concurrent `wgpu::Instance` creation segfaults on RADV/RDNA1. The fix: every test shares a single `GpuDevice` via `LazyLock`:

```rust
// In tests — always use this, never create your own:
static TEST_DEV: LazyLock<GpuDevice> = LazyLock::new(|| GpuDevice::f500().unwrap());
```

Never create a separate `OnceLock<GpuDevice>` or `LazyLock<GpuDevice>` in a new test module. Add your tests to the existing shared-device pattern. Creating a second adapter concurrently will crash the process on RADV.

This is tracked at commit [`e124fbb`](https://github.com/cochranblock/any-gpu/commit/e124fbb).

### No arrayLength() in WGSL

`arrayLength()` in WGSL compiles to `OpArrayLength` in SPIR-V, which crashes some RADV drivers. All shaders in any-gpu use uniform params for buffer length instead:

```wgsl
// Wrong — crashes RADV:
let n = arrayLength(&buf);

// Right — pass n via uniform:
@group(0) @binding(2) var<uniform> p: Params;
// p.n is the element count
```

### No enable f16 in WGSL

`enable f16;` in WGSL is not supported by Naga (the WGSL compiler in wgpu) as of gfx-rs/wgpu#4384. any-gpu uses packed u32 storage with `unpack2x16float` for f16 dequantization:

- `t540 = GpuBufferF16` — packed f16 data: 2 f16 elements per u32
- `f771` — upload &[u16] → t540
- `f772` — dequant t540 → t501 via `unpack2x16float` WGSL intrinsic

### No enumerate_adapters()

Calling `wgpu::Instance::enumerate_adapters()` with GL backend probing crashes on Linux when RADV is present. `t500::f500()` uses `request_adapter(HighPerformance)` only — no enumeration.

### No concurrent adapter requests

`WGPU_BACKEND=vulkan cargo test --release` is preferred on AMD to avoid the driver spending time probing other backends.

## wgpu Backend Auto-Selection

| Platform | Backend | Notes |
|----------|---------|-------|
| Linux | Vulkan | AMD (RADV), NVIDIA, Intel |
| macOS | Metal | Apple Silicon and Intel Mac |
| Windows | DX12 | AMD, NVIDIA, Intel |
| Any | OpenGL | Fallback, disabled in any-gpu to avoid RADV crashes |

Override: `WGPU_BACKEND=vulkan` (or `metal`, `dx12`).

## Design Target

The primary design target is the bt node: AMD Radeon RX 5700 XT, 8 GB VRAM, RDNA1, RADV driver, Debian 13. Every WGSL quirk fix and VRAM budget decision (fused SDPA, LayerPager, f16 storage) targets this configuration. If it runs correctly and efficiently on the 5700 XT, it runs on everything.
