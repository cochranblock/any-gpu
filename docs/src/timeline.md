# Timeline of Invention

See [TIMELINE_OF_INVENTION.md](https://github.com/cochranblock/any-gpu/blob/main/TIMELINE_OF_INVENTION.md) for the full dated commit-level record.

## Summary Table

| Date | Entry | Tests | Key Commits |
|------|-------|-------|-------------|
| 2026-05-17 | Sprint 7 complete: fused SDPA + tokenizer + LLM inference stack + serve runtime | 239→256 | ae8a688 (S7.6), 522a63a (S7.7) |
| 2026-05-17 | S7.4 LayerPager + S7.5 GpuBufferF16 | 228→233 | — |
| 2026-05-16 | Self-licking test audit + hardcoded-reference backstops | 214→221 | — |
| 2026-05-16 | Safetensors loader (Sprint 7, step 3) | 195→214 | — |
| 2026-05-15 | Causal SDPA + RoPE + KV cache (Sprint 7, step 2) | 174→195 | — |
| 2026-05-15 | Transformer inference primitives (Sprint 7, step 1) | 145→174 | — |
| 2026-04-09 | Autograd, training loop, pipeline caching | 62→145 | 0ca243d, dd55772, 5137d40, c09b0ee, 9511a61, 9f0b567, 6d93866, a905cd1 |
| 2026-04-03 | NanoSign + full doc update | — | 5e58eb3 |
| 2026-04-02 | CPU-validated test suite | 27→54 | 801c4de |
| 2026-04-02 | 15 diffusion training ops | — | 8aa9fc1, 56976a7 |
| 2026-04-02 | CUDA/Metal comparison benchmarks | — | d6ab4ec |
| 2026-04-02 | AMD RADV segfault fix | — | 35c75ef, e124fbb, 56976a7 |
| 2026-04-02 | 4-GPU benchmark matrix | — | 1a93e7f |
| 2026-04-02 | Sprint 1: wgpu compute backend | — | e1a6d96 |
