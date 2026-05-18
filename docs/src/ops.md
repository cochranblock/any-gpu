# Ops Catalog

All ops dispatch WGSL compute shaders via wgpu. Full token↔name mapping: [docs/compression_map.md](https://github.com/cochranblock/any-gpu/blob/main/docs/compression_map.md).

## Elementwise (f550–f563)

| Token | Op | Notes |
|-------|----|-------|
| f550 | add | element-wise A + B |
| f551 | sub | element-wise A - B |
| f552 | mul | element-wise A * B |
| f553 | scale | multiply every element by a scalar |
| f554 | relu | max(0, x) |
| f555 | sigmoid | 1/(1+e^-x) |
| f556 | swish / silu | x * sigmoid(x) |
| f557 | tanh_act | tanh(x) |
| f558 | gelu | GELU tanh approximation (GPT-2/PyTorch `approximate="tanh"`) |
| f559 | relu_backward | upstream_grad * (x > 0 ? 1 : 0) |
| f560 | sigmoid_backward | grad * sigmoid(x) * (1 - sigmoid(x)) |
| f561 | swish_backward | grad * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))) |
| f562 | tanh_backward | grad * (1 - tanh(x)^2) |

## Convolution / Matmul (f580–f585)

| Token | Op | Notes |
|-------|----|-------|
| f580 | matmul | tiled 16x16 shared-memory matmul, [m,k] @ [k,n] |
| f581 | batch_matmul | [batch, m, k] @ [batch, k, n] |
| f582 | conv2d | im2col-style, any kernel size, any stride/padding |
| f583 | conv_transpose2d | transposed convolution (deconv) |
| f584 | conv2d_grad_weight | gradient w.r.t. conv2d weight (pub crate) |
| f585 | conv2d_grad_bias | gradient w.r.t. conv2d bias (pub crate) |

## Normalization (f600–f603)

| Token | Op | Notes |
|-------|----|-------|
| f600 | group_norm | two-pass: stats per group → normalize + affine |
| f601 | group_norm_backward | gradient through group_norm |
| f602 | layer_norm | two-pass: per-row mean/inv_std → normalize + affine |
| f603 | rms_norm | two-pass: per-row inv_rms → scale + per-col weight; Llama/Mistral standard |

## Attention (f620–f629)

| Token | Op | Notes |
|-------|----|-------|
| f620 | softmax | two-pass numerically stable (max subtraction → exp/sum → divide) |
| f621 | scaled_dot_product_attention | Q K^T / sqrt(d) → softmax → V |
| f622 | mse_loss | mean squared error, scalar output |
| f623 | scaled_dot_product_attention_causal | SDPA with asymmetric q/kv seq-len (prefill + decode) |
| f624 | apply_causal_mask | in-place causal mask: scores[i,j]=-1e30 for j>i+offset |
| f625 | rope | rotary position embeddings, adjacent-pair, parameterized by start_pos + base |
| f626 | scaled_dot_product_attention_fused | online-softmax fused SDPA — no N×N alloc, handles long contexts within 8 GB VRAM |
| f627 | split_heads | [seq, n*hd] → [n, seq, hd] |
| f628 | merge_heads | [n, seq, hd] → [seq, n*hd] |
| f629 | repeat_kv | GQA expansion [n_kv, kv_seq, hd] → [n, kv_seq, hd] |

## Tensor Ops (f640–f646)

| Token | Op | Notes |
|-------|----|-------|
| f640 | concat | concat along any axis |
| f641 | transpose | swap two axes |
| f642 | add_broadcast | add [batch, d] to [d] (bias add pattern) |
| f643 | slice_per_block | pub(crate) slice helper |
| f644 | sum_inner | pub(crate) inner reduction |
| f645 | add_per_col | add column vector to each row |
| f646 | sum_rows | pub(crate) row reduction |

## Spatial (f660–f661)

| Token | Op | Notes |
|-------|----|-------|
| f660 | upsample_nearest2d | 2D nearest-neighbor upsampling |
| f661 | upsample_nearest2d_backward | gradient through upsample |

## Transformer Primitives (f670–f677)

| Token | Op | Notes |
|-------|----|-------|
| f670 | embedding_lookup | gather rows from [vocab, d_model] by token id |
| f671 | argmax | argmax along last dim (LM-head greedy sampler) |
| f672 | KVCache::new | allocate K and V buffers for max_seq tokens |
| f673 | KVCache::append | strided-write WGSL shader, increments cursor |
| f674 | KVCache::reset | zero cursor (reuse buffers without realloc) |
| f675 | KVCache::cursor | current sequence length |
| f676 | KVCache::k_buffer | read-only handle to K buffer |
| f677 | KVCache::v_buffer | read-only handle to V buffer |

## f16 / Staging (f768–f774)

| Token | Op | Notes |
|-------|----|-------|
| f768 | LayerPager::new | pinned MAP_WRITE\|COPY_SRC staging buffer |
| f769 | LayerPager::upload | f32 slice → VRAM via staging |
| f770 | LayerPager::page_layer | named tensors from SafetensorsModel → VRAM map |
| f771 | GpuDevice::upload_f16 | &[u16] → t540 (packs 2 f16 per u32) |
| f772 | GpuDevice::f16_to_f32 | t540 → t501 via unpack2x16float kernel |
| f773 | LayerPager::upload_f16_raw | &[u16] → t540 via staging |
| f774 | LayerPager::page_layer_f16 | named tensors → HashMap\<String, t540\> |

Note: `enable f16` in WGSL is NOT supported by Naga/wgpu ([gfx-rs/wgpu#4384](https://github.com/gfx-rs/wgpu/issues/4384)). any-gpu uses packed u32 storage with `unpack2x16float` for f16 dequant instead.

## SafeTensors Loader (f760–f767)

| Token | Op | Notes |
|-------|----|-------|
| f760 | SafetensorsModel::load | load from path; NanoSign-aware (HF unsigned files pass through) |
| f761 | SafetensorsModel::from_bytes | in-memory parse |
| f762 | SafetensorsModel::names | list tensor names |
| f763 | SafetensorsModel::shape | shape of named tensor |
| f764 | SafetensorsModel::data | CPU f32 slice |
| f765 | SafetensorsModel::upload | RAM → GPU as t501 |
| f766 | bf16_to_f32 | bit-shift (bf16 IS the top 16 bits of f32) |
| f767 | f16_to_f32 | full IEEE 754 binary16 decode (normals, denormals, zeros, infs, NaNs) |

Supported dtypes: F32, BF16, F16. GGUF, PyTorch .bin, and ONNX are not supported and not planned.

## Optimizer (f720–f721)

| Token | Op | Notes |
|-------|----|-------|
| f720 | AdamW::new | allocate momentum + velocity buffers per param |
| f721 | AdamW::step | single WGSL dispatch: momentum, velocity, weight decay, bias correction |

## Loss

| Token | Op | Notes |
|-------|----|-------|
| f622 | mse_loss | mean squared error (lives in attention module, f622) |
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
