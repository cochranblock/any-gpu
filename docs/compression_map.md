<!-- Unlicense — cochranblock.org -->
<!-- Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7 -->

# any-gpu Compression Map

Tokenization for traceability. Convention: workspace tokenization rules (see `kova/assets/prompts/tokenization.mdc`).

Numbering range: any-gpu uses **t500+** for types and **f500+** for functions so it never collides with [kova's map](../../kova/docs/compression_map.md) (max kova: T215, f402).

Preserved (not compressed): Rust std types, primitives, traits, ecosystem types (`Arc`, `HashMap`, `Mutex`, `Path`, `PathBuf`, `wgpu::*`, `bytemuck::Pod`, `anyhow::Result`, `blake3::Hash`). Local file-private `struct P { ... }`, `struct Params`, and `struct Dims` shader-uniform wrappers are already-compressed local-scope names and stay as-is.

## Types (tN)

| Token | Human name | Module |
|-------|------------|--------|
| t500 | GpuDevice | device |
| t501 | GpuBuffer | device |
| t502 | Tensor | tensor |
| t503 | TensorId | autograd |
| t504 | Op | autograd |
| t505 | TapeEntry | autograd |
| t506 | Tape | autograd |
| t507 | AdamW | optim |
| t508 | AdamWParams | optim (private) |
| t509 | StepResult | train |
| t510 | NanoSignResult | nanosign |
| t511 | ElemParams | ops/mod (private) |
| t512 | ScaleParams | ops/elementwise (private) |
| t513 | ReduceParams | ops/attention (private) |
| t514 | SoftmaxParams | ops/attention (private) |
| t515 | ConcatParams | ops/tensor_ops (private) |
| t516 | SliceParams | ops/tensor_ops (private) |
| t517 | BroadcastAddParams | ops/tensor_ops (private) |
| t518 | SumInnerParams | ops/tensor_ops (private) |
| t519 | AddPerColParams | ops/tensor_ops (private) |
| t520 | SumRowsParams | ops/tensor_ops (private) |
| t521 | TransposeParams | ops/tensor_ops (private) |
| t522 | GNStatsParams | ops/norm (private) |
| t523 | LNParams | ops/norm (private) |
| t524 | EmbedParams | ops/transformer (private) |
| t525 | ArgmaxParams | ops/transformer (private) |
| t526 | UpsampleParams | ops/upsample (private) |
| t527 | UpsampleBackwardParams | ops/upsample (private) |
| t528 | MatmulDims | ops/conv (private) |
| t529 | BatchMatmulDims | ops/conv (private) |
| t530 | Conv2dParams | ops/conv (private) |
| t531 | ConvTranspose2dParams | ops/conv (private) |
| t532 | Conv2dGradParams | ops/conv (private) |
| t533 | Conv2dGradBiasParams | ops/conv (private) |
| t534 | KVCache | ops/transformer (Sprint 7 step 2) |
| t535 | CausalMaskParams | ops/attention (private, Sprint 7 step 2) |
| t536 | RopeParams | ops/attention (private, Sprint 7 step 2) |
| t537 | KVAppendParams | ops/transformer (private, Sprint 7 step 2) |
| t538 | SafetensorsModel | safetensors (Sprint 7 step 3) |
| t539 | LayerPager | pager (Sprint 7 step 4) |
| t540 | GpuBufferF16 | device (Sprint 7 step 5) |
| t541 | FusedSdpaParams | ops/attention (private, Sprint 7 step 6) |
| t542 | HeadParams | ops/attention (private, Sprint 7 step 7) |
| t543 | RepeatKvParams | ops/attention (private, Sprint 7 step 7) |
| t544 | Tokenizer | tokenizer (Sprint 7 step 7) |
| t545 | Module | module (trait, Sprint 7 step 7) |
| t546 | Linear | module (Sprint 7 step 7) |
| t547 | LmConfig | lm (Sprint 7 step 7) |
| t548 | CausalLM | lm (Sprint 7 step 7) |
| t549 | GpuBatch | device (batch-dispatch recording guard) |
| t550 | GpuParams | train (GPU-resident param set for f734) |
| t551 | BatchKvPool | lm — batch KV pool for f788b/f789b (continuous batching) |
| t552 | CopyToSlotParams | ops/transformer — uniform params for SHADER_COPY_TO_SLOT |
| t553 | BatchDecodeAppendParams | ops/transformer — uniform params for SHADER_BATCH_DECODE_APPEND |
| t554 | BatchDecodeSdpaParams | ops/attention — uniform params for fused batch-decode SDPA |
| t555 | RopeBatchParams | ops/attention — uniform params for SHADER_ROPE_BATCH |
| t556 | DecodeSlot | lm — CPU-only; tracks one active request in the batch pool |
| t557 | CopySlotParams | ops/transformer — uniform params for slot-migrate shaders (f807) |

## Functions (fN)

### device (f500–f519)

| Token | Human name | Notes |
|-------|------------|-------|
| f500 | gpu | Blocking init, desktop only |
| f501 | gpu_async | Async init, desktop + wasm |
| f502 | upload | Upload &[f32] to GPU |
| f503 | alloc | Empty GPU buffer of N f32s |
| f504 | read | Blocking readback to Vec<f32> |
| f505 | read_async | Async readback |
| f506 | upload_uniform | pub(crate) uniform buffer helper |
| f507 | pipeline | pub(crate) compiled-pipeline cache lookup |
| f508 | pipeline_cache_len | test-only |
| f509 | begin | Enter batch recording mode; returns t549 |
| f510 | execute | Submit batch without CPU poll (t549 method) |
| f511 | sync | Submit batch + poll until GPU done (t549 method) |

### tensor (f520–f539)

| Token | Human name |
|-------|------------|
| f520 | Tensor::new |
| f521 | Tensor::from_buf |
| f522 | Tensor::zeros |
| f523 | Tensor::shape |
| f524 | Tensor::ndim |
| f525 | Tensor::numel |
| f526 | Tensor::to_vec |
| f527 | Tensor::to_vec_async |
| f528 | Tensor::buffer |
| f529 | Tensor::reshape |
| f530 | Tensor::dim |
| f531 | Tensor::matmul |
| f532 | Tensor::relu |
| f533 | Tensor::softmax |
| f534 | Tensor::mse_loss |
| f535 | Tensor::conv2d |

### ops/mod helpers (f540–f549)

| Token | Human name |
|-------|------------|
| f540 | dispatch_1d |
| f541 | unary_op |
| f542 | binary_op |
| f543 | dispatch_shader |
| f544 | assert_approx (test) |

### ops/elementwise (f550–f579)

| Token | Human name |
|-------|------------|
| f550 | add |
| f551 | sub |
| f552 | mul |
| f553 | scale |
| f554 | relu |
| f555 | sigmoid |
| f556 | swish |
| f557 | tanh_act |
| f558 | gelu |
| f559 | relu_backward |
| f560 | sigmoid_backward |
| f561 | swish_backward |
| f562 | tanh_backward |
| f563 | gelu (tanh approximation) |

### ops/conv (f580–f599)

| Token | Human name |
|-------|------------|
| f580 | matmul |
| f581 | batch_matmul |
| f582 | conv2d |
| f583 | conv_transpose2d |
| f584 | conv2d_grad_weight (pub crate) |
| f585 | conv2d_grad_bias (pub crate) |

### ops/norm (f600–f619)

| Token | Human name |
|-------|------------|
| f600 | group_norm |
| f601 | group_norm_backward |
| f602 | layer_norm |
| f603 | rms_norm |

### ops/attention (f620–f639)

| Token | Human name |
|-------|------------|
| f620 | softmax |
| f621 | scaled_dot_product_attention |
| f622 | mse_loss |
| f623 | scaled_dot_product_attention_causal (Sprint 7 step 2) |
| f624 | apply_causal_mask (Sprint 7 step 2) |
| f625 | rope — rotary position embeddings (Sprint 7 step 2) |
| f626 | scaled_dot_product_attention_fused — online-softmax causal SDPA, no N×N alloc (Sprint 7 step 6) |
| f627 | split_heads — [seq, n*hd] → [n, seq, hd] (Sprint 7 step 7) |
| f628 | merge_heads — [n, seq, hd] → [seq, n*hd] (Sprint 7 step 7) |
| f629 | repeat_kv — GQA key/value expansion [n_kv, kv_seq, hd] → [n, kv_seq, hd] (Sprint 7 step 7) |
| f630 | fused_sdpa_batch_decode — wave64 or scalar batch-decode SDPA; Q:[B*nh,hd], K/V pool, per-slot kv_lens |
| f631 | rope_batch_decode — batch RoPE with per-request start_pos_buf[B] |

### ops/tensor_ops (f640–f659)

| Token | Human name |
|-------|------------|
| f640 | concat |
| f641 | transpose |
| f642 | add_broadcast |
| f643 | slice_per_block (pub crate) |
| f644 | sum_inner (pub crate) |
| f645 | add_per_col |
| f646 | sum_rows (pub crate) |

### ops/upsample (f660–f669)

| Token | Human name |
|-------|------------|
| f660 | upsample_nearest2d |
| f661 | upsample_nearest2d_backward |

### ops/transformer (f670–f679, f800–f807)

| Token | Human name |
|-------|------------|
| f670 | embedding_lookup |
| f671 | argmax |
| f672 | KVCache::new (Sprint 7 step 2) |
| f673 | KVCache::append (Sprint 7 step 2) |
| f674 | KVCache::reset (Sprint 7 step 2) |
| f675 | KVCache::cursor (Sprint 7 step 2) |
| f676 | KVCache::k_buffer (Sprint 7 step 2) |
| f677 | KVCache::v_buffer (Sprint 7 step 2) |
| f800 | BatchKvPool::new |
| f801 | BatchKvPool::copy_from_kvcache — copy prefilled t534 into a pool slot |
| f802 | BatchKvPool::reset_slot |
| f803 | BatchKvPool::cursor |
| f804k/f804v | BatchKvPool::k_buf / v_buf |
| f805 | BatchKvPool::batch_decode_append — append one token per active slot |
| f806 | BatchKvPool::kv_lens_buf — cursors[0..n] as f32 GPU buffer |
| f807 | BatchKvPool::migrate_slot — copy slot src→dst via staging buffers (compaction) |

### autograd (f680–f719)

| Token | Human name |
|-------|------------|
| f680 | Tape::new |
| f681 | Tape::leaf |
| f682 | Tape::read |
| f683 | Tape::read_grad |
| f684 | Tape::push_result (private) |
| f685 | Tape::buf (private) |
| f686 | Tape::add |
| f687 | Tape::sub |
| f688 | Tape::mul |
| f689 | Tape::scale |
| f690 | Tape::relu |
| f691 | Tape::sigmoid |
| f692 | Tape::swish |
| f693 | Tape::tanh_act |
| f694 | Tape::matmul |
| f695 | Tape::mse_loss |
| f696 | Tape::conv2d |
| f697 | Tape::concat |
| f698 | Tape::group_norm |
| f699 | Tape::upsample_nearest2d |
| f700 | Tape::add_broadcast |
| f701 | Tape::add_per_col |
| f702 | Tape::backward |
| f703 | Tape::accum_grad (private) |

### optim (f720–f729)

| Token | Human name |
|-------|------------|
| f720 | AdamW::new |
| f721 | AdamW::step |

### train (f730–f739)

| Token | Human name |
|-------|------------|
| f730 | train_step |
| f731 | GpuParams::new |
| f732 | GpuParams::checkpoint |
| f733 | GpuParams::len |
| f734 | train_step_gpu |

### nanosign (f740–f759)

| Token | Human name |
|-------|------------|
| f740 | sign |
| f741 | verify |
| f742 | verify_bytes |
| f743 | sign_bytes |
| f744 | strip_bytes |
| f745 | save_signed |
| f746 | load_verified |
| f747 | hex (private) |

### safetensors loader (f760–f779) — Sprint 7 step 3

| Token | Human name |
|-------|------------|
| f760 | SafetensorsModel::load (from path; signature-aware via f746) |
| f761 | SafetensorsModel::from_bytes |
| f762 | SafetensorsModel::names |
| f763 | SafetensorsModel::shape |
| f764 | SafetensorsModel::data |
| f765 | SafetensorsModel::upload (to GPU as t501) |
| f766 | bf16_to_f32 (free fn) |
| f767 | f16_to_f32 (free fn) |

### tokenizer (f775–f779) — Sprint 7 step 7

| Token | Human name |
|-------|------------|
| f775 | Tokenizer::load (from_file) |
| f775b | Tokenizer::from_bytes |
| f776 | Tokenizer::encode |
| f777 | Tokenizer::decode |
| f778 | Tokenizer::vocab_size |
| f779 | Tokenizer::eos_id |

### module (f780–f789) — Sprint 7 step 7

| Token | Human name |
|-------|------------|
| f780 | Linear::from_weights (pre-transposed weight) |
| f781 | Linear::from_f32 (HF [out,in] → transpose → [in,out]) |

### lm (f782–f786) — Sprint 7 step 7

| Token | Human name |
|-------|------------|
| f782 | LmConfig::from_json |
| f783 | CausalLM::load |
| f784 | CausalLM::prefill |
| f785 | CausalLM::decode_one |
| f786 | CausalLM::generate |
| f788b | CausalLM::prefill_slot — prefill into a batch pool slot |
| f789b | CausalLM::batch_decode_step — one decode step for all active slots |

### pager (f768–f774) — Sprint 7 step 4

| Token | Human name |
|-------|------------|
| f768 | LayerPager::new |
| f769 | LayerPager::upload (f32 slice → VRAM via staging) |
| f770 | LayerPager::page_layer (named tensors from SafetensorsModel → VRAM map) |
| f771 | GpuDevice::upload_f16 (&[u16] → t540; packs 2 f16 per u32) |
| f772 | GpuDevice::f16_to_f32 (t540 → t501 via unpack2x16float kernel) |
| f773 | LayerPager::upload_f16_raw (&[u16] → t540 via staging) |
| f774 | LayerPager::page_layer_f16 (named tensors → HashMap<String, t540>) |

## Struct fields (sN)

| Token | Human name | Owner type |
|-------|------------|------------|
| s500 | device | t500 (GpuDevice) |
| s501 | queue | t500 |
| s502 | adapter_name | t500 |
| s503 | backend | t500 |
| s504 | pipeline_cache | t500 |
| s505 | buffer | t501 (GpuBuffer) |
| s506 | size | t501 |
| s507 | len | t501 |
| s508 | buf | t502 (Tensor) |
| s509 | dims | t502 |
| s510 | ndim | t502 |
| s511 | k | t534 (KVCache) |
| s512 | v | t534 |
| s513 | cursor | t534 |
| s514 | max_seq | t534 |
| s515 | batch_heads | t534 |
| s516 | head_dim | t534 |
| s517 | tensors | t538 (HashMap<String, Vec<f32>>) |
| s518 | shapes | t538 (HashMap<String, Vec<u32>>) |
| s519 | staging | t539 (LayerPager staging buffer: MAP_WRITE \| COPY_SRC) |
| s520 | cap | t539 (staging capacity in bytes) |
| s521 | has_f16 | t500 (SHADER_F16 device feature supported) |
| s508 | batch_state | t500 (Mutex<Option<t549State>> — None=eager, Some=batch recording) |
| s509 | has_subgroup | t500 (SUBGROUP feature; enables subgroupMax/subgroupAdd shaders) |
| s522 | buf | t540 (wgpu::Buffer for packed f16 data) |
| s523 | size | t540 (byte length = ceil(s524/2)*4) |
| s524 | len | t540 (number of f16 elements) |

(Internal shader-uniform field names like `n`, `rows`, `cols`, `eps` stay as-is — they map directly to WGSL uniform struct fields and renaming would desync Rust/WGSL.)

## Crate-root allow attribute

`src/lib.rs` carries `#![allow(non_camel_case_types, non_snake_case, dead_code, unused_imports)]` per the workspace tokenization rule.

## Documentation pattern

Every tokenized item carries a doc comment mapping back to the human name, per the workspace convention:

```rust
/// f580 = matmul. C = A @ B where A is [m,k] and B is [k,n].
pub fn f580(&self, ...) { ... }
```

## Test naming

Tests in any-gpu use the function-under-test token as a prefix: `f580_basic`, `f580_vs_cpu`, `f602_constant_input`. The existing test names migrate via mechanical rename keyed by the function they exercise.
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
