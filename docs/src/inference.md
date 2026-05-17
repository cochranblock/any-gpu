# Inference

Sprint 7 (complete 2026-05-17) ships a full LLaMA-compatible inference stack: tokenizer, module layer, LM head, and an HTTP serve binary.

## Types

| Token | Human Name | Purpose |
|-------|------------|---------|
| t544 | Tokenizer | HuggingFace tokenizers crate wrapper |
| t545 | Module | trait: `forward(&self, dev, x) -> Result<Tensor>` |
| t546 | Linear | GPU linear layer — wraps a weight Tensor |
| t547 | LmConfig | parsed config.json (vocab_size, hidden_size, num_heads, etc.) |
| t548 | CausalLM | LLaMA-compatible model: embedding + layers + LM head + KV cache |

## Tokenizer (t544)

```rust
use any_gpu::t544;

let tok = t544::f775(Path::new("tokenizer.json"))?;
let ids: Vec<u32> = tok.f776("hello world")?;
let text: String = tok.f777(&ids)?;
let eos: u32 = tok.f779();
```

Wraps the HuggingFace `tokenizers` crate (version 0.21). Supports any tokenizer.json from the HuggingFace hub. `f776 = encode`, `f777 = decode`, `f778 = vocab_size`, `f779 = eos_id`.

## Linear Layer (t546)

```rust
use any_gpu::t546;

// Load pre-transposed weight (already [in, out])
let linear = t546::f780(weight_tensor);

// Load from HuggingFace format [out, in] — transposed at load time
let linear = t546::f781(&dev, &safetensors_model, "model.layers.0.self_attn.q_proj")?;
```

HuggingFace stores linear weights as `[out_features, in_features]`. `f781` transposes to `[in_features, out_features]` at load time so forward pass is just `f580(x, weight)`.

## LmConfig (t547)

Config JSON format (LLaMA-compatible):

```json
{
  "vocab_size": 32000,
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

`num_key_value_heads < num_attention_heads` enables GQA (Grouped Query Attention). KV heads are expanded via `f629 = repeat_kv` before SDPA.

Load:

```rust
use any_gpu::t547;
let config = t547::f782(Path::new("config.json"))?;
```

## CausalLM (t548)

LLaMA-compatible forward pass. Architecture:

1. `f670` embedding lookup — token ids → hidden states
2. For each layer:
   - `f603` RMSNorm on input
   - `f627` split_heads on Q, K, V projections
   - `f625` RoPE on Q and K
   - `f673` KVCache::append K and V
   - `f629` repeat_kv if GQA (num_kv_heads < num_heads)
   - `f626` fused SDPA (online-softmax, no N×N alloc)
   - `f628` merge_heads
   - Output projection
   - Residual add
   - `f603` RMSNorm on input to MLP
   - MLP: gate proj * silu(up proj), then down proj
   - Residual add
3. Final `f603` RMSNorm
4. LM head linear projection → logits
5. `f671` argmax → next token id

**Prefill** (`f784`): processes the full prompt in one forward pass. Returns the last logit position.

**Decode** (`f785`): processes one token at a time. KVCache holds all previous K/V. Returns the next token id.

**Generate** (`f786`): calls prefill once, then decode in a loop until EOS or max_new_tokens.

## Fused SDPA (f626)

`f626 = scaled_dot_product_attention_fused` uses online softmax to avoid materializing the full N×N attention score matrix. For a sequence of length N with head_dim D:

- Standard SDPA: allocates `[batch_heads, N, N]` — O(N²) VRAM
- Fused SDPA: streams through K/V in tiles, keeps only running max/sum — O(N) VRAM

This is what allows any-gpu to handle long contexts within 8 GB VRAM (the bt node's RX 5700 XT limit).

## Running the Serve Binary

```bash
cargo build --release --bin any-gpu-serve

./target/release/any-gpu-serve \
  --model /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer /path/to/tokenizer.json \
  --port 8080
```

Routes:

- `GET /health` — returns `{"status":"ok"}`
- `POST /generate` — body: `{"prompt":"...", "max_new_tokens":50}` — returns `{"text":"..."}`

The server loads the model once at startup (weights streamed to VRAM via `t539 = LayerPager`), then handles requests sequentially. Each request gets a fresh KV cache reset.

## HuggingFace Weight Naming

any-gpu expects LLaMA-style weight names:

```
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

This matches LLaMA 2, LLaMA 3, Mistral, and Qwen2 safetensors files from the HuggingFace hub (use `--ignore-patterns "*.bin" "*.gguf"` when downloading to get safetensors only).
