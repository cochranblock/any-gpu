// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// t547 = LmConfig. Architecture parameters for a decoder-only Transformer.
// t548 = CausalLM.  LLaMA-style causal language model (MHA or GQA).
//   Weights follow HuggingFace naming: model.embed_tokens.weight,
//   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight, model.layers.{i}.mlp.*,
//   model.layers.{i}.{input,post}_attention_layernorm.weight,
//   model.norm.weight, lm_head.weight.
//   All projection weights are transposed at load time (HF [out,in] → [in,out]).
//
// f782 = LmConfig::from_json  (parse config.json string)
// f783 = CausalLM::load       (safetensors + pager → GPU weights + KV caches)
// f784 = CausalLM::prefill    (tokenize + run prompt through all layers)
// f785 = CausalLM::decode_one (single autoregressive decode step)
// f786 = CausalLM::generate   (prefill + decode loop → String)
// t556 = DecodeSlot (CPU-only; tracks one active request in the batch pool)
// f788b = CausalLM::prefill_slot  (prefill into a batch pool slot)
// f789b = CausalLM::batch_decode_step (one decode step for all active slots)

use crate::device::{t500, t501};
use crate::ops::transformer::{t534, t551};
use crate::pager::t539;
use crate::safetensors::t538;
use crate::tokenizer::t544;
use anyhow::{ensure, Result};

/// t547 = LmConfig.
#[derive(Clone, serde::Deserialize)]
pub struct t547 {
    /// Residual stream width. LLaMA-7B=4096, TinyLlama=2048.
    pub hidden_dim:   usize,
    /// Total query heads (all heads attend). LLaMA-7B=32.
    pub num_heads:    usize,
    /// KV heads for GQA. Equals num_heads for MHA. LLaMA-7B=32, TinyLlama=4.
    pub num_kv_heads: usize,
    /// Number of transformer layers.
    pub num_layers:   usize,
    /// FFN intermediate width. LLaMA-7B=11008, TinyLlama=5632.
    pub ff_dim:       usize,
    /// Vocabulary size.
    pub vocab_size:   usize,
    /// Maximum sequence length (KV cache pre-allocated to this).
    pub max_seq:      usize,
    /// RoPE rotation base. Llama default 10000.0.
    #[serde(default = "default_rope_base")]
    pub rope_base:    f32,
    /// RMSNorm epsilon. Default 1e-5.
    #[serde(default = "default_norm_eps")]
    pub norm_eps:     f32,
    /// Maximum concurrent decode requests for batch inference. Default 1 (single-request mode).
    /// Allocates `max_batch` KV slots in the batch pool. Larger values use more VRAM.
    #[serde(default = "default_max_batch")]
    pub max_batch:    usize,
}

fn default_rope_base() -> f32 { 10000.0 }
fn default_norm_eps()   -> f32 { 1e-5 }
fn default_max_batch()  -> usize { 1 }

impl t547 {
    /// f782 = LmConfig::from_json. Parse an architecture config from a JSON string.
    pub fn f782(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn head_dim(&self) -> usize { self.hidden_dim / self.num_heads }
}

/// Per-layer weight buffers.
struct LmLayer {
    attn_norm_w: t501,  // [hidden_dim]
    q_w:         t501,  // [hidden_dim, num_heads*head_dim]  (pre-transposed)
    k_w:         t501,  // [hidden_dim, num_kv_heads*head_dim]
    v_w:         t501,
    o_w:         t501,  // [num_heads*head_dim, hidden_dim]  (pre-transposed)
    ffn_norm_w:  t501,  // [hidden_dim]
    gate_w:      t501,  // [hidden_dim, ff_dim]
    up_w:        t501,
    down_w:      t501,  // [ff_dim, hidden_dim]
}

/// t556 = DecodeSlot. CPU-only state for one active request in the batch pool.
#[derive(Clone)]
pub struct t556 {
    /// Slot index in the batch pool (0..max_batch).
    pub slot: usize,
    /// Tokens generated so far (not including the prefill prompt).
    pub generated: Vec<u32>,
    /// Next token to feed as input for the next decode step.
    pub next_token: u32,
    /// Remaining decode steps allowed.
    pub tokens_left: usize,
}

/// t548 = CausalLM. LLaMA-style decoder-only transformer with KV cache.
pub struct t548 {
    pub cfg:          t547,
    embed_w:          t501,        // [vocab_size, hidden_dim]
    layers:           Vec<LmLayer>,
    final_norm_w:     t501,        // [hidden_dim]
    lm_head_w:        t501,        // [hidden_dim, vocab_size]  (pre-transposed)
    kv_caches:        Vec<t534>,   // single-request KV (f784/f785/f786)
    pub batch_kv:     Vec<t551>,   // batch KV pool (f788b/f789b): one per layer
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Upload a weight from the safetensors map, transpose from [out, in] → [in, out].
fn upload_and_transpose(dev: &t500, pager: &t539, model: &t538, key: &str,
                        out_f: usize, in_f: usize) -> Result<t501> {
    let data = model.f764(key)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {key}"))?;
    ensure!(data.len() == out_f * in_f, "weight {key}: expected {}×{} got {}", out_f, in_f, data.len());
    let raw = pager.f769(dev, data)?;
    // f641: [1, out_f, in_f, 1] → [1, in_f, out_f, 1]
    dev.f641(&raw, 1, out_f as u32, in_f as u32, 1)
}

/// Upload a 1-D weight (norm / bias) without transposing.
fn upload_1d(dev: &t500, pager: &t539, model: &t538, key: &str, len: usize) -> Result<t501> {
    let data = model.f764(key)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {key}"))?;
    ensure!(data.len() == len, "weight {key}: expected {len} got {}", data.len());
    pager.f769(dev, data)
}

// ─── CausalLM implementation ─────────────────────────────────────────────────

impl t548 {
    /// f783 = CausalLM::load. Upload all model weights from a SafetensorsModel.
    /// `pager` is a pre-allocated LayerPager (use DEFAULT_STAGE_BYTES for large models).
    pub fn f783(dev: &t500, model: &t538, pager: &t539, cfg: t547) -> Result<Self> {
        let hd = cfg.head_dim();
        let n_kv_dim = cfg.num_kv_heads * hd;
        let n_q_dim  = cfg.num_heads    * hd;

        let embed_w = upload_1d(dev, pager, model, "model.embed_tokens.weight",
                                cfg.vocab_size * cfg.hidden_dim)
            .or_else(|_| upload_1d(dev, pager, model, "transformer.wte.weight",
                                   cfg.vocab_size * cfg.hidden_dim))?;

        // Actually embed table is 2-D but stored flat: [vocab_size * hidden_dim]
        // upload_1d works because we just treat it as a 1-D slice.

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{i}");
            layers.push(LmLayer {
                attn_norm_w: upload_1d(dev, pager, model,
                    &format!("{p}.input_layernorm.weight"), cfg.hidden_dim)?,
                q_w: upload_and_transpose(dev, pager, model,
                    &format!("{p}.self_attn.q_proj.weight"), n_q_dim, cfg.hidden_dim)?,
                k_w: upload_and_transpose(dev, pager, model,
                    &format!("{p}.self_attn.k_proj.weight"), n_kv_dim, cfg.hidden_dim)?,
                v_w: upload_and_transpose(dev, pager, model,
                    &format!("{p}.self_attn.v_proj.weight"), n_kv_dim, cfg.hidden_dim)?,
                o_w: upload_and_transpose(dev, pager, model,
                    &format!("{p}.self_attn.o_proj.weight"), cfg.hidden_dim, n_q_dim)?,
                ffn_norm_w: upload_1d(dev, pager, model,
                    &format!("{p}.post_attention_layernorm.weight"), cfg.hidden_dim)?,
                gate_w: upload_and_transpose(dev, pager, model,
                    &format!("{p}.mlp.gate_proj.weight"), cfg.ff_dim, cfg.hidden_dim)?,
                up_w: upload_and_transpose(dev, pager, model,
                    &format!("{p}.mlp.up_proj.weight"), cfg.ff_dim, cfg.hidden_dim)?,
                down_w: upload_and_transpose(dev, pager, model,
                    &format!("{p}.mlp.down_proj.weight"), cfg.hidden_dim, cfg.ff_dim)?,
            });
        }

        let final_norm_w = upload_1d(dev, pager, model, "model.norm.weight", cfg.hidden_dim)?;
        let lm_head_w = upload_and_transpose(dev, pager, model, "lm_head.weight",
                                             cfg.vocab_size, cfg.hidden_dim)?;

        // Pre-allocate single-request KV caches (f784/f785/f786)
        let kv_caches = (0..cfg.num_layers)
            .map(|_| t534::f672(dev, cfg.max_seq as u32,
                                cfg.num_kv_heads as u32, hd as u32))
            .collect();

        // Pre-allocate batch KV pool (f788b/f789b); max_batch slots per layer
        let max_batch = cfg.max_batch.max(1) as u32;
        let batch_kv = (0..cfg.num_layers)
            .map(|_| t551::f800(dev, max_batch, cfg.num_kv_heads as u32,
                                cfg.max_seq as u32, hd as u32))
            .collect();

        Ok(Self { cfg, embed_w, layers, final_norm_w, lm_head_w, kv_caches, batch_kv })
    }

    /// Single transformer layer forward. `x` is [seq_len, hidden_dim].
    /// Updates `kv` in-place with the new tokens; `start_pos` is the absolute
    /// position of the first token in `x` (for RoPE and causal mask offset).
    fn layer_forward(
        &mut self, dev: &t500, x: &t501,
        layer_idx: usize, seq_len: u32, start_pos: u32,
    ) -> Result<t501> {
        let cfg = &self.cfg;
        let hd   = cfg.head_dim() as u32;
        let nh   = cfg.num_heads as u32;
        let nkvh = cfg.num_kv_heads as u32;
        let hdim = cfg.hidden_dim as u32;
        let lyr  = &self.layers[layer_idx];

        // 1. Pre-attention RMSNorm
        let normed = dev.f603(x, &lyr.attn_norm_w, seq_len, hdim, cfg.norm_eps)?;

        // 2. Q/K/V projections  [seq, hidden] @ [hidden, proj_dim] = [seq, proj_dim]
        let q_flat = dev.f580(&normed, &lyr.q_w, seq_len, nh * hd, hdim)?;
        let k_flat = dev.f580(&normed, &lyr.k_w, seq_len, nkvh * hd, hdim)?;
        let v_flat = dev.f580(&normed, &lyr.v_w, seq_len, nkvh * hd, hdim)?;

        // 3. split_heads: [seq, n*hd] → [n, seq, hd]
        let q = dev.f627(&q_flat, nh, seq_len, hd)?;
        let k = dev.f627(&k_flat, nkvh, seq_len, hd)?;
        let v = dev.f627(&v_flat, nkvh, seq_len, hd)?;

        // 4. RoPE on Q and K
        let q = dev.f625(&q, nh, seq_len, hd, start_pos, cfg.rope_base)?;
        let k = dev.f625(&k, nkvh, seq_len, hd, start_pos, cfg.rope_base)?;

        // 5. Append to KV cache
        self.kv_caches[layer_idx].f673(dev, &k, &v, seq_len)?;
        let kv_len = self.kv_caches[layer_idx].f675();
        let k_full = self.kv_caches[layer_idx].f676();
        let v_full = self.kv_caches[layer_idx].f677();

        // 6. Expand KV for GQA if needed: [nkvh, kv_len, hd] → [nh, kv_len, hd]
        let (k_exp, v_exp) = if nh == nkvh {
            // MHA: use cache buffers directly (cloned as t501 references)
            // repeat_kv with n_rep=1 returns a copy — avoid for MHA
            (k_full.clone_via(dev), v_full.clone_via(dev))
        } else {
            (dev.f629(k_full, nh, nkvh, kv_len, hd)?,
             dev.f629(v_full, nh, nkvh, kv_len, hd)?)
        };

        // 7. Fused causal SDPA: [nh, seq, hd] (Q) × [nh, kv, hd] (K,V) → [nh, seq, hd]
        let attn = dev.f626(&q, &k_exp, &v_exp, nh, seq_len, kv_len, hd)?;

        // 8. merge_heads: [nh, seq, hd] → [seq, nh*hd]
        let attn_flat = dev.f628(&attn, nh, seq_len, hd)?;

        // 9. Output projection
        let attn_out = dev.f580(&attn_flat, &lyr.o_w, seq_len, hdim, nh * hd)?;

        // 10. Residual
        let x = dev.f550(x, &attn_out)?;

        // 11. Pre-FFN RMSNorm
        let normed2 = dev.f603(&x, &lyr.ffn_norm_w, seq_len, hdim, cfg.norm_eps)?;

        // 12. FFN: SwiGLU — gate * swish(gate) element-wise mul with up, then down
        let gate  = dev.f580(&normed2, &lyr.gate_w, seq_len, cfg.ff_dim as u32, hdim)?;
        let up    = dev.f580(&normed2, &lyr.up_w,   seq_len, cfg.ff_dim as u32, hdim)?;
        let sgate = dev.f556(&gate)?;            // swish(gate)
        let ffn   = dev.f552(&sgate, &up)?;      // element-wise multiply
        let ffn_out = dev.f580(&ffn, &lyr.down_w, seq_len, hdim, cfg.ff_dim as u32)?;

        // 13. Residual
        dev.f550(&x, &ffn_out)
    }

    /// f784 = CausalLM::prefill. Run the prompt tokens through all layers.
    /// Resets all KV caches, fills them from the prompt, and returns the
    /// logits for the last token (shape [1, vocab_size]).
    pub fn f784(&mut self, dev: &t500, token_ids: &[u32]) -> Result<t501> {
        ensure!(!token_ids.is_empty(), "prefill: empty token list");
        ensure!(token_ids.len() <= self.cfg.max_seq,
            "prefill: {} tokens exceeds max_seq {}", token_ids.len(), self.cfg.max_seq);

        // Reset all caches
        for kv in &mut self.kv_caches { kv.f674(); }

        let seq_len = token_ids.len() as u32;
        let hdim    = self.cfg.hidden_dim as u32;
        let vsz     = self.cfg.vocab_size as u32;

        // Upload token ids as f32 and embed
        let ids_f32: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
        let ids_buf = dev.f502(&ids_f32);
        let mut x = dev.f670(&ids_buf, &self.embed_w, seq_len, vsz, hdim)?;

        for i in 0..self.cfg.num_layers {
            x = self.layer_forward(dev, &x, i, seq_len, 0)?;
        }

        // Final norm on last token only
        let last_start = ((seq_len - 1) * hdim) as usize;
        let last_tok = x.slice(dev, last_start, hdim as usize)?;
        let normed = dev.f603(&last_tok, &self.final_norm_w, 1, hdim, self.cfg.norm_eps)?;

        // LM head: [1, hidden] @ [hidden, vocab] = [1, vocab]
        dev.f580(&normed, &self.lm_head_w, 1, vsz, hdim)
    }

    /// f785 = CausalLM::decode_one. Single decode step given the next token id.
    /// KV caches must already contain the context from prefill or prior decode steps.
    /// Returns logits [1, vocab_size].
    pub fn f785(&mut self, dev: &t500, token_id: u32) -> Result<t501> {
        let start_pos = self.kv_caches[0].f675(); // current cache length = next absolute pos
        ensure!(start_pos < self.cfg.max_seq as u32,
            "decode_one: KV cache is full ({} tokens)", start_pos);

        let hdim = self.cfg.hidden_dim as u32;
        let vsz  = self.cfg.vocab_size as u32;

        let id_buf = dev.f502(&[token_id as f32]);
        let mut x = dev.f670(&id_buf, &self.embed_w, 1, vsz, hdim)?;

        for i in 0..self.cfg.num_layers {
            x = self.layer_forward(dev, &x, i, 1, start_pos)?;
        }

        let normed = dev.f603(&x, &self.final_norm_w, 1, hdim, self.cfg.norm_eps)?;
        dev.f580(&normed, &self.lm_head_w, 1, vsz, hdim)
    }

    /// f786 = CausalLM::generate. Tokenize `prompt`, prefill, then decode up to
    /// `max_new_tokens` tokens (stops early at EOS). Returns the generated text only
    /// (not including the prompt).
    pub fn f786(
        &mut self,
        dev: &t500,
        tok: &t544,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String> {
        let ids = tok.f776(prompt, true)?;
        let eos = tok.f779();

        // Prefill: returns logits for position after the last prompt token
        let logits = self.f784(dev, &ids)?;
        let mut next_id = argmax_logits(dev, &logits, self.cfg.vocab_size as u32)?;

        let mut generated: Vec<u32> = vec![next_id];

        for _ in 1..max_new_tokens {
            if Some(next_id) == eos { break; }
            let logits = self.f785(dev, next_id)?;
            next_id = argmax_logits(dev, &logits, self.cfg.vocab_size as u32)?;
            generated.push(next_id);
        }

        // Strip trailing EOS if present
        if generated.last().copied() == eos {
            generated.pop();
        }

        tok.f777(&generated)
    }

    // ─── batch inference (f788b / f789b) ─────────────────────────────────────

    /// One transformer layer in batch-decode mode.
    /// `x`:            [active_batch, hidden_dim]
    /// `start_pos_buf`: [active_batch] as f32 — absolute KV position before this step
    /// Returns: [active_batch, hidden_dim]
    fn layer_forward_batch_decode(
        &mut self,
        dev: &t500,
        x: &t501,
        layer_idx: usize,
        active_batch: u32,
        start_pos_buf: &t501,
    ) -> Result<t501> {
        let hd   = self.cfg.head_dim() as u32;
        let nh   = self.cfg.num_heads as u32;
        let nkvh = self.cfg.num_kv_heads as u32;
        let hdim = self.cfg.hidden_dim as u32;

        // Step 1: Pre-attention RMSNorm
        let normed = dev.f603(x, &self.layers[layer_idx].attn_norm_w,
                              active_batch, hdim, self.cfg.norm_eps)?;

        // Step 2: Q/K/V projections → [B, nh*hd] and [B, nkvh*hd]
        // These are identical to [B*nh, hd] and [B*nkvh, hd] in memory (same flat layout).
        let q_flat = dev.f580(&normed, &self.layers[layer_idx].q_w,
                              active_batch, nh * hd, hdim)?;
        let k_flat = dev.f580(&normed, &self.layers[layer_idx].k_w,
                              active_batch, nkvh * hd, hdim)?;
        let v_flat = dev.f580(&normed, &self.layers[layer_idx].v_w,
                              active_batch, nkvh * hd, hdim)?;

        // Step 3: Batch RoPE on Q ([B*nh, hd]) and K ([B*nkvh, hd])
        let q = dev.f631(&q_flat, start_pos_buf, active_batch * nh, nh, hd, self.cfg.rope_base)?;
        let k = dev.f631(&k_flat, start_pos_buf, active_batch * nkvh, nkvh, hd, self.cfg.rope_base)?;

        // Step 4: Append new K/V tokens to the batch KV pool (advances cursors)
        self.batch_kv[layer_idx].f805(dev, &k, &v_flat, active_batch as usize)?;

        // Step 5: Build kv_lens buffer (post-append cursors)
        let kv_lens_buf = self.batch_kv[layer_idx].f806(dev, active_batch as usize);
        let max_kv_seq  = self.cfg.max_seq as u32;

        // Step 6: Batch decode SDPA — Q [B*nh, hd] × K/V [max_batch*nkvh, max_kv_seq, hd]
        let k_buf = self.batch_kv[layer_idx].f804k().clone();
        let v_buf = self.batch_kv[layer_idx].f804v().clone();
        let attn = dev.f630(&q, &k_buf, &v_buf, &kv_lens_buf,
                            active_batch, nh, nkvh, max_kv_seq, hd)?;

        // Step 7: Output projection: [B*nh, hd] treated as [B, nh*hd] @ [nh*hd, hdim]
        let attn_out = dev.f580(&attn, &self.layers[layer_idx].o_w,
                                active_batch, hdim, nh * hd)?;

        // Step 8: Residual
        let x = dev.f550(x, &attn_out)?;

        // Step 9: Pre-FFN RMSNorm
        let normed2 = dev.f603(&x, &self.layers[layer_idx].ffn_norm_w,
                               active_batch, hdim, self.cfg.norm_eps)?;

        // Step 10: SwiGLU FFN
        let gate    = dev.f580(&normed2, &self.layers[layer_idx].gate_w,
                               active_batch, self.cfg.ff_dim as u32, hdim)?;
        let up      = dev.f580(&normed2, &self.layers[layer_idx].up_w,
                               active_batch, self.cfg.ff_dim as u32, hdim)?;
        let sgate   = dev.f556(&gate)?;
        let ffn     = dev.f552(&sgate, &up)?;
        let ffn_out = dev.f580(&ffn, &self.layers[layer_idx].down_w,
                               active_batch, hdim, self.cfg.ff_dim as u32)?;

        // Step 11: Residual
        dev.f550(&x, &ffn_out)
    }

    /// f788b = CausalLM::prefill_slot. Prefill `token_ids` into batch pool slot `slot`.
    /// Runs the normal single-request prefill (f784), then copies each layer's KV
    /// into the batch pool. After this, batch_kv[layer].cursor(slot) == token_ids.len().
    /// Returns the logits for the position after the last prefill token.
    pub fn f788b(&mut self, dev: &t500, slot: usize, token_ids: &[u32]) -> Result<t501> {
        ensure!(slot < self.cfg.max_batch.max(1),
            "f788b: slot {} >= max_batch {}", slot, self.cfg.max_batch.max(1));
        // Run standard prefill — fills self.kv_caches
        let logits = self.f784(dev, token_ids)?;
        // Copy each layer's prefilled KV into the batch pool at `slot`
        for layer_idx in 0..self.cfg.num_layers {
            let cursor     = self.kv_caches[layer_idx].s513;
            let max_seq_src = self.kv_caches[layer_idx].s514;
            let k_src = self.kv_caches[layer_idx].s511.clone();
            let v_src = self.kv_caches[layer_idx].s512.clone();
            self.batch_kv[layer_idx].f801(dev, slot, &k_src, &v_src, cursor, max_seq_src)?;
        }
        Ok(logits)
    }

    /// f789b = CausalLM::batch_decode_step. Run one decode step for `next_tokens.len()`
    /// concurrent requests (slots 0..active_batch-1 of the batch pool).
    /// Reads start positions from batch_kv[0].s536 (cursor before append).
    /// Returns logits: [active_batch, vocab_size].
    pub fn f789b(&mut self, dev: &t500, next_tokens: &[u32]) -> Result<t501> {
        let active_batch = next_tokens.len() as u32;
        ensure!(active_batch > 0, "f789b: empty next_tokens");
        ensure!(active_batch <= self.cfg.max_batch.max(1) as u32,
            "f789b: active_batch {} > max_batch {}", active_batch, self.cfg.max_batch.max(1));

        let hdim = self.cfg.hidden_dim as u32;
        let vsz  = self.cfg.vocab_size as u32;

        // Read start positions (= current KV cursor before this step) for RoPE
        let start_pos_f32: Vec<f32> = (0..active_batch as usize)
            .map(|s| self.batch_kv[0].s536[s] as f32)
            .collect();
        let start_pos_buf = dev.f502(&start_pos_f32);

        // Embed all next tokens: [active_batch, hidden_dim]
        let ids_f32: Vec<f32> = next_tokens.iter().map(|&id| id as f32).collect();
        let ids_buf = dev.f502(&ids_f32);
        let mut x = dev.f670(&ids_buf, &self.embed_w, active_batch, vsz, hdim)?;

        // Run all transformer layers in batch-decode mode
        for layer_idx in 0..self.cfg.num_layers {
            x = self.layer_forward_batch_decode(dev, &x, layer_idx, active_batch, &start_pos_buf)?;
        }

        // Final norm: [active_batch, hidden_dim]
        let normed = dev.f603(&x, &self.final_norm_w, active_batch, hdim, self.cfg.norm_eps)?;

        // LM head: [active_batch, hidden_dim] @ [hidden_dim, vocab_size] = [active_batch, vocab_size]
        dev.f580(&normed, &self.lm_head_w, active_batch, vsz, hdim)
    }
}

/// Read argmax from a logits buffer [1, vocab_size] back to CPU and return the token id.
fn argmax_logits(dev: &t500, logits: &t501, vocab_size: u32) -> Result<u32> {
    let idx_buf = dev.f671(logits, 1, vocab_size)?;
    let idx_f32 = dev.f504(&idx_buf)?;
    Ok(idx_f32[0] as u32)
}

// ─── t501 slice helper ────────────────────────────────────────────────────────

/// Extension trait on t501 for GPU-side sub-slicing and cloning.
trait GpuBufExt {
    /// Return a new t501 containing elements [start, start+len).
    fn slice(&self, dev: &t500, start: usize, len: usize) -> Result<t501>;
    /// Return a new t501 that is a full copy of self.
    fn clone_via(&self, dev: &t500) -> t501;
}

impl GpuBufExt for t501 {
    fn slice(&self, dev: &t500, start: usize, len: usize) -> Result<t501> {
        ensure!(start + len <= self.s507, "slice: out of bounds");
        let dst = dev.f503(len);
        let mut enc = dev.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(
            &self.s505, (start * 4) as u64,
            &dst.s505,  0,
            (len * 4) as u64,
        );
        dev.s501.submit(Some(enc.finish()));
        Ok(dst)
    }

    fn clone_via(&self, dev: &t500) -> t501 {
        let dst = dev.f503(self.s507);
        let mut enc = dev.s500.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.s505, 0, &dst.s505, 0, self.s506);
        dev.s501.submit(Some(enc.finish()));
        dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;

    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    // f782: parse a minimal JSON config.
    #[test]
    fn f782_parse_config() {
        let json = r#"{
            "hidden_dim": 64, "num_heads": 2, "num_kv_heads": 2,
            "num_layers": 2, "ff_dim": 128, "vocab_size": 32, "max_seq": 512
        }"#;
        let cfg = t547::f782(json).unwrap();
        assert_eq!(cfg.hidden_dim, 64);
        assert_eq!(cfg.num_heads, 2);
        assert_eq!(cfg.head_dim(), 32);
        assert!((cfg.rope_base - 10000.0).abs() < 1.0);
        assert!((cfg.norm_eps - 1e-5).abs() < 1e-10);
    }

    // GpuBufExt::slice: extract last element from a buffer.
    #[test]
    fn slice_last_element() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let buf = dev().f502(&v);
        let last = buf.slice(dev(), 3, 1).unwrap();
        let got = dev().f504(&last).unwrap();
        f544(&got, &[4.0], 1e-6);
    }

    // GpuBufExt::clone_via: copy produces equal buffer.
    #[test]
    fn clone_via_equals_original() {
        let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let buf = dev().f502(&v);
        let copy = buf.clone_via(dev());
        let got = dev().f504(&copy).unwrap();
        f544(&got, &v, 1e-6);
    }
}
