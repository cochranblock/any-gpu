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

use crate::device::{t500, t501};
use crate::ops::transformer::t534;
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
}

fn default_rope_base() -> f32 { 10000.0 }
fn default_norm_eps()   -> f32 { 1e-5 }

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

/// t548 = CausalLM. LLaMA-style decoder-only transformer with KV cache.
pub struct t548 {
    cfg:          t547,
    embed_w:      t501,        // [vocab_size, hidden_dim]
    layers:       Vec<LmLayer>,
    final_norm_w: t501,        // [hidden_dim]
    lm_head_w:    t501,        // [hidden_dim, vocab_size]  (pre-transposed)
    kv_caches:    Vec<t534>,
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

        // Pre-allocate KV caches for all layers
        let kv_caches = (0..cfg.num_layers)
            .map(|_| t534::f672(dev, cfg.max_seq as u32,
                                cfg.num_kv_heads as u32, hd as u32))
            .collect();

        Ok(Self { cfg, embed_w, layers, final_norm_w, lm_head_w, kv_caches })
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
