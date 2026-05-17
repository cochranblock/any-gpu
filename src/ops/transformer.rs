// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// f670=embedding_lookup, f671=argmax. Transformer inference primitives.
// t534=KVCache + f672..f677 KV cache methods (Sprint 7 step 2).
//
// Token IDs are passed as f32 — exact for vocab sizes up to 2^24 (~16.7M), which covers
// every published tokenizer. A typed u32 buffer is a later refactor; the f32 path keeps
// the t501 surface uniform.

use crate::device::{t500, t501};
use anyhow::{ensure, Result};

/// t524 = EmbedParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t524 {
    n: u32,
    n_ids: u32,
    d_model: u32,
    vocab_size: u32,
}

// One thread per output element. Reads token id, looks up that row of the weight matrix.
// Out-of-range ids are clamped to the last vocab entry rather than corrupting memory.
const SHADER_EMBED: &str = "
struct P { n: u32, n_ids: u32, d_model: u32, vocab_size: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ids: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if idx >= p.n { return; }
    let id_idx = idx / p.d_model;
    let d = idx % p.d_model;
    let token_id = min(u32(ids[id_idx]), p.vocab_size - 1u);
    out[idx] = weights[token_id * p.d_model + d];
}
";

/// t525 = ArgmaxParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t525 {
    rows: u32,
    cols: u32,
    _pad: [u32; 2],
}

// One thread per row. Scans cols, returns the index of the max as f32.
// Tie-breaking: lowest index wins (only `>` updates the running best).
const SHADER_ARGMAX: &str = "
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x + gid.y * 65535u * 256u;
    if r >= p.rows { return; }
    let base = r * p.cols;
    var best_idx: u32 = 0u;
    var best_val: f32 = input[base];
    for (var c: u32 = 1u; c < p.cols; c++) {
        let v = input[base + c];
        if v > best_val {
            best_val = v;
            best_idx = c;
        }
    }
    out[r] = f32(best_idx);
}
";

impl t500 {
    /// f670 = embedding_lookup. For each token id in `p0`, return that row of `p1`.
    /// `p1`: [vocab_size, d_model]. `p0`: [n_ids] (f32 holding integer values).
    /// Returns [n_ids, d_model]. For a transformer with input shape [batch, seq],
    /// pass n_ids = batch * seq.
    pub fn f670(
        &self,
        p0: &t501,
        p1: &t501,
        p2: u32, p3: u32, p4: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == p2 as usize, "embedding: ids len {} != n_ids {}", p0.s507, p2);
        ensure!(
            p1.s507 == (p3 * p4) as usize,
            "embedding: weights len {} != vocab_size*d_model {}",
            p1.s507, p3 * p4,
        );
        ensure!(p3 > 0, "embedding: vocab_size must be > 0");

        let v0 = p2 * p4;
        let v1 = self.f503(v0 as usize);
        let v2 = t524 { n: v0, n_ids: p2, d_model: p4, vocab_size: p3 };
        self.f543(
            SHADER_EMBED, Some("embedding_lookup"),
            &v2, &[p0, p1], &v1,
            super::f540(v0),
        );
        Ok(v1)
    }

    /// f671 = argmax. Along the last dim: input [rows, cols] -> indices [rows] (as f32).
    /// For LM-head logits with shape [batch*seq, vocab_size], returns one token id per row.
    pub fn f671(
        &self,
        p0: &t501,
        p1: u32, p2: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p1 * p2) as usize, "argmax: input size mismatch");
        ensure!(p2 > 0, "argmax: cols must be > 0");
        let v0 = self.f503(p1 as usize);
        let v1 = t525 { rows: p1, cols: p2, _pad: [0; 2] };
        self.f543(
            SHADER_ARGMAX, Some("argmax"),
            &v1, &[p0], &v0,
            super::f540(p1),
        );
        Ok(v0)
    }
}

// --- KVCache (Sprint 7 step 2) ---
//
// Append-only persistent K and V buffers for autoregressive decoding. Layout for both:
// `[batch_heads, max_seq, head_dim]` flat. `s513` (cursor) tracks how many tokens are valid.
// Prefill writes the full prompt in one append; each subsequent decode step appends one row.

/// t537 = KVAppendParams. Uniform for the f673 (append) shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t537 {
    batch_heads: u32,
    max_seq: u32,
    head_dim: u32,
    cursor: u32,
    new_tokens: u32,
    _pad: [u32; 3],
}

// Strided write: copies [bh, new_tokens, head_dim] from new_data into cache at
// position [bh, cursor..cursor+new_tokens, :].
const SHADER_KV_APPEND: &str = "
struct P {
    batch_heads: u32, max_seq: u32, head_dim: u32, cursor: u32,
    new_tokens: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> new_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch_heads * p.new_tokens * p.head_dim;
    if idx >= total { return; }

    let d = idx % p.head_dim;
    let s = (idx / p.head_dim) % p.new_tokens;
    let bh = idx / (p.new_tokens * p.head_dim);

    let cache_idx = bh * p.max_seq * p.head_dim + (p.cursor + s) * p.head_dim + d;
    cache[cache_idx] = new_data[idx];
}
";

/// t534 = KVCache. Append-only K and V storage for autoregressive decoding.
pub struct t534 {
    pub(crate) s511: t501,    // k
    pub(crate) s512: t501,    // v
    pub(crate) s513: u32,     // cursor (tokens stored)
    pub(crate) s514: u32,     // max_seq
    pub(crate) s515: u32,     // batch_heads
    pub(crate) s516: u32,     // head_dim
}

impl t534 {
    /// f672 = KVCache::new. Allocate empty cache for up to `p1` (max_seq) tokens.
    /// `p2` = batch_heads, `p3` = head_dim.
    pub fn f672(p0: &t500, p1: u32, p2: u32, p3: u32) -> Self {
        let v0 = (p2 * p1 * p3) as usize;
        let v1 = p0.f503(v0);
        let v2 = p0.f503(v0);
        Self { s511: v1, s512: v2, s513: 0, s514: p1, s515: p2, s516: p3 }
    }

    /// f673 = KVCache::append. Write `p1` (new K) and `p2` (new V), each
    /// shaped [batch_heads, p3 new_tokens, head_dim], at the current cursor; advance.
    pub fn f673(&mut self, p0: &t500, p1: &t501, p2: &t501, p3: u32) -> Result<()> {
        ensure!(self.s513 + p3 <= self.s514,
            "KVCache::append: cursor {} + new {} exceeds max_seq {}",
            self.s513, p3, self.s514);
        let v0 = (self.s515 * p3 * self.s516) as usize;
        ensure!(p1.s507 == v0, "KVCache::append: K size mismatch (got {}, expected {})", p1.s507, v0);
        ensure!(p2.s507 == v0, "KVCache::append: V size mismatch (got {}, expected {})", p2.s507, v0);

        let v1 = t537 {
            batch_heads: self.s515, max_seq: self.s514, head_dim: self.s516,
            cursor: self.s513, new_tokens: p3, _pad: [0; 3],
        };
        let v2 = self.s515 * p3 * self.s516;

        // Append into K
        p0.f543(
            SHADER_KV_APPEND, Some("kv_append_k"),
            &v1, &[p1], &self.s511,
            super::f540(v2),
        );
        // Append into V
        p0.f543(
            SHADER_KV_APPEND, Some("kv_append_v"),
            &v1, &[p2], &self.s512,
            super::f540(v2),
        );

        self.s513 += p3;
        Ok(())
    }

    /// f674 = KVCache::reset. Move the cursor back to zero. Underlying buffers are
    /// not zeroed — subsequent appends overwrite from offset 0.
    pub fn f674(&mut self) {
        self.s513 = 0;
    }

    /// f675 = KVCache::cursor. Number of tokens currently stored.
    pub fn f675(&self) -> u32 { self.s513 }

    /// f676 = KVCache::k_buffer. Borrow the underlying K buffer.
    pub fn f676(&self) -> &t501 { &self.s511 }

    /// f677 = KVCache::v_buffer. Borrow the underlying V buffer.
    pub fn f677(&self) -> &t501 { &self.s512 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;
    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    // --- embedding_lookup ---

    #[test]
    fn f670_basic() {
        let v0 = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        let v1 = vec![2.0f32, 0.0, 3.0];
        let v2 = dev().f504(&dev().f670(
            &dev().f502(&v1), &dev().f502(&v0),
            3, 4, 3,
        ).unwrap()).unwrap();
        assert_eq!(v2, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn f670_single_token() {
        let v0: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let v1 = vec![7.0f32];
        let v2 = dev().f504(&dev().f670(
            &dev().f502(&v1), &dev().f502(&v0),
            1, 10, 5,
        ).unwrap()).unwrap();
        assert_eq!(v2, vec![35.0, 36.0, 37.0, 38.0, 39.0]);
    }

    #[test]
    fn f670_repeat_token() {
        let v0 = vec![10.0, 20.0, 99.0, 88.0];
        let v1 = vec![1.0f32, 1.0, 1.0];
        let v2 = dev().f504(&dev().f670(
            &dev().f502(&v1), &dev().f502(&v0),
            3, 2, 2,
        ).unwrap()).unwrap();
        assert_eq!(v2, vec![99.0, 88.0, 99.0, 88.0, 99.0, 88.0]);
    }

    #[test]
    fn f670_out_of_range_clamped() {
        let v0 = vec![1.0, 2.0, 3.0, 4.0];
        let v1 = vec![5.0f32];
        let v2 = dev().f504(&dev().f670(
            &dev().f502(&v1), &dev().f502(&v0),
            1, 2, 2,
        ).unwrap()).unwrap();
        assert_eq!(v2, vec![3.0, 4.0]);
    }

    #[test]
    fn f670_transformer_shape() {
        let v0 = 1024;
        let v1 = 128;
        let v2 = 4;
        let v3: Vec<f32> = (0..v0 * v1).map(|i| (i as f32) * 0.001).collect();
        let v4 = vec![10.0f32, 500.0, 0.0, 1023.0];
        let v5 = dev().f504(&dev().f670(
            &dev().f502(&v4), &dev().f502(&v3),
            v2, v0 as u32, v1 as u32,
        ).unwrap()).unwrap();
        for (v6, &v7) in (0..v2 as usize).zip(&v4) {
            let v8 = v7 as usize;
            for v9 in 0..v1 {
                let v10 = v3[v8 * v1 + v9];
                let v11 = v5[v6 * v1 + v9];
                assert!((v11 - v10).abs() < 1e-6, "row {v6} col {v9}: got {v11} expected {v10}");
            }
        }
    }

    #[test]
    fn f670_size_mismatch() {
        let v0 = vec![1.0; 6];
        let v1 = vec![0.0f32, 1.0];
        assert!(dev().f670(&dev().f502(&v1), &dev().f502(&v0), 2, 3, 3).is_err());
    }

    #[test]
    fn f670_ids_length_mismatch() {
        let v0 = vec![1.0; 6];
        let v1 = vec![0.0f32, 1.0, 1.0];
        assert!(dev().f670(&dev().f502(&v1), &dev().f502(&v0), 2, 2, 3).is_err());
    }

    // --- argmax ---

    #[test]
    fn f671_basic() {
        let v0 = vec![
            1.0f32, 5.0, 2.0, 3.0,
            7.0, 1.0, 9.0, 8.0,
            -1.0, -5.0, -3.0, -2.0,
        ];
        let v1 = dev().f504(&dev().f671(&dev().f502(&v0), 3, 4).unwrap()).unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn f671_single_row() {
        let v0 = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let v1 = dev().f504(&dev().f671(&dev().f502(&v0), 1, 7).unwrap()).unwrap();
        assert_eq!(v1, vec![5.0]);
    }

    #[test]
    fn f671_single_column() {
        let v0 = vec![42.0f32, -7.0, 100.0];
        let v1 = dev().f504(&dev().f671(&dev().f502(&v0), 3, 1).unwrap()).unwrap();
        assert_eq!(v1, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn f671_ties_pick_lowest_index() {
        let v0 = vec![5.0f32, 5.0, 5.0, 5.0];
        let v1 = dev().f504(&dev().f671(&dev().f502(&v0), 1, 4).unwrap()).unwrap();
        assert_eq!(v1, vec![0.0]);
    }

    #[test]
    fn f671_lm_head_shape() {
        let v0 = 8u32;
        let v1 = 32u32;
        let mut v2 = vec![0.0f32; (v0 * v1) as usize];
        let mut v3 = Vec::with_capacity(v0 as usize);
        for v4 in 0..v0 {
            let v5 = (v4 * 3 + 7) % v1;
            for v6 in 0..v1 {
                v2[(v4 * v1 + v6) as usize] = (v6 as f32) * 0.01;
            }
            v2[(v4 * v1 + v5) as usize] = 99.0;
            v3.push(v5 as f32);
        }
        let v7 = dev().f504(&dev().f671(&dev().f502(&v2), v0, v1).unwrap()).unwrap();
        f544(&v7, &v3, 1e-6);
    }

    #[test]
    fn f671_size_mismatch() {
        let v0 = vec![1.0f32; 10];
        assert!(dev().f671(&dev().f502(&v0), 3, 4).is_err());
    }

    #[test]
    fn f671_zero_cols() {
        let v0 = vec![1.0f32; 0];
        assert!(dev().f671(&dev().f502(&v0), 3, 0).is_err());
    }

    // --- KVCache (f672..f677) ---

    #[test]
    fn f672_starts_empty() {
        let v0 = t534::f672(dev(), 16, 2, 4);
        assert_eq!(v0.f675(), 0);
        // K and V buffers should be alloc'd at full size and zero-initialized
        assert_eq!(v0.f676().s507, 2 * 16 * 4);
        assert_eq!(v0.f677().s507, 2 * 16 * 4);
    }

    #[test]
    fn f673_append_single_token() {
        // 1 head, max_seq=4, head_dim=2 -> cache layout flat: 8 floats per K/V
        let mut v0 = t534::f672(dev(), 4, 1, 2);
        let v1 = dev().f502(&[7.0f32, 8.0]); // [bh=1, new=1, dim=2]
        let v2 = dev().f502(&[70.0f32, 80.0]);
        v0.f673(dev(), &v1, &v2, 1).unwrap();
        assert_eq!(v0.f675(), 1);
        let v3 = dev().f504(v0.f676()).unwrap();
        let v4 = dev().f504(v0.f677()).unwrap();
        // First two slots of K = [7, 8], rest stays zero
        assert_eq!(&v3[..2], &[7.0, 8.0]);
        assert!(v3[2..].iter().all(|&x| x == 0.0));
        assert_eq!(&v4[..2], &[70.0, 80.0]);
        assert!(v4[2..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn f673_append_multi_then_single() {
        // Prefill 3 tokens, then decode-step 1 token. Verify layout.
        let mut v0 = t534::f672(dev(), 8, 1, 2);
        let v1 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]); // 3 tokens
        let v2 = dev().f502(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]);
        v0.f673(dev(), &v1, &v2, 3).unwrap();
        assert_eq!(v0.f675(), 3);

        let v3 = dev().f502(&[99.0f32, 100.0]); // 1 token
        let v4 = dev().f502(&[999.0f32, 1000.0]);
        v0.f673(dev(), &v3, &v4, 1).unwrap();
        assert_eq!(v0.f675(), 4);

        let v5 = dev().f504(v0.f676()).unwrap();
        assert_eq!(&v5[..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0, 100.0]);
        // Slots 8..16 (positions 4..7) still zero
        assert!(v5[8..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn f673_multi_head_layout() {
        // 2 batch_heads, max_seq=3, head_dim=2 -> 12 floats per K/V.
        // Append [bh=2, new=1, dim=2] = [head0: 1,2  head1: 3,4]
        // Cache layout: [bh=0, max=3, dim=2] flat = [1,2,0,0,0,0,3,4,0,0,0,0]
        let mut v0 = t534::f672(dev(), 3, 2, 2);
        let v1 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0]);
        let v2 = dev().f502(&[10.0f32, 20.0, 30.0, 40.0]);
        v0.f673(dev(), &v1, &v2, 1).unwrap();
        let v3 = dev().f504(v0.f676()).unwrap();
        assert_eq!(v3, vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn f674_reset_zeroes_cursor() {
        let mut v0 = t534::f672(dev(), 4, 1, 2);
        let v1 = dev().f502(&[1.0f32, 2.0]);
        let v2 = dev().f502(&[1.0f32, 2.0]);
        v0.f673(dev(), &v1, &v2, 1).unwrap();
        assert_eq!(v0.f675(), 1);
        v0.f674();
        assert_eq!(v0.f675(), 0);
    }

    #[test]
    fn f673_overflow_errors() {
        let mut v0 = t534::f672(dev(), 2, 1, 2);
        let v1 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]); // 3 tokens > max_seq=2
        let v2 = dev().f502(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(v0.f673(dev(), &v1, &v2, 3).is_err());
    }

    #[test]
    fn f673_kv_cache_then_causal_attention() {
        // End-to-end: prefill 2 tokens, decode 1 token, run causal SDPA over the cached buffers.
        // Compare against a direct CPU reference using the full 3-token K/V.
        let dk = 4u32;
        let max_seq = 8u32;
        let mut v0 = t534::f672(dev(), max_seq, 1, dk);

        // Prefill: K_prefill [2, 4], V_prefill [2, 4]
        let k_prefill: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
        let v_prefill: Vec<f32> = (0..8).map(|i| (i as f32) * 0.2 - 0.5).collect();
        v0.f673(dev(),
            &dev().f502(&k_prefill),
            &dev().f502(&v_prefill),
            2).unwrap();

        // Decode step: append 1 more token
        let k_decode = vec![0.7f32, 0.8, 0.9, 1.0];
        let v_decode = vec![-0.1f32, 0.3, 0.6, 0.9];
        v0.f673(dev(),
            &dev().f502(&k_decode),
            &dev().f502(&v_decode),
            1).unwrap();
        assert_eq!(v0.f675(), 3);

        // Build Q for the current decode step (1 row)
        let q_now = vec![0.5f32, -0.2, 0.3, 0.1];

        // Run causal SDPA: q_seq=1, kv_seq=3
        // The cache buffer is sized [max_seq=8, dk=4] = 32 floats. To use it as a kv_seq=3 input,
        // we need to slice the first 3 tokens out. For this test we read the cache buffers,
        // slice on CPU, and re-upload — verifying both the cache writes and causal SDPA work.
        let k_full = dev().f504(v0.f676()).unwrap();
        let v_full = dev().f504(v0.f677()).unwrap();
        let k_used = &k_full[..(3 * dk as usize)];
        let v_used = &v_full[..(3 * dk as usize)];

        let attn_gpu = dev().f504(&dev().f623(
            &dev().f502(&q_now),
            &dev().f502(k_used),
            &dev().f502(v_used),
            1, 1, 3, dk,
        ).unwrap()).unwrap();

        // CPU reference using the full prefilled+decoded K/V
        let mut k_all = k_prefill.clone();
        k_all.extend_from_slice(&k_decode);
        let mut v_all = v_prefill.clone();
        v_all.extend_from_slice(&v_decode);
        let attn_cpu = cpu_decode_attention(&q_now, &k_all, &v_all, 3, dk as usize);
        f544(&attn_gpu, &attn_cpu, 1e-3);
    }

    // CPU reference for the decode case (q_seq=1, kv_seq=N, no masking applies).
    fn cpu_decode_attention(q: &[f32], k: &[f32], v: &[f32], kv: usize, dk: usize) -> Vec<f32> {
        let scale = 1.0 / (dk as f32).sqrt();
        let mut scores = vec![0.0f32; kv];
        for j in 0..kv {
            let mut s = 0.0;
            for d in 0..dk { s += q[d] * k[j * dk + d]; }
            scores[j] = s * scale;
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter().map(|&x| (x - mx).exp()).sum();
        let attn: Vec<f32> = scores.iter().map(|&x| (x - mx).exp() / sum).collect();
        let mut out = vec![0.0f32; dk];
        for d in 0..dk {
            for j in 0..kv { out[d] += attn[j] * v[j * dk + d]; }
        }
        out
    }
}
