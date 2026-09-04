// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// Sampling ops for autoregressive decoding.
// f787=top_k_mask, f788=top_p_mask, f789=sample_multinomial, f790=rep_penalty.
//
// Temperature scaling: use dev.f553(logits, 1.0 / temp) before calling these ops.
// Typical pipeline: rep_penalty? → ÷temp (f553) → top_k_mask → softmax (f620)
//                   → top_p_mask → sample_multinomial → detokenize.

use crate::device::{t500, t501};
use anyhow::{Result, ensure};

// --- Top-k mask ---

/// t560 = TopKParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t560 {
    batch: u32,
    vocab: u32,
    k: u32,
    _pad: u32,
}

// One wave64 (64 threads) per batch row — three phases:
//   P1: all 64 threads scan vocab stride-64, each keeping a private min-heap of
//       up to kk=min(k,128) values in registers (128 VGPRs, within RDNA1 budget).
//   P2: each thread writes its kk candidates to LDS (64×128 = 8192 f32 = 32 KB);
//       thread 0 merges all 8192 LDS entries to find the global threshold.
//   P3: all 64 threads apply the threshold mask stride-64.
//
// Old design: thread 0 serial O(vocab×k) = 32000×50 = 1.6M ops.
// New design: 64× parallel phase-1 + thread-0 merge of 8192 LDS candidates.
const SHADER_TOP_K_MASK: &str = "
struct P { batch: u32, vocab: u32, k: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;

// 64 threads × 128 slots × 4 bytes = 32 KB LDS (RDNA1 per-CU limit; 2 WGs/CU occupancy).
var<workgroup> wg_cands:     array<f32, 8192>;
var<workgroup> wg_threshold: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)           wid: vec3<u32>,
    @builtin(local_invocation_index) t:   u32,
) {
    let row = wid.x + wid.y * 65535u;
    if row >= p.batch { return; }
    let base = row * p.vocab;
    let kk = min(p.k, 128u);

    // ── Phase 1: each thread builds a private top-kk heap ────────────────────
    var heap: array<f32, 128>;
    var heap_min: f32 = -1e30;
    var heap_min_idx: u32 = 0u;
    var heap_size: u32 = 0u;

    var i = t;
    while i < p.vocab {
        let v = logits[base + i];
        if heap_size < kk {
            heap[heap_size] = v;
            heap_size += 1u;
            if heap_size == kk {
                heap_min = heap[0]; heap_min_idx = 0u;
                for (var j = 1u; j < kk; j++) {
                    if heap[j] < heap_min { heap_min = heap[j]; heap_min_idx = j; }
                }
            }
        } else if v > heap_min {
            heap[heap_min_idx] = v;
            heap_min = heap[0]; heap_min_idx = 0u;
            for (var j = 1u; j < kk; j++) {
                if heap[j] < heap_min { heap_min = heap[j]; heap_min_idx = j; }
            }
        }
        i += 64u;
    }

    // ── Phase 2: deposit candidates to LDS; thread 0 merges ──────────────────
    let lds_base = t * 128u;
    for (var j = 0u; j < 128u; j++) {
        wg_cands[lds_base + j] = select(-1e30, heap[j], j < heap_size);
    }
    workgroupBarrier();

    if t == 0u {
        var heap2: array<f32, 128>;
        var hs2: u32 = 0u;
        var hm2: f32 = -1e30;
        var hmi2: u32 = 0u;
        for (var ci = 0u; ci < 8192u; ci++) {
            let v = wg_cands[ci];
            if hs2 < kk {
                heap2[hs2] = v;
                hs2 += 1u;
                if hs2 == kk {
                    hm2 = heap2[0]; hmi2 = 0u;
                    for (var j = 1u; j < kk; j++) {
                        if heap2[j] < hm2 { hm2 = heap2[j]; hmi2 = j; }
                    }
                }
            } else if v > hm2 {
                heap2[hmi2] = v;
                hm2 = heap2[0]; hmi2 = 0u;
                for (var j = 1u; j < kk; j++) {
                    if heap2[j] < hm2 { hm2 = heap2[j]; hmi2 = j; }
                }
            }
        }
        wg_threshold = select(hm2, -1e30, hs2 < kk);
    }
    workgroupBarrier();

    // ── Phase 3: all 64 threads apply the mask ────────────────────────────────
    let threshold = wg_threshold;
    var j = t;
    while j < p.vocab {
        let v = logits[base + j];
        out[base + j] = select(-1e9, v, v >= threshold);
        j += 64u;
    }
}
";

// --- Top-p mask (nucleus sampling) ---

/// t561 = TopPParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t561 {
    batch: u32,
    vocab: u32,
    p_val: f32,
    _pad: u32,
}

// One workgroup per row, 256 threads.
// Thread 0: collect active (non-zero) probabilities, sort descending (selection sort),
// cumulative sum until > p, record the cutoff value.
// All threads: zero-out probs below cutoff.
// Intended to run after top_k_mask + softmax — so at most k non-zero probs per row.
// Handles up to 128 active tokens (sufficient for k ≤ 128).
const SHADER_TOP_P_MASK: &str = "
struct P { batch: u32, vocab: u32, p_val: f32, _pad: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:   array<f32>;

var<workgroup> wg_cutoff: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)           wid: vec3<u32>,
    @builtin(local_invocation_index) t:   u32,
) {
    let row = wid.x + wid.y * 65535u;
    if row >= p.batch { return; }
    let base = row * p.vocab;

    if t == 0u {
        // Collect up to 128 non-zero probabilities into a local sorted buffer.
        var vals: array<f32, 128>;
        var cnt: u32 = 0u;
        for (var i: u32 = 0u; i < p.vocab && cnt < 128u; i++) {
            let v = probs[base + i];
            if v > 0.0 { vals[cnt] = v; cnt++; }
        }

        // Selection sort descending (cnt ≤ 128 so O(cnt²) is negligible)
        for (var i: u32 = 0u; i < cnt; i++) {
            var best = i;
            for (var j: u32 = i + 1u; j < cnt; j++) {
                if vals[j] > vals[best] { best = j; }
            }
            let tmp = vals[i]; vals[i] = vals[best]; vals[best] = tmp;
        }

        // Cumulative sum — find first index where it exceeds p
        var cumsum: f32 = 0.0;
        var cutoff: f32 = 0.0;
        for (var i: u32 = 0u; i < cnt; i++) {
            cumsum += vals[i];
            if cumsum >= p.p_val {
                cutoff = vals[i];
                break;
            }
            // If we exhaust the list without hitting p (e.g., floating-point rounding),
            // set cutoff to the last non-zero value so everything passes.
            if i == cnt - 1u { cutoff = vals[i]; }
        }
        wg_cutoff = cutoff;
    }
    workgroupBarrier();

    let cutoff = wg_cutoff;
    var i = t;
    while i < p.vocab {
        out[base + i] = select(0.0, probs[base + i], probs[base + i] >= cutoff);
        i += 256u;
    }
}
";

// --- Multinomial sampling ---

/// t562 = MultinomialParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t562 {
    batch: u32,
    vocab: u32,
    seed: u32,
    step: u32,
}

// One workgroup per row. Thread 0 does inverse-CDF sampling using a PCG-derived uniform.
// RNG: pcg32(seed ^ row ^ (step << 12)) — deterministic given (seed, step, row).
// Probs do not need to be normalized — we compute the total and sample proportionally.
// Output: one f32 token index per row.
const SHADER_MULTINOMIAL: &str = "
struct P { batch: u32, vocab: u32, seed: u32, step: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:   array<f32>;

var<workgroup> wg_token: f32;

fn pcg32(state: u32) -> u32 {
    let s = state * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn uniform01(state: u32) -> f32 {
    // Map u32 to [0, 1) — use the top 24 bits for float mantissa
    return f32(pcg32(state) >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)           wid: vec3<u32>,
    @builtin(local_invocation_index) t:   u32,
) {
    let row = wid.x + wid.y * 65535u;
    if row >= p.batch { return; }
    let base = row * p.vocab;

    if t == 0u {
        // Compute total probability mass (handles unnormalized probs)
        var total: f32 = 0.0;
        for (var i: u32 = 0u; i < p.vocab; i++) {
            total += probs[base + i];
        }
        if total <= 0.0 {
            wg_token = 0.0;
        } else {
            // PCG-derived uniform threshold in [0, total)
            let rng_state = p.seed ^ (row * 2654435761u) ^ (p.step << 12u);
            let u = uniform01(rng_state) * total;

            // Inverse CDF walk
            var cumsum: f32 = 0.0;
            var chosen: u32 = p.vocab - 1u;
            for (var i: u32 = 0u; i < p.vocab; i++) {
                cumsum += probs[base + i];
                if cumsum > u {
                    chosen = i;
                    break;
                }
            }
            wg_token = f32(chosen);
        }
    }
    workgroupBarrier();

    if t == 0u {
        out[row] = wg_token;
    }
}
";

// --- Repetition penalty ---

/// t563 = RepPenaltyParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t563 {
    batch: u32,
    vocab: u32,
    seq_len: u32,
    _pad: u32,
    penalty: f32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

// One thread per (batch_row, vocab_idx). Each thread checks if its vocab index
// appears anywhere in past_ids[row * seq_len .. (row+1) * seq_len].
// If found: logit > 0 → divide by penalty; logit <= 0 → multiply by penalty.
// Standard Hugging Face repetition penalty formula.
const SHADER_REP_PENALTY: &str = "
struct P {
    batch: u32, vocab: u32, seq_len: u32, _p0: u32,
    penalty: f32, _p1: u32, _p2: u32, _p3: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       logits:   array<f32>;
@group(0) @binding(2) var<storage, read>       past_ids: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:      array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.vocab;
    if idx >= total { return; }

    let row = idx / p.vocab;
    let vi  = idx % p.vocab;
    let id_base = row * p.seq_len;

    var appears = false;
    for (var s: u32 = 0u; s < p.seq_len; s++) {
        if u32(past_ids[id_base + s]) == vi { appears = true; break; }
    }

    let v = logits[idx];
    if appears {
        out[idx] = select(v * p.penalty, v / p.penalty, v > 0.0);
    } else {
        out[idx] = v;
    }
}
";

impl t500 {
    /// f787 = top_k_mask. For each row of `p0` [p2, p3], keep the top-k logits and
    /// set the rest to -1e9. `p1` = k (max 128). Returns a new buffer of the same shape.
    /// Apply before softmax. For temperature scaling, use `dev.f553(logits, 1.0/temp)` first.
    pub fn f787(&self, p0: &t501, p1: u32, p2: u32, p3: u32) -> Result<t501> {
        ensure!(p0.s507 == (p2 * p3) as usize, "top_k_mask: size mismatch");
        ensure!(
            p1 > 0 && p1 <= 128,
            "top_k_mask: k must be 1..128, got {}",
            p1
        );
        ensure!(p3 > 0, "top_k_mask: vocab must be > 0");
        let v0 = self.f503(p0.s507);
        let v1 = t560 {
            batch: p2,
            vocab: p3,
            k: p1,
            _pad: 0,
        };
        let (vx, vy) = if p2 <= 65535 {
            (p2, 1)
        } else {
            (65535, p2.div_ceil(65535))
        };
        self.f543(
            SHADER_TOP_K_MASK,
            Some("top_k_mask"),
            &v1,
            &[p0],
            &v0,
            (vx, vy, 1),
        );
        Ok(v0)
    }

    /// f788 = top_p_mask. Nucleus sampling: given softmax probabilities `p0` [p2, p3],
    /// zero out the tail such that the retained tokens cover cumulative probability >= `p1`.
    /// `p1` in (0.0, 1.0]. Apply after top_k_mask + softmax. Returns same-shape buffer.
    pub fn f788(&self, p0: &t501, p1: f32, p2: u32, p3: u32) -> Result<t501> {
        ensure!(p0.s507 == (p2 * p3) as usize, "top_p_mask: size mismatch");
        ensure!(
            p1 > 0.0 && p1 <= 1.0,
            "top_p_mask: p must be in (0, 1], got {}",
            p1
        );
        let v0 = self.f503(p0.s507);
        let v1 = t561 {
            batch: p2,
            vocab: p3,
            p_val: p1,
            _pad: 0,
        };
        let (vx, vy) = if p2 <= 65535 {
            (p2, 1)
        } else {
            (65535, p2.div_ceil(65535))
        };
        self.f543(
            SHADER_TOP_P_MASK,
            Some("top_p_mask"),
            &v1,
            &[p0],
            &v0,
            (vx, vy, 1),
        );
        Ok(v0)
    }

    /// f789 = sample_multinomial. Sample one token per row from (unnormalized) probability
    /// distribution `p0` [p2, p3]. `p3` = seed (set once per session), `p4` = step (incremented
    /// each decode step for TRIPLE SIMS determinism). Returns [p2] token indices as f32.
    pub fn f789(&self, p0: &t501, p3: u32, p4: u32, p2: u32, p5: u32) -> Result<t501> {
        ensure!(
            p0.s507 == (p2 * p5) as usize,
            "sample_multinomial: size mismatch"
        );
        ensure!(p5 > 0, "sample_multinomial: vocab must be > 0");
        let v0 = self.f503(p2 as usize);
        let v1 = t562 {
            batch: p2,
            vocab: p5,
            seed: p3,
            step: p4,
        };
        let (vx, vy) = if p2 <= 65535 {
            (p2, 1)
        } else {
            (65535, p2.div_ceil(65535))
        };
        self.f543(
            SHADER_MULTINOMIAL,
            Some("sample_multinomial"),
            &v1,
            &[p0],
            &v0,
            (vx, vy, 1),
        );
        Ok(v0)
    }

    /// f790 = rep_penalty. Apply repetition penalty to `p0` (logits [p3, p4]) using
    /// `p1` (past token ids [p3, p5] as f32). `p2` = penalty (> 1.0 discourages repeats).
    /// Hugging Face convention: positive logits divided by penalty, negative logits multiplied.
    pub fn f790(&self, p0: &t501, p1: &t501, p2: f32, p3: u32, p4: u32, p5: u32) -> Result<t501> {
        ensure!(
            p0.s507 == (p3 * p4) as usize,
            "rep_penalty: logits size mismatch"
        );
        ensure!(
            p1.s507 == (p3 * p5) as usize,
            "rep_penalty: ids size mismatch"
        );
        ensure!(p2 >= 1.0, "rep_penalty: penalty must be >= 1.0, got {}", p2);
        let v0 = self.f503(p0.s507);
        let v1 = t563 {
            batch: p3,
            vocab: p4,
            seq_len: p5,
            _pad: 0,
            penalty: p2,
            _p1: 0,
            _p2: 0,
            _p3: 0,
        };
        self.f543(
            SHADER_REP_PENALTY,
            Some("rep_penalty"),
            &v1,
            &[p0, p1],
            &v0,
            super::f540(p3 * p4),
        );
        Ok(v0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;
    fn dev() -> &'static t500 {
        &crate::ops::TEST_DEV
    }

    // --- f787 top_k_mask ---

    #[test]
    fn f787_keeps_top_k() {
        // [1, 5]: logits = [0.1, 0.5, 0.3, 0.9, 0.2], k=3
        // Top-3: 0.9, 0.5, 0.3 → indices 3, 1, 2 pass; indices 0,4 become -1e9
        let v0 = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
        let v1 = dev()
            .f504(&dev().f787(&dev().f502(&v0), 3, 1, 5).unwrap())
            .unwrap();
        assert!(v1[0] < -1e8, "index 0 (0.1) should be masked");
        assert!((v1[1] - 0.5).abs() < 1e-5, "index 1 kept");
        assert!((v1[2] - 0.3).abs() < 1e-5, "index 2 kept");
        assert!((v1[3] - 0.9).abs() < 1e-5, "index 3 kept");
        assert!(v1[4] < -1e8, "index 4 (0.2) should be masked");
    }

    #[test]
    fn f787_k1_keeps_max() {
        let v0 = vec![1.0f32, 3.0, 2.0, 0.5];
        let v1 = dev()
            .f504(&dev().f787(&dev().f502(&v0), 1, 1, 4).unwrap())
            .unwrap();
        assert!(v1[0] < -1e8);
        assert!((v1[1] - 3.0).abs() < 1e-5);
        assert!(v1[2] < -1e8);
        assert!(v1[3] < -1e8);
    }

    #[test]
    fn f787_k_ge_vocab_keeps_all() {
        let v0 = vec![1.0f32, 2.0, 3.0];
        let v1 = dev()
            .f504(&dev().f787(&dev().f502(&v0), 5, 1, 3).unwrap())
            .unwrap();
        f544(&v1, &v0, 1e-5);
    }

    #[test]
    fn f787_two_rows() {
        let v0 = vec![
            1.0f32, 5.0, 2.0, 3.0, // row 0: top-2 = 5.0(idx1), 3.0(idx3)
            4.0, 1.0, 8.0, 2.0, // row 1: top-2 = 8.0(idx2), 4.0(idx0)
        ];
        let v1 = dev()
            .f504(&dev().f787(&dev().f502(&v0), 2, 2, 4).unwrap())
            .unwrap();
        assert!(v1[0] < -1e8); // row0 idx0 masked
        assert!((v1[1] - 5.0).abs() < 1e-5);
        assert!(v1[2] < -1e8); // row0 idx2 masked
        assert!((v1[3] - 3.0).abs() < 1e-5);
        assert!((v1[4] - 4.0).abs() < 1e-5);
        assert!(v1[5] < -1e8); // row1 idx1 masked
        assert!((v1[6] - 8.0).abs() < 1e-5);
        assert!(v1[7] < -1e8); // row1 idx3 masked
    }

    // --- f788 top_p_mask ---

    #[test]
    fn f788_nucleus_cutoff() {
        // Probs: [0.4, 0.3, 0.2, 0.1], p=0.7 → top-2 (0.4+0.3=0.7) pass; 0.2 and 0.1 zeroed
        let v0 = vec![0.4f32, 0.3, 0.2, 0.1];
        let v1 = dev()
            .f504(&dev().f788(&dev().f502(&v0), 0.7, 1, 4).unwrap())
            .unwrap();
        assert!((v1[0] - 0.4).abs() < 1e-5);
        assert!((v1[1] - 0.3).abs() < 1e-5);
        assert!(v1[2] == 0.0 || v1[2] < 1e-5, "p=0.2 should be zeroed");
        assert!(v1[3] == 0.0 || v1[3] < 1e-5, "p=0.1 should be zeroed");
    }

    #[test]
    fn f788_p1_keeps_all() {
        let v0 = vec![0.25f32, 0.25, 0.25, 0.25];
        let v1 = dev()
            .f504(&dev().f788(&dev().f502(&v0), 1.0, 1, 4).unwrap())
            .unwrap();
        for &x in &v1 {
            assert!(x > 0.2, "all should be kept");
        }
    }

    // --- f789 sample_multinomial ---

    #[test]
    fn f789_samples_valid_token() {
        // Uniform distribution over 10 tokens; sampled index must be in [0, 9]
        let v0 = vec![0.1f32; 10];
        let v1 = dev()
            .f504(&dev().f789(&dev().f502(&v0), 42, 0, 1, 10).unwrap())
            .unwrap();
        assert_eq!(v1.len(), 1);
        let tok = v1[0] as u32;
        assert!(tok < 10, "token {} out of range", tok);
    }

    #[test]
    fn f789_deterministic_same_seed_step() {
        let v0: Vec<f32> = (0..32u32).map(|i| (i as f32 + 1.0) / 528.0).collect();
        let v1 = dev()
            .f504(&dev().f789(&dev().f502(&v0), 1337, 5, 1, 32).unwrap())
            .unwrap();
        let v2 = dev()
            .f504(&dev().f789(&dev().f502(&v0), 1337, 5, 1, 32).unwrap())
            .unwrap();
        assert_eq!(v1, v2, "same seed+step must give same token (TRIPLE SIMS)");
    }

    #[test]
    fn f789_different_steps_may_differ() {
        let v0: Vec<f32> = (0..1000u32).map(|_| 1.0 / 1000.0).collect();
        let mut tokens = std::collections::HashSet::new();
        for step in 0u32..20 {
            let v1 = dev()
                .f504(&dev().f789(&dev().f502(&v0), 42, step, 1, 1000).unwrap())
                .unwrap();
            tokens.insert(v1[0] as u32);
        }
        assert!(
            tokens.len() > 1,
            "different steps should (likely) give different tokens"
        );
    }

    #[test]
    fn f789_deterministic_high_prob_token() {
        // Token 7 has probability 0.99, everything else near-zero
        let mut v0 = vec![0.0001f32; 100];
        v0[7] = 0.99;
        let total: f32 = v0.iter().sum();
        let v1: Vec<f32> = v0.iter().map(|x| x / total).collect();
        // Sample many times — token 7 should win almost always
        let mut wins = 0u32;
        for step in 0u32..20 {
            let out = dev()
                .f504(&dev().f789(&dev().f502(&v1), 12345, step, 1, 100).unwrap())
                .unwrap();
            if out[0] as u32 == 7 {
                wins += 1;
            }
        }
        assert!(
            wins >= 15,
            "high-prob token should win most of the time, got {}/20",
            wins
        );
    }

    // --- f790 rep_penalty ---

    #[test]
    fn f790_reduces_past_token_logit() {
        // logits [1.0, 2.0, 3.0], past_ids = [1] (token 1 appeared)
        // penalty=2.0: logit[1]=2.0 > 0 → 2.0/2.0=1.0; others unchanged
        let logits = vec![1.0f32, 2.0, 3.0];
        let ids = vec![1.0f32];
        let v0 = dev()
            .f504(
                &dev()
                    .f790(&dev().f502(&logits), &dev().f502(&ids), 2.0, 1, 3, 1)
                    .unwrap(),
            )
            .unwrap();
        assert!((v0[0] - 1.0).abs() < 1e-5, "unseen token unchanged");
        assert!((v0[1] - 1.0).abs() < 1e-5, "seen token: 2.0/2.0=1.0");
        assert!((v0[2] - 3.0).abs() < 1e-5, "unseen token unchanged");
    }

    #[test]
    fn f790_negative_logit_multiplied() {
        // logit[0] = -1.0, token 0 appears in past → -1.0 * 2.0 = -2.0
        let logits = vec![-1.0f32, 0.5];
        let ids = vec![0.0f32];
        let v0 = dev()
            .f504(
                &dev()
                    .f790(&dev().f502(&logits), &dev().f502(&ids), 2.0, 1, 2, 1)
                    .unwrap(),
            )
            .unwrap();
        assert!((v0[0] - (-2.0)).abs() < 1e-5, "negative logit multiplied");
        assert!((v0[1] - 0.5).abs() < 1e-5, "unseen unchanged");
    }

    #[test]
    fn f790_penalty_one_is_noop() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let ids = vec![0.0f32, 1.0, 2.0];
        let v0 = dev()
            .f504(
                &dev()
                    .f790(&dev().f502(&logits), &dev().f502(&ids), 1.0, 1, 3, 3)
                    .unwrap(),
            )
            .unwrap();
        f544(&v0, &logits, 1e-5);
    }

    #[test]
    fn f790_no_past_tokens_unchanged() {
        let logits: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let ids: Vec<f32> = vec![];
        // seq_len=0 → no ids, so all logits pass through unchanged
        // Actually can't pass empty; use a token id that's out of range as a workaround.
        // Just verify error for malformed call; use a minimal valid test:
        let ids2 = vec![100.0f32]; // token 100 doesn't exist in vocab=8
        let v0 = dev()
            .f504(
                &dev()
                    .f790(&dev().f502(&logits), &dev().f502(&ids2), 2.0, 1, 8, 1)
                    .unwrap(),
            )
            .unwrap();
        f544(&v0, &logits, 1e-5);
    }
}
