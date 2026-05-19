// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// Comprehensive inference-ops benchmark at LLaMA-7B scales.
// All timings are compute+readback (no upload), warmed up with 5 shader-cache priming
// runs before measurement so first-compile latency doesn't contaminate numbers.
//
// LLaMA-7B reference dimensions:
//   d_model=4096, n_heads=32, head_dim=128, n_kv_heads=32, vocab=32000

use any_gpu::t500;
use std::time::Instant;

const WARMUP: usize = 5;
const ITERS:  usize = 20;

// ── helpers ──────────────────────────────────────────────────────────────────

fn ms_per_iter(elapsed: std::time::Duration) -> f64 {
    elapsed.as_secs_f64() * 1e3 / ITERS as f64
}

fn bw_gb_s(bytes: u64, ms: f64) -> f64 {
    bytes as f64 / (ms * 1e-3) / 1e9
}

fn header(title: &str) {
    println!("\n── {title} ──");
}

// ── rms_norm ─────────────────────────────────────────────────────────────────

fn bench_rms(dev: &t500, rows: u32, cols: u32) {
    let inp: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let w: Vec<f32> = vec![1.0f32; cols as usize];
    let gi = dev.f502(&inp);
    let gw = dev.f502(&w);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f603(&gi, &gw, rows, cols, 1e-5).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f603(&gi, &gw, rows, cols, 1e-5).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    // read input + write output
    let bytes = (rows * cols * 4 * 2) as u64;
    println!("  rms_norm  [{rows:>4}×{cols}]  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

// ── layer_norm ────────────────────────────────────────────────────────────────

fn bench_ln(dev: &t500, rows: u32, cols: u32) {
    let inp: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let g: Vec<f32> = vec![1.0f32; cols as usize];
    let b: Vec<f32> = vec![0.0f32; cols as usize];
    let gi = dev.f502(&inp);
    let gg = dev.f502(&g);
    let gb = dev.f502(&b);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f602(&gi, &gg, &gb, rows, cols, 1e-5).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f602(&gi, &gg, &gb, rows, cols, 1e-5).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (rows * cols * 4 * 2) as u64;
    println!("  layer_norm[{rows:>4}×{cols}]  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

// ── softmax ───────────────────────────────────────────────────────────────────

fn bench_softmax(dev: &t500, rows: u32, cols: u32, label: &str) {
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
    let buf = dev.f502(&data);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f620(&buf, rows, cols).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f620(&buf, rows, cols).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (rows * cols * 4 * 2) as u64;
    println!("  softmax   {label:<20}  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

// ── matmul ────────────────────────────────────────────────────────────────────

fn bench_matmul(dev: &t500, m: u32, n: u32, k: u32) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.0005).collect();
    let ga = dev.f502(&a);
    let gb = dev.f502(&b);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f580(&ga, &gb, m, n, k).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f580(&ga, &gb, m, n, k).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    println!("  matmul    [{m:>4}×{k}×{n}]  {ms:>7.3} ms  {:>6.1} GFLOPS", flops / (ms * 1e-3) / 1e9);
}

// ── batch_matmul ──────────────────────────────────────────────────────────────

fn bench_batch_matmul(dev: &t500, batch: u32, m: u32, n: u32, k: u32, label: &str) {
    let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.0005).collect();
    let ga = dev.f502(&a);
    let gb = dev.f502(&b);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f581(&ga, &gb, batch, m, n, k).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f581(&ga, &gb, batch, m, n, k).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let flops = 2.0 * batch as f64 * m as f64 * n as f64 * k as f64;
    println!("  batch_mm  {label:<22}  {ms:>7.3} ms  {:>6.1} GFLOPS", flops / (ms * 1e-3) / 1e9);
}

// ── fused SDPA (f626) ─────────────────────────────────────────────────────────

fn bench_sdpa(dev: &t500, batch_heads: u32, q_seq: u32, kv_seq: u32, head_dim: u32, label: &str) {
    let n = (batch_heads * q_seq * head_dim) as usize;
    let nkv = (batch_heads * kv_seq * head_dim) as usize;
    let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let k: Vec<f32> = (0..nkv).map(|i| (i as f32) * 0.001).collect();
    let v = k.clone();
    let gq = dev.f502(&q);
    let gk = dev.f502(&k);
    let gv = dev.f502(&v);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f626(&gq, &gk, &gv, batch_heads, q_seq, kv_seq, head_dim).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f626(&gq, &gk, &gv, batch_heads, q_seq, kv_seq, head_dim).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    // Q+K+V read + output write
    let bytes = ((n + 2 * nkv + n) * 4) as u64;
    println!("  fused_sdpa {label:<20}  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

// ── RoPE ─────────────────────────────────────────────────────────────────────

fn bench_rope(dev: &t500, batch_heads: u32, seq: u32, head_dim: u32) {
    let n = (batch_heads * seq * head_dim) as usize;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let buf = dev.f502(&data);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f625(&buf, batch_heads, seq, head_dim, 0, 10000.0).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f625(&buf, batch_heads, seq, head_dim, 0, 10000.0).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (n * 4 * 2) as u64;
    println!("  rope      [{batch_heads:>2}×{seq:>4}×{head_dim}]  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

// ── embedding lookup ──────────────────────────────────────────────────────────

fn bench_embedding(dev: &t500, vocab: u32, dim: u32, seq: u32) {
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.001).collect();
    let ids: Vec<u32> = (0..seq).map(|i| i % vocab).collect();
    let gt = dev.f502(&table);
    let gi = dev.f502(bytemuck::cast_slice(&ids));
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f670(&gi, &gt, seq, vocab, dim).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f670(&gi, &gt, seq, vocab, dim).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (seq * dim * 4) as u64;
    println!("  embedding [vocab={vocab} dim={dim} seq={seq:>3}]  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

// ── sampler ops ───────────────────────────────────────────────────────────────

fn bench_top_k(dev: &t500, batch: u32, vocab: u32, k: u32) {
    let logits: Vec<f32> = (0..batch * vocab).map(|i| (i as f32) * 0.001 - 16.0).collect();
    let gl = dev.f502(&logits);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f787(&gl, k, batch, vocab).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f787(&gl, k, batch, vocab).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (batch * vocab * 4 * 2) as u64;
    println!("  top_k_mask  [b={batch} v={vocab} k={k:>3}]  {ms:>7.3} ms  {:>5.1} GB/s",
        bw_gb_s(bytes, ms));
}

fn bench_top_p(dev: &t500, batch: u32, vocab: u32, p: f32) {
    // probs after top-k + softmax
    let mut probs: Vec<f32> = vec![0.0; (batch * vocab) as usize];
    for row in 0..batch as usize {
        let base = row * vocab as usize;
        let sum: f32 = 50.0; // k=50 non-zero slots, equal weight
        for i in 0..50usize {
            probs[base + i] = 1.0 / sum;
        }
    }
    let gp = dev.f502(&probs);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f788(&gp, p, batch, vocab).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f788(&gp, p, batch, vocab).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (batch * vocab * 4 * 2) as u64;
    println!("  top_p_mask  [b={batch} v={vocab} p={p:.2}]  {ms:>7.3} ms  {:>5.1} GB/s",
        bw_gb_s(bytes, ms));
}

fn bench_multinomial(dev: &t500, batch: u32, vocab: u32) {
    let mut probs: Vec<f32> = vec![0.0; (batch * vocab) as usize];
    for row in 0..batch as usize {
        let base = row * vocab as usize;
        for i in 0..50usize { probs[base + i] = 1.0 / 50.0; }
    }
    let gp = dev.f502(&probs);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f789(&gp, 42, 0, batch, vocab).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f789(&gp, 42, 0, batch, vocab).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    println!("  multinomial [b={batch} v={vocab}]         {ms:>7.3} ms");
}

fn bench_rep_penalty(dev: &t500, batch: u32, vocab: u32, seq_len: u32) {
    let logits: Vec<f32> = (0..batch * vocab).map(|i| (i as f32) * 0.001 - 8.0).collect();
    let ids: Vec<f32> = (0..batch * seq_len).map(|i| (i % vocab) as f32).collect();
    let gl = dev.f502(&logits);
    let gi = dev.f502(&ids);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f790(&gl, &gi, 1.3, batch, vocab, seq_len).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f790(&gl, &gi, 1.3, batch, vocab, seq_len).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (batch * vocab * 4 * 2) as u64;
    println!("  rep_penalty [b={batch} v={vocab} seq={seq_len:>3}]  {ms:>7.3} ms  {:>5.1} GB/s",
        bw_gb_s(bytes, ms));
}

// Full decode-step sampling pipeline: top_k → softmax → top_p → sample
fn bench_sample_pipeline(dev: &t500, batch: u32, vocab: u32, k: u32, p: f32) {
    let logits: Vec<f32> = (0..batch * vocab).map(|i| (i as f32) * 0.001 - 16.0).collect();
    let gl = dev.f502(&logits);
    let run = || {
        let masked = dev.f787(&gl, k, batch, vocab).unwrap();
        let probs  = dev.f620(&masked, batch, vocab).unwrap();
        let pruned = dev.f788(&probs, p, batch, vocab).unwrap();
        dev.f504(&dev.f789(&pruned, 42, 0, batch, vocab).unwrap()).unwrap()
    };
    for _ in 0..WARMUP { let _ = run(); }
    let t0 = Instant::now();
    for _ in 0..ITERS { let _ = run(); }
    let ms = ms_per_iter(t0.elapsed());
    println!("  pipeline    [b={batch} v={vocab} k={k:>3} p={p:.2}]  {ms:>7.3} ms  (top_k→sm→top_p→sample)");
}

// ── backward shaders ─────────────────────────────────────────────────────────

fn bench_ln_backward(dev: &t500, rows: u32, cols: u32) {
    let grad: Vec<f32> = vec![0.01f32; (rows * cols) as usize];
    let inp:  Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let g:    Vec<f32> = vec![1.0f32; cols as usize];
    let gg = dev.f502(&grad);
    let gi = dev.f502(&inp);
    let gw = dev.f502(&g);
    for _ in 0..WARMUP {
        let (a, b, c) = dev.f791(&gg, &gi, &gw, rows, cols, 1e-5).unwrap();
        let _ = (dev.f504(&a).unwrap(), dev.f504(&b).unwrap(), dev.f504(&c).unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let (a, b, c) = dev.f791(&gg, &gi, &gw, rows, cols, 1e-5).unwrap();
        let _ = (dev.f504(&a).unwrap(), dev.f504(&b).unwrap(), dev.f504(&c).unwrap());
    }
    let ms = ms_per_iter(t0.elapsed());
    // grad_out + input + gamma read; grad_input + grad_gamma + grad_beta write
    let bytes = ((rows * cols * 2 + cols) * 4 * 2) as u64;
    println!("  ln_backward [{rows:>4}×{cols}]  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

fn bench_rms_backward(dev: &t500, rows: u32, cols: u32) {
    let grad: Vec<f32> = vec![0.01f32; (rows * cols) as usize];
    let inp:  Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let g:    Vec<f32> = vec![1.0f32; cols as usize];
    let gg = dev.f502(&grad);
    let gi = dev.f502(&inp);
    let gw = dev.f502(&g);
    for _ in 0..WARMUP {
        let (a, b) = dev.f792(&gg, &gi, &gw, rows, cols, 1e-5).unwrap();
        let _ = (dev.f504(&a).unwrap(), dev.f504(&b).unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let (a, b) = dev.f792(&gg, &gi, &gw, rows, cols, 1e-5).unwrap();
        let _ = (dev.f504(&a).unwrap(), dev.f504(&b).unwrap());
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = ((rows * cols * 2 + cols) * 4 * 2) as u64;
    println!("  rn_backward [{rows:>4}×{cols}]  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

fn bench_softmax_backward(dev: &t500, rows: u32, cols: u32, label: &str) {
    let grad: Vec<f32> = vec![0.01f32; (rows * cols) as usize];
    let fwd:  Vec<f32> = (0..rows * cols).map(|i| {
        let v = (i % cols) as f32;
        (v * 0.01).exp() / 10.0  // rough softmax output
    }).collect();
    let gg = dev.f502(&grad);
    let gf = dev.f502(&fwd);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f794(&gg, &gf, rows, cols).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f794(&gg, &gf, rows, cols).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (rows * cols * 4 * 3) as u64; // grad + fwd + output
    println!("  sm_backward {label:<20}  {ms:>7.3} ms  {:>5.1} GB/s", bw_gb_s(bytes, ms));
}

fn bench_rope_backward(dev: &t500, batch_heads: u32, seq: u32, head_dim: u32) {
    let n = (batch_heads * seq * head_dim) as usize;
    let grad: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let gg = dev.f502(&grad);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f796(&gg, batch_heads, seq, head_dim, 0, 10000.0).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f796(&gg, batch_heads, seq, head_dim, 0, 10000.0).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    let bytes = (n * 4 * 2) as u64;
    println!("  rope_bwd    [{batch_heads:>2}×{seq:>4}×{head_dim}]  {ms:>7.3} ms  {:>5.1} GB/s",
        bw_gb_s(bytes, ms));
}

fn bench_embed_backward(dev: &t500, vocab: u32, dim: u32, seq: u32) {
    let grad: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.001).collect();
    let ids: Vec<f32>  = (0..seq).map(|i| (i % vocab) as f32).collect();
    let gg = dev.f502(&grad);
    let gi = dev.f502(&ids);
    for _ in 0..WARMUP {
        let _ = dev.f504(&dev.f793(&gg, &gi, seq, vocab, dim).unwrap()).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = dev.f504(&dev.f793(&gg, &gi, seq, vocab, dim).unwrap()).unwrap();
    }
    let ms = ms_per_iter(t0.elapsed());
    println!("  embed_bwd   [v={vocab} d={dim} seq={seq:>3}]  {ms:>7.3} ms");
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let dev = t500::f500().expect("no GPU");
    println!("any-gpu ops benchmark");
    println!("═══════════════════════════════════════════════════");
    println!("device:    {} ({})", dev.s502, dev.s503);
    println!("subgroup:  {}", dev.s509);
    println!("warmup:    {WARMUP} runs   measured: {ITERS} runs");
    println!("═══════════════════════════════════════════════════");

    // ── Norm: LLaMA-7B d_model=4096 ──────────────────────────────────────────
    header("RMSNorm  (P3 fused, d_model=4096)");
    for &seq in &[1u32, 8, 64, 512] {
        bench_rms(&dev, seq, 4096);
    }

    header("LayerNorm  (P3 fused, d_model=4096)");
    for &seq in &[1u32, 8, 64, 512] {
        bench_ln(&dev, seq, 4096);
    }

    // ── Softmax ───────────────────────────────────────────────────────────────
    header("Softmax  (subgroup-fused, P2)");
    bench_softmax(&dev, 32,  128,  "[32×128  attn decode]");
    bench_softmax(&dev, 32,  512,  "[32×512  attn prefill]");
    bench_softmax(&dev, 32, 2048,  "[32×2048 long context]");
    bench_softmax(&dev, 512, 32000, "[512×32k lm-head]");

    // ── Matmul: projection layers ─────────────────────────────────────────────
    header("Matmul  (wave64 4×4, P1)");
    bench_matmul(&dev,   1, 4096, 4096);  // decode: 1 token → QKV projection
    bench_matmul(&dev,  64, 4096, 4096);  // small prefill
    bench_matmul(&dev, 512, 4096, 4096);  // medium prefill
    bench_matmul(&dev,   1, 4096, 11008); // decode: MLP gate
    bench_matmul(&dev, 512, 4096, 11008); // prefill: MLP gate

    // ── Batch matmul: attention QK^T and AV ──────────────────────────────────
    header("Batch matmul  (wave64 4×4, P4) — LLaMA-7B 32 heads hd=128");
    bench_batch_matmul(&dev, 32,   1, 128, 128,  "[decode  Q·K^T  1×128]");
    bench_batch_matmul(&dev, 32,  64,  64, 128,  "[prefill Q·K^T 64×64 ]");
    bench_batch_matmul(&dev, 32, 512, 512, 128,  "[prefill Q·K^T 512×512]");
    bench_batch_matmul(&dev, 32,   1, 128,  64,  "[decode  A·V    1×128 ]");
    bench_batch_matmul(&dev, 32,  64, 128,  64,  "[prefill A·V   64×128 ]");

    // ── Fused SDPA (f626) ─────────────────────────────────────────────────────
    header("Fused SDPA  (online-softmax, no N×N alloc)");
    bench_sdpa(&dev, 32,   1, 128, 128, "[decode    1q / 128kv]");
    bench_sdpa(&dev, 32,   1, 512, 128, "[decode    1q / 512kv]");
    bench_sdpa(&dev, 32,  64,  64, 128, "[prefill  64q /  64kv]");
    bench_sdpa(&dev, 32, 512, 512, 128, "[prefill 512q / 512kv]");

    // ── RoPE ─────────────────────────────────────────────────────────────────
    header("RoPE");
    bench_rope(&dev, 32,   1, 128);
    bench_rope(&dev, 32,  64, 128);
    bench_rope(&dev, 32, 512, 128);

    // ── Embedding lookup ──────────────────────────────────────────────────────
    header("Embedding lookup");
    bench_embedding(&dev, 32000, 4096,   1);
    bench_embedding(&dev, 32000, 4096, 512);

    // ── Sampler ops ───────────────────────────────────────────────────────────
    header("Sampler ops  (LLM decode step, vocab=32000)");
    bench_top_k(&dev, 1, 32000, 50);
    bench_top_k(&dev, 8, 32000, 50);
    bench_top_p(&dev, 1, 32000, 0.9);
    bench_top_p(&dev, 8, 32000, 0.9);
    bench_multinomial(&dev, 1, 32000);
    bench_multinomial(&dev, 8, 32000);
    bench_rep_penalty(&dev, 1, 32000, 128);
    bench_rep_penalty(&dev, 8, 32000, 128);

    header("Decode sampling pipeline  (top_k → softmax → top_p → sample)");
    bench_sample_pipeline(&dev, 1, 32000, 50, 0.9);
    bench_sample_pipeline(&dev, 8, 32000, 50, 0.9);

    // ── Backward shaders ─────────────────────────────────────────────────────
    header("LayerNorm backward  (3-pass)");
    bench_ln_backward(&dev,   1, 4096);
    bench_ln_backward(&dev,  64, 4096);
    bench_ln_backward(&dev, 512, 4096);

    header("RMSNorm backward  (3-pass)");
    bench_rms_backward(&dev,   1, 4096);
    bench_rms_backward(&dev,  64, 4096);
    bench_rms_backward(&dev, 512, 4096);

    header("Softmax backward  (dot-reduce + grad)");
    bench_softmax_backward(&dev, 32,   128, "[32×128  attn decode]");
    bench_softmax_backward(&dev, 32,   512, "[32×512  attn prefill]");
    bench_softmax_backward(&dev, 512, 32000, "[512×32k lm-head]");

    header("RoPE backward  (conjugate rotation)");
    bench_rope_backward(&dev, 32,   1, 128);
    bench_rope_backward(&dev, 32,  64, 128);
    bench_rope_backward(&dev, 32, 512, 128);

    // vocab=512 keeps the readback at 8 MB instead of 512 MB so PCIe transfer
    // doesn't swamp the scatter-add time. Real LLM training keeps grad_weight on GPU.
    header("Embedding backward  (f32 CAS scatter-add, vocab=512)");
    bench_embed_backward(&dev, 512, 4096,   1);
    bench_embed_backward(&dev, 512, 4096, 128);
    bench_embed_backward(&dev, 512, 4096, 512);

    println!("\n═══════════════════════════════════════════════════");
}
