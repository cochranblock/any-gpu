// Benchmark subgroup-fused softmax (f620) at realistic attention-head sizes.
use any_gpu::t500;
use std::time::Instant;

fn bench_softmax(dev: &t500, rows: u32, cols: u32) {
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
    let buf = dev.f502(&data);

    // Warmup
    for _ in 0..3 {
        let _ = dev.f620(&buf, rows, cols).unwrap();
        let _ = dev.f504(&dev.f620(&buf, rows, cols).unwrap()).unwrap();
    }

    let n = 20;
    let t0 = Instant::now();
    for _ in 0..n {
        let out = dev.f620(&buf, rows, cols).unwrap();
        let _ = dev.f504(&out).unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    let gb_s = (rows * cols * 4 * 2) as f64 / (ms * 1e-3) / 1e9;
    println!("  softmax [{rows}x{cols}]  {ms:>8.3}ms  {gb_s:.1} GB/s");
}

fn main() {
    let dev = t500::f500().expect("no GPU");
    println!("softmax benchmark — subgroup={}", dev.s509);
    println!("device: {} ({})\n", dev.s502, dev.s503);

    // Attention softmax: rows=batch_heads, cols=seq_len
    for &seq in &[128u32, 512, 1024, 2048] {
        bench_softmax(&dev, 32, seq);
    }
    println!();
    // LM head softmax: rows=batch*seq, cols=vocab_size
    bench_softmax(&dev, 16, 32000);
}
