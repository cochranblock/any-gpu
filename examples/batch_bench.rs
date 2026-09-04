// Unlicense — cochranblock.org
// Benchmark: eager dispatch (one submit per op) vs batched dispatch (one submit total).
// Runs a chain of N matmuls to simulate a forward pass through a stack of linear layers.

use any_gpu::t500;
use std::time::Instant;

fn bench_chain(dev: &t500, size: usize, n_ops: usize) {
    let data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
    let weight: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.0001).collect();

    let x = dev.f502(&data);
    let w = dev.f502(&weight);

    // --- Eager: one submit per matmul ---
    let t0 = Instant::now();
    let mut out = dev
        .f580(&x, &w, size as u32, size as u32, size as u32)
        .unwrap();
    for _ in 1..n_ops {
        out = dev
            .f580(&out, &w, size as u32, size as u32, size as u32)
            .unwrap();
    }
    let _ = dev.f504(&out).unwrap();
    let eager_ms = t0.elapsed().as_secs_f64() * 1e3;

    // --- Batched: all ops into one encoder, one submit ---
    let t0 = Instant::now();
    let batch = dev.f509();
    let mut out = dev
        .f580(&x, &w, size as u32, size as u32, size as u32)
        .unwrap();
    for _ in 1..n_ops {
        out = dev
            .f580(&out, &w, size as u32, size as u32, size as u32)
            .unwrap();
    }
    batch.f511();
    let _ = dev.f504(&out).unwrap();
    let batch_ms = t0.elapsed().as_secs_f64() * 1e3;

    let speedup = eager_ms / batch_ms;
    println!(
        "  {n_ops:>3}× matmul {size}²  eager={eager_ms:>8.2}ms  batch={batch_ms:>8.2}ms  speedup={speedup:.2}×"
    );
}

fn main() {
    println!("any-gpu batch dispatch benchmark");
    println!("=================================\n");

    let dev = t500::f500().expect("no GPU");
    println!("device: {} ({})\n", dev.s502, dev.s503);

    println!("chain of N matmuls (simulating N linear layers in a forward pass):\n");
    for &size in &[256usize, 512, 1024] {
        for &n in &[1usize, 4, 8, 16] {
            bench_chain(&dev, size, n);
        }
        println!();
    }
}
