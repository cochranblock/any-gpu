// Candle matmul benchmark — compare against any-gpu on the same hardware.
// Build with: cargo run --release --example candle_bench --features cuda
// Or:         cargo run --release --example candle_bench --features metal

use candle_core::{Device, Tensor};
use std::time::Instant;

fn bench_matmul(dev: &Device, size: usize, warmup: usize, iters: usize) {
    let a = Tensor::randn(0f32, 1.0, (size, size), dev).unwrap();
    let b = Tensor::randn(0f32, 1.0, (size, size), dev).unwrap();

    // Warmup — pipeline/kernel compilation
    for _ in 0..warmup {
        let _ = a.matmul(&b).unwrap();
    }
    // Sync before timing
    if let Device::Cuda(_) = dev {
        let _ = a.matmul(&b).unwrap().to_vec2::<f32>().unwrap();
    }

    // Timed runs
    let t0 = Instant::now();
    for _ in 0..iters {
        let c = a.matmul(&b).unwrap();
        // Force sync by reading one element
        let _ = c.flatten_all().unwrap().get(0).unwrap().to_scalar::<f32>().unwrap();
    }
    let elapsed = t0.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1e3 / iters as f64;
    let flops = 2.0 * size as f64 * size as f64 * size as f64;
    let gflops = flops / (avg_ms / 1e3) / 1e9;

    println!(
        "  {size:>5}x{size:<5}  {:>8.2} ms  ({:.2} GFLOPS)  [{} iters]",
        avg_ms, gflops, iters
    );
}

fn main() {
    // Try CUDA first, then Metal, then CPU
    let (dev, backend_name) = if cfg!(feature = "cuda") {
        match Device::new_cuda(0) {
            Ok(d) => (d, "CUDA"),
            Err(e) => {
                eprintln!("CUDA init failed: {e}");
                (Device::Cpu, "CPU")
            }
        }
    } else if cfg!(feature = "metal") {
        match Device::new_metal(0) {
            Ok(d) => (d, "Metal"),
            Err(e) => {
                eprintln!("Metal init failed: {e}");
                (Device::Cpu, "CPU")
            }
        }
    } else {
        (Device::Cpu, "CPU")
    };

    println!("candle matmul benchmark ({backend_name})");
    println!("===================================\n");

    for &size in &[128, 256, 512, 1024] {
        let iters = if size <= 256 { 100 } else if size <= 512 { 50 } else { 20 };
        bench_matmul(&dev, size, 3, iters);
    }
}
