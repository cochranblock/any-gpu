// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Benchmark: CPU naive matmul vs Vulkan compute shader matmul.

use candle_vulkan::VulkanDevice;
use rand::Rng;
use std::time::Instant;

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    out
}

fn bench_size(dev: &VulkanDevice, size: usize) {
    let m = size;
    let n = size;
    let k = size;

    let mut rng = rand::thread_rng();
    let a_data: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b_data: Vec<f32> = (0..k * n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // CPU
    let t0 = Instant::now();
    let cpu_result = cpu_matmul(&a_data, &b_data, m, n, k);
    let cpu_time = t0.elapsed();

    // GPU — include upload + compute + readback
    let t0 = Instant::now();
    let a_gpu = dev.upload(&a_data);
    let b_gpu = dev.upload(&b_data);
    let c_gpu = dev
        .matmul(&a_gpu, &b_gpu, m as u32, n as u32, k as u32)
        .unwrap();
    let gpu_result = dev.read(&c_gpu).unwrap();
    let gpu_total = t0.elapsed();

    // GPU — compute only (data already on device)
    let t0 = Instant::now();
    let c_gpu2 = dev
        .matmul(&a_gpu, &b_gpu, m as u32, n as u32, k as u32)
        .unwrap();
    // Force sync by reading one element
    let _ = dev.read(&c_gpu2).unwrap();
    let gpu_compute = t0.elapsed();

    // Verify correctness (spot check)
    let max_err = cpu_result
        .iter()
        .zip(gpu_result.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let cpu_gflops = flops / cpu_time.as_secs_f64() / 1e9;
    let gpu_gflops_total = flops / gpu_total.as_secs_f64() / 1e9;
    let gpu_gflops_compute = flops / gpu_compute.as_secs_f64() / 1e9;

    println!("matmul {m}x{k} * {k}x{n}:");
    println!("  CPU:              {:>8.2} ms  ({:.2} GFLOPS)", cpu_time.as_secs_f64() * 1e3, cpu_gflops);
    println!("  GPU (w/ transfer):{:>8.2} ms  ({:.2} GFLOPS)", gpu_total.as_secs_f64() * 1e3, gpu_gflops_total);
    println!("  GPU (compute):    {:>8.2} ms  ({:.2} GFLOPS)", gpu_compute.as_secs_f64() * 1e3, gpu_gflops_compute);
    println!("  max error:        {max_err:.6}");
    println!("  speedup:          {:.1}x (total), {:.1}x (compute)",
        cpu_time.as_secs_f64() / gpu_total.as_secs_f64(),
        cpu_time.as_secs_f64() / gpu_compute.as_secs_f64(),
    );
    println!();
}

fn main() {
    println!("candle-vulkan benchmark");
    println!("=======================\n");

    let dev = VulkanDevice::new().expect("no Vulkan GPU available");
    println!(
        "device: {} ({})\n",
        dev.adapter_name, dev.backend
    );

    for &size in &[64, 128, 256, 512, 1024] {
        bench_size(&dev, size);
    }
}
