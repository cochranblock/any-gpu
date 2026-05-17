use any_gpu::t500;
use std::time::Instant;

fn main() {
    let dev = t500::f500().expect("no GPU");
    println!("conv2d benchmark on {}", dev.s502);
    println!("=================================\n");

    // Warmup
    let w = dev.f502(&[1.0; 9]);
    let inp = dev.f502(&[1.0; 16]);
    let _ = dev.f582(&inp, &w, None, 1, 1, 4, 4, 1, 3, 3, (1,1), (1,1), (1,1), 1);

    for &(c_in, c_out, h, w_size, kh) in &[
        (3u32, 64, 32, 32, 3),    // first layer
        (64, 128, 16, 16, 3),     // downsample
        (128, 256, 8, 8, 3),      // bottleneck
        (256, 128, 8, 8, 3),      // upsample path
        (128, 64, 16, 16, 3),     // decoder
        (64, 3, 32, 32, 3),       // output layer
    ] {
        let n_in = (1 * c_in * h * w_size) as usize;
        let n_w = (c_out * c_in * kh * kh) as usize;
        let input = dev.f502(&vec![0.01f32; n_in]);
        let weight = dev.f502(&vec![0.01f32; n_w]);

        let t0 = Instant::now();
        let iters = 10;
        for _ in 0..iters {
            let out = dev.f582(&input, &weight, None,
                1, c_in, h, w_size, c_out, kh, kh, (1,1), (1,1), (1,1), 1).unwrap();
            let _ = dev.f504(&out).unwrap();
        }
        let avg = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let out_h = h;
        let out_w = w_size;
        let flops = 2.0 * 1.0 * c_out as f64 * out_h as f64 * out_w as f64 * c_in as f64 * kh as f64 * kh as f64;
        let gflops = flops / (avg / 1e3) / 1e9;
        println!("  conv2d {c_in:>3}x{h:>2}x{w_size:>2} -> {c_out:>3}x{out_h:>2}x{out_w:>2}  k={kh}x{kh}  {:>8.2} ms  ({:.2} GFLOPS)", avg, gflops);
    }
}
