// Unlicense — cochranblock.org
//
// Sprint 7 step 2 demo: exercises causal SDPA + RoPE + KV cache using ONLY
// the public any-gpu API. Compilation alone proves the symbols are reachable
// from outside the crate. `cargo run --release --example decode_step2`
// to also exercise the pipeline on a real GPU.

use any_gpu::{t500, t501, t534};

fn main() -> anyhow::Result<()> {
    let dev = t500::f500()?;
    println!("device: {} ({})", dev.s502, dev.s503);

    let bh: u32 = 1;
    let d_k: u32 = 4;
    let max_seq: u32 = 8;

    let mut cache = t534::f672(&dev, max_seq, bh, d_k);

    // ----- Prefill: 2 tokens at positions {0, 1} -----
    let q_prefill = dev.f502(&[
        0.10, 0.20, 0.30, 0.40,
        0.50, 0.60, 0.70, 0.80,
    ]);
    let k_prefill_raw = dev.f502(&[
        1.00, 0.00, 0.50, -0.50,
        0.20, 0.40, -0.30, 0.70,
    ]);
    let v_prefill = dev.f502(&[
        0.90, -0.10, 0.20, 0.30,
        -0.40, 0.50, 0.60, -0.20,
    ]);

    let q_prefill_rot = dev.f625(&q_prefill, bh, 2, d_k, 0, 10000.0)?;
    let k_prefill_rot = dev.f625(&k_prefill_raw, bh, 2, d_k, 0, 10000.0)?;
    cache.f673(&dev, &k_prefill_rot, &v_prefill, 2)?;

    // Causal SDPA over the prefill (q_seq=2, kv_seq=2 — full triangular mask)
    let k_used: t501 = dev.f502(&dev.f504(cache.f676())?[..(2 * d_k) as usize]);
    let v_used: t501 = dev.f502(&dev.f504(cache.f677())?[..(2 * d_k) as usize]);
    let prefill_out = dev.f623(&q_prefill_rot, &k_used, &v_used, bh, 2, 2, d_k)?;
    let prefill_vec = dev.f504(&prefill_out)?;
    println!("prefill ({} f32):", prefill_vec.len());
    for row in prefill_vec.chunks(d_k as usize) {
        println!("  {:?}", row);
    }
    assert!(prefill_vec.iter().all(|x| x.is_finite()),
        "prefill output must be finite (no NaN/INF from mask leak)");

    // ----- Decode: 1 token at absolute position 2 -----
    let q_decode = dev.f502(&[-0.20, 0.70, 0.40, -0.10]);
    let k_decode_raw = dev.f502(&[0.30, -0.60, 0.80, 0.10]);
    let v_decode = dev.f502(&[0.00, 1.00, -1.00, 0.50]);

    let q_decode_rot = dev.f625(&q_decode, bh, 1, d_k, 2, 10000.0)?;
    let k_decode_rot = dev.f625(&k_decode_raw, bh, 1, d_k, 2, 10000.0)?;
    cache.f673(&dev, &k_decode_rot, &v_decode, 1)?;

    let kv_len = cache.f675();
    let k_used: t501 = dev.f502(&dev.f504(cache.f676())?[..(kv_len * d_k) as usize]);
    let v_used: t501 = dev.f502(&dev.f504(cache.f677())?[..(kv_len * d_k) as usize]);
    let decode_out = dev.f623(&q_decode_rot, &k_used, &v_used, bh, 1, kv_len, d_k)?;
    let decode_vec = dev.f504(&decode_out)?;
    println!("decode (kv_len={kv_len}): {:?}", decode_vec);
    assert!(decode_vec.iter().all(|x| x.is_finite()), "decode output finite");
    assert!(decode_vec.iter().any(|&x| x.abs() > 0.01), "decode output non-trivial");

    // ----- Reset and reuse the cache -----
    cache.f674();
    assert_eq!(cache.f675(), 0);
    cache.f673(&dev, &k_prefill_rot, &v_prefill, 2)?;
    assert_eq!(cache.f675(), 2);

    println!("step 2 public-API smoke: prefill + decode + reset OK");
    Ok(())
}
