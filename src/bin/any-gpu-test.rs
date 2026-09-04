// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// any-gpu-test — exopack TRIPLE SIMS gate for the any-gpu crate.
//
// Runs a deterministic smoke test against the public API 3× back-to-back
// (the TRIPLE SIMS discipline used across the cochranblock fleet). A single
// failure across any of the three passes fails the whole gate. Designed to
// surface flaky shader bindings, KV-cache state leaks across runs, and
// non-deterministic dequant paths.
//
// Build: cargo build --release --bin any-gpu-test --features tests
// Run:   cargo run   --release --bin any-gpu-test --features tests
//
// On RADV/RDNA1 the shared GpuDevice lives in a LazyLock so all three sims
// reuse the same `wgpu::Instance` — same reason the in-crate test harness uses
// TEST_DEV. Without that, concurrent instance creation segfaults Mesa.

#![allow(non_camel_case_types, non_snake_case)]

use std::sync::LazyLock;
use std::time::Instant;

use any_gpu::{t500, t501, t534, t538};
use safetensors::tensor::{Dtype, TensorView, serialize};

static DEV: LazyLock<t500> = LazyLock::new(|| t500::f500().expect("any-gpu-test: GPU init"));

/// One deterministic pass over the public Sprint 7 API surface.
/// Returns true iff every step produces the bit-exact expected output.
/// Each call constructs fresh inputs — no state leaks between sims.
fn run_once() -> bool {
    // ----- Sprint 7 step 3: build a safetensors file in memory, parse, upload -----
    let v0: Vec<f32> = (0..16).map(|i| (i as f32) * 0.125 - 1.0).collect();
    let mut v1: Vec<u8> = Vec::with_capacity(v0.len() * 4);
    for v2 in &v0 {
        v1.extend_from_slice(&v2.to_le_bytes());
    }
    let v3 = match TensorView::new(Dtype::F32, vec![4, 4], v1.as_slice()) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let v4 = match serialize(vec![("w", v3)], &None) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let v5 = match t538::f761(&v4) {
        Ok(p) => p,
        Err(_) => return false,
    };
    if v5.f763("w") != Some(&[4u32, 4][..]) {
        return false;
    }
    if v5.f764("w") != Some(v0.as_slice()) {
        return false;
    }

    let v6: t501 = match v5.f765(&DEV, "w") {
        Ok(p) => p,
        Err(_) => return false,
    };
    let v7 = match DEV.f504(&v6) {
        Ok(p) => p,
        Err(_) => return false,
    };
    if v7 != v0 {
        return false;
    }

    // ----- Sprint 7 step 1+2 chain on the loaded weights -----
    // Treat the [4, 4] tensor as Q, K, V for a 1-head attention with seq=4, d_k=4.
    // RoPE both Q and K at start_pos=0, then run causal SDPA — fully deterministic.
    let v8 = match DEV.f625(&v6, 1, 4, 4, 0, 10000.0) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let v9 = match DEV.f625(&v6, 1, 4, 4, 0, 10000.0) {
        Ok(p) => p,
        Err(_) => return false,
    };
    // RoPE is deterministic — two calls with same input must produce identical output.
    let v10 = match DEV.f504(&v8) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let v11 = match DEV.f504(&v9) {
        Ok(p) => p,
        Err(_) => return false,
    };
    if v10 != v11 {
        return false;
    }

    // Causal SDPA: q_seq=4, kv_seq=4, full triangular mask. Row 0 must equal V[0].
    let v12 = match DEV.f623(&v8, &v9, &v6, 1, 4, 4, 4) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let v13 = match DEV.f504(&v12) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // Row 0 of attention output must equal V row 0 (causal mask, only j=0 attended).
    for v14 in 0..4 {
        if (v13[v14] - v0[v14]).abs() > 1e-3 {
            eprintln!("row 0 dim {v14}: got {} expected {}", v13[v14], v0[v14]);
            return false;
        }
    }
    if v13.iter().any(|p| !p.is_finite()) {
        return false;
    }

    // ----- Sprint 7 step 2 KV cache: prefill 2 tokens, decode 1, read back -----
    let mut v15 = t534::f672(&DEV, 8, 1, 4);
    let v16 = DEV.f502(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let v17 = DEV.f502(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    if v15.f673(&DEV, &v16, &v17, 2).is_err() {
        return false;
    }
    if v15.f675() != 2 {
        return false;
    }

    let v18 = DEV.f502(&[9.0, 9.0, 9.0, 9.0]);
    let v19 = DEV.f502(&[90.0, 90.0, 90.0, 90.0]);
    if v15.f673(&DEV, &v18, &v19, 1).is_err() {
        return false;
    }
    if v15.f675() != 3 {
        return false;
    }

    let v20 = match DEV.f504(v15.f676()) {
        Ok(p) => p,
        Err(_) => return false,
    };
    if &v20[..8] != &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
        return false;
    }
    if &v20[8..12] != &[9.0, 9.0, 9.0, 9.0] {
        return false;
    }
    if v20[12..].iter().any(|&p| p != 0.0) {
        return false;
    }

    true
}

fn main() {
    // exopack 0.2.1 exposes the compressed alias only — the v0.3+ `run` is a
    // forwarder that doesn't exist yet here. Pin to f60 so the 0.2 line works.
    let v0 = pollster::block_on(exopack::triple_sims::f60(|| async { run_once() }));
    let v1 = Instant::now();
    println!(
        "any-gpu-test TRIPLE SIMS: {} (device: {})",
        if v0 { "PASS" } else { "FAIL" },
        DEV.s502
    );
    let _ = v1; // exopack already times each pass internally
    std::process::exit(if v0 { 0 } else { 1 });
}
