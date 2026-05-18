// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// any-gpu — Stratagems CLI.
// Usage: any-gpu <COMMAND>
//   info              Print GPU device info
//   bench             Benchmark key ops at LLaMA-7B scales
//   train <stratagem> Run a named training stratagem
//
// Stratagems: subatomic

use anyhow::{bail, Result};
use any_gpu::t500;
use clap::{Parser, Subcommand};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "any-gpu", about = "Tensor engine CLI — info, bench, train")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print GPU adapter, backend, and feature flags
    Info,
    /// Benchmark key ops at LLaMA-7B scales
    Bench,
    /// Run a named training stratagem
    Train {
        /// Stratagem name (available: subatomic)
        stratagem: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let gpu = t500::f500()?;
    match cli.cmd {
        Cmd::Info => cmd_info(&gpu),
        Cmd::Bench => cmd_bench(&gpu),
        Cmd::Train { stratagem } => cmd_train(&gpu, &stratagem),
    }
}

// ── info ─────────────────────────────────────────────────────────────────────

fn cmd_info(gpu: &t500) -> Result<()> {
    println!("adapter  : {}", gpu.s502);
    println!("backend  : {}", gpu.s503);
    println!("subgroup : {}", gpu.s509);
    println!("f16      : {}", gpu.s521);
    Ok(())
}

// ── bench ────────────────────────────────────────────────────────────────────

const WARMUP: usize = 3;
const ITERS: usize = 10;

// Each closure must call f504 to fence the GPU; timings are compute+readback.
fn time_op(f: impl Fn()) -> f64 {
    for _ in 0..WARMUP { f(); }
    let t = Instant::now();
    for _ in 0..ITERS { f(); }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn cmd_bench(gpu: &t500) -> Result<()> {
    println!("device: {} ({})", gpu.s502, gpu.s503);
    println!();

    // matmul prefill: 512×4096 × 4096×4096 → 512×4096
    {
        let a = gpu.f502(&vec![0.1f32; 512 * 4096]);
        let b = gpu.f502(&vec![0.1f32; 4096 * 4096]);
        let ms = time_op(|| { gpu.f504(&gpu.f580(&a, &b, 512, 4096, 4096).unwrap()).unwrap(); });
        let gflops = 2.0 * 512.0 * 4096.0 * 4096.0 / (ms * 1e-3) / 1e9;
        println!("matmul prefill 512×4096×4096 : {ms:>7.2} ms  {gflops:.0} GFLOPS");
    }

    // matmul decode GEMV: 1×4096 × 4096×4096 → 1×4096
    {
        let a = gpu.f502(&vec![0.1f32; 4096]);
        let b = gpu.f502(&vec![0.1f32; 4096 * 4096]);
        let ms = time_op(|| { gpu.f504(&gpu.f580(&a, &b, 1, 4096, 4096).unwrap()).unwrap(); });
        let gb_s = 4096.0 * 4096.0 * 4.0 / (ms * 1e-3) / 1e9;
        println!("matmul decode  1×4096×4096   : {ms:>7.2} ms  {gb_s:.1} GB/s");
    }

    println!();

    // rms_norm 512×4096
    {
        let x = gpu.f502(&vec![0.1f32; 512 * 4096]);
        let w = gpu.f502(&vec![1.0f32; 4096]);
        let ms = time_op(|| { gpu.f504(&gpu.f603(&x, &w, 512, 4096, 1e-5).unwrap()).unwrap(); });
        let gb_s = 512.0 * 4096.0 * 4.0 * 2.0 / (ms * 1e-3) / 1e9;
        println!("rms_norm   512×4096          : {ms:>7.3} ms  {gb_s:.1} GB/s");
    }

    // layer_norm 512×4096
    {
        let x = gpu.f502(&vec![0.1f32; 512 * 4096]);
        let w = gpu.f502(&vec![1.0f32; 4096]);
        let b = gpu.f502(&vec![0.0f32; 4096]);
        let ms = time_op(|| { gpu.f504(&gpu.f602(&x, &w, &b, 512, 4096, 1e-5).unwrap()).unwrap(); });
        let gb_s = 512.0 * 4096.0 * 4.0 * 2.0 / (ms * 1e-3) / 1e9;
        println!("layer_norm 512×4096          : {ms:>7.3} ms  {gb_s:.1} GB/s");
    }

    println!();

    // fused SDPA decode: 32 heads, 1q, 512kv, head_dim=128
    {
        let bh: u32 = 32; let q: u32 = 1; let kv: u32 = 512; let hd: u32 = 128;
        let qb = gpu.f502(&vec![0.1f32; (bh * q * hd) as usize]);
        let kb = gpu.f502(&vec![0.1f32; (bh * kv * hd) as usize]);
        let vb = gpu.f502(&vec![0.1f32; (bh * kv * hd) as usize]);
        let ms = time_op(|| { gpu.f504(&gpu.f626(&qb, &kb, &vb, bh, q, kv, hd).unwrap()).unwrap(); });
        println!("fused_sdpa decode  32h 1q/512kv/128d   : {ms:>7.3} ms");
    }

    // fused SDPA prefill: 32 heads, 512q, 512kv, head_dim=128
    {
        let bh: u32 = 32; let q: u32 = 512; let kv: u32 = 512; let hd: u32 = 128;
        let qb = gpu.f502(&vec![0.1f32; (bh * q * hd) as usize]);
        let kb = gpu.f502(&vec![0.1f32; (bh * kv * hd) as usize]);
        let vb = gpu.f502(&vec![0.1f32; (bh * kv * hd) as usize]);
        let ms = time_op(|| { gpu.f504(&gpu.f626(&qb, &kb, &vb, bh, q, kv, hd).unwrap()).unwrap(); });
        println!("fused_sdpa prefill 32h 512q/512kv/128d : {ms:>7.2} ms");
    }

    Ok(())
}

// ── train ────────────────────────────────────────────────────────────────────

fn cmd_train(gpu: &t500, stratagem: &str) -> Result<()> {
    match stratagem {
        "subatomic" => subatomic::run(gpu),
        other => bail!("unknown stratagem: {other}\navailable: subatomic"),
    }
}

// ── stratagem: subatomic ─────────────────────────────────────────────────────
// Three tiny language classifiers trained on GPU:
//   slop_detector    — clean prose vs AI slop (binary)
//   code_vs_english  — code snippet vs English prose (binary)
//   lang_detector    — Rust / Python / JS / Go / Shell (5-class)

mod subatomic {
    use any_gpu::t500;
    use anyhow::Result;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::Instant;

    const DIM: usize = 256;

    fn featurize(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 3 {
            let mut h = DefaultHasher::new(); text.hash(&mut h);
            v[(h.finish() as usize) % DIM] = 1.0; return v;
        }
        for w in chars.windows(3) {
            let mut h = DefaultHasher::new(); (w[0], w[1], w[2]).hash(&mut h);
            v[(h.finish() as usize) % DIM] += 1.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for f in &mut v { *f /= norm; } }
        v
    }

    fn train(
        gpu: &t500, name: &str,
        feats: &[Vec<f32>], labels: &[usize], nc: usize,
        epochs: usize, lr: f32,
    ) -> (Vec<f32>, Vec<f32>, f32) {
        let n = feats.len();
        let scale = (2.0 / (DIM + nc) as f64).sqrt() as f32;
        let mut w = vec![0.0f32; nc * DIM];
        let mut b = vec![0.0f32; nc];
        for (i, x) in w.iter_mut().enumerate() {
            let mut h = DefaultHasher::new(); i.hash(&mut h);
            *x = ((h.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0) * scale;
        }

        let t0 = Instant::now();
        let mut best_acc = 0.0f32;

        for epoch in 0..epochs {
            let mut correct = 0usize;
            let mut total_loss = 0.0f32;

            for (i, feat) in feats.iter().enumerate() {
                let target = labels[i];
                let x_gpu = gpu.f502(feat);

                // logits = feat @ W^T  ([1,DIM] @ [DIM,nc] = [1,nc])
                let mut wt = vec![0.0f32; DIM * nc];
                for c in 0..nc { for d in 0..DIM { wt[d * nc + c] = w[c * DIM + d]; } }
                let wt_gpu = gpu.f502(&wt);
                let logits_gpu = gpu.f580(&x_gpu, &wt_gpu, 1, nc as u32, DIM as u32).unwrap();
                let probs_gpu = gpu.f620(&logits_gpu, 1, nc as u32).unwrap();
                let probs = gpu.f504(&probs_gpu).unwrap();

                total_loss -= probs[target].max(1e-10).ln();
                let pred = probs.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                if pred == target { correct += 1; }

                let mut grad = probs.clone();
                grad[target] -= 1.0;

                let grad_gpu = gpu.f502(&grad);
                let gw_gpu = gpu.f580(&grad_gpu, &x_gpu, nc as u32, DIM as u32, 1).unwrap();
                let gw = gpu.f504(&gw_gpu).unwrap();
                for j in 0..w.len() { w[j] -= lr * gw[j]; }
                for c in 0..nc { b[c] -= lr * grad[c]; }
            }

            let acc = correct as f32 / n as f32;
            if acc > best_acc { best_acc = acc; }
            if epoch % 10 == 0 || epoch == epochs - 1 {
                eprintln!("  [{name}] epoch {}/{epochs}: loss={:.4} acc={:.1}%",
                    epoch + 1, total_loss / n as f32, acc * 100.0);
            }
        }

        eprintln!("  [{name}] done: {:.2}s  {} params  {:.1}% best",
            t0.elapsed().as_secs_f64(), nc * DIM + nc, best_acc * 100.0);
        (w, b, best_acc)
    }

    fn predict(w: &[f32], b: &[f32], text: &str, nc: usize, names: &[&str]) -> (&'static str, f32) {
        let feat = featurize(text);
        let mut logits = b.to_vec();
        for c in 0..nc { for d in 0..DIM { logits[c] += w[c * DIM + d] * feat[d]; } }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = probs.iter().sum(); for p in &mut probs { *p /= sum; }
        let i = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        // SAFETY: names is always a static slice literal in callers below
        let name: &'static str = unsafe { &*(names[i] as *const str) };
        (name, probs[i])
    }

    fn save(name: &str, w: &[f32], b: &[f32], names: &[&str], acc: f32) {
        let dir = format!("models/{name}");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(format!("{dir}/weights.bin"),
            w.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        std::fs::write(format!("{dir}/bias.bin"),
            b.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        let cfg = format!(
            r#"{{"name":"{name}","feature_dim":{DIM},"num_classes":{},"class_names":{names:?},"best_accuracy":{acc}}}"#,
            names.len());
        std::fs::write(format!("{dir}/config.json"), cfg).unwrap();
        eprintln!("  [{name}] saved → {dir}/");
    }

    pub fn run(gpu: &t500) -> Result<()> {
        eprintln!("=== subatomic stratagem — GPU: {} ===\n", gpu.s502);

        // ── slop_detector ─────────────────────────────────
        eprintln!("--- slop_detector ---");
        const CLEAN: &[&str] = &[
            "This function reads the file and returns the contents.",
            "The build passed with zero warnings.",
            "Fixed the off-by-one error in the loop.",
            "Added a test for the edge case.",
            "Refactored the parser to handle nested blocks.",
            "The server starts on port 8080.",
            "Removed the unused import.",
            "Updated the config to use the new path.",
            "The binary is 27 MB after stripping.",
            "Moved the struct to a separate module.",
            "Changed the return type to Result.",
            "The test covers both success and error paths.",
            "Split the function into smaller parts.",
            "Added timeout handling for SSH connections.",
            "The CI pipeline runs clippy and tests.",
        ];
        const SLOP: &[&str] = &[
            "utilize", "leverage", "optimize", "comprehensive", "robust",
            "seamlessly", "scalable", "paradigm", "synergy", "cutting-edge",
            "streamline", "empower", "delve", "foster", "harness",
            "groundbreaking", "innovative", "transform", "revolutionize", "unprecedented",
        ];
        const TMPLS: &[&str] = &[
            "We need to {} the codebase.", "This {} solution improves everything.",
            "Our {} approach handles all cases.", "This {} architecture supports all platforms.",
            "The system is designed to {} workflows.",
        ];
        let mut sf: Vec<Vec<f32>> = CLEAN.iter().map(|s| featurize(s)).collect();
        let mut sl: Vec<usize> = vec![0; CLEAN.len()];
        for word in SLOP { for t in TMPLS {
            sf.push(featurize(&t.replacen("{}", word, 1))); sl.push(1);
        }}
        let (sw, sb, sa) = train(gpu, "slop_detector", &sf, &sl, 2, 50, 0.01);
        save("slop_detector", &sw, &sb, &["clean", "slop"], sa);
        let (cls, conf) = predict(&sw, &sb, "We need to leverage the synergy of our paradigm.", 2, &["clean", "slop"]);
        eprintln!("  sample: '{cls}' ({:.1}%)\n", conf * 100.0);

        // ── code_vs_english ───────────────────────────────
        eprintln!("--- code_vs_english ---");
        const CODE: &[&str] = &[
            "fn main() { println!(\"hello\"); }", "let mut v: Vec<i32> = Vec::new();",
            "def train(model, data):", "import numpy as np",
            "function handleClick(event) {}", "const express = require('express');",
            "func main() { fmt.Println(\"hello\") }", "#!/bin/bash\nset -e",
            "SELECT * FROM users WHERE id = ?", "docker run -d -p 8080:8080 myapp",
        ];
        const ENGLISH: &[&str] = &[
            "The project uses a single binary architecture.",
            "All tests must pass before merging a pull request.",
            "The documentation should be updated when adding new features.",
            "Worker nodes communicate over SSH with host certificate authentication.",
            "Binary size was reduced from 54 MB to 27 MB with LTO.",
            "The agent loop continues until the model stops calling tools.",
            "Each commit message should explain why the change was made.",
            "The config file lives in the home directory under a hidden folder.",
            "Training data comes from tournament results and execution traces.",
            "The starter nanobyte ships embedded in the binary for zero setup.",
        ];
        let mut cf: Vec<Vec<f32>> = ENGLISH.iter().map(|s| featurize(s)).collect();
        let mut cl: Vec<usize> = vec![0; ENGLISH.len()];
        for s in CODE { cf.push(featurize(s)); cl.push(1); }
        let (cw, cb, ca) = train(gpu, "code_vs_english", &cf, &cl, 2, 50, 0.01);
        save("code_vs_english", &cw, &cb, &["english", "code"], ca);
        let (cls, conf) = predict(&cw, &cb, "fn main() { println!(\"hello\"); }", 2, &["english", "code"]);
        eprintln!("  sample: '{cls}' ({:.1}%)\n", conf * 100.0);

        // ── lang_detector ─────────────────────────────────
        eprintln!("--- lang_detector ---");
        const RUST: &[&str] = &[
            "fn main() { println!(\"hello\"); }", "let mut v: Vec<i32> = Vec::new();",
            "impl Display for Error {}", "pub async fn serve(port: u16) -> Result<()> {}",
            "#[derive(Debug, Clone)]", "match result { Ok(v) => v, Err(e) => Err(e) }",
        ];
        const PYTHON: &[&str] = &[
            "def train(model, data, epochs=10):", "import numpy as np",
            "from transformers import AutoTokenizer", "loss = criterion(output, target)",
            "optimizer.zero_grad()", "for i, (x, y) in enumerate(dataloader):",
        ];
        const JS: &[&str] = &[
            "const express = require('express');", "function handleClick(event) {}",
            "const [state, setState] = useState(null);", "export default function App() {}",
            "fetch('/api').then(res => res.json());", "npm install express cors",
        ];
        const GO: &[&str] = &[
            "func main() { fmt.Println(\"hello\") }", "package main; import \"fmt\"",
            "if err != nil { return fmt.Errorf(\"failed: %w\", err) }",
            "ch := make(chan string, 10)", "go func() { result <- process(data) }()",
            "type Config struct { Port int }",
        ];
        const SHELL: &[&str] = &[
            "#!/bin/bash\nset -euo pipefail", "for f in *.rs; do wc -l \"$f\"; done",
            "export PATH=\"$HOME/.cargo/bin:$PATH\"", "ssh lf 'cargo build --release'",
            "rsync -avz --exclude target src/ remote:src/", "kill $(pgrep -f 'kova serve')",
        ];
        let langs: &[(&[&str], usize, &str)] = &[
            (RUST, 0, "rust"), (PYTHON, 1, "python"), (JS, 2, "javascript"),
            (GO, 3, "go"), (SHELL, 4, "shell"),
        ];
        let mut lf: Vec<Vec<f32>> = Vec::new();
        let mut ll: Vec<usize> = Vec::new();
        for (samples, label, _) in langs {
            for s in *samples { lf.push(featurize(s)); ll.push(*label); }
        }
        let names: Vec<&str> = langs.iter().map(|(_, _, n)| *n).collect();
        let (lw, lb, la) = train(gpu, "lang_detector", &lf, &ll, 5, 80, 0.005);
        save("lang_detector", &lw, &lb, &names, la);
        let (cls, conf) = predict(&lw, &lb, "fn main() { println!(\"hello\"); }", 5, &names);
        eprintln!("  sample: '{cls}' ({:.1}%)\n", conf * 100.0);

        eprintln!("=== done. models saved to models/ ===");
        eprintln!("slop={:.1}%  code={:.1}%  lang={:.1}%", sa*100.0, ca*100.0, la*100.0);
        Ok(())
    }
}
