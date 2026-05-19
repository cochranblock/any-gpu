// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// Nanobyte — DDPM training on 32×32 RGBA pixel art sprites.
// NanoUNet: CHANNELS=[32,64,64], TIME_DIM=64, T=1000 steps, ~1.09M params.
// Mirrors pixel-forge's TinyUNet (Cinder) architecture in any-gpu native API.
//
// Usage:
//   cargo run --release --example nanobyte -- --data /path/to/pixel-forge/data_v3_32
//   cargo run --release --example nanobyte -- --data ... --steps 20000 --save nb.bin
//   cargo run --release --example nanobyte -- --data ... --resume nb.bin --sample out.png

use anyhow::{bail, Result};
use any_gpu::{
    autograd::{t503, t506},
    optim::t507,
    t500,
    train::{f734, t550},
};
use clap::Parser;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::time::Instant;

// ── constants ─────────────────────────────────────────────────────────────────

const CH: [u32; 3] = [32, 64, 64];  // U-Net channel widths
const TD: usize = 64;                // time embedding dim
const DDPM_T: u32 = 1000;           // diffusion timesteps
const SP: u32 = 32;                  // spatial size
const IC: u32 = 4;                   // input channels (RGBA)

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "nanobyte", about = "Train NanoUNet on 32×32 pixel art")]
struct Cli {
    #[arg(long)]                       data: String,
    #[arg(long, default_value = "10000")] steps: u32,
    #[arg(long, default_value = "0.0001")] lr: f32,
    #[arg(long, default_value = "4")]  batch: u32,
    #[arg(long)]                       save: Option<String>,
    #[arg(long)]                       resume: Option<String>,
    #[arg(long)]                       sample: Option<String>,
    #[arg(long, default_value = "200")] log_every: u32,
}

// ── DDPM noise schedule ───────────────────────────────────────────────────────

struct Sched { alpha_bar: Vec<f32>, sigma: Vec<f32> }

fn make_sched() -> Sched {
    let (b0, b1) = (1e-4f32, 0.02f32);
    let mut acc = 1.0f32;
    let mut alpha_bar = Vec::with_capacity(DDPM_T as usize);
    for t in 0..DDPM_T as usize {
        let beta = b0 + (b1 - b0) * t as f32 / (DDPM_T as f32 - 1.0);
        acc *= 1.0 - beta;
        alpha_bar.push(acc);
    }
    let sigma = alpha_bar.iter().map(|&ab| (1.0 - ab).sqrt()).collect();
    Sched { alpha_bar, sigma }
}

fn q_sample(x0: &[f32], t: u32, eps: &[f32], s: &Sched) -> Vec<f32> {
    let ab = s.alpha_bar[t as usize];
    let sig = s.sigma[t as usize];
    x0.iter().zip(eps).map(|(&x, &e)| ab.sqrt() * x + sig * e).collect()
}

// ── sinusoidal timestep embedding ─────────────────────────────────────────────

fn timestep_emb(ts: &[u32]) -> Vec<f32> {
    let half = TD / 2;
    let mut out = vec![0.0f32; ts.len() * TD];
    for (b, &t) in ts.iter().enumerate() {
        for i in 0..half {
            let freq = t as f32 / (10000.0f32).powf(2.0 * i as f32 / TD as f32);
            out[b * TD + i] = freq.cos();
            out[b * TD + half + i] = freq.sin();
        }
    }
    out
}

// ── data loading ─────────────────────────────────────────────────────────────

fn load_sprites(data_path: &str) -> Result<Vec<Vec<f32>>> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() { walk(&p, out); }
            else if p.extension().and_then(|s| s.to_str()) == Some("png") { out.push(p); }
        }
    }
    let mut paths = Vec::new();
    walk(std::path::Path::new(data_path), &mut paths);
    let mut sprites = Vec::new();
    for path in &paths {
        let img = match image::open(path) { Ok(i) => i.into_rgba8(), Err(_) => continue };
        if img.width() != SP || img.height() != SP { continue; }
        let raw = img.as_raw();
        let n = (SP * SP) as usize;
        let mut chw = vec![0.0f32; (IC as usize) * n];
        for i in 0..n {
            for c in 0..4usize { chw[c * n + i] = raw[4 * i + c] as f32 / 127.5 - 1.0; }
        }
        sprites.push(chw);
    }
    if sprites.is_empty() { bail!("no 32×32 sprites found in {data_path}"); }
    eprintln!("loaded {} sprites", sprites.len());
    Ok(sprites)
}

// ── standard normal sample ────────────────────────────────────────────────────

fn randn(rng: &mut impl Rng) -> f32 {
    let u1 = rng.gen_range(0.0f32..1.0f32).max(1e-7);
    let u2 = rng.gen_range(0.0f32..1.0f32);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ── param layout ─────────────────────────────────────────────────────────────

fn group_count(ch: u32) -> u32 {
    if ch % 8 == 0 { 8 } else if ch % 4 == 0 { 4 } else { 1 }
}

fn xavier(n: usize, fan_in: usize, seed: &mut u64) -> Vec<f32> {
    let bound = (6.0 / fan_in.max(1) as f64).sqrt() as f32;
    (0..n).map(|_| {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * bound
    }).collect()
}

#[derive(Clone)]
struct RbIdx {
    skip_w: Option<usize>, skip_b: Option<usize>,
    gn1g: usize, gn1b: usize,
    c1w: usize,  c1b: usize,
    tpw:  usize, tpb:  usize,
    gn2g: usize, gn2b: usize,
    c2w:  usize, c2b:  usize,
    in_ch: u32, out_ch: u32,
}

struct Layout {
    tl:  [(usize, usize); 2],  // time_lin0, time_lin1 (w, b)
    ci:  (usize, usize),       // conv_in (w, b)
    enc: Vec<[RbIdx; 2]>,
    dn:  Vec<(usize, usize)>,  // downsample convs
    mid: [RbIdx; 2],
    dec: Vec<[RbIdx; 2]>,
    up:  Vec<(usize, usize)>,  // upsample convs
    gno: (usize, usize),       // gn_out
    co:  (usize, usize),       // conv_out (w, b)
}

fn make_rb(ps: &mut Vec<Vec<f32>>, seed: &mut u64, in_ch: u32, out_ch: u32) -> RbIdx {
    macro_rules! pp { ($e:expr) => {{ let i = ps.len(); ps.push($e); i }} }
    let (skip_w, skip_b) = if in_ch != out_ch {
        (Some(pp!(xavier((out_ch * in_ch) as usize, in_ch as usize, seed))),
         Some(pp!(vec![0.0f32; out_ch as usize])))
    } else { (None, None) };
    let gn1g = pp!(vec![1.0f32; in_ch as usize]);
    let gn1b = pp!(vec![0.0f32; in_ch as usize]);
    let c1w  = pp!(xavier((out_ch * in_ch * 9) as usize, (in_ch * 9) as usize, seed));
    let c1b  = pp!(vec![0.0f32; out_ch as usize]);
    let tpw  = pp!(xavier(TD * out_ch as usize, TD, seed));
    let tpb  = pp!(vec![0.0f32; out_ch as usize]);
    let gn2g = pp!(vec![1.0f32; out_ch as usize]);
    let gn2b = pp!(vec![0.0f32; out_ch as usize]);
    let c2w  = pp!(xavier((out_ch * out_ch * 9) as usize, (out_ch * 9) as usize, seed));
    let c2b  = pp!(vec![0.0f32; out_ch as usize]);
    RbIdx { skip_w, skip_b, gn1g, gn1b, c1w, c1b, tpw, tpb, gn2g, gn2b, c2w, c2b, in_ch, out_ch }
}

fn init_params() -> (Vec<Vec<f32>>, Layout) {
    let mut ps: Vec<Vec<f32>> = Vec::new();
    let mut seed: u64 = 0xCAFE_BABE_DEAD_BEEF;
    macro_rules! pp { ($e:expr) => {{ let i = ps.len(); ps.push($e); i }} }

    let tl0w = pp!(xavier(TD * TD, TD, &mut seed)); let tl0b = pp!(vec![0.0f32; TD]);
    let tl1w = pp!(xavier(TD * TD, TD, &mut seed)); let tl1b = pp!(vec![0.0f32; TD]);
    let tl = [(tl0w, tl0b), (tl1w, tl1b)];

    let ciw = pp!(xavier((CH[0] * IC * 9) as usize, (IC * 9) as usize, &mut seed));
    let cib = pp!(vec![0.0f32; CH[0] as usize]);
    let ci = (ciw, cib);

    // Encoder: 3 levels × 2 ResBlocks; downsamples between levels 0→1 and 1→2
    let mut enc = Vec::new();
    let mut dn  = Vec::new();
    let mut ch_in = CH[0];
    for (i, &ch_out) in CH.iter().enumerate() {
        let r1 = make_rb(&mut ps, &mut seed, ch_in, ch_out);
        let r2 = make_rb(&mut ps, &mut seed, ch_out, ch_out);
        enc.push([r1, r2]);
        if i < CH.len() - 1 {
            let dnw = pp!(xavier((ch_out * ch_out * 9) as usize, (ch_out * 9) as usize, &mut seed));
            let dnb = pp!(vec![0.0f32; ch_out as usize]);
            dn.push((dnw, dnb));
        }
        ch_in = ch_out;
    }

    let mid_ch = *CH.last().unwrap();
    let mid = [
        make_rb(&mut ps, &mut seed, mid_ch, mid_ch),
        make_rb(&mut ps, &mut seed, mid_ch, mid_ch),
    ];

    // Decoder: reversed levels; upsample convs for levels 1 and 2
    let rev: Vec<u32> = CH.iter().copied().rev().collect();
    let mut dec = Vec::new();
    let mut up  = Vec::new();
    for (i, &ch_out) in rev.iter().enumerate() {
        if i > 0 {
            let prev_ch = rev[i - 1];
            let upw = pp!(xavier((prev_ch * prev_ch * 9) as usize, (prev_ch * 9) as usize, &mut seed));
            let upb = pp!(vec![0.0f32; prev_ch as usize]);
            up.push((upw, upb));
        }
        let concat_in = if i == 0 { mid_ch + ch_out } else { rev[i - 1] + ch_out };
        let r1 = make_rb(&mut ps, &mut seed, concat_in, ch_out);
        let r2 = make_rb(&mut ps, &mut seed, ch_out, ch_out);
        dec.push([r1, r2]);
    }

    let gnow = pp!(vec![1.0f32; CH[0] as usize]);
    let gnob = pp!(vec![0.0f32; CH[0] as usize]);
    let gno = (gnow, gnob);

    let cow = pp!(xavier((IC * CH[0] * 9) as usize, (CH[0] * 9) as usize, &mut seed));
    let cob = pp!(vec![0.0f32; IC as usize]);
    let co = (cow, cob);

    let total: usize = ps.iter().map(|v| v.len()).sum();
    eprintln!("NanoUNet: {} params ({} tensors)", total, ps.len());

    let layout = Layout { tl, ci, enc, dn, mid, dec, up, gno, co };
    (ps, layout)
}

// ── ResBlock forward ──────────────────────────────────────────────────────────

fn resblock_fwd(
    tape: &mut t506, ids: &[t503], rb: &RbIdx,
    x: t503, t_emb: t503, batch: u32, ih: u32, iw: u32,
) -> Result<t503> {
    let in_ch  = rb.in_ch;
    let out_ch = rb.out_ch;
    let gcin  = group_count(in_ch);
    let gcout = group_count(out_ch);

    // Skip branch (1×1 conv when channels change)
    let skip = if let (Some(sw), Some(sb)) = (rb.skip_w, rb.skip_b) {
        tape.f696(x, ids[sw], Some(ids[sb]), batch, in_ch, ih, iw, out_ch, 1, 1, (1,1), (0,0), (1,1), 1)?
    } else { x };

    // Main branch: GN → swish → conv1 → time_inject → GN → swish → conv2
    let h = tape.f698(x, ids[rb.gn1g], ids[rb.gn1b], batch, in_ch, ih * iw, gcin, 1e-5)?;
    let h = tape.f692(h)?;  // swish
    let h = tape.f696(h, ids[rb.c1w], Some(ids[rb.c1b]), batch, in_ch, ih, iw, out_ch, 3, 3, (1,1), (1,1), (1,1), 1)?;

    // Time conditioning: linear(t_emb) → swish → add_broadcast per channel
    let tp = tape.f694(t_emb, ids[rb.tpw], batch, out_ch as u32, TD as u32)?;
    let tp = tape.f701(tp, ids[rb.tpb], batch, out_ch)?;
    let tp = tape.f692(tp)?;
    let h = tape.f700(h, tp, batch * out_ch, ih * iw)?;

    let h = tape.f698(h, ids[rb.gn2g], ids[rb.gn2b], batch, out_ch, ih * iw, gcout, 1e-5)?;
    let h = tape.f692(h)?;
    let h = tape.f696(h, ids[rb.c2w], Some(ids[rb.c2b]), batch, out_ch, ih, iw, out_ch, 3, 3, (1,1), (1,1), (1,1), 1)?;

    tape.f686(h, skip)
}

// ── NanoUNet forward pass ─────────────────────────────────────────────────────

fn unet_fwd(
    tape: &mut t506, ids: &[t503], lo: &Layout,
    x: t503, t_sin: t503, batch: u32,
) -> Result<t503> {
    // Time MLP: sin_emb → linear → swish → linear → t_emb
    let t = tape.f694(t_sin, ids[lo.tl[0].0], batch, TD as u32, TD as u32)?;
    let t = tape.f701(t, ids[lo.tl[0].1], batch, TD as u32)?;
    let t = tape.f692(t)?;
    let t = tape.f694(t, ids[lo.tl[1].0], batch, TD as u32, TD as u32)?;
    let t_emb = tape.f701(t, ids[lo.tl[1].1], batch, TD as u32)?;

    // conv_in: [batch, IC, 32, 32] → [batch, CH[0], 32, 32]
    let mut h = tape.f696(x, ids[lo.ci.0], Some(ids[lo.ci.1]),
        batch, IC, SP, SP, CH[0], 3, 3, (1,1), (1,1), (1,1), 1)?;

    // Encoder: 3 levels, 2 ResBlocks each, downsample between
    let spatials = [SP, SP / 2, SP / 4];   // spatial size at each level
    let mut skips: Vec<(t503, u32, u32)> = Vec::new(); // (id, ch, spatial)
    let mut cur_ch = CH[0];

    for (lv, rb_pair) in lo.enc.iter().enumerate() {
        let sp = spatials[lv];
        h = resblock_fwd(tape, ids, &rb_pair[0], h, t_emb, batch, sp, sp)?;
        h = resblock_fwd(tape, ids, &rb_pair[1], h, t_emb, batch, sp, sp)?;
        cur_ch = CH[lv];
        skips.push((h, cur_ch, sp));
        if lv < lo.dn.len() {
            let (dnw, dnb) = lo.dn[lv];
            h = tape.f696(h, ids[dnw], Some(ids[dnb]),
                batch, cur_ch, sp, sp, cur_ch, 3, 3, (2,2), (1,1), (1,1), 1)?;
        }
    }

    // Mid: 2 ResBlocks at 8×8
    let mid_sp = spatials[2];
    h = resblock_fwd(tape, ids, &lo.mid[0], h, t_emb, batch, mid_sp, mid_sp)?;
    h = resblock_fwd(tape, ids, &lo.mid[1], h, t_emb, batch, mid_sp, mid_sp)?;

    // Decoder: reverse levels with upsample + skip concat
    let mut dec_sp = mid_sp;
    for (i, rb_pair) in lo.dec.iter().enumerate() {
        if i > 0 {
            let (upw, upb) = lo.up[i - 1];
            h = tape.f699(h, batch, cur_ch, dec_sp, dec_sp, 2, 2)?;
            dec_sp *= 2;
            h = tape.f696(h, ids[upw], Some(ids[upb]),
                batch, cur_ch, dec_sp, dec_sp, cur_ch, 3, 3, (1,1), (1,1), (1,1), 1)?;
        }
        let (skip_id, skip_ch, _) = skips[skips.len() - 1 - i];
        let hw = dec_sp * dec_sp;
        h = tape.f697(h, skip_id, batch, cur_ch * hw, skip_ch * hw)?;
        let concat_ch = cur_ch + skip_ch;
        h = resblock_fwd(tape, ids, &rb_pair[0], h, t_emb, batch, dec_sp, dec_sp)?;
        h = resblock_fwd(tape, ids, &rb_pair[1], h, t_emb, batch, dec_sp, dec_sp)?;
        cur_ch = rb_pair[1].out_ch;
        let _ = concat_ch;
    }

    // Output: GN → swish → conv_out (IC=4 channels)
    let gc_out = group_count(cur_ch);
    let hw = dec_sp * dec_sp;
    let h = tape.f698(h, ids[lo.gno.0], ids[lo.gno.1], batch, cur_ch, hw, gc_out, 1e-5)?;
    let h = tape.f692(h)?;
    tape.f696(h, ids[lo.co.0], Some(ids[lo.co.1]),
        batch, cur_ch, dec_sp, dec_sp, IC, 3, 3, (1,1), (1,1), (1,1), 1)
}

// ── checkpoint save / load ────────────────────────────────────────────────────

fn save_model(params: &[Vec<f32>], path: &str) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"NBYT")?;
    f.write_all(&(params.len() as u32).to_le_bytes())?;
    for p in params {
        f.write_all(&(p.len() as u32).to_le_bytes())?;
        for &v in p { f.write_all(&v.to_le_bytes())?; }
    }
    eprintln!("saved → {path}");
    Ok(())
}

fn load_model(path: &str) -> Result<Vec<Vec<f32>>> {
    let data = std::fs::read(path)?;
    let mut pos = 0;
    if &data[pos..pos+4] != b"NBYT" { bail!("not a nanobyte checkpoint: {path}"); }
    pos += 4;
    let num = u32::from_le_bytes(data[pos..pos+4].try_into()?) as usize; pos += 4;
    let mut params = Vec::with_capacity(num);
    for _ in 0..num {
        let len = u32::from_le_bytes(data[pos..pos+4].try_into()?) as usize; pos += 4;
        let mut p = Vec::with_capacity(len);
        for _ in 0..len {
            p.push(f32::from_le_bytes(data[pos..pos+4].try_into()?)); pos += 4;
        }
        params.push(p);
    }
    eprintln!("resumed from {path} ({num} tensors)");
    Ok(params)
}

// ── denoising sample (one sprite) ────────────────────────────────────────────

fn sample_one(
    gpu: &t500, gp: &t550, lo: &Layout, sched: &Sched, path: &str,
) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(42);
    let n = (IC * SP * SP) as usize;
    let mut x: Vec<f32> = (0..n).map(|_| randn(&mut rng)).collect();

    // Reverse DDPM process: T→1
    for step in (0..DDPM_T).rev() {
        let t_emb = timestep_emb(&[step]);
        let pred = {
            let mut tape = t506::f680(gpu);
            let x_id = tape.f681(&x);
            let t_id = tape.f681(&t_emb);
            // Inject resident params
            let pids: Vec<t503> = gp.params.iter().map(|p| tape.f681r(p)).collect();
            let out = unet_fwd(&mut tape, &pids, lo, x_id, t_id, 1)?;
            tape.f682(out)?
        };

        // DDPM reverse step
        let ab  = sched.alpha_bar[step as usize];
        let ab1 = if step > 0 { sched.alpha_bar[(step - 1) as usize] } else { 1.0 };
        let sig = sched.sigma[step as usize];
        let alpha = ab / ab1;
        let noise: Vec<f32> = if step > 0 { (0..n).map(|_| randn(&mut rng)).collect() } else { vec![0.0; n] };
        let sigma_q = ((1.0 - ab1) / (1.0 - ab) * (1.0 - alpha)).sqrt();

        for i in 0..n {
            let x_pred = (x[i] - sig * pred[i]) / ab.sqrt();
            x[i] = alpha.sqrt() * x_pred + sigma_q * noise[i];
        }
    }

    // Clamp to [-1,1] then save as RGBA PNG
    let hw = (SP * SP) as usize;
    let mut pixels = Vec::with_capacity(hw * 4);
    for i in 0..hw {
        for c in 0..4usize {
            let v = x[c * hw + i].clamp(-1.0, 1.0);
            pixels.push(((v + 1.0) * 127.5) as u8);
        }
    }
    image::RgbaImage::from_raw(SP, SP, pixels)
        .ok_or_else(|| anyhow::anyhow!("invalid sample buffer"))?
        .save(path)?;
    eprintln!("sample → {path}");
    Ok(())
}

// ── training ─────────────────────────────────────────────────────────────────

fn train(gpu: &t500, cli: &Cli) -> Result<()> {
    let sched  = make_sched();
    let data   = load_sprites(&cli.data)?;
    let sprite_n = (IC * SP * SP) as usize;

    // Init or resume params
    let (init_ps, layout) = init_params();
    let flat_ps = if let Some(path) = &cli.resume {
        let loaded = load_model(path)?;
        if loaded.len() != init_ps.len() {
            bail!("checkpoint has {} tensors, model expects {}", loaded.len(), init_ps.len());
        }
        loaded
    } else {
        init_ps
    };

    let refs: Vec<&[f32]> = flat_ps.iter().map(|v| v.as_slice()).collect();
    let mut gparams = t550::f731(gpu, &refs);
    let mut opt = t507::f720(cli.lr);
    opt.weight_decay = 1e-5;

    let mut rng = StdRng::seed_from_u64(1337);
    let t0 = Instant::now();
    let batch = cli.batch;

    for step in 0..cli.steps {
        // Random mini-batch
        let mut x0 = vec![0.0f32; batch as usize * sprite_n];
        for b in 0..batch as usize {
            let idx = rng.gen_range(0..data.len());
            x0[b * sprite_n..(b + 1) * sprite_n].copy_from_slice(&data[idx]);
        }

        // Random timesteps and noise
        let ts: Vec<u32> = (0..batch).map(|_| rng.gen_range(0..DDPM_T)).collect();
        let eps: Vec<f32> = (0..batch as usize * sprite_n).map(|_| randn(&mut rng)).collect();

        // Noisy images x_t (batch concat)
        let mut xt = vec![0.0f32; batch as usize * sprite_n];
        for b in 0..batch as usize {
            let x0b = &x0[b * sprite_n..(b + 1) * sprite_n];
            let epsb = &eps[b * sprite_n..(b + 1) * sprite_n];
            let xtb  = q_sample(x0b, ts[b], epsb, &sched);
            xt[b * sprite_n..(b + 1) * sprite_n].copy_from_slice(&xtb);
        }

        let t_emb = timestep_emb(&ts);

        let result = f734(gpu, &mut opt, &mut gparams, step, |tape, param_ids| {
            let x_id   = tape.f681(&xt);
            let t_id   = tape.f681(&t_emb);
            let eps_id = tape.f681(&eps);
            let pred   = unet_fwd(tape, param_ids, &layout, x_id, t_id, batch)?;
            tape.f695(pred, eps_id)
        })?;

        if step % cli.log_every == 0 || step == cli.steps - 1 {
            let elapsed = t0.elapsed().as_secs_f64();
            let sps = (step + 1) as f64 / elapsed;
            eprintln!("step {:>6}/{}: loss={:.4}  {:.1} steps/s", step, cli.steps, result.loss, sps);
        }
    }

    // Checkpoint
    if let Some(path) = &cli.save {
        let updated = gparams.f732(gpu)?;
        save_model(&updated, path)?;
    }

    // Sample
    if let Some(path) = &cli.sample {
        sample_one(gpu, &gparams, &layout, &sched, path)?;
    }

    Ok(())
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    let gpu = t500::f500()?;
    train(&gpu, &cli)
}
