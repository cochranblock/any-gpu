// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// any-gpu-serve — GPU inference server with continuous batching.
// Up to max_batch requests decode in parallel; prefill is single-request sequential.
//
// Active slots are kept dense (indices 0..n-1 = pool slots 0..n-1).
// On completion, the last slot migrates into the freed slot via f807.
//
// Usage: any-gpu-serve --model model.safetensors --config config.json \
//                      --tokenizer tokenizer.json [--port 8080]
//
// POST /generate  body: {"prompt":"...","max_new_tokens":128}
//                reply: {"output":"..."}
// GET  /health    reply: {"status":"ok","device":"...","active_slots":N}

use anyhow::Result;
use clap::Parser;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;
use std::collections::VecDeque;

use any_gpu::{t500, t547, t548, t539, DEFAULT_STAGE_BYTES, t538, t544, t556};

#[derive(Parser)]
#[command(name = "any-gpu-serve", about = "GPU inference server (continuous batching)")]
struct Cli {
    #[arg(long)] model:     String,
    #[arg(long)] config:    String,
    #[arg(long)] tokenizer: String,
    #[arg(long, default_value = "8080")] port: u16,
    #[arg(long)] max_seq:   Option<usize>,
}

struct PendingReq {
    stream:     TcpStream,
    token_ids:  Vec<u32>,
    max_new:    usize,
}

// Active is dense: active[i].slot.slot == i (pool slot == vec index).
struct InFlight {
    stream: TcpStream,
    slot:   t556,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    eprintln!("[any-gpu-serve] initialising GPU…");
    let dev = t500::f500()?;
    eprintln!("[any-gpu-serve] device: {} ({})", dev.s502, dev.s503);

    eprintln!("[any-gpu-serve] loading config {}…", cli.config);
    let cfg_str = std::fs::read_to_string(&cli.config)?;
    let mut cfg = t547::f782(&cfg_str)?;
    if let Some(ms) = cli.max_seq { cfg.max_seq = ms; }
    let max_batch = cfg.max_batch.max(1);

    eprintln!("[any-gpu-serve] loading tokenizer {}…", cli.tokenizer);
    let tok = t544::f775(&cli.tokenizer)?;
    let eos = tok.f779();

    eprintln!("[any-gpu-serve] loading model {}…", cli.model);
    let model_bytes = std::fs::read(&cli.model)?;
    let st = t538::f761(&model_bytes)?;
    let pager = t539::f768(&dev, DEFAULT_STAGE_BYTES);
    let mut lm = t548::f783(&dev, &st, &pager, cfg)?;

    eprintln!("[any-gpu-serve] max_batch={max_batch}, ready — listening on 0.0.0.0:{}", cli.port);
    let listener = TcpListener::bind(format!("0.0.0.0:{}", cli.port))?;
    listener.set_nonblocking(true)?;

    let mut pending: VecDeque<PendingReq> = VecDeque::new();
    // Dense vec: active[i].slot.slot == i always.
    let mut active: Vec<InFlight> = Vec::with_capacity(max_batch);
    let dev_name = format!("{} ({})", dev.s502, dev.s503);

    loop {
        // 1. Accept new connections (non-blocking).
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
                    match parse_request(&stream, &tok) {
                        Ok(Some(pr)) => pending.push_back(pr),
                        Ok(None)     => send_health(&stream, &dev_name, active.len()),
                        Err(e)       => eprintln!("[serve] parse error: {e}"),
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => eprintln!("[serve] accept error: {e}"),
            }
        }

        // 2. Prefill pending requests into free slots.
        while !pending.is_empty() && active.len() < max_batch {
            let pr = pending.pop_front().unwrap();
            let pool_slot = active.len(); // dense: next slot = current length
            eprintln!("[serve] prefilling slot {pool_slot}, {} tokens", pr.token_ids.len());
            match lm.f788b(&dev, pool_slot, &pr.token_ids) {
                Err(e) => {
                    eprintln!("[serve] prefill error: {e}");
                    let _ = write_err(&pr.stream, &e.to_string());
                }
                Ok(logits) => {
                    let next = greedy_sample(&dev, &logits, lm.cfg.vocab_size as u32)?;
                    let mut slot_state = t556 {
                        slot: pool_slot,
                        generated: vec![next],
                        next_token: next,
                        tokens_left: pr.max_new.saturating_sub(1),
                    };
                    if Some(next) == eos || slot_state.tokens_left == 0 {
                        if Some(next) == eos { slot_state.generated.pop(); }
                        finish_slot(&tok, &pr.stream, &slot_state.generated)?;
                        lm.batch_kv.iter_mut().for_each(|bkv| bkv.f802(pool_slot));
                        // Slot was never pushed to active, so nothing to compact.
                    } else {
                        active.push(InFlight { stream: pr.stream, slot: slot_state });
                    }
                }
            }
        }

        // 3. If there are active slots, run one batch decode step.
        let n_active = active.len();
        if n_active > 0 {
            let next_tokens: Vec<u32> = active.iter().map(|inf| inf.slot.next_token).collect();
            match lm.f789b(&dev, &next_tokens) {
                Err(e) => eprintln!("[serve] decode error: {e}"),
                Ok(logits_buf) => {
                    let logits_cpu = dev.f504(&logits_buf)?;
                    let vocab = lm.cfg.vocab_size;

                    // First pass: update state, collect completions.
                    let mut completed: Vec<usize> = Vec::new();
                    for i in 0..active.len() {
                        let next = argmax(&logits_cpu[i * vocab..(i + 1) * vocab]);
                        let inf = &mut active[i];
                        inf.slot.generated.push(next);
                        if Some(next) == eos || inf.slot.tokens_left == 0 {
                            completed.push(i);
                        } else {
                            inf.slot.next_token = next;
                            inf.slot.tokens_left -= 1;
                        }
                    }

                    // Second pass: retire completed slots in REVERSE order so
                    // swap_remove indices remain valid as we shrink the vec.
                    for &i in completed.iter().rev() {
                        let toks = {
                            let mut t = active[i].slot.generated.clone();
                            if Some(*t.last().unwrap()) == eos { t.pop(); }
                            t
                        };
                        let _ = finish_slot(&tok, &active[i].stream, &toks);
                        eprintln!("[serve] retired slot {i}");

                        let last = active.len() - 1;
                        if i < last {
                            // Migrate last slot's KV into slot i, then shrink.
                            lm.batch_kv.iter_mut().for_each(|bkv| bkv.f807(&dev, last, i));
                            lm.batch_kv.iter_mut().for_each(|bkv| bkv.f802(last));
                            active.swap_remove(i);
                            active[i].slot.slot = i;
                        } else {
                            lm.batch_kv.iter_mut().for_each(|bkv| bkv.f802(i));
                            active.pop();
                        }
                    }
                }
            }
        }

        // 4. If nothing is happening, sleep briefly to avoid busy-polling.
        if pending.is_empty() && active.is_empty() {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn greedy_sample(dev: &t500, logits: &any_gpu::t501, vocab_size: u32) -> Result<u32> {
    let idx_buf = dev.f671(logits, 1, vocab_size)?;
    let v = dev.f504(&idx_buf)?;
    Ok(v[0] as u32)
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn finish_slot(tok: &t544, stream: &TcpStream, generated: &[u32]) -> Result<()> {
    let text = tok.f777(generated)?;
    eprintln!("[serve] done: {} tokens", generated.len());
    let body = serde_json::json!({"output": text}).to_string();
    write_response(stream, 200, "OK", &body)
}

fn write_err(stream: &TcpStream, msg: &str) -> Result<()> {
    let body = serde_json::json!({"error": msg}).to_string();
    write_response(stream, 500, "Internal Server Error", &body)
}

fn send_health(stream: &TcpStream, dev_name: &str, active: usize) {
    let body = serde_json::json!({"status":"ok","device":dev_name,"active_slots":active}).to_string();
    let _ = write_response(stream, 200, "OK", &body);
}

/// Returns Ok(Some(PendingReq)) for POST /generate, Ok(None) for GET /health, Err otherwise.
fn parse_request(stream: &TcpStream, tok: &t544) -> Result<Option<PendingReq>> {
    let mut reader = BufReader::new(stream);
    let mut req_line = String::new();
    reader.read_line(&mut req_line)?;
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    anyhow::ensure!(parts.len() >= 2, "bad request line");
    let method = parts[0];
    let path   = parts[1];

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() { break; }
        if trimmed.to_lowercase().starts_with("content-length:") {
            content_length = trimmed.split(':').nth(1)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
        }
    }

    match (method, path) {
        ("GET", "/health") => Ok(None),
        ("POST", "/generate") => {
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body)?;
            let req: serde_json::Value = serde_json::from_slice(&body)?;
            let prompt  = req["prompt"].as_str().unwrap_or("").to_string();
            let max_new = req["max_new_tokens"].as_u64().unwrap_or(128) as usize;
            eprintln!("[serve] /generate prompt={:?} max_new={max_new}",
                      &prompt[..prompt.len().min(60)]);
            let token_ids = tok.f776(&prompt, true)?;
            Ok(Some(PendingReq { stream: stream.try_clone()?, token_ids, max_new }))
        }
        _ => anyhow::bail!("unknown endpoint: {method} {path}"),
    }
}

fn write_response(w: &TcpStream, status: u16, reason: &str, body: &str) -> Result<()> {
    let resp = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    );
    let mut w = w.try_clone()?;
    w.write_all(resp.as_bytes())?;
    Ok(())
}
