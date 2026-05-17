// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// any-gpu-serve — GPU inference HTTP server.
// Usage: any-gpu-serve --model model.safetensors --config config.json \
//                      --tokenizer tokenizer.json [--port 8080]
//
// POST /generate  body: {"prompt":"...","max_new_tokens":128}
//                reply: {"output":"..."}
// GET  /health    reply: {"status":"ok","device":"..."}

use anyhow::Result;
use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::Mutex;

use any_gpu::{t500, t547, t548, t539, DEFAULT_STAGE_BYTES, t538, t544};

#[derive(Parser)]
#[command(name = "any-gpu-serve", about = "GPU inference server for any-gpu models")]
struct Cli {
    /// Path to model.safetensors (NanoSign-verified or unsigned)
    #[arg(long)]
    model: String,

    /// Path to config.json (LmConfig JSON)
    #[arg(long)]
    config: String,

    /// Path to tokenizer.json (HuggingFace format)
    #[arg(long)]
    tokenizer: String,

    /// TCP port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Maximum sequence length override (defaults to config max_seq)
    #[arg(long)]
    max_seq: Option<usize>,
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

    eprintln!("[any-gpu-serve] loading tokenizer {}…", cli.tokenizer);
    let tok = t544::f775(&cli.tokenizer)?;

    eprintln!("[any-gpu-serve] loading model {}…", cli.model);
    let model_bytes = std::fs::read(&cli.model)?;
    let st = t538::f761(&model_bytes)?;
    let pager = t539::f768(&dev, DEFAULT_STAGE_BYTES);
    let mut lm = t548::f783(&dev, &st, &pager, cfg)?;

    eprintln!("[any-gpu-serve] ready — listening on 0.0.0.0:{}", cli.port);
    let listener = TcpListener::bind(format!("0.0.0.0:{}", cli.port))?;
    let lm = Mutex::new(lm);
    let dev_name = format!("{} ({})", dev.s502, dev.s503);

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => { eprintln!("[serve] accept error: {e}"); continue; }
        };
        if let Err(e) = handle(&stream, &dev, &tok, &lm, &dev_name) {
            eprintln!("[serve] handler error: {e}");
        }
    }
    Ok(())
}

fn handle(
    stream: &std::net::TcpStream,
    dev: &t500,
    tok: &t544,
    lm: &Mutex<t548>,
    dev_name: &str,
) -> Result<()> {
    let mut reader = BufReader::new(stream);
    let mut writer = stream.try_clone()?;

    // Parse request line
    let mut req_line = String::new();
    reader.read_line(&mut req_line)?;
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 { return Ok(()); }
    let method = parts[0];
    let path   = parts[1];

    // Read headers
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() { break; }
        let lower = trimmed.to_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }

    match (method, path) {
        ("GET", "/health") => {
            let body = format!(r#"{{"status":"ok","device":{}}}"#,
                               serde_json::to_string(dev_name)?);
            write_response(&mut writer, 200, "OK", &body)?;
        }
        ("POST", "/generate") => {
            let mut body_bytes = vec![0u8; content_length];
            use std::io::Read;
            reader.read_exact(&mut body_bytes)?;
            let req: serde_json::Value = serde_json::from_slice(&body_bytes)?;
            let prompt = req["prompt"].as_str().unwrap_or("").to_string();
            let max_new: usize = req["max_new_tokens"].as_u64().unwrap_or(128) as usize;

            eprintln!("[serve] generate prompt={:?} max_new={max_new}", &prompt[..prompt.len().min(60)]);
            let output = lm.lock().unwrap().f786(dev, tok, &prompt, max_new)?;
            eprintln!("[serve] generated {} chars", output.len());

            let resp = serde_json::json!({"output": output});
            write_response(&mut writer, 200, "OK", &resp.to_string())?;
        }
        _ => {
            write_response(&mut writer, 404, "Not Found", r#"{"error":"not found"}"#)?;
        }
    }
    Ok(())
}

fn write_response(w: &mut impl Write, status: u16, reason: &str, body: &str) -> Result<()> {
    let resp = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    );
    w.write_all(resp.as_bytes())?;
    Ok(())
}
