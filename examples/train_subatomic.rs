//! Train subatomic models on GPU via any-gpu.
//! Proves: any-gpu tensor ops on AMD Vulkan → trained model → inference.
//!
//! Architecture: trigram hash features → GPU matmul → softmax → cross-entropy.
//! Backward: manual gradient computation via GPU matmul.
//!
//! Usage: cargo run --release --example train_subatomic
// Unlicense — cochranblock.org

use any_gpu::GpuDevice;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

const FEATURE_DIM: usize = 256;

// ── Slop words (P12) ───────────────────────────────────────

const SLOP_WORDS: &[&str] = &[
    "utilize", "leverage", "optimize", "comprehensive", "robust",
    "seamlessly", "scalable", "paradigm", "synergy", "cutting-edge",
    "streamline", "empower", "delve", "foster", "harness",
    "groundbreaking", "innovative", "transform", "revolutionize",
    "unprecedented",
];

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
    "cargo build --release", "git commit -m 'fix bug'", "fn main() {}",
    "let x = 42;", "assert_eq!(result, expected);", "use std::path::Path;",
    "impl Display for Error {}", "#[test] fn it_works() {}",
    "pub struct Config { port: u16 }", "Each worker node has its own sled database.",
    "Run the tests before pushing.", "This module handles HTTP routing.",
];

const SLOP_TEMPLATES: &[&str] = &[
    "We need to {s} the codebase for better results.",
    "This {s} solution will improve everything.",
    "The system is designed to {s} workflows.",
    "Our {s} approach handles all cases.",
    "This provides a {s} way to handle errors.",
    "The tool will {s} your development process.",
    "We've built a {s} framework for testing.",
    "This {s} architecture supports all platforms.",
    "The engine is designed to be {s} and reliable.",
    "We can {s} the deployment pipeline.",
];

// ── Trigram hash featurizer ────────────────────────────────

fn featurize(text: &str, dim: usize) -> Vec<f32> {
    let mut features = vec![0.0f32; dim];
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 3 {
        let mut h = DefaultHasher::new();
        text.hash(&mut h);
        features[(h.finish() as usize) % dim] = 1.0;
        return features;
    }
    for w in chars.windows(3) {
        let mut h = DefaultHasher::new();
        (w[0], w[1], w[2]).hash(&mut h);
        features[(h.finish() as usize) % dim] += 1.0;
    }
    let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for f in &mut features { *f /= norm; }
    }
    features
}

// ── GPU-accelerated training ───────────────────────────────

fn train_on_gpu(
    gpu: &GpuDevice,
    name: &str,
    features: &[Vec<f32>],
    labels: &[usize],
    num_classes: usize,
    epochs: usize,
    lr: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
    let n = features.len();
    let dim = FEATURE_DIM;

    // Xavier init weights.
    let scale = (2.0 / (dim + num_classes) as f64).sqrt() as f32;
    let mut weights = vec![0.0f32; num_classes * dim]; // [nc, dim]
    let mut bias = vec![0.0f32; num_classes];
    for (i, w) in weights.iter_mut().enumerate() {
        let mut h = DefaultHasher::new();
        i.hash(&mut h);
        *w = ((h.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0) * scale;
    }

    let t0 = Instant::now();
    let mut best_acc = 0.0f32;

    for epoch in 0..epochs {
        let mut correct = 0usize;
        let mut total_loss = 0.0f32;

        // Batch: process all examples, accumulate gradients on GPU.
        // Upload weights matrix to GPU.
        let w_gpu = gpu.upload(&weights);

        for (i, feat) in features.iter().enumerate() {
            let target = labels[i];

            // Upload feature vector as [1, dim] row.
            let x_gpu = gpu.upload(feat);

            // Forward: logits = x @ W^T → [1, nc]
            // matmul(a, b, m, n, k): a=[m,k], b=[k,n] → [m,n]
            // We want [1, dim] @ [dim, nc] = [1, nc]
            // But weights is [nc, dim], so transpose conceptually:
            // Actually matmul does a[m,k] * b[k,n]. We have x=[1,dim], W=[nc,dim].
            // We need W^T = [dim, nc]. Let's just store W as [dim, nc].
            // Wait — weights is [nc * dim] stored row-major as nc rows of dim.
            // matmul(x, W^T, 1, nc, dim) where W^T is [dim, nc].
            // Transpose W: from [nc, dim] to [dim, nc].
            let mut w_t = vec![0.0f32; dim * num_classes];
            for c in 0..num_classes {
                for d in 0..dim {
                    w_t[d * num_classes + c] = weights[c * dim + d];
                }
            }
            let wt_gpu = gpu.upload(&w_t);

            let logits_gpu = gpu.matmul(&x_gpu, &wt_gpu, 1, num_classes as u32, dim as u32)
                .expect("matmul failed");

            // Softmax on GPU: [1, nc]
            let probs_gpu = gpu.softmax(&logits_gpu, 1, num_classes as u32)
                .expect("softmax failed");

            // Download probs for loss computation and gradient.
            let probs = gpu.read(&probs_gpu).expect("read probs");

            // Loss.
            total_loss -= probs[target].max(1e-10).ln();
            let pred = probs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == target { correct += 1; }

            // Backward: grad = probs - one_hot(target)
            let mut grad = probs.clone();
            grad[target] -= 1.0;

            // Weight update: W[c][d] -= lr * grad[c] * feat[d]
            // GPU: grad_w = grad^T @ x → [nc, 1] @ [1, dim] = [nc, dim]
            let grad_gpu = gpu.upload(&grad);
            let grad_w_gpu = gpu.matmul(&grad_gpu, &x_gpu, num_classes as u32, dim as u32, 1)
                .expect("grad matmul failed");
            let grad_w = gpu.read(&grad_w_gpu).expect("read grad_w");

            // Apply update (CPU — tiny operation).
            for j in 0..weights.len() {
                weights[j] -= lr * grad_w[j];
            }
            for c in 0..num_classes {
                bias[c] -= lr * grad[c];
            }
        }

        let acc = correct as f32 / n as f32;
        if acc > best_acc { best_acc = acc; }

        if epoch % 10 == 0 || epoch == epochs - 1 {
            eprintln!(
                "  [{name}] epoch {}/{}: loss={:.4}, acc={:.1}%",
                epoch + 1, epochs,
                total_loss / n as f32,
                acc * 100.0
            );
        }
    }

    let elapsed = t0.elapsed();
    let total_params = num_classes * dim + num_classes;
    eprintln!(
        "  [{name}] done: {:.2}s, {} params, {:.1}% accuracy",
        elapsed.as_secs_f64(), total_params, best_acc * 100.0
    );

    (weights, bias, best_acc)
}

// ── Inference (CPU, microseconds) ──────────────────────────

fn predict(weights: &[f32], bias: &[f32], text: &str, nc: usize, class_names: &[&str]) -> (String, f32) {
    let feat = featurize(text, FEATURE_DIM);
    let mut logits = vec![0.0f32; nc];
    for c in 0..nc {
        logits[c] = bias[c];
        for d in 0..FEATURE_DIM {
            logits[c] += weights[c * FEATURE_DIM + d] * feat[d];
        }
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; nc];
    let mut sum = 0.0f32;
    for c in 0..nc {
        probs[c] = (logits[c] - max).exp();
        sum += probs[c];
    }
    for p in &mut probs { *p /= sum; }
    let pred = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    (class_names[pred].to_string(), probs[pred])
}

// ── Save model ─────────────────────────────────────────────

fn save_model(name: &str, weights: &[f32], bias: &[f32], class_names: &[&str], acc: f32) {
    let dir = format!("models/{}", name);
    std::fs::create_dir_all(&dir).unwrap();

    let wb: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(format!("{}/weights.bin", dir), &wb).unwrap();

    let bb: Vec<u8> = bias.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(format!("{}/bias.bin", dir), &bb).unwrap();

    let config = format!(
        r#"{{"name":"{}","feature_dim":{},"num_classes":{},"class_names":{:?},"total_params":{},"best_accuracy":{},"architecture":"trigram_hash_linear","trained_on":"AMD Radeon RX 5700 XT via any-gpu Vulkan"}}"#,
        name, FEATURE_DIM, class_names.len(), class_names,
        FEATURE_DIM * class_names.len() + class_names.len(), acc
    );
    std::fs::write(format!("{}/config.json", dir), config).unwrap();
    eprintln!("  [{name}] saved to {dir}/");
}

fn main() {
    eprintln!("=== Subatomic Model Training on AMD Vulkan ===\n");

    let gpu = GpuDevice::gpu().expect("failed to init GPU");
    eprintln!();

    // ── 1. Slop Detector (binary: clean=0, slop=1) ─────────

    eprintln!("--- Model 1: slop_detector ---");
    let mut slop_features = Vec::new();
    let mut slop_labels = Vec::new();

    for s in CLEAN {
        slop_features.push(featurize(s, FEATURE_DIM));
        slop_labels.push(0);
    }
    for word in SLOP_WORDS {
        for tmpl in SLOP_TEMPLATES {
            let text = tmpl.replace("{s}", word);
            slop_features.push(featurize(&text, FEATURE_DIM));
            slop_labels.push(1);
        }
    }
    eprintln!("  data: {} clean + {} slop = {} total",
        CLEAN.len(), SLOP_WORDS.len() * SLOP_TEMPLATES.len(), slop_features.len());

    let (sw, sb, sa) = train_on_gpu(&gpu, "slop_detector", &slop_features, &slop_labels, 2, 50, 0.01);
    save_model("slop_detector", &sw, &sb, &["clean", "slop"], sa);

    // Test inference.
    let t = Instant::now();
    let (cls, conf) = predict(&sw, &sb, "We need to leverage the synergy of our paradigm.", 2, &["clean", "slop"]);
    let us = t.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  inference: '{}' ({:.1}%) in {:.1}us", cls, conf * 100.0, us);
    let t = Instant::now();
    let (cls, conf) = predict(&sw, &sb, "Fixed the off-by-one error in the parser.", 2, &["clean", "slop"]);
    let us = t.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  inference: '{}' ({:.1}%) in {:.1}us\n", cls, conf * 100.0, us);

    // ── 2. Code vs English (binary: english=0, code=1) ──────

    eprintln!("--- Model 2: code_vs_english ---");
    let code_samples = [
        "fn main() { println!(\"hello\"); }", "let mut v: Vec<i32> = Vec::new();",
        "impl Display for Error {", "use std::collections::HashMap;",
        "pub async fn serve(port: u16) -> Result<()> {", "#[derive(Debug, Clone)]",
        "match result { Ok(v) => v, Err(e) => return Err(e) }",
        "for (i, item) in items.iter().enumerate() {", "if let Some(val) = map.get(&key) {",
        "type Result<T> = std::result::Result<T, Error>;",
        "const MAX: u32 = 10;", "struct Config { pub port: u16, pub host: String }",
        "trait Handler: Send + Sync { fn handle(&self); }",
        "enum State { Idle, Running, Failed(String) }", "let handle = tokio::spawn(async move {",
        "#[cfg(test)] mod tests { use super::*; }", "impl From<std::io::Error> for AppError {",
        "def train(model, data):", "import numpy as np", "function handleClick(event) {",
        "const express = require('express');", "func main() { fmt.Println(\"hello\") }",
        "#!/bin/bash\nset -e", "for f in *.rs; do wc -l \"$f\"; done",
        "SELECT * FROM users WHERE id = ?", "docker run -d -p 8080:8080 myapp",
    ];
    let english_samples = [
        "The project uses a single binary architecture for deployment.",
        "All tests must pass before merging a pull request.",
        "The documentation should be updated when adding new features.",
        "This approach reduces complexity by keeping everything in one crate.",
        "Worker nodes communicate over SSH with host certificate authentication.",
        "The tournament results show that smaller models can be more accurate.",
        "Each commit message should explain why the change was made.",
        "The config file lives in the home directory under a hidden folder.",
        "Binary size was reduced from 54 MB to 27 MB with link time optimization.",
        "The agent loop continues until the model stops calling tools.",
        "This is a production augment engine that runs on a single binary.",
        "The web client is compiled to WASM and embedded in the server.",
        "Context compaction summarizes older messages when the window fills up.",
        "Permission gates prompt the user before running shell commands.",
        "The pyramid architecture replaces external API calls with local models.",
        "Training data comes from tournament results and real execution traces.",
        "The starter nanobyte ships embedded in the binary for zero setup.",
        "Noodle is a companion penguin that reacts to session events.",
        "Each node in the cluster has a tokenized command set for SSH access.",
        "The unblock daemon auto approves prompts and flushes pasted text.",
        "File checkpoints are stored in sled before every write operation.",
        "The compression protocol maps every public symbol to a short token.",
        "Code review uses the inference backend to score diffs by severity.",
        "The gauntlet stress test has five phases of increasing difficulty.",
        "Deployment uses rsync to copy the binary and models to worker nodes.",
        "The feedback loop generates harder challenges from recorded failures.",
    ];

    let mut cve_features = Vec::new();
    let mut cve_labels = Vec::new();
    for s in &english_samples { cve_features.push(featurize(s, FEATURE_DIM)); cve_labels.push(0); }
    for s in &code_samples { cve_features.push(featurize(s, FEATURE_DIM)); cve_labels.push(1); }
    eprintln!("  data: {} english + {} code = {} total", english_samples.len(), code_samples.len(), cve_features.len());

    let (cw, cb, ca) = train_on_gpu(&gpu, "code_vs_english", &cve_features, &cve_labels, 2, 50, 0.01);
    save_model("code_vs_english", &cw, &cb, &["english", "code"], ca);

    let t = Instant::now();
    let (cls, conf) = predict(&cw, &cb, "fn main() { println!(\"hello\"); }", 2, &["english", "code"]);
    let us = t.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  inference: '{}' ({:.1}%) in {:.1}us", cls, conf * 100.0, us);
    let t = Instant::now();
    let (cls, conf) = predict(&cw, &cb, "The binary is 27 MB after stripping.", 2, &["english", "code"]);
    let us = t.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  inference: '{}' ({:.1}%) in {:.1}us\n", cls, conf * 100.0, us);

    // ── 3. Language Detector (5-class) ──────────────────────

    eprintln!("--- Model 3: lang_detector ---");
    let rust = ["fn main() { println!(\"hello\"); }", "let mut v: Vec<i32> = Vec::new();",
        "impl Display for Error {", "use std::collections::HashMap;",
        "pub async fn serve(port: u16) -> anyhow::Result<()> {", "#[derive(Debug, Clone)]",
        "match result { Ok(v) => v, Err(e) => return Err(e.into()) }",
        "for (i, item) in items.iter().enumerate() {", "if let Some(val) = map.get(&key) {",
        "type Result<T> = std::result::Result<T, Error>;",
        "const MAX: u32 = 10;", "struct Config { pub port: u16 }",
        "trait Handler: Send + Sync {}", "enum State { Idle, Running }",
        "let handle = tokio::spawn(async move {", "#[cfg(test)] mod tests {}",
        "impl From<std::io::Error> for AppError {", "let db = sled::open(&path)?;",
        "pub static TOOLS: &[t101] = &[", "let rx = mpsc::channel::<Arc<str>>();"];
    let python = ["def train(model, data, epochs=10):", "import numpy as np",
        "from transformers import AutoTokenizer", "class DataLoader: def __init__(self):",
        "if __name__ == '__main__': main()", "for i, (x, y) in enumerate(dataloader):",
        "loss = criterion(output, target)", "optimizer.zero_grad()",
        "import os; os.path.join(base, 'models')", "print(f'epoch {epoch}: loss={loss:.4f}')",
        "x = np.array([[1, 2], [3, 4]])", "def forward(self, x): return self.linear(x)",
        "pip install torch transformers", "with open('data.json') as f: data = json.load(f)",
        "@dataclass class Config: lr: float = 3e-4", "yield from self._generate(data)",
        "except ValueError as e: logger.error(e)", "lambda x: x ** 2",
        "self.weights = nn.Parameter(torch.randn(256))", "model.eval()"];
    let javascript = ["const express = require('express');", "function handleClick(event) {}",
        "const [state, setState] = useState(null);", "export default function App() {}",
        "fetch('/api').then(res => res.json());", "document.getElementById('root')",
        "const app = express(); app.listen(3000);", "module.exports = { config };",
        "async function fetchData(url) { return await fetch(url); }",
        "console.log(`port ${PORT}`);", "npm install express cors",
        "const router = express.Router();", "arr.map(x => x * 2).filter(x => x > 10)",
        "try { JSON.parse(input) } catch (e) {}", "window.localStorage.setItem('k', v);",
        "new Promise((resolve) => setTimeout(resolve, 1000));",
        "Object.keys(obj).forEach(key => delete obj[key]);",
        "import { createServer } from 'http';", "const { data } = useSWR('/api');",
        "class EventEmitter extends EventTarget {}"];
    let go = ["func main() { fmt.Println(\"hello\") }", "package main; import \"fmt\"",
        "func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {",
        "if err != nil { return fmt.Errorf(\"failed: %w\", err) }",
        "ch := make(chan string, 10)", "go func() { result <- process(data) }()",
        "type Config struct { Port int }", "defer file.Close()",
        "for _, item := range items { process(item) }", "ctx, cancel := context.WithTimeout()",
        "http.HandleFunc(\"/health\", handler)", "log.Fatal(http.ListenAndServe(\":8080\", nil))",
        "var wg sync.WaitGroup", "select { case msg := <-ch: handle(msg) }",
        "func NewClient(addr string) (*Client, error) {", "json.NewDecoder(r.Body).Decode(&req)",
        "bytes, err := ioutil.ReadAll(resp.Body)", "go build -o myapp ./cmd/server",
        "interface{ Error() string }", "mu.Lock() defer mu.Unlock()"];
    let shell = ["#!/bin/bash\nset -euo pipefail", "for f in *.rs; do wc -l \"$f\"; done",
        "export PATH=\"$HOME/.cargo/bin:$PATH\"", "if [ -f \"$CONFIG\" ]; then source \"$CONFIG\"; fi",
        "curl -sSf https://sh.rustup.rs | sh", "find . -name '*.rs' -exec grep -l TODO {} +",
        "tar -czf backup.tar.gz --exclude=target .", "ssh lf 'cargo build --release'",
        "rsync -avz --exclude target src/ remote:src/", "echo \"$VAR\" | grep -q 'pattern'",
        "kill $(pgrep -f 'kova serve')", "systemctl --user restart kova-serve",
        "cat /proc/cpuinfo | grep 'model name'", "awk '{print $1}' /etc/hosts | sort -u",
        "scp -r ./dist user@host:/var/www/", "chmod +x scripts/deploy.sh",
        "nohup ./server &> server.log &", "while read -r line; do process \"$line\"; done < in.txt",
        "[ -z \"$API_KEY\" ] && echo 'not set' && exit 1", "ln -sf /usr/local/bin/kova /usr/bin/kova"];

    let mut lang_features = Vec::new();
    let mut lang_labels = Vec::new();
    for s in &rust { lang_features.push(featurize(s, FEATURE_DIM)); lang_labels.push(0); }
    for s in &python { lang_features.push(featurize(s, FEATURE_DIM)); lang_labels.push(1); }
    for s in &javascript { lang_features.push(featurize(s, FEATURE_DIM)); lang_labels.push(2); }
    for s in &go { lang_features.push(featurize(s, FEATURE_DIM)); lang_labels.push(3); }
    for s in &shell { lang_features.push(featurize(s, FEATURE_DIM)); lang_labels.push(4); }
    eprintln!("  data: {} rust + {} python + {} js + {} go + {} shell = {} total",
        rust.len(), python.len(), javascript.len(), go.len(), shell.len(), lang_features.len());

    let class_names = &["rust", "python", "javascript", "go", "shell"];
    let (lw, lb, la) = train_on_gpu(&gpu, "lang_detector", &lang_features, &lang_labels, 5, 80, 0.005);
    save_model("lang_detector", &lw, &lb, class_names, la);

    let t = Instant::now();
    let (cls, conf) = predict(&lw, &lb, "fn main() { println!(\"hello\"); }", 5, class_names);
    let us = t.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  inference: '{}' ({:.1}%) in {:.1}us", cls, conf * 100.0, us);
    let t = Instant::now();
    let (cls, conf) = predict(&lw, &lb, "import numpy as np", 5, class_names);
    let us = t.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  inference: '{}' ({:.1}%) in {:.1}us", cls, conf * 100.0, us);
    let t = Instant::now();
    let (cls, conf) = predict(&lw, &lb, "#!/bin/bash\nset -e", 5, class_names);
    let us = t.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  inference: '{}' ({:.1}%) in {:.1}us", cls, conf * 100.0, us);

    eprintln!("\n=== Training complete. Models saved to models/ ===");
}
