# Training

any-gpu has reverse-mode autograd, AdamW, and a `train_step` function. No framework — one call, one shot.

## How the Flat Tape Works

`t506 = Tape` is a flat vector of `t505 = TapeEntry`. Each entry holds:
- The output `t503 = TensorId` (index into the tape)
- The `t504 = Op` enum variant that produced it
- The input TensorIds it consumed

No trait objects. No dynamic dispatch. The `Op` enum covers all differentiable ops. Backward pass: topological sort in reverse, accumulate gradients into each input's grad buffer using GPU shaders.

```
Forward: tape.add(a, b) → c   records Op::Add { a, b } → c
Backward: for each entry (reverse order):
    match op {
        Op::Add { a, b } => {
            accum_grad(a, c.grad)
            accum_grad(b, c.grad)
        }
        ...
    }
```

Gradient accumulation (`f703 = accum_grad`) is a GPU elementwise add — all gradient math stays on the GPU.

## Differentiable Ops

The following ops are on the autograd tape and have backward shaders:

| Tape fn | Op | Backward |
|---------|----|---------|
| f686 | Tape::add | grad passes through to both inputs |
| f687 | Tape::sub | grad passes through (negated for subtrahend) |
| f688 | Tape::mul | grad * other_input |
| f689 | Tape::scale | grad * scalar |
| f690 | Tape::relu | grad * (x > 0) — relu_backward shader |
| f691 | Tape::sigmoid | grad * sigmoid * (1 - sigmoid) — sigmoid_backward shader |
| f692 | Tape::swish | grad * (sigmoid + x * sigmoid * (1 - sigmoid)) — swish_backward shader |
| f693 | Tape::tanh_act | grad * (1 - tanh²) — tanh_backward shader |
| f694 | Tape::matmul | grad_a = grad @ B^T, grad_b = A^T @ grad |
| f695 | Tape::mse_loss | 2/n * (pred - target) |
| f696 | Tape::conv2d | grad_weight via conv2d_grad_weight shader, grad_input via conv_transpose2d |
| f697 | Tape::concat | slice grad back to each input |
| f698 | Tape::group_norm | group_norm_backward shader |
| f699 | Tape::upsample_nearest2d | upsample_nearest2d_backward shader |
| f700 | Tape::add_broadcast | sum grad over batch dim for bias |
| f701 | Tape::add_per_col | same as add_broadcast (column direction) |

## AdamW (t507)

Single WGSL shader per step. In-place weight update:

```
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad²
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
w = w * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)
```

All of this runs in one dispatch per parameter tensor. No CPU round-trips.

```rust
use any_gpu::{t506, t507};

let opt = t507::f720(&dev, &params, lr, beta1, beta2, eps, wd)?;
opt.f721(&tape, &params, step)?;  // one call per optimizer step
```

## train_step (f730)

`f730 = train_step` wraps forward + backward + optimizer in one call:

```rust
use any_gpu::train;

let result = train::f730(&dev, &mut tape, &params, &mut opt, &mut step,
    |tape, x| {
        // your forward pass here
        // return (loss_tensor_id, predictions_tensor_id)
    }
)?;
// result.loss: f32 — current loss value
// result.step: u32 — optimizer step count
```

The closure receives the tape and input tensor id. Inside it, record your forward pass using tape ops. Return the loss. `f730` calls `f702 = tape.backward()`, then `f721 = opt.step()`, then reads the scalar loss back to CPU.

## Linear Regression Example

```rust
use any_gpu::{t500, t502, t506, t507};

let dev = t500::f500()?;

// Data: y = 2x + 1 + noise
let x_data = dev.f502(&[0.0, 1.0, 2.0, 3.0]);
let y_data = dev.f502(&[1.0, 3.0, 5.0, 7.0]);

// Parameters
let w = t502::f522(&dev, &[1])?;  // weight, initialized to zero
let b = t502::f522(&dev, &[1])?;  // bias, initialized to zero

let params = vec![w.clone(), b.clone()];
let mut opt = t507::f720(&dev, &params, 0.01, 0.9, 0.999, 1e-8, 0.0)?;
let mut step = 0u32;

for epoch in 0..1000 {
    let mut tape = t506::f680(&dev);
    let xw = tape.f681(w.clone());
    let xb = tape.f681(b.clone());
    let xi = tape.f681(t502::f521(&dev, x_data.clone(), &[4])?);
    let yi = tape.f681(t502::f521(&dev, y_data.clone(), &[4])?);

    // pred = x * w + b
    let scaled = tape.f689(xi, 1.0)?;  // identity for now
    let pred = tape.f688(xi, xw)?;
    let pred = tape.f700(pred, xb)?;
    let loss = tape.f695(pred, yi)?;

    step += 1;
    let loss_val = train::f730_inner(&dev, &mut tape, &params, &mut opt, step)?;
    if epoch % 100 == 0 {
        println!("epoch {epoch}: loss={loss_val:.6}");
    }
}
```

The tape records every op in the forward pass. `backward()` walks it in reverse and accumulates gradients. `opt.step()` updates `w` and `b` in place on the GPU. No CPU allocation of gradient tensors.
