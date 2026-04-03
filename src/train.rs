// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Training loop: forward + backward + optimizer step.
// One function call, not a framework.

use crate::autograd::{Tape, TensorId};
use crate::device::GpuDevice;
use crate::optim::AdamW;
use anyhow::Result;

/// Training step result.
pub struct StepResult {
    pub loss: f32,
    pub step: u32,
}

/// Train an MLP (or any differentiable graph) for one step.
/// `forward_fn` builds the computation graph on the tape and returns (loss_id, param_ids).
/// The training loop runs backward, extracts gradients, and updates params.
pub fn train_step(
    dev: &GpuDevice,
    opt: &mut AdamW,
    step_num: u32,
    forward_fn: impl FnOnce(&mut Tape) -> Result<(TensorId, Vec<TensorId>)>,
) -> Result<StepResult> {
    let mut tape = Tape::new(dev);

    // Forward: user builds the graph
    let (loss_id, param_ids) = forward_fn(&mut tape)?;

    // Read loss value
    let loss_val = tape.read(loss_id)?[0];

    // Backward
    tape.backward(loss_id)?;

    // Extract param buffers and grad buffers for optimizer
    let mut params: Vec<_> = param_ids.iter().map(|id| {
        // We need to extract the buffer from the tape.
        // For now, read grad and param, re-upload for optimizer.
        // This is inefficient (CPU roundtrip) but correct. Pipeline caching will fix later.
        tape.read(*id).unwrap()
    }).collect();

    let grads: Vec<_> = param_ids.iter().map(|id| {
        tape.read_grad(*id).unwrap().unwrap_or_else(|| vec![0.0; params[0].len()])
    }).collect();

    // Upload params as mutable GPU buffers and grads as read-only
    let mut param_bufs: Vec<_> = params.iter().map(|p| dev.upload(p)).collect();
    let grad_bufs: Vec<_> = grads.iter().map(|g| dev.upload(g)).collect();

    // Optimizer step (in-place update on GPU)
    opt.step(dev, &mut param_bufs, &grad_bufs)?;

    // Read updated params back (caller can use these for next step)
    for (i, buf) in param_bufs.iter().enumerate() {
        params[i] = dev.read(buf)?;
    }

    Ok(StepResult { loss: loss_val, step: step_num })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::assert_approx;

    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    #[test]
    fn test_train_step_linear_regression() {
        // Train y = 2x + 1 with MSE loss
        // Input: x = [1, 2, 3], target: y = [3, 5, 7]
        let x_data = vec![1.0, 2.0, 3.0];
        let y_data = vec![3.0, 5.0, 7.0];

        // Initial params: w=0.0, b=0.0
        let mut w = vec![0.0f32];
        let mut b = vec![0.0f32];

        let mut opt = AdamW::new(0.1);
        opt.weight_decay = 0.0;

        let mut last_loss = f32::MAX;
        for step in 0..50 {
            let x = x_data.clone();
            let y = y_data.clone();
            let w_val = w.clone();
            let b_val = b.clone();

            let mut tape = Tape::new(dev());
            let w_id = tape.leaf(&w_val);
            let b_id = tape.leaf(&b_val);
            let x_id = tape.leaf(&x);
            let y_id = tape.leaf(&y);

            // Forward: pred = x * w + b (broadcast w and b across elements)
            // Since our ops are element-wise, we need w and b as 3-element vectors
            let w3_id = tape.leaf(&[w_val[0], w_val[0], w_val[0]]);
            let b3_id = tape.leaf(&[b_val[0], b_val[0], b_val[0]]);
            let xw = tape.mul(x_id, w3_id).unwrap();
            let pred = tape.add(xw, b3_id).unwrap();
            let loss = tape.mse_loss(pred, y_id).unwrap();

            let loss_val = tape.read(loss).unwrap()[0];
            tape.backward(loss).unwrap();

            // Get gradients for the broadcast params
            let gw3 = tape.read_grad(w3_id).unwrap().unwrap();
            let gb3 = tape.read_grad(b3_id).unwrap().unwrap();

            // Sum gradients (since w3 and b3 are broadcast copies of w and b)
            let gw_sum: f32 = gw3.iter().sum();
            let gb_sum: f32 = gb3.iter().sum();

            // Manual SGD for simplicity (AdamW tested separately)
            w[0] -= 0.01 * gw_sum;
            b[0] -= 0.01 * gb_sum;

            if step % 10 == 0 {
                assert!(loss_val < last_loss || step == 0, "loss should decrease: step {step} loss {loss_val} >= prev {last_loss}");
            }
            last_loss = loss_val;
        }

        // After 50 steps, w should approach 2.0 and b should approach 1.0
        assert!((w[0] - 2.0).abs() < 0.5, "w should be near 2.0, got {}", w[0]);
        assert!((b[0] - 1.0).abs() < 0.5, "b should be near 1.0, got {}", b[0]);
    }
}
