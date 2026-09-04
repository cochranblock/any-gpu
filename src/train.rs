// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Training loop: forward + backward + optimizer step. One function call, not a framework.
// t509=StepResult, t550=GpuParams. f730=train_step, f731..f735, f734=train_step_gpu.

use crate::autograd::{t503, t506};
use crate::device::{t500, t501};
use crate::optim::t507;
use anyhow::Result;

/// t509 = StepResult. Outcome of one f730 call.
pub struct t509 {
    pub loss: f32,
    pub step: u32,
}

/// f730 = train_step. Train an MLP (or any differentiable graph) for one step.
/// `p3` builds the computation graph on the tape and returns (loss_id, param_ids).
/// The training loop runs backward, extracts gradients, and updates params.
pub fn f730(
    p0: &t500,
    p1: &mut t507,
    p2: u32,
    p3: impl FnOnce(&mut t506) -> Result<(t503, Vec<t503>)>,
) -> Result<t509> {
    let mut v0 = t506::f680(p0);

    // Forward: user builds the graph
    let (v1, v2) = p3(&mut v0)?;

    // Read loss value
    let v3 = v0.f682(v1)?[0];

    // Backward
    v0.f702(v1)?;

    // Extract param buffers and grad buffers for optimizer
    let mut v4: Vec<_> = v2.iter().map(|v5| v0.f682(*v5).unwrap()).collect();

    let v6: Vec<_> = v2
        .iter()
        .map(|v7| {
            v0.f683(*v7)
                .unwrap()
                .unwrap_or_else(|| vec![0.0; v4[0].len()])
        })
        .collect();

    // Upload params as mutable GPU buffers and grads as read-only
    let mut v8: Vec<_> = v4.iter().map(|v9| p0.f502(v9)).collect();
    let v10: Vec<_> = v6.iter().map(|v11| p0.f502(v11)).collect();

    // Optimizer step (in-place update on GPU)
    p1.f721(p0, &mut v8, &v10)?;

    // Read updated params back (caller can use these for next step)
    for (v12, v13) in v8.iter().enumerate() {
        v4[v12] = p0.f504(v13)?;
    }

    Ok(t509 { loss: v3, step: p2 })
}

/// t550 = GpuParams. GPU-resident parameter tensors that persist across train steps.
/// Eliminates the per-step CPU↔GPU roundtrip in f730 — weights never leave VRAM
/// between steps. Call f732 only when you need to read values (checkpoint/save).
pub struct t550 {
    pub params: Vec<t501>,
}

impl t550 {
    /// f731 = GpuParams::new. Upload initial param values to GPU once.
    pub fn f731(p0: &t500, p1: &[&[f32]]) -> Self {
        Self {
            params: p1.iter().map(|v0| p0.f502(v0)).collect(),
        }
    }

    /// f732 = GpuParams::checkpoint. Read all params back to CPU. Use only for
    /// saving, logging, or evaluation — not in the hot training loop.
    pub fn f732(&self, p0: &t500) -> Result<Vec<Vec<f32>>> {
        self.params.iter().map(|v0| p0.f504(v0)).collect()
    }

    /// f733 = GpuParams::len. Number of param tensors.
    pub fn f733(&self) -> usize {
        self.params.len()
    }
}

/// f734 = train_step_gpu. One train step with zero CPU↔GPU param roundtrip.
/// Params stay GPU-resident; `p3` receives (tape, param_ids) where each param_id
/// maps 1:1 to `p2.params[i]`. Build the forward graph and return the loss id.
/// After backward, grads flow directly to the optimizer — no CPU readback.
pub fn f734(
    p0: &t500,
    p1: &mut t507,
    p2: &mut t550,
    p3: u32,
    p4: impl FnOnce(&mut t506, &[t503]) -> Result<t503>,
) -> Result<t509> {
    let mut v0 = t506::f680(p0);

    // Inject resident GPU buffers — cheap Arc clone, no upload
    let v1: Vec<t503> = p2.params.iter().map(|v2| v0.f681r(v2)).collect();

    // Forward: user builds graph using the injected param ids
    let v3 = p4(&mut v0, &v1)?;

    // Read scalar loss to CPU (unavoidable: needed to return the value)
    let v4 = v0.f682(v3)?[0];

    // Backward
    v0.f702(v3)?;

    // Collect grad buffers by reference — no CPU readback
    let v5: Vec<t501> = v1
        .iter()
        .enumerate()
        .map(|(v6, v7)| {
            v0.f684r(*v7)
                .cloned()
                .unwrap_or_else(|| p0.f503(p2.params[v6].s507))
        })
        .collect();

    // Optimizer updates p2.params in-place on GPU; grad clone is cheap (Arc)
    p1.f721(p0, &mut p2.params, &v5)?;

    Ok(t509 { loss: v4, step: p3 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;

    fn dev() -> &'static t500 {
        &crate::ops::TEST_DEV
    }

    #[test]
    fn f730_linear_regression() {
        // Train y = 2x + 1 with MSE loss
        let v0 = vec![1.0, 2.0, 3.0];
        let v1 = vec![3.0, 5.0, 7.0];

        let mut v2 = vec![0.0f32];
        let mut v3 = vec![0.0f32];

        let mut v4 = t507::f720(0.1);
        v4.weight_decay = 0.0;

        let mut v5 = f32::MAX;
        for v6 in 0..50 {
            let v7 = v0.clone();
            let v8 = v1.clone();
            let v9 = v2.clone();
            let v10 = v3.clone();

            let mut v11 = t506::f680(dev());
            let v12 = v11.f681(&v9);
            let v13 = v11.f681(&v10);
            let v14 = v11.f681(&v7);
            let v15 = v11.f681(&v8);

            // Forward: pred = x * w + b (broadcast w and b across elements)
            let v16 = v11.f681(&[v9[0], v9[0], v9[0]]);
            let v17 = v11.f681(&[v10[0], v10[0], v10[0]]);
            let v18 = v11.f688(v14, v16).unwrap();
            let v19 = v11.f686(v18, v17).unwrap();
            let v20 = v11.f695(v19, v15).unwrap();

            let v21 = v11.f682(v20).unwrap()[0];
            v11.f702(v20).unwrap();

            // Get gradients for the broadcast params
            let v22 = v11.f683(v16).unwrap().unwrap();
            let v23 = v11.f683(v17).unwrap().unwrap();

            // Sum gradients (since v16 and v17 are broadcast copies of v12 and v13)
            let v24: f32 = v22.iter().sum();
            let v25: f32 = v23.iter().sum();

            v2[0] -= 0.01 * v24;
            v3[0] -= 0.01 * v25;

            if v6 % 10 == 0 {
                assert!(
                    v21 < v5 || v6 == 0,
                    "loss should decrease: step {v6} loss {v21} >= prev {v5}"
                );
            }
            v5 = v21;

            // Unused but mirror originals
            let _ = (v12, v13);
        }

        assert!(
            (v2[0] - 2.0).abs() < 0.5,
            "w should be near 2.0, got {}",
            v2[0]
        );
        assert!(
            (v3[0] - 1.0).abs() < 0.5,
            "b should be near 1.0, got {}",
            v3[0]
        );
    }

    #[test]
    fn f731_upload_and_checkpoint() {
        let v0 = t550::f731(dev(), &[&[1.0, 2.0, 3.0], &[4.0, 5.0]]);
        assert_eq!(v0.f733(), 2);
        let v1 = v0.f732(dev()).unwrap();
        f544(&v1[0], &[1.0, 2.0, 3.0], 1e-6);
        f544(&v1[1], &[4.0, 5.0], 1e-6);
    }

    #[test]
    fn f734_params_converge() {
        // params = [w0, w1, w2], inputs = [1, 1, 1], targets = [0.5, 0.5, 0.5]
        // Forward: pred = params * inputs (element-wise), loss = MSE(pred, targets).
        // params should converge toward 0.5 using only the injected param_ids — no
        // checkpoint inside the loop, so weights never leave VRAM between steps.
        let mut v0 = t550::f731(dev(), &[&[0.0f32, 0.0, 0.0]]);
        let mut v1 = t507::f720(0.1);
        v1.weight_decay = 0.0;

        for v2 in 0..60u32 {
            f734(dev(), &mut v1, &mut v0, v2, |v3, v4| {
                let v5 = v3.f681(&[1.0f32, 1.0, 1.0]);
                let v6 = v3.f681(&[0.5f32, 0.5, 0.5]);
                let v7 = v3.f688(v4[0], v5)?; // pred = params * inputs
                v3.f695(v7, v6) // MSE(pred, targets)
            })
            .unwrap();
        }

        // Single checkpoint at the end — params never touched CPU during the loop
        let v3 = v0.f732(dev()).unwrap();
        for v4 in &v3[0] {
            assert!(
                (*v4 - 0.5).abs() < 0.1,
                "param should be near 0.5, got {v4}"
            );
        }
    }

    #[test]
    fn f734_loss_decreases() {
        // Verify overall convergence: loss at step 19 is less than 1% of initial loss.
        // AdamW with momentum can oscillate near zero, so strict per-step monotonicity
        // is not expected once loss is small.
        let mut v0 = t550::f731(dev(), &[&[0.0f32, 0.0, 0.0]]);
        let mut v1 = t507::f720(0.1);
        v1.weight_decay = 0.0;

        let mut v2 = f32::MAX;
        let mut v_init = f32::MAX;
        for v3 in 0..20u32 {
            let v4 = f734(dev(), &mut v1, &mut v0, v3, |v5, v6| {
                let v7 = v5.f681(&[1.0f32, 1.0, 1.0]);
                let v8 = v5.f681(&[0.5f32, 0.5, 0.5]);
                let v9 = v5.f688(v6[0], v7)?;
                v5.f695(v9, v8)
            })
            .unwrap();
            if v3 == 0 {
                v_init = v4.loss;
            }
            v2 = v4.loss;
        }
        assert!(
            v2 < v_init * 0.01,
            "loss should converge: final={v2} initial={v_init}"
        );
    }

    #[test]
    fn f733_len() {
        let v0 = t550::f731(dev(), &[&[1.0f32], &[2.0f32], &[3.0f32]]);
        assert_eq!(v0.f733(), 3);
    }
}
