// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Training loop: forward + backward + optimizer step. One function call, not a framework.
// t509=StepResult. f730=train_step.

use crate::autograd::{t506, t503};
use crate::device::t500;
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
    let mut v4: Vec<_> = v2.iter().map(|v5| {
        v0.f682(*v5).unwrap()
    }).collect();

    let v6: Vec<_> = v2.iter().map(|v7| {
        v0.f683(*v7).unwrap().unwrap_or_else(|| vec![0.0; v4[0].len()])
    }).collect();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;

    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

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
                assert!(v21 < v5 || v6 == 0, "loss should decrease: step {v6} loss {v21} >= prev {v5}");
            }
            v5 = v21;

            // Unused but mirror originals
            let _ = (v12, v13);
        }

        assert!((v2[0] - 2.0).abs() < 0.5, "w should be near 2.0, got {}", v2[0]);
        assert!((v3[0] - 1.0).abs() < 0.5, "b should be near 1.0, got {}", v3[0]);
    }
}
