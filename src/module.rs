// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// t545 = Module trait. Composable GPU layer: forward(dev, x) → y.
// t546 = Linear. Dense projection: x[m, in_f] @ W[in_f, out_f] + bias[out_f].
//   Weight stored pre-transposed (row-major [in_f, out_f]) for direct f580 use.
//   f780 = Linear::from_weights (caller provides pre-transposed weight buffer).
//   f781 = Linear::from_f32 (transposes [out_f, in_f] weight slice on upload).

use crate::device::{t500, t501};
use anyhow::Result;

/// t545 = Module. Composable GPU forward pass.
pub trait t545: Send + Sync {
    /// Run the layer: x [m, in_features] → y [m, out_features].
    fn forward(&self, dev: &t500, x: &t501) -> Result<t501>;
}

/// t546 = Linear. y = x @ weight + bias. Weight stored as [in_f, out_f].
pub struct t546 {
    /// Weight [in_f, out_f] — pre-transposed from the HuggingFace [out_f, in_f] format.
    pub weight: t501,
    /// Optional bias [out_f].
    pub bias: Option<t501>,
    pub in_f: usize,
    pub out_f: usize,
}

impl t546 {
    /// f780 = Linear::from_weights. Caller provides weight already in [in_f, out_f] order.
    pub fn f780(weight: t501, bias: Option<t501>, in_f: usize, out_f: usize) -> Self {
        Self {
            weight,
            bias,
            in_f,
            out_f,
        }
    }

    /// f781 = Linear::from_f32. Upload a HuggingFace weight [out_f, in_f] + optional bias.
    /// Transposes the weight to [in_f, out_f] using f641 so f580 can be called directly.
    pub fn f781(
        dev: &t500,
        w: &[f32],
        b: Option<&[f32]>,
        in_f: usize,
        out_f: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            w.len() == in_f * out_f,
            "Linear::from_f32: weight len {} != in_f*out_f {}",
            w.len(),
            in_f * out_f
        );
        // Upload W [out_f, in_f], then transpose to [in_f, out_f]
        let raw = dev.f502(w);
        let weight = dev.f641(&raw, 1, out_f as u32, in_f as u32, 1)?;
        let bias = match b {
            Some(bv) => {
                anyhow::ensure!(bv.len() == out_f, "Linear::from_f32: bias len mismatch");
                Some(dev.f502(bv))
            }
            None => None,
        };
        Ok(Self {
            weight,
            bias,
            in_f,
            out_f,
        })
    }
}

impl t545 for t546 {
    /// Linear forward: x[m, in_f] @ weight[in_f, out_f] → [m, out_f], then + bias if present.
    fn forward(&self, dev: &t500, x: &t501) -> Result<t501> {
        let m = x.s507 / self.in_f;
        anyhow::ensure!(x.s507 == m * self.in_f, "Linear::forward: x size mismatch");
        let mut out = dev.f580(
            x,
            &self.weight,
            m as u32,
            self.out_f as u32,
            self.in_f as u32,
        )?;
        if let Some(ref b) = self.bias {
            out = dev.f645(&out, b, m as u32, self.out_f as u32)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;

    fn dev() -> &'static t500 {
        &crate::ops::TEST_DEV
    }

    // f781 + forward: 2×3 linear, no bias.
    #[test]
    fn f781_forward_no_bias() {
        // W [out_f=2, in_f=3] in HF format (rows = output neurons)
        let w = vec![
            1.0f32, 0.0, 0.0, // row 0 -> selects input dim 0
            0.0, 1.0, 0.0,
        ]; // row 1 -> selects input dim 1
        let lin = t546::f781(dev(), &w, None, 3, 2).unwrap();
        // x [1, 3] = [5, 7, 0]
        let x = dev().f502(&[5.0f32, 7.0, 0.0]);
        let y = lin.forward(dev(), &x).unwrap();
        let got = dev().f504(&y).unwrap();
        // y = x @ W^T where W^T [in=3, out=2]: col0=[1,0,0], col1=[0,1,0]
        // y[0,0] = 5*1 + 7*0 + 0*0 = 5
        // y[0,1] = 5*0 + 7*1 + 0*0 = 7
        f544(&got, &[5.0, 7.0], 1e-4);
    }

    // f781 + forward: with bias.
    #[test]
    fn f781_forward_with_bias() {
        let w = vec![
            1.0f32, 0.0, // [out=2, in=2] identity
            0.0, 1.0,
        ];
        let b = vec![10.0f32, 20.0];
        let lin = t546::f781(dev(), &w, Some(&b), 2, 2).unwrap();
        let x = dev().f502(&[3.0f32, 4.0]);
        let y = lin.forward(dev(), &x).unwrap();
        let got = dev().f504(&y).unwrap();
        f544(&got, &[13.0, 24.0], 1e-4);
    }

    // f780: pre-transposed weight (pass-through, already [in, out]).
    #[test]
    fn f780_pre_transposed() {
        // Weight already [in_f=2, out_f=2] = identity
        let w_t = vec![1.0f32, 0.0, 0.0, 1.0];
        let weight = dev().f502(&w_t);
        let lin = t546::f780(weight, None, 2, 2);
        let x = dev().f502(&[6.0f32, 9.0]);
        let got = dev().f504(&lin.forward(dev(), &x).unwrap()).unwrap();
        f544(&got, &[6.0, 9.0], 1e-4);
    }

    // Batch input: m=3 rows.
    #[test]
    fn f781_batch_rows() {
        // Scale by 2: W [1, 1] = [[2.0]]
        let lin = t546::f781(dev(), &[2.0f32], None, 1, 1).unwrap();
        let x = dev().f502(&[1.0f32, 3.0, 5.0]); // [3, 1]
        let got = dev().f504(&lin.forward(dev(), &x).unwrap()).unwrap();
        f544(&got, &[2.0, 6.0, 10.0], 1e-4);
    }
}
