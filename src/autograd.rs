// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Autograd: reverse-mode automatic differentiation.
// Flat tape, enum ops, no trait objects. The tape owns all tensors.
// t503=TensorId, t504=Op, t505=TapeEntry, t506=Tape.

use crate::device::{t500, t501};
use anyhow::{Result, ensure};

/// t503 = TensorId. Index into the tape's tensor storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct t503(pub u32);

/// t504 = Op. Recorded operation for backward pass. Variant names stay descriptive
/// (Add, Sub, ...) — they're enum tags, not part of the function-token map.
#[derive(Copy, Clone, Debug)]
pub enum t504 {
    /// Leaf tensor (parameter or input). No backward.
    Leaf,
    Add { a: t503, b: t503 },
    Sub { a: t503, b: t503 },
    Mul { a: t503, b: t503 },
    Scale { a: t503, s: f32 },
    Relu { a: t503 },
    Sigmoid { a: t503 },
    Swish { a: t503 },
    Tanh { a: t503 },
    Matmul { a: t503, b: t503, m: u32, n: u32, k: u32 },
    MseLoss { pred: t503, target: t503 },
    Conv2d {
        input: t503,
        weight: t503,
        bias: Option<t503>,
        batch: u32, in_c: u32, in_h: u32, in_w: u32,
        out_c: u32, out_h: u32, out_w: u32,
        kh: u32, kw: u32,
        stride_h: u32, stride_w: u32,
        pad_h: u32, pad_w: u32,
        dil_h: u32, dil_w: u32,
        groups: u32,
    },
    /// Concat a[outer, a_inner] and b[outer, b_inner] along trailing axis.
    Concat { a: t503, b: t503, outer: u32, a_inner: u32, b_inner: u32 },
    /// GroupNorm with learnable affine (gamma, beta).
    GroupNorm {
        input: t503, gamma: t503, beta: t503,
        batch: u32, channels: u32, spatial: u32, groups: u32, eps: f32,
    },
    /// Nearest-neighbor 2D upsample.
    UpsampleNearest2d {
        input: t503,
        batch: u32, channels: u32, in_h: u32, in_w: u32,
        scale_h: u32, scale_w: u32,
    },
    /// Broadcast add: out[outer, inner] = a[outer, inner] + b[outer].
    AddBroadcast { a: t503, b: t503, outer: u32, inner: u32 },
    /// Per-column add: out[rows, cols] = a[rows, cols] + b[cols] (Linear bias).
    AddPerCol { a: t503, b: t503, rows: u32, cols: u32 },
    LayerNorm {
        input: t503, gamma: t503, beta: t503,
        rows: u32, cols: u32, eps: f32,
    },
    RmsNorm {
        input: t503, gamma: t503,
        rows: u32, cols: u32, eps: f32,
    },
    EmbedLookup {
        ids: t503, weight: t503,
        n_ids: u32, vocab_size: u32, d_model: u32,
    },
    Softmax {
        input: t503, rows: u32, cols: u32,
    },
    CausalMask {
        input: t503, batch_heads: u32, q_seq: u32, kv_seq: u32,
    },
    Rope {
        input: t503, batch_heads: u32, seq: u32, head_dim: u32, start_pos: u32, base: f32,
    },
}

/// t505 = TapeEntry. One recorded operation.
struct t505 {
    op: t504,
    output: t503,
}

/// t506 = Tape. Records forward operations, runs backward to compute gradients.
pub struct t506<'d> {
    dev: &'d t500,
    entries: Vec<t505>,
    bufs: Vec<t501>,
    grads: Vec<Option<t501>>,
}

impl<'d> t506<'d> {
    /// f680 = Tape::new. Fresh tape bound to a device.
    pub fn f680(p0: &'d t500) -> Self {
        Self {
            dev: p0,
            entries: Vec::new(),
            bufs: Vec::new(),
            grads: Vec::new(),
        }
    }

    /// f681 = Tape::leaf. Register a leaf tensor (parameter or input data).
    /// No backward through this.
    pub fn f681(&mut self, p0: &[f32]) -> t503 {
        let v0 = self.dev.f502(p0);
        let v1 = t503(self.bufs.len() as u32);
        self.bufs.push(v0);
        self.grads.push(None);
        self.entries.push(t505 { op: t504::Leaf, output: v1 });
        v1
    }

    /// f681r = Tape::leaf_resident. Inject a pre-existing GPU buffer as a leaf without
    /// uploading from CPU. The handle is cloned (cheap Arc refcount bump), so the caller's
    /// original buffer and the tape share the same GPU allocation. After the optimizer
    /// updates the caller's buffer in-place the change is already reflected — no readback.
    pub fn f681r(&mut self, p0: &t501) -> t503 {
        let v0 = t503(self.bufs.len() as u32);
        self.bufs.push(p0.clone());
        self.grads.push(None);
        self.entries.push(t505 { op: t504::Leaf, output: v0 });
        v0
    }

    /// f684r = Tape::grad_buf. Return a reference to the gradient buffer for a tensor
    /// without reading to CPU. Used by f734 (train_step_gpu) to pass grads directly to
    /// the optimizer without a CPU round-trip.
    pub fn f684r(&self, p0: t503) -> Option<&t501> {
        self.grads[p0.0 as usize].as_ref()
    }

    /// f682 = Tape::read. Read tensor data back to CPU.
    pub fn f682(&self, p0: t503) -> Result<Vec<f32>> {
        self.dev.f504(&self.bufs[p0.0 as usize])
    }

    /// f683 = Tape::read_grad. Read gradient data back to CPU. Returns None if no
    /// gradient computed.
    pub fn f683(&self, p0: t503) -> Result<Option<Vec<f32>>> {
        match &self.grads[p0.0 as usize] {
            Some(v0) => Ok(Some(self.dev.f504(v0)?)),
            None => Ok(None),
        }
    }

    /// f684 = Tape::push_result. Append a result buffer + op to the tape.
    fn f684(&mut self, p0: t501, p1: t504) -> t503 {
        let v0 = t503(self.bufs.len() as u32);
        self.bufs.push(p0);
        self.grads.push(None);
        self.entries.push(t505 { op: p1, output: v0 });
        v0
    }

    /// f685 = Tape::buf. Borrow the buffer for a tape id.
    fn f685(&self, p0: t503) -> &t501 {
        &self.bufs[p0.0 as usize]
    }

    // --- Forward ops (recorded on tape) ---

    /// f686 = Tape::add.
    pub fn f686(&mut self, p0: t503, p1: t503) -> Result<t503> {
        let v0 = self.dev.f550(self.f685(p0), self.f685(p1))?;
        Ok(self.f684(v0, t504::Add { a: p0, b: p1 }))
    }

    /// f687 = Tape::sub.
    pub fn f687(&mut self, p0: t503, p1: t503) -> Result<t503> {
        let v0 = self.dev.f551(self.f685(p0), self.f685(p1))?;
        Ok(self.f684(v0, t504::Sub { a: p0, b: p1 }))
    }

    /// f688 = Tape::mul.
    pub fn f688(&mut self, p0: t503, p1: t503) -> Result<t503> {
        let v0 = self.dev.f552(self.f685(p0), self.f685(p1))?;
        Ok(self.f684(v0, t504::Mul { a: p0, b: p1 }))
    }

    /// f689 = Tape::scale.
    pub fn f689(&mut self, p0: t503, p1: f32) -> Result<t503> {
        let v0 = self.dev.f553(self.f685(p0), p1)?;
        Ok(self.f684(v0, t504::Scale { a: p0, s: p1 }))
    }

    /// f690 = Tape::relu.
    pub fn f690(&mut self, p0: t503) -> Result<t503> {
        let v0 = self.dev.f554(self.f685(p0))?;
        Ok(self.f684(v0, t504::Relu { a: p0 }))
    }

    /// f691 = Tape::sigmoid.
    pub fn f691(&mut self, p0: t503) -> Result<t503> {
        let v0 = self.dev.f555(self.f685(p0))?;
        Ok(self.f684(v0, t504::Sigmoid { a: p0 }))
    }

    /// f692 = Tape::swish.
    pub fn f692(&mut self, p0: t503) -> Result<t503> {
        let v0 = self.dev.f556(self.f685(p0))?;
        Ok(self.f684(v0, t504::Swish { a: p0 }))
    }

    /// f693 = Tape::tanh_act.
    pub fn f693(&mut self, p0: t503) -> Result<t503> {
        let v0 = self.dev.f557(self.f685(p0))?;
        Ok(self.f684(v0, t504::Tanh { a: p0 }))
    }

    /// f694 = Tape::matmul.
    pub fn f694(&mut self, p0: t503, p1: t503, p2: u32, p3: u32, p4: u32) -> Result<t503> {
        let v0 = self.dev.f580(self.f685(p0), self.f685(p1), p2, p3, p4)?;
        Ok(self.f684(v0, t504::Matmul { a: p0, b: p1, m: p2, n: p3, k: p4 }))
    }

    /// f695 = Tape::mse_loss.
    pub fn f695(&mut self, p0: t503, p1: t503) -> Result<t503> {
        let v0 = self.dev.f622(self.f685(p0), self.f685(p1))?;
        Ok(self.f684(v0, t504::MseLoss { pred: p0, target: p1 }))
    }

    /// f696 = Tape::conv2d.
    pub fn f696(
        &mut self,
        p0: t503,
        p1: t503,
        p2: Option<t503>,
        p3: u32, p4: u32, p5: u32, p6: u32,
        p7: u32, p8: u32, p9: u32,
        p10: (u32, u32), p11: (u32, u32),
        p12: (u32, u32), p13: u32,
    ) -> Result<t503> {
        let v0 = (p5 + 2 * p11.0 - p12.0 * (p8 - 1) - 1) / p10.0 + 1;
        let v1 = (p6 + 2 * p11.1 - p12.1 * (p9 - 1) - 1) / p10.1 + 1;
        let v2 = self.dev.f582(
            self.f685(p0), self.f685(p1),
            p2.map(|v3| &self.bufs[v3.0 as usize]).as_deref(),
            p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
        )?;
        Ok(self.f684(v2, t504::Conv2d {
            input: p0, weight: p1, bias: p2,
            batch: p3, in_c: p4, in_h: p5, in_w: p6,
            out_c: p7, out_h: v0, out_w: v1,
            kh: p8, kw: p9,
            stride_h: p10.0, stride_w: p10.1,
            pad_h: p11.0, pad_w: p11.1,
            dil_h: p12.0, dil_w: p12.1,
            groups: p13,
        }))
    }

    /// f697 = Tape::concat. Two tensors along trailing axis: a[outer, a_inner] +
    /// b[outer, b_inner] -> out[outer, a_inner + b_inner].
    pub fn f697(
        &mut self,
        p0: t503, p1: t503,
        p2: u32, p3: u32, p4: u32,
    ) -> Result<t503> {
        let v0 = self.dev.f640(self.f685(p0), self.f685(p1), p2, p3, p4)?;
        Ok(self.f684(v0, t504::Concat { a: p0, b: p1, outer: p2, a_inner: p3, b_inner: p4 }))
    }

    /// f698 = Tape::group_norm. GroupNorm with learnable affine. Input shape
    /// [batch, channels, spatial].
    pub fn f698(
        &mut self,
        p0: t503, p1: t503, p2: t503,
        p3: u32, p4: u32, p5: u32, p6: u32, p7: f32,
    ) -> Result<t503> {
        let v0 = self.dev.f600(
            self.f685(p0), self.f685(p1), self.f685(p2),
            p3, p4, p5, p6, p7,
        )?;
        Ok(self.f684(v0, t504::GroupNorm {
            input: p0, gamma: p1, beta: p2,
            batch: p3, channels: p4, spatial: p5, groups: p6, eps: p7,
        }))
    }

    /// f699 = Tape::upsample_nearest2d. input: [batch, channels, in_h, in_w]
    /// -> [batch, channels, in_h * scale_h, in_w * scale_w].
    pub fn f699(
        &mut self,
        p0: t503,
        p1: u32, p2: u32, p3: u32, p4: u32,
        p5: u32, p6: u32,
    ) -> Result<t503> {
        let v0 = self.dev.f660(
            self.f685(p0), p1, p2, p3, p4, p5, p6,
        )?;
        Ok(self.f684(v0, t504::UpsampleNearest2d {
            input: p0, batch: p1, channels: p2, in_h: p3, in_w: p4, scale_h: p5, scale_w: p6,
        }))
    }

    /// f700 = Tape::add_broadcast. out[outer, inner] = a[outer, inner] + b[outer].
    /// For bias add: outer = channels, inner = batch * spatial.
    /// For time conditioning: outer = batch * channels, inner = spatial.
    pub fn f700(
        &mut self,
        p0: t503, p1: t503,
        p2: u32, p3: u32,
    ) -> Result<t503> {
        let v0 = self.dev.f642(self.f685(p0), self.f685(p1), p2, p3)?;
        Ok(self.f684(v0, t504::AddBroadcast { a: p0, b: p1, outer: p2, inner: p3 }))
    }

    /// f701 = Tape::add_per_col. out[rows, cols] = a[rows, cols] + b[cols]. Linear bias.
    pub fn f701(
        &mut self,
        p0: t503, p1: t503,
        p2: u32, p3: u32,
    ) -> Result<t503> {
        let v0 = self.dev.f645(self.f685(p0), self.f685(p1), p2, p3)?;
        Ok(self.f684(v0, t504::AddPerCol { a: p0, b: p1, rows: p2, cols: p3 }))
    }

    /// f712 = Tape::layer_norm. rows×cols input, cols-wide gamma+beta.
    pub fn f712(&mut self, p0: t503, p1: t503, p2: t503, p3: u32, p4: u32, p5: f32) -> Result<t503> {
        let v0 = self.dev.f602(self.f685(p0), self.f685(p1), self.f685(p2), p3, p4, p5)?;
        Ok(self.f684(v0, t504::LayerNorm { input: p0, gamma: p1, beta: p2, rows: p3, cols: p4, eps: p5 }))
    }

    /// f713 = Tape::rms_norm. rows×cols input, cols-wide gamma.
    pub fn f713(&mut self, p0: t503, p1: t503, p2: u32, p3: u32, p4: f32) -> Result<t503> {
        let v0 = self.dev.f603(self.f685(p0), self.f685(p1), p2, p3, p4)?;
        Ok(self.f684(v0, t504::RmsNorm { input: p0, gamma: p1, rows: p2, cols: p3, eps: p4 }))
    }

    /// f714 = Tape::embed_lookup. ids [n_ids], weight [vocab, d_model] -> [n_ids, d_model].
    pub fn f714(&mut self, p0: t503, p1: t503, p2: u32, p3: u32, p4: u32) -> Result<t503> {
        let v0 = self.dev.f670(self.f685(p0), self.f685(p1), p2, p3, p4)?;
        Ok(self.f684(v0, t504::EmbedLookup { ids: p0, weight: p1, n_ids: p2, vocab_size: p3, d_model: p4 }))
    }

    /// f715 = Tape::softmax. [rows, cols] -> same shape.
    pub fn f715(&mut self, p0: t503, p1: u32, p2: u32) -> Result<t503> {
        let v0 = self.dev.f620(self.f685(p0), p1, p2)?;
        Ok(self.f684(v0, t504::Softmax { input: p0, rows: p1, cols: p2 }))
    }

    /// f716 = Tape::causal_mask. Makes a copy of the scores tensor then applies the in-place mask.
    /// scores [batch_heads, q_seq, kv_seq] -> same shape with future positions set to -1e30.
    pub fn f716(&mut self, p0: t503, p1: u32, p2: u32, p3: u32) -> Result<t503> {
        let v0 = self.dev.f553(self.f685(p0), 1.0)?;
        self.dev.f624(&v0, p1, p2, p3)?;
        Ok(self.f684(v0, t504::CausalMask { input: p0, batch_heads: p1, q_seq: p2, kv_seq: p3 }))
    }

    /// f717 = Tape::rope_fwd. Applies RoPE to [batch_heads, seq, head_dim].
    pub fn f717(&mut self, p0: t503, p1: u32, p2: u32, p3: u32, p4: u32, p5: f32) -> Result<t503> {
        let v0 = self.dev.f625(self.f685(p0), p1, p2, p3, p4, p5)?;
        Ok(self.f684(v0, t504::Rope { input: p0, batch_heads: p1, seq: p2, head_dim: p3, start_pos: p4, base: p5 }))
    }

    // --- Backward ---

    /// f703 = Tape::accum_grad. Accumulate gradient into a tensor's grad buffer.
    fn f703(&mut self, p0: t503, p1: t501) -> Result<()> {
        match &self.grads[p0.0 as usize] {
            Some(v0) => {
                let v1 = self.dev.f550(v0, &p1)?;
                self.grads[p0.0 as usize] = Some(v1);
            }
            None => {
                self.grads[p0.0 as usize] = Some(p1);
            }
        }
        Ok(())
    }

    /// f702 = Tape::backward. Run backward pass from a loss tensor. Computes
    /// gradients for all tensors on the tape.
    pub fn f702(&mut self, p0: t503) -> Result<()> {
        ensure!(self.bufs[p0.0 as usize].s507 == 1, "backward: loss must be a scalar (1 element)");

        // Seed: d(loss)/d(loss) = 1.0
        self.grads[p0.0 as usize] = Some(self.dev.f502(&[1.0]));

        // Walk tape in reverse
        for v0 in (0..self.entries.len()).rev() {
            let v1 = &self.entries[v0];
            let v2 = v1.output;

            // Skip if no gradient flows to this node
            let v3 = match &self.grads[v2.0 as usize] {
                Some(v4) => v4,
                None => continue,
            };

            match v1.op {
                t504::Leaf => {}

                t504::Add { a, b } => {
                    let v5 = self.dev.f553(v3, 1.0)?;
                    let v6 = self.dev.f553(v3, 1.0)?;
                    self.f703(a, v5)?;
                    self.f703(b, v6)?;
                }

                t504::Sub { a, b } => {
                    let v5 = self.dev.f553(v3, 1.0)?;
                    let v6 = self.dev.f553(v3, -1.0)?;
                    self.f703(a, v5)?;
                    self.f703(b, v6)?;
                }

                t504::Mul { a, b } => {
                    let v5 = self.dev.f552(v3, &self.bufs[b.0 as usize])?;
                    let v6 = self.dev.f552(v3, &self.bufs[a.0 as usize])?;
                    self.f703(a, v5)?;
                    self.f703(b, v6)?;
                }

                t504::Scale { a, s } => {
                    let v5 = self.dev.f553(v3, s)?;
                    self.f703(a, v5)?;
                }

                t504::Relu { a } => {
                    let v5 = self.dev.f559(v3, &self.bufs[a.0 as usize])?;
                    self.f703(a, v5)?;
                }

                t504::Sigmoid { a } => {
                    let v5 = self.dev.f560(v3, &self.bufs[v2.0 as usize])?;
                    self.f703(a, v5)?;
                }

                t504::Swish { a } => {
                    let v5 = self.dev.f561(v3, &self.bufs[a.0 as usize])?;
                    self.f703(a, v5)?;
                }

                t504::Tanh { a } => {
                    let v5 = self.dev.f562(v3, &self.bufs[v2.0 as usize])?;
                    self.f703(a, v5)?;
                }

                t504::Matmul { a, b, m, n, k } => {
                    // grad_a = grad_out @ B^T
                    let v5 = self.dev.f641(&self.bufs[b.0 as usize], 1, k, n, 1)?;
                    let v6 = self.dev.f580(v3, &v5, m, k, n)?;
                    // grad_b = A^T @ grad_out
                    let v7 = self.dev.f641(&self.bufs[a.0 as usize], 1, m, k, 1)?;
                    let v8 = self.dev.f580(&v7, v3, k, n, m)?;
                    self.f703(a, v6)?;
                    self.f703(b, v8)?;
                }

                t504::MseLoss { pred, target } => {
                    // grad_pred = 2 * (pred - target) / n
                    let v5 = self.bufs[pred.0 as usize].s507 as f32;
                    let v6 = self.dev.f551(&self.bufs[pred.0 as usize], &self.bufs[target.0 as usize])?;
                    let v7 = self.dev.f553(&v6, 2.0 / v5)?;
                    self.f703(pred, v7)?;
                }

                t504::Conv2d { input, weight, bias, batch, in_c, in_h, in_w, out_c, out_h, out_w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups } => {
                    // grad_input via f583 (conv_transpose2d).
                    let v5 = (in_h as i32)
                        - ((out_h as i32 - 1) * stride_h as i32
                            - 2 * pad_h as i32
                            + dil_h as i32 * (kh as i32 - 1)
                            + 1);
                    let v6 = (in_w as i32)
                        - ((out_w as i32 - 1) * stride_w as i32
                            - 2 * pad_w as i32
                            + dil_w as i32 * (kw as i32 - 1)
                            + 1);
                    ensure!(v5 >= 0 && v6 >= 0, "negative output_pad in conv backward");
                    let v7 = self.dev.f583(
                        v3,
                        &self.bufs[weight.0 as usize],
                        None,
                        batch, out_c, out_h, out_w,
                        in_c, kh, kw,
                        (stride_h, stride_w),
                        (pad_h, pad_w),
                        (v5 as u32, v6 as u32),
                        (dil_h, dil_w),
                        groups,
                    )?;
                    // grad_weight
                    let v8 = self.dev.f584(
                        &self.bufs[input.0 as usize],
                        v3,
                        batch, in_c, in_h, in_w,
                        out_c, out_h, out_w, kh, kw,
                        stride_h, stride_w, pad_h, pad_w,
                        dil_h, dil_w, groups,
                    )?;
                    // grad_bias
                    let v9 = if bias.is_some() {
                        Some(self.dev.f585(v3, batch, out_c, out_h, out_w)?)
                    } else {
                        None
                    };
                    self.f703(input, v7)?;
                    self.f703(weight, v8)?;
                    if let (Some(v10), Some(v11)) = (bias, v9) {
                        self.f703(v10, v11)?;
                    }
                }

                t504::Concat { a, b, outer, a_inner, b_inner } => {
                    let v5 = a_inner + b_inner;
                    let v6 = self.dev.f643(v3, outer, a_inner, 0, v5)?;
                    let v7 = self.dev.f643(v3, outer, b_inner, a_inner, v5)?;
                    self.f703(a, v6)?;
                    self.f703(b, v7)?;
                }

                t504::GroupNorm { input, gamma, beta, batch, channels, spatial, groups, eps } => {
                    let (v5, v6, v7) = self.dev.f601(
                        v3,
                        &self.bufs[input.0 as usize],
                        &self.bufs[gamma.0 as usize],
                        batch, channels, spatial, groups, eps,
                    )?;
                    self.f703(input, v5)?;
                    self.f703(gamma, v6)?;
                    self.f703(beta, v7)?;
                }

                t504::UpsampleNearest2d { input, batch, channels, in_h, in_w, scale_h, scale_w } => {
                    let v5 = self.dev.f661(
                        v3, batch, channels, in_h, in_w, scale_h, scale_w,
                    )?;
                    self.f703(input, v5)?;
                }

                t504::AddBroadcast { a, b, outer, inner } => {
                    let v5 = self.dev.f553(v3, 1.0)?;
                    let v6 = self.dev.f644(v3, outer, inner)?;
                    self.f703(a, v5)?;
                    self.f703(b, v6)?;
                }

                t504::AddPerCol { a, b, rows, cols } => {
                    let v5 = self.dev.f553(v3, 1.0)?;
                    let v6 = self.dev.f646(v3, rows, cols)?;
                    self.f703(a, v5)?;
                    self.f703(b, v6)?;
                }

                t504::LayerNorm { input, gamma, beta, rows, cols, eps } => {
                    let (v5, v6, v7) = self.dev.f791(v3, &self.bufs[input.0 as usize], &self.bufs[gamma.0 as usize], rows, cols, eps)?;
                    self.f703(input, v5)?;
                    self.f703(gamma, v6)?;
                    self.f703(beta, v7)?;
                }

                t504::RmsNorm { input, gamma, rows, cols, eps } => {
                    let (v5, v6) = self.dev.f792(v3, &self.bufs[input.0 as usize], &self.bufs[gamma.0 as usize], rows, cols, eps)?;
                    self.f703(input, v5)?;
                    self.f703(gamma, v6)?;
                }

                t504::EmbedLookup { ids, weight, n_ids, vocab_size, d_model } => {
                    let v5 = self.dev.f793(v3, &self.bufs[ids.0 as usize], n_ids, vocab_size, d_model)?;
                    self.f703(weight, v5)?;
                }

                t504::Softmax { input, rows, cols } => {
                    let v5 = self.dev.f794(v3, &self.bufs[v2.0 as usize], rows, cols)?;
                    self.f703(input, v5)?;
                }

                t504::CausalMask { input, batch_heads, q_seq, kv_seq } => {
                    let v5 = self.dev.f553(v3, 1.0)?;
                    self.dev.f624(&v5, batch_heads, q_seq, kv_seq)?;
                    self.f703(input, v5)?;
                }

                t504::Rope { input, batch_heads, seq, head_dim, start_pos, base } => {
                    let v5 = self.dev.f796(v3, batch_heads, seq, head_dim, start_pos, base)?;
                    self.f703(input, v5)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;

    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    #[test]
    fn f686_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[1.0, 2.0, 3.0]);
        let v2 = v0.f681(&[4.0, 5.0, 6.0]);
        let v3 = v0.f686(v1, v2).unwrap();
        let v4 = v0.f681(&[0.0, 0.0, 0.0]);
        let v5 = v0.f695(v3, v4).unwrap();
        v0.f702(v5).unwrap();

        let v6 = v0.f682(v5).unwrap();
        f544(&v6, &[155.0 / 3.0], 1e-3);

        let v7 = v0.f683(v1).unwrap().unwrap();
        let v8 = v0.f683(v2).unwrap().unwrap();
        f544(&v7, &[10.0/3.0, 14.0/3.0, 18.0/3.0], 1e-3);
        f544(&v8, &[10.0/3.0, 14.0/3.0, 18.0/3.0], 1e-3);
    }

    #[test]
    fn f688_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[2.0, 3.0]);
        let v2 = v0.f681(&[4.0, 5.0]);
        let v3 = v0.f688(v1, v2).unwrap();
        let v4 = v0.f681(&[0.0, 0.0]);
        let v5 = v0.f695(v3, v4).unwrap();
        v0.f702(v5).unwrap();

        let v6 = v0.f682(v5).unwrap();
        f544(&v6, &[144.5], 1e-3);

        let v7 = v0.f683(v1).unwrap().unwrap();
        let v8 = v0.f683(v2).unwrap().unwrap();
        f544(&v7, &[32.0, 75.0], 1e-3);
        f544(&v8, &[16.0, 45.0], 1e-3);
    }

    #[test]
    fn f694_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[1.0, 2.0]);
        let v2 = v0.f681(&[3.0, 4.0]);
        let v3 = v0.f694(v1, v2, 1, 1, 2).unwrap();
        let v4 = v0.f681(&[0.0]);
        let v5 = v0.f695(v3, v4).unwrap();
        v0.f702(v5).unwrap();

        let v6 = v0.f682(v5).unwrap();
        f544(&v6, &[121.0], 1e-3);

        let v7 = v0.f683(v1).unwrap().unwrap();
        let v8 = v0.f683(v2).unwrap().unwrap();
        f544(&v7, &[66.0, 88.0], 1e-3);
        f544(&v8, &[22.0, 44.0], 1e-3);
    }

    #[test]
    fn f690_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[-1.0, 2.0, -3.0, 4.0]);
        let v2 = v0.f690(v1).unwrap();
        let v3 = v0.f681(&[0.0, 0.0, 0.0, 0.0]);
        let v4 = v0.f695(v2, v3).unwrap();
        v0.f702(v4).unwrap();

        let v5 = v0.f682(v4).unwrap();
        f544(&v5, &[5.0], 1e-3);

        let v6 = v0.f683(v1).unwrap().unwrap();
        f544(&v6, &[0.0, 1.0, 0.0, 2.0], 1e-3);
    }

    #[test]
    fn f689_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[1.0, 2.0, 3.0]);
        let v2 = v0.f689(v1, 3.0).unwrap();
        let v3 = v0.f681(&[0.0, 0.0, 0.0]);
        let v4 = v0.f695(v2, v3).unwrap();
        v0.f702(v4).unwrap();
        let v5 = v0.f683(v1).unwrap().unwrap();
        f544(&v5, &[6.0, 12.0, 18.0], 1e-3);
    }

    #[test]
    fn f687_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[5.0, 10.0]);
        let v2 = v0.f681(&[1.0, 2.0]);
        let v3 = v0.f687(v1, v2).unwrap();
        let v4 = v0.f681(&[0.0, 0.0]);
        let v5 = v0.f695(v3, v4).unwrap();
        v0.f702(v5).unwrap();
        let v6 = v0.f683(v1).unwrap().unwrap();
        let v7 = v0.f683(v2).unwrap().unwrap();
        f544(&v6, &[4.0, 8.0], 1e-3);
        f544(&v7, &[-4.0, -8.0], 1e-3);
    }

    #[test]
    fn f691_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[0.0, 1.0, -1.0]);
        let v2 = v0.f691(v1).unwrap();
        let v3 = v0.f681(&[0.0, 0.0, 0.0]);
        let v4 = v0.f695(v2, v3).unwrap();
        v0.f702(v4).unwrap();

        let v5 = [0.5f32, 0.7311, 0.2689];
        let v6: Vec<f32> = (0..3).map(|i| 2.0 * v5[i] / 3.0 * v5[i] * (1.0 - v5[i])).collect();
        let v7 = v0.f683(v1).unwrap().unwrap();
        f544(&v7, &v6, 1e-3);
    }

    #[test]
    fn f693_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[0.0, 1.0, -1.0]);
        let v2 = v0.f693(v1).unwrap();
        let v3 = v0.f681(&[0.0, 0.0, 0.0]);
        let v4 = v0.f695(v2, v3).unwrap();
        v0.f702(v4).unwrap();

        let v5 = [0.0f32, 0.7616, -0.7616];
        let v6: Vec<f32> = (0..3).map(|i| 2.0 * v5[i] / 3.0 * (1.0 - v5[i] * v5[i])).collect();
        let v7 = v0.f683(v1).unwrap().unwrap();
        f544(&v7, &v6, 1e-2);
    }

    #[test]
    fn f692_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[0.0, 1.0, -1.0]);
        let v2 = v0.f692(v1).unwrap();
        let v3 = v0.f681(&[0.0, 0.0, 0.0]);
        let v4 = v0.f695(v2, v3).unwrap();
        v0.f702(v4).unwrap();

        let v5 = [0.0f32, 1.0, -1.0];
        let v6: Vec<f32> = v5.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        let v7: Vec<f32> = (0..3).map(|i| {
            let s = 1.0 / (1.0 + (-v5[i]).exp());
            let d = s + v5[i] * s * (1.0 - s);
            2.0 * v6[i] / 3.0 * d
        }).collect();
        let v8 = v0.f683(v1).unwrap().unwrap();
        f544(&v8, &v7, 1e-2);
    }

    #[test]
    fn f683_before_backward() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[1.0, 2.0]);
        assert!(v0.f683(v1).unwrap().is_none());
    }

    #[test]
    fn f702_non_scalar_loss() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[1.0, 2.0]);
        assert!(v0.f702(v1).is_err());
    }

    #[test]
    fn f702_diamond_graph() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[1.0]);
        let v2 = v0.f689(v1, 2.0).unwrap();
        let v3 = v0.f689(v1, 3.0).unwrap();
        let v4 = v0.f686(v2, v3).unwrap();
        let v5 = v0.f681(&[0.0]);
        let v6 = v0.f695(v4, v5).unwrap();
        v0.f702(v6).unwrap();

        let v7 = v0.f683(v1).unwrap().unwrap();
        f544(&v7, &[50.0], 1e-3);
    }

    #[test]
    fn f681_data_roundtrip() {
        let mut v0 = t506::f680(dev());
        let v1 = vec![1.5, -2.7, 0.0, 99.9];
        let v2 = v0.f681(&v1);
        assert_eq!(v0.f682(v2).unwrap(), v1);
    }

    #[test]
    fn f696_forward() {
        let mut v0 = t506::f680(dev());
        let v1: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let v2 = v0.f681(&v1);
        let v3 = v0.f681(&[1.0f32]);
        let v4 = v0.f681(&[0.0f32]);
        let v5 = v0.f696(v2, v3, Some(v4), 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
        let v6 = v0.f682(v5).unwrap();
        f544(&v6, &v1, 1e-5);
    }

    #[test]
    fn f696_backward_weight_grad() {
        let v0 = 1e-3f32;
        let v1: Vec<f32> = (1..=9).map(|x| x as f32 * 0.1).collect();
        let v2 = vec![0.5f32];

        let v3 = |v4: f32| -> f32 {
            let mut v5 = t506::f680(dev());
            let v6 = v5.f681(&v1);
            let v7 = v5.f681(&[v4]);
            let v8 = v5.f696(v6, v7, None, 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
            let v9 = v5.f681(&vec![0.0f32; 9]);
            let v10 = v5.f695(v8, v9).unwrap();
            v5.f682(v10).unwrap()[0]
        };

        let mut v11 = t506::f680(dev());
        let v12 = v11.f681(&v1);
        let v13 = v11.f681(&v2);
        let v14 = v11.f696(v12, v13, None, 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
        let v15 = v11.f681(&vec![0.0f32; 9]);
        let v16 = v11.f695(v14, v15).unwrap();
        v11.f702(v16).unwrap();
        let v17 = v11.f683(v13).unwrap().unwrap();

        let v18 = (v3(v2[0] + v0) - v3(v2[0] - v0)) / (2.0 * v0);
        assert!((v17[0] - v18).abs() < 1e-2,
            "weight grad: analytical={}, numeric={}", v17[0], v18);
    }

    #[test]
    fn f696_backward_input_grad() {
        let v0 = 1e-3f32;
        let v1: Vec<f32> = (1..=9).map(|x| x as f32 * 0.1).collect();
        let v2 = vec![0.5f32];

        let v3 = |v4: f32, v5: usize| -> f32 {
            let mut v6 = v1.clone();
            v6[v5] = v4;
            let mut v7 = t506::f680(dev());
            let v8 = v7.f681(&v6);
            let v9 = v7.f681(&v2);
            let v10 = v7.f696(v8, v9, None, 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
            let v11 = v7.f681(&vec![0.0f32; 9]);
            let v12 = v7.f695(v10, v11).unwrap();
            v7.f682(v12).unwrap()[0]
        };

        let mut v13 = t506::f680(dev());
        let v14 = v13.f681(&v1);
        let v15 = v13.f681(&v2);
        let v16 = v13.f696(v14, v15, None, 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
        let v17 = v13.f681(&vec![0.0f32; 9]);
        let v18 = v13.f695(v16, v17).unwrap();
        v13.f702(v18).unwrap();
        let v19 = v13.f683(v14).unwrap().unwrap();

        for v20 in 0..9 {
            let v21 = (v3(v1[v20] + v0, v20) - v3(v1[v20] - v0, v20)) / (2.0 * v0);
            assert!((v19[v20] - v21).abs() < 1e-2,
                "input grad[{v20}]: analytical={}, numeric={}", v19[v20], v21);
        }
    }

    #[test]
    fn f696_backward_bias_grad() {
        let mut v0 = t506::f680(dev());
        let v1 = v0.f681(&[1.0f32, 2.0, 3.0, 4.0]);
        let v2 = v0.f681(&[1.0f32]);
        let v3 = v0.f681(&[0.0f32]);
        let v4 = v0.f696(v1, v2, Some(v3), 1, 1, 2, 2, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
        let v5 = v0.f681(&[0.0f32; 4]);
        let v6 = v0.f695(v4, v5).unwrap();
        v0.f702(v6).unwrap();

        let v7 = v0.f683(v3).unwrap().unwrap();
        f544(&v7, &[5.0], 1e-3);
    }
}
