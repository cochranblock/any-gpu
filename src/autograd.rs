// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Autograd: reverse-mode automatic differentiation.
// Flat tape, enum ops, no trait objects. The tape owns all tensors.

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{Result, ensure};

/// Tensor ID — index into the tape's tensor storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TensorId(pub u32);

/// Recorded operation for backward pass.
#[derive(Copy, Clone, Debug)]
pub enum Op {
    /// Leaf tensor (parameter or input). No backward.
    Leaf,
    Add { a: TensorId, b: TensorId },
    Sub { a: TensorId, b: TensorId },
    Mul { a: TensorId, b: TensorId },
    Scale { a: TensorId, s: f32 },
    Relu { a: TensorId },
    Sigmoid { a: TensorId },
    Swish { a: TensorId },
    Tanh { a: TensorId },
    Matmul { a: TensorId, b: TensorId, m: u32, n: u32, k: u32 },
    MseLoss { pred: TensorId, target: TensorId },
    Conv2d {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        batch: u32, in_c: u32, in_h: u32, in_w: u32,
        out_c: u32, out_h: u32, out_w: u32,
        kh: u32, kw: u32,
        stride_h: u32, stride_w: u32,
        pad_h: u32, pad_w: u32,
        dil_h: u32, dil_w: u32,
        groups: u32,
    },
    /// Concat a[outer, a_inner] and b[outer, b_inner] along trailing axis.
    Concat { a: TensorId, b: TensorId, outer: u32, a_inner: u32, b_inner: u32 },
    /// GroupNorm with learnable affine (gamma, beta).
    GroupNorm {
        input: TensorId, gamma: TensorId, beta: TensorId,
        batch: u32, channels: u32, spatial: u32, groups: u32, eps: f32,
    },
    /// Nearest-neighbor 2D upsample.
    UpsampleNearest2d {
        input: TensorId,
        batch: u32, channels: u32, in_h: u32, in_w: u32,
        scale_h: u32, scale_w: u32,
    },
    /// Broadcast add: out[outer, inner] = a[outer, inner] + b[outer].
    AddBroadcast { a: TensorId, b: TensorId, outer: u32, inner: u32 },
    /// Per-column add: out[rows, cols] = a[rows, cols] + b[cols] (Linear bias).
    AddPerCol { a: TensorId, b: TensorId, rows: u32, cols: u32 },
}

/// Tape entry: one recorded operation.
struct TapeEntry {
    op: Op,
    output: TensorId,
}

/// Autograd tape. Records forward operations, runs backward to compute gradients.
pub struct Tape<'d> {
    dev: &'d GpuDevice,
    entries: Vec<TapeEntry>,
    bufs: Vec<GpuBuffer>,
    grads: Vec<Option<GpuBuffer>>,
}

impl<'d> Tape<'d> {
    pub fn new(dev: &'d GpuDevice) -> Self {
        Self {
            dev,
            entries: Vec::new(),
            bufs: Vec::new(),
            grads: Vec::new(),
        }
    }

    /// Register a leaf tensor (parameter or input data). No backward through this.
    pub fn leaf(&mut self, data: &[f32]) -> TensorId {
        let buf = self.dev.upload(data);
        let id = TensorId(self.bufs.len() as u32);
        self.bufs.push(buf);
        self.grads.push(None);
        self.entries.push(TapeEntry { op: Op::Leaf, output: id });
        id
    }

    /// Read tensor data back to CPU.
    pub fn read(&self, id: TensorId) -> Result<Vec<f32>> {
        self.dev.read(&self.bufs[id.0 as usize])
    }

    /// Read gradient data back to CPU. Returns None if no gradient computed.
    pub fn read_grad(&self, id: TensorId) -> Result<Option<Vec<f32>>> {
        match &self.grads[id.0 as usize] {
            Some(buf) => Ok(Some(self.dev.read(buf)?)),
            None => Ok(None),
        }
    }

    fn push_result(&mut self, buf: GpuBuffer, op: Op) -> TensorId {
        let id = TensorId(self.bufs.len() as u32);
        self.bufs.push(buf);
        self.grads.push(None);
        self.entries.push(TapeEntry { op, output: id });
        id
    }

    fn buf(&self, id: TensorId) -> &GpuBuffer {
        &self.bufs[id.0 as usize]
    }

    // --- Forward ops (recorded on tape) ---

    pub fn add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        let out = self.dev.add(self.buf(a), self.buf(b))?;
        Ok(self.push_result(out, Op::Add { a, b }))
    }

    pub fn sub(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        let out = self.dev.sub(self.buf(a), self.buf(b))?;
        Ok(self.push_result(out, Op::Sub { a, b }))
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        let out = self.dev.mul(self.buf(a), self.buf(b))?;
        Ok(self.push_result(out, Op::Mul { a, b }))
    }

    pub fn scale(&mut self, a: TensorId, s: f32) -> Result<TensorId> {
        let out = self.dev.scale(self.buf(a), s)?;
        Ok(self.push_result(out, Op::Scale { a, s }))
    }

    pub fn relu(&mut self, a: TensorId) -> Result<TensorId> {
        let out = self.dev.relu(self.buf(a))?;
        Ok(self.push_result(out, Op::Relu { a }))
    }

    pub fn sigmoid(&mut self, a: TensorId) -> Result<TensorId> {
        let out = self.dev.sigmoid(self.buf(a))?;
        Ok(self.push_result(out, Op::Sigmoid { a }))
    }

    pub fn swish(&mut self, a: TensorId) -> Result<TensorId> {
        let out = self.dev.swish(self.buf(a))?;
        Ok(self.push_result(out, Op::Swish { a }))
    }

    pub fn tanh_act(&mut self, a: TensorId) -> Result<TensorId> {
        let out = self.dev.tanh_act(self.buf(a))?;
        Ok(self.push_result(out, Op::Tanh { a }))
    }

    pub fn matmul(&mut self, a: TensorId, b: TensorId, m: u32, n: u32, k: u32) -> Result<TensorId> {
        let out = self.dev.matmul(self.buf(a), self.buf(b), m, n, k)?;
        Ok(self.push_result(out, Op::Matmul { a, b, m, n, k }))
    }

    pub fn mse_loss(&mut self, pred: TensorId, target: TensorId) -> Result<TensorId> {
        let out = self.dev.mse_loss(self.buf(pred), self.buf(target))?;
        Ok(self.push_result(out, Op::MseLoss { pred, target }))
    }

    pub fn conv2d(
        &mut self,
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        batch: u32, in_c: u32, in_h: u32, in_w: u32,
        out_c: u32, kh: u32, kw: u32,
        stride: (u32, u32), padding: (u32, u32),
        dilation: (u32, u32), groups: u32,
    ) -> Result<TensorId> {
        let out_h = (in_h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;
        let out = self.dev.conv2d(
            self.buf(input), self.buf(weight),
            bias.map(|id| &self.bufs[id.0 as usize]).as_deref(),
            batch, in_c, in_h, in_w, out_c, kh, kw, stride, padding, dilation, groups,
        )?;
        Ok(self.push_result(out, Op::Conv2d {
            input, weight, bias,
            batch, in_c, in_h, in_w,
            out_c, out_h, out_w,
            kh, kw,
            stride_h: stride.0, stride_w: stride.1,
            pad_h: padding.0, pad_w: padding.1,
            dil_h: dilation.0, dil_w: dilation.1,
            groups,
        }))
    }

    /// Concat two tensors along trailing axis: a[outer, a_inner] + b[outer, b_inner]
    /// -> out[outer, a_inner + b_inner].
    pub fn concat(
        &mut self,
        a: TensorId, b: TensorId,
        outer: u32, a_inner: u32, b_inner: u32,
    ) -> Result<TensorId> {
        let out = self.dev.concat(self.buf(a), self.buf(b), outer, a_inner, b_inner)?;
        Ok(self.push_result(out, Op::Concat { a, b, outer, a_inner, b_inner }))
    }

    /// Group normalization with learnable affine. input shape [batch, channels, spatial].
    pub fn group_norm(
        &mut self,
        input: TensorId, gamma: TensorId, beta: TensorId,
        batch: u32, channels: u32, spatial: u32, groups: u32, eps: f32,
    ) -> Result<TensorId> {
        let out = self.dev.group_norm(
            self.buf(input), self.buf(gamma), self.buf(beta),
            batch, channels, spatial, groups, eps,
        )?;
        Ok(self.push_result(out, Op::GroupNorm {
            input, gamma, beta, batch, channels, spatial, groups, eps,
        }))
    }

    /// Nearest-neighbor 2D upsample. input: [batch, channels, in_h, in_w]
    /// -> [batch, channels, in_h * scale_h, in_w * scale_w].
    pub fn upsample_nearest2d(
        &mut self,
        input: TensorId,
        batch: u32, channels: u32, in_h: u32, in_w: u32,
        scale_h: u32, scale_w: u32,
    ) -> Result<TensorId> {
        let out = self.dev.upsample_nearest2d(
            self.buf(input), batch, channels, in_h, in_w, scale_h, scale_w,
        )?;
        Ok(self.push_result(out, Op::UpsampleNearest2d {
            input, batch, channels, in_h, in_w, scale_h, scale_w,
        }))
    }

    /// Broadcast add: out[outer, inner] = a[outer, inner] + b[outer].
    /// For bias add: outer = channels, inner = batch * spatial.
    /// For time conditioning: outer = batch * channels, inner = spatial.
    pub fn add_broadcast(
        &mut self,
        a: TensorId, b: TensorId,
        outer: u32, inner: u32,
    ) -> Result<TensorId> {
        let out = self.dev.add_broadcast(self.buf(a), self.buf(b), outer, inner)?;
        Ok(self.push_result(out, Op::AddBroadcast { a, b, outer, inner }))
    }

    /// Per-column add: out[rows, cols] = a[rows, cols] + b[cols]. Linear bias.
    pub fn add_per_col(
        &mut self,
        a: TensorId, b: TensorId,
        rows: u32, cols: u32,
    ) -> Result<TensorId> {
        let out = self.dev.add_per_col(self.buf(a), self.buf(b), rows, cols)?;
        Ok(self.push_result(out, Op::AddPerCol { a, b, rows, cols }))
    }

    // --- Backward ---

    /// Accumulate gradient into a tensor's grad buffer.
    fn accum_grad(&mut self, id: TensorId, grad: GpuBuffer) -> Result<()> {
        match &self.grads[id.0 as usize] {
            Some(existing) => {
                let summed = self.dev.add(existing, &grad)?;
                self.grads[id.0 as usize] = Some(summed);
            }
            None => {
                self.grads[id.0 as usize] = Some(grad);
            }
        }
        Ok(())
    }

    /// Run backward pass from a loss tensor. Computes gradients for all tensors on the tape.
    pub fn backward(&mut self, loss: TensorId) -> Result<()> {
        ensure!(self.bufs[loss.0 as usize].len == 1, "backward: loss must be a scalar (1 element)");

        // Seed: d(loss)/d(loss) = 1.0
        self.grads[loss.0 as usize] = Some(self.dev.upload(&[1.0]));

        // Walk tape in reverse
        for i in (0..self.entries.len()).rev() {
            let entry = &self.entries[i];
            let out_id = entry.output;

            // Skip if no gradient flows to this node
            let grad_out = match &self.grads[out_id.0 as usize] {
                Some(g) => g,
                None => continue,
            };

            // Clone the grad_out reference data we need before mutating self
            // We need to read grad_out's buffer info before calling accum_grad
            match entry.op {
                Op::Leaf => {} // no backward for leaves

                Op::Add { a, b } => {
                    // grad_a = grad_out, grad_b = grad_out
                    let ga = self.dev.scale(grad_out, 1.0)?; // copy
                    let gb = self.dev.scale(grad_out, 1.0)?;
                    self.accum_grad(a, ga)?;
                    self.accum_grad(b, gb)?;
                }

                Op::Sub { a, b } => {
                    // grad_a = grad_out, grad_b = -grad_out
                    let ga = self.dev.scale(grad_out, 1.0)?;
                    let gb = self.dev.scale(grad_out, -1.0)?;
                    self.accum_grad(a, ga)?;
                    self.accum_grad(b, gb)?;
                }

                Op::Mul { a, b } => {
                    // grad_a = grad_out * b, grad_b = grad_out * a
                    let ga = self.dev.mul(grad_out, &self.bufs[b.0 as usize])?;
                    let gb = self.dev.mul(grad_out, &self.bufs[a.0 as usize])?;
                    self.accum_grad(a, ga)?;
                    self.accum_grad(b, gb)?;
                }

                Op::Scale { a, s } => {
                    // grad_a = grad_out * s
                    let ga = self.dev.scale(grad_out, s)?;
                    self.accum_grad(a, ga)?;
                }

                Op::Relu { a } => {
                    // grad_a = grad_out * (input > 0)
                    let ga = self.dev.relu_backward(grad_out, &self.bufs[a.0 as usize])?;
                    self.accum_grad(a, ga)?;
                }

                Op::Sigmoid { a } => {
                    // grad_a = grad_out * sig * (1 - sig) where sig = output
                    let ga = self.dev.sigmoid_backward(grad_out, &self.bufs[out_id.0 as usize])?;
                    self.accum_grad(a, ga)?;
                }

                Op::Swish { a } => {
                    // grad_a = grad_out * (sig + x * sig * (1 - sig)) where sig = sigmoid(x)
                    let ga = self.dev.swish_backward(grad_out, &self.bufs[a.0 as usize])?;
                    self.accum_grad(a, ga)?;
                }

                Op::Tanh { a } => {
                    // grad_a = grad_out * (1 - tanh(x)^2) where tanh(x) = output
                    let ga = self.dev.tanh_backward(grad_out, &self.bufs[out_id.0 as usize])?;
                    self.accum_grad(a, ga)?;
                }

                Op::Matmul { a, b, m, n, k } => {
                    // grad_a = grad_out @ B^T  (grad_out is m x n, B is k x n, B^T is n x k -> grad_a is m x k)
                    let bt = self.dev.transpose(&self.bufs[b.0 as usize], 1, k, n, 1)?;
                    let ga = self.dev.matmul(grad_out, &bt, m, k, n)?;
                    // grad_b = A^T @ grad_out  (A is m x k, A^T is k x m, grad_out is m x n -> grad_b is k x n)
                    let at = self.dev.transpose(&self.bufs[a.0 as usize], 1, m, k, 1)?;
                    let gb = self.dev.matmul(&at, grad_out, k, n, m)?;
                    self.accum_grad(a, ga)?;
                    self.accum_grad(b, gb)?;
                }

                Op::MseLoss { pred, target } => {
                    // grad_pred = 2 * (pred - target) / n
                    let n = self.bufs[pred.0 as usize].len as f32;
                    let diff = self.dev.sub(&self.bufs[pred.0 as usize], &self.bufs[target.0 as usize])?;
                    let ga = self.dev.scale(&diff, 2.0 / n)?;
                    self.accum_grad(pred, ga)?;
                }

                Op::Conv2d { input, weight, bias, batch, in_c, in_h, in_w, out_c, out_h, out_w, kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups } => {
                    // grad_input via conv_transpose2d.
                    // For stride > 1, we must provide output_padding so the transpose
                    // conv recovers the original input dims. For a forward conv:
                    //   out_h = (in_h + 2*pad - dil*(k-1) - 1) / stride + 1
                    // The inverse transpose conv gives:
                    //   recovered_h = (out_h - 1)*stride - 2*pad + dil*(k-1) + output_pad + 1
                    // Solve for output_pad so recovered_h == in_h.
                    let out_pad_h = (in_h as i32)
                        - ((out_h as i32 - 1) * stride_h as i32
                            - 2 * pad_h as i32
                            + dil_h as i32 * (kh as i32 - 1)
                            + 1);
                    let out_pad_w = (in_w as i32)
                        - ((out_w as i32 - 1) * stride_w as i32
                            - 2 * pad_w as i32
                            + dil_w as i32 * (kw as i32 - 1)
                            + 1);
                    ensure!(out_pad_h >= 0 && out_pad_w >= 0, "negative output_pad in conv backward");
                    let ga = self.dev.conv_transpose2d(
                        grad_out,
                        &self.bufs[weight.0 as usize],
                        None,
                        batch, out_c, out_h, out_w,
                        in_c, kh, kw,
                        (stride_h, stride_w),
                        (pad_h, pad_w),
                        (out_pad_h as u32, out_pad_w as u32),
                        (dil_h, dil_w),
                        groups,
                    )?;
                    // grad_weight
                    let gw = self.dev.conv2d_grad_weight(
                        &self.bufs[input.0 as usize],
                        grad_out,
                        batch, in_c, in_h, in_w,
                        out_c, out_h, out_w, kh, kw,
                        stride_h, stride_w, pad_h, pad_w,
                        dil_h, dil_w, groups,
                    )?;
                    // grad_bias
                    let gb = if bias.is_some() {
                        Some(self.dev.conv2d_grad_bias(grad_out, batch, out_c, out_h, out_w)?)
                    } else {
                        None
                    };
                    self.accum_grad(input, ga)?;
                    self.accum_grad(weight, gw)?;
                    if let (Some(bias_id), Some(gb_buf)) = (bias, gb) {
                        self.accum_grad(bias_id, gb_buf)?;
                    }
                }

                Op::Concat { a, b, outer, a_inner, b_inner } => {
                    // grad_a = first a_inner of each outer block in grad_out
                    // grad_b = last b_inner of each outer block in grad_out
                    let combined = a_inner + b_inner;
                    let ga = self.dev.slice_per_block(grad_out, outer, a_inner, 0, combined)?;
                    let gb = self.dev.slice_per_block(grad_out, outer, b_inner, a_inner, combined)?;
                    self.accum_grad(a, ga)?;
                    self.accum_grad(b, gb)?;
                }

                Op::GroupNorm { input, gamma, beta, batch, channels, spatial, groups, eps } => {
                    let (di, dg, db) = self.dev.group_norm_backward(
                        grad_out,
                        &self.bufs[input.0 as usize],
                        &self.bufs[gamma.0 as usize],
                        batch, channels, spatial, groups, eps,
                    )?;
                    self.accum_grad(input, di)?;
                    self.accum_grad(gamma, dg)?;
                    self.accum_grad(beta, db)?;
                }

                Op::UpsampleNearest2d { input, batch, channels, in_h, in_w, scale_h, scale_w } => {
                    let gi = self.dev.upsample_nearest2d_backward(
                        grad_out, batch, channels, in_h, in_w, scale_h, scale_w,
                    )?;
                    self.accum_grad(input, gi)?;
                }

                Op::AddBroadcast { a, b, outer, inner } => {
                    // grad_a = grad_out (same shape)
                    // grad_b[o] = sum over inner dim of grad_out[o, :]
                    let ga = self.dev.scale(grad_out, 1.0)?;
                    let gb = self.dev.sum_inner(grad_out, outer, inner)?;
                    self.accum_grad(a, ga)?;
                    self.accum_grad(b, gb)?;
                }

                Op::AddPerCol { a, b, rows, cols } => {
                    // grad_a = grad_out (same shape)
                    // grad_b[c] = sum over rows of grad_out[:, c]
                    let ga = self.dev.scale(grad_out, 1.0)?;
                    let gb = self.dev.sum_rows(grad_out, rows, cols)?;
                    self.accum_grad(a, ga)?;
                    self.accum_grad(b, gb)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::assert_approx;

    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    #[test]
    fn test_backward_add() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[1.0, 2.0, 3.0]);
        let b = tape.leaf(&[4.0, 5.0, 6.0]);
        let c = tape.add(a, b).unwrap();
        // sum c to scalar for loss
        // c = [5, 7, 9], loss = mean(c^2) - but let's use a simpler path
        // Just test: loss = sum(c) via scale trick: loss_val = c[0]+c[1]+c[2]
        // Actually, let's test with mse against zero
        let target = tape.leaf(&[0.0, 0.0, 0.0]);
        let loss = tape.mse_loss(c, target).unwrap();
        tape.backward(loss).unwrap();

        // MSE = (5^2 + 7^2 + 9^2)/3 = (25+49+81)/3 = 155/3
        let loss_val = tape.read(loss).unwrap();
        assert_approx(&loss_val, &[155.0 / 3.0], 1e-3);

        // d(MSE)/d(pred) = 2*(pred-target)/n = 2*[5,7,9]/3
        // d(pred)/d(a) = 1, d(pred)/d(b) = 1
        // So grad_a = grad_b = 2*[5,7,9]/3
        let ga = tape.read_grad(a).unwrap().unwrap();
        let gb = tape.read_grad(b).unwrap().unwrap();
        assert_approx(&ga, &[10.0/3.0, 14.0/3.0, 18.0/3.0], 1e-3);
        assert_approx(&gb, &[10.0/3.0, 14.0/3.0, 18.0/3.0], 1e-3);
    }

    #[test]
    fn test_backward_mul() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[2.0, 3.0]);
        let b = tape.leaf(&[4.0, 5.0]);
        let c = tape.mul(a, b).unwrap(); // c = [8, 15]
        let target = tape.leaf(&[0.0, 0.0]);
        let loss = tape.mse_loss(c, target).unwrap();
        tape.backward(loss).unwrap();

        // MSE = (64 + 225)/2 = 144.5
        let loss_val = tape.read(loss).unwrap();
        assert_approx(&loss_val, &[144.5], 1e-3);

        // d(MSE)/d(c) = 2*[8,15]/2 = [8, 15]
        // d(c)/d(a) = b = [4, 5], d(c)/d(b) = a = [2, 3]
        // grad_a = [8,15] * [4,5] = [32, 75]
        // grad_b = [8,15] * [2,3] = [16, 45]
        let ga = tape.read_grad(a).unwrap().unwrap();
        let gb = tape.read_grad(b).unwrap().unwrap();
        assert_approx(&ga, &[32.0, 75.0], 1e-3);
        assert_approx(&gb, &[16.0, 45.0], 1e-3);
    }

    #[test]
    fn test_backward_matmul() {
        let mut tape = Tape::new(dev());
        // A = [[1, 2]], B = [[3], [4]] -> C = [[11]]
        let a = tape.leaf(&[1.0, 2.0]); // 1x2
        let b = tape.leaf(&[3.0, 4.0]); // 2x1
        let c = tape.matmul(a, b, 1, 1, 2).unwrap(); // 1x1 = [[11]]
        let target = tape.leaf(&[0.0]);
        let loss = tape.mse_loss(c, target).unwrap();
        tape.backward(loss).unwrap();

        // MSE = 121/1 = 121
        let loss_val = tape.read(loss).unwrap();
        assert_approx(&loss_val, &[121.0], 1e-3);

        // d(MSE)/d(c) = 2*11/1 = 22
        // grad_a = grad_out @ B^T = [22] @ [3, 4] = [66, 88]
        // grad_b = A^T @ grad_out = [[1],[2]] @ [22] = [22, 44]
        let ga = tape.read_grad(a).unwrap().unwrap();
        let gb = tape.read_grad(b).unwrap().unwrap();
        assert_approx(&ga, &[66.0, 88.0], 1e-3);
        assert_approx(&gb, &[22.0, 44.0], 1e-3);
    }

    #[test]
    fn test_backward_relu() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[-1.0, 2.0, -3.0, 4.0]);
        let b = tape.relu(a).unwrap(); // [0, 2, 0, 4]
        let target = tape.leaf(&[0.0, 0.0, 0.0, 0.0]);
        let loss = tape.mse_loss(b, target).unwrap();
        tape.backward(loss).unwrap();

        // MSE = (0 + 4 + 0 + 16)/4 = 5
        let loss_val = tape.read(loss).unwrap();
        assert_approx(&loss_val, &[5.0], 1e-3);

        // d(MSE)/d(b) = 2*[0,2,0,4]/4 = [0, 1, 0, 2]
        // d(relu)/d(a) = [0, 1, 0, 1] (mask where a > 0)
        // grad_a = [0, 1, 0, 2] * [0, 1, 0, 1] = [0, 1, 0, 2]
        let ga = tape.read_grad(a).unwrap().unwrap();
        assert_approx(&ga, &[0.0, 1.0, 0.0, 2.0], 1e-3);
    }

    #[test]
    fn test_backward_scale() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[1.0, 2.0, 3.0]);
        let b = tape.scale(a, 3.0).unwrap();
        let target = tape.leaf(&[0.0, 0.0, 0.0]);
        let loss = tape.mse_loss(b, target).unwrap();
        tape.backward(loss).unwrap();
        let ga = tape.read_grad(a).unwrap().unwrap();
        assert_approx(&ga, &[6.0, 12.0, 18.0], 1e-3);
    }

    #[test]
    fn test_backward_sub() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[5.0, 10.0]);
        let b = tape.leaf(&[1.0, 2.0]);
        let c = tape.sub(a, b).unwrap(); // [4, 8]
        let target = tape.leaf(&[0.0, 0.0]);
        let loss = tape.mse_loss(c, target).unwrap();
        tape.backward(loss).unwrap();
        // d(MSE)/d(c) = 2*[4,8]/2 = [4, 8]
        // grad_a = [4, 8] * 1 = [4, 8], grad_b = [4, 8] * (-1) = [-4, -8]
        let ga = tape.read_grad(a).unwrap().unwrap();
        let gb = tape.read_grad(b).unwrap().unwrap();
        assert_approx(&ga, &[4.0, 8.0], 1e-3);
        assert_approx(&gb, &[-4.0, -8.0], 1e-3);
    }

    #[test]
    fn test_backward_sigmoid() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[0.0, 1.0, -1.0]);
        let b = tape.sigmoid(a).unwrap();
        let target = tape.leaf(&[0.0, 0.0, 0.0]);
        let loss = tape.mse_loss(b, target).unwrap();
        tape.backward(loss).unwrap();

        // sig(0)=0.5, sig(1)=0.7311, sig(-1)=0.2689
        // d(MSE)/d(b) = 2*[0.5, 0.7311, 0.2689]/3
        // d(sig)/d(a) = sig*(1-sig) = [0.25, 0.1966, 0.1966]
        // grad_a = d(MSE)/d(b) * d(sig)/d(a)
        let s = [0.5f32, 0.7311, 0.2689];
        let expected: Vec<f32> = (0..3).map(|i| 2.0 * s[i] / 3.0 * s[i] * (1.0 - s[i])).collect();
        let ga = tape.read_grad(a).unwrap().unwrap();
        assert_approx(&ga, &expected, 1e-3);
    }

    #[test]
    fn test_backward_tanh() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[0.0, 1.0, -1.0]);
        let b = tape.tanh_act(a).unwrap();
        let target = tape.leaf(&[0.0, 0.0, 0.0]);
        let loss = tape.mse_loss(b, target).unwrap();
        tape.backward(loss).unwrap();

        // tanh(0)=0, tanh(1)=0.7616, tanh(-1)=-0.7616
        // d(MSE)/d(b) = 2*[0, 0.7616, -0.7616]/3
        // d(tanh)/d(a) = 1-tanh^2 = [1, 0.4200, 0.4200]
        let t = [0.0f32, 0.7616, -0.7616];
        let expected: Vec<f32> = (0..3).map(|i| 2.0 * t[i] / 3.0 * (1.0 - t[i] * t[i])).collect();
        let ga = tape.read_grad(a).unwrap().unwrap();
        assert_approx(&ga, &expected, 1e-2);
    }

    #[test]
    fn test_backward_swish() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[0.0, 1.0, -1.0]);
        let b = tape.swish(a).unwrap();
        let target = tape.leaf(&[0.0, 0.0, 0.0]);
        let loss = tape.mse_loss(b, target).unwrap();
        tape.backward(loss).unwrap();

        // swish(x) = x*sig(x), d(swish)/d(x) = sig(x) + x*sig(x)*(1-sig(x))
        let x = [0.0f32, 1.0, -1.0];
        let sw: Vec<f32> = x.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        let expected: Vec<f32> = (0..3).map(|i| {
            let s = 1.0 / (1.0 + (-x[i]).exp());
            let d_swish = s + x[i] * s * (1.0 - s);
            2.0 * sw[i] / 3.0 * d_swish
        }).collect();
        let ga = tape.read_grad(a).unwrap().unwrap();
        assert_approx(&ga, &expected, 1e-2);
    }

    #[test]
    fn test_read_grad_before_backward() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[1.0, 2.0]);
        assert!(tape.read_grad(a).unwrap().is_none());
    }

    #[test]
    fn test_backward_non_scalar_loss() {
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[1.0, 2.0]);
        // Try backward on a non-scalar — should error
        assert!(tape.backward(a).is_err());
    }

    #[test]
    fn test_backward_diamond_graph() {
        // a -> b = a*2, a -> c = a*3, d = b+c, loss = mse(d, target)
        // Tests gradient accumulation: a receives grad from both b and c paths
        let mut tape = Tape::new(dev());
        let a = tape.leaf(&[1.0]); // scalar
        let b = tape.scale(a, 2.0).unwrap(); // 2
        let c = tape.scale(a, 3.0).unwrap(); // 3
        let d = tape.add(b, c).unwrap(); // 5
        let target = tape.leaf(&[0.0]);
        let loss = tape.mse_loss(d, target).unwrap();
        tape.backward(loss).unwrap();

        // d=5, MSE=25, d(MSE)/d(d)=10
        // grad_b = 10, grad_c = 10
        // grad_a from b path: 10*2 = 20
        // grad_a from c path: 10*3 = 30
        // total grad_a = 50
        let ga = tape.read_grad(a).unwrap().unwrap();
        assert_approx(&ga, &[50.0], 1e-3);
    }

    #[test]
    fn test_tape_leaf_data_roundtrip() {
        let mut tape = Tape::new(dev());
        let data = vec![1.5, -2.7, 0.0, 99.9];
        let a = tape.leaf(&data);
        assert_eq!(tape.read(a).unwrap(), data);
    }

    #[test]
    fn test_tape_conv2d_forward() {
        // 1x1x3x3 input, 1x1x1x1 weight=1, bias=0 -> output == input
        let mut tape = Tape::new(dev());
        let input_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let inp = tape.leaf(&input_data);
        let w = tape.leaf(&[1.0f32]);
        let b = tape.leaf(&[0.0f32]);
        let out = tape.conv2d(inp, w, Some(b), 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
        let result = tape.read(out).unwrap();
        assert_approx(&result, &input_data, 1e-5);
    }

    #[test]
    fn test_tape_conv2d_backward_weight_grad() {
        let eps = 1e-3f32;
        let input_data: Vec<f32> = (1..=9).map(|x| x as f32 * 0.1).collect();
        let weight_data = vec![0.5f32];

        let run = |w_val: f32| -> f32 {
            let mut tape = Tape::new(dev());
            let inp = tape.leaf(&input_data);
            let w = tape.leaf(&[w_val]);
            let out = tape.conv2d(inp, w, None, 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
            let target = tape.leaf(&vec![0.0f32; 9]);
            let loss = tape.mse_loss(out, target).unwrap();
            tape.read(loss).unwrap()[0]
        };

        let mut tape = Tape::new(dev());
        let inp = tape.leaf(&input_data);
        let w = tape.leaf(&weight_data);
        let out = tape.conv2d(inp, w, None, 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
        let target = tape.leaf(&vec![0.0f32; 9]);
        let loss = tape.mse_loss(out, target).unwrap();
        tape.backward(loss).unwrap();
        let gw = tape.read_grad(w).unwrap().unwrap();

        let numeric = (run(weight_data[0] + eps) - run(weight_data[0] - eps)) / (2.0 * eps);
        assert!((gw[0] - numeric).abs() < 1e-2,
            "weight grad: analytical={}, numeric={}", gw[0], numeric);
    }

    #[test]
    fn test_tape_conv2d_backward_input_grad() {
        let eps = 1e-3f32;
        let input_data: Vec<f32> = (1..=9).map(|x| x as f32 * 0.1).collect();
        let weight_data = vec![0.5f32];

        let run = |x_val: f32, idx: usize| -> f32 {
            let mut inp_data = input_data.clone();
            inp_data[idx] = x_val;
            let mut tape = Tape::new(dev());
            let inp = tape.leaf(&inp_data);
            let w = tape.leaf(&weight_data);
            let out = tape.conv2d(inp, w, None, 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
            let target = tape.leaf(&vec![0.0f32; 9]);
            let loss = tape.mse_loss(out, target).unwrap();
            tape.read(loss).unwrap()[0]
        };

        let mut tape = Tape::new(dev());
        let inp = tape.leaf(&input_data);
        let w = tape.leaf(&weight_data);
        let out = tape.conv2d(inp, w, None, 1, 1, 3, 3, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
        let target = tape.leaf(&vec![0.0f32; 9]);
        let loss = tape.mse_loss(out, target).unwrap();
        tape.backward(loss).unwrap();
        let gi = tape.read_grad(inp).unwrap().unwrap();

        for i in 0..9 {
            let numeric = (run(input_data[i] + eps, i) - run(input_data[i] - eps, i)) / (2.0 * eps);
            assert!((gi[i] - numeric).abs() < 1e-2,
                "input grad[{i}]: analytical={}, numeric={}", gi[i], numeric);
        }
    }

    #[test]
    fn test_tape_conv2d_backward_bias_grad() {
        // 1x1x2x2 input, 1x1x1x1 kernel, with bias
        // out is 2x2, grad_bias = sum of grad_out over spatial
        let mut tape = Tape::new(dev());
        let inp = tape.leaf(&[1.0f32, 2.0, 3.0, 4.0]);
        let w = tape.leaf(&[1.0f32]);
        let b = tape.leaf(&[0.0f32]);
        let out = tape.conv2d(inp, w, Some(b), 1, 1, 2, 2, 1, 1, 1, (1,1), (0,0), (1,1), 1).unwrap();
        let target = tape.leaf(&[0.0f32; 4]);
        let loss = tape.mse_loss(out, target).unwrap();
        tape.backward(loss).unwrap();

        // output = input (1x1 kernel=1, bias=0), target=0
        // MSE grad = 2*output/4 = output/2 = [0.5, 1.0, 1.5, 2.0]
        // grad_bias = sum = 0.5 + 1.0 + 1.5 + 2.0 = 5.0
        let gb = tape.read_grad(b).unwrap().unwrap();
        assert_approx(&gb, &[5.0], 1e-3);
    }
}
