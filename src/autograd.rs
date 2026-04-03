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
        let b = tape.scale(a, 3.0).unwrap(); // [3, 6, 9]
        let target = tape.leaf(&[0.0, 0.0, 0.0]);
        let loss = tape.mse_loss(b, target).unwrap();
        tape.backward(loss).unwrap();

        // grad_a = d(MSE)/d(b) * d(b)/d(a) = 2*[3,6,9]/3 * 3 = 2*[3,6,9] = [6, 12, 18]
        let ga = tape.read_grad(a).unwrap().unwrap();
        assert_approx(&ga, &[6.0, 12.0, 18.0], 1e-3);
    }
}
