// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// t502 = Tensor: shaped view over a t501 (GpuBuffer). Tracks dimensions for op dispatch.

use crate::device::{t500, t501};
use anyhow::{Result, ensure};

/// t502 = Tensor. GPU tensor with shape metadata. Wraps a t501.
/// Shape is stored inline (max 6 dims covers batch x channel x D x H x W + extra).
pub struct t502 {
    /// s508 = buf. Underlying GPU storage.
    pub(crate) s508: t501,
    /// s509 = dims. Fixed-size shape buffer; only the first `s510` slots are valid.
    s509: [u32; 6],
    /// s510 = ndim. Number of valid entries in `s509`.
    s510: u8,
}

impl t502 {
    /// f520 = Tensor::new. Create a tensor from data with the given shape.
    pub fn f520(p0: &t500, p1: &[f32], p2: &[u32]) -> Result<Self> {
        let v0: u32 = p2.iter().product();
        ensure!(
            p1.len() == v0 as usize,
            "shape {:?} needs {} elements, got {}",
            p2,
            v0,
            p1.len()
        );
        ensure!(p2.len() <= 6, "max 6 dimensions, got {}", p2.len());
        let v1 = p0.f502(p1);
        let mut v2 = [0u32; 6];
        v2[..p2.len()].copy_from_slice(p2);
        Ok(Self {
            s508: v1,
            s509: v2,
            s510: p2.len() as u8,
        })
    }

    /// f521 = Tensor::from_buf. Create a tensor from an existing t501 with the given shape.
    pub fn f521(p0: t501, p1: &[u32]) -> Result<Self> {
        let v0: u32 = p1.iter().product();
        ensure!(
            p0.s507 == v0 as usize,
            "buffer has {} elements, shape needs {}",
            p0.s507,
            v0
        );
        ensure!(p1.len() <= 6, "max 6 dimensions");
        let mut v1 = [0u32; 6];
        v1[..p1.len()].copy_from_slice(p1);
        Ok(Self {
            s508: p0,
            s509: v1,
            s510: p1.len() as u8,
        })
    }

    /// f522 = Tensor::zeros. Create a zero tensor with the given shape.
    pub fn f522(p0: &t500, p1: &[u32]) -> Result<Self> {
        let v0: u32 = p1.iter().product();
        ensure!(p1.len() <= 6, "max 6 dimensions");
        let v1 = p0.f503(v0 as usize);
        let mut v2 = [0u32; 6];
        v2[..p1.len()].copy_from_slice(p1);
        Ok(Self {
            s508: v1,
            s509: v2,
            s510: p1.len() as u8,
        })
    }

    /// f523 = Tensor::shape. Shape as a slice.
    #[inline]
    pub fn f523(&self) -> &[u32] {
        &self.s509[..self.s510 as usize]
    }

    /// f524 = Tensor::ndim. Number of dimensions.
    #[inline]
    pub fn f524(&self) -> usize {
        self.s510 as usize
    }

    /// f525 = Tensor::numel. Total number of elements.
    #[inline]
    pub fn f525(&self) -> usize {
        self.s508.s507
    }

    /// f526 = Tensor::to_vec. Read tensor data back to CPU. Desktop-only;
    /// browser callers must use f527.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn f526(&self, p0: &t500) -> Result<Vec<f32>> {
        p0.f504(&self.s508)
    }

    /// f527 = Tensor::to_vec_async. Read tensor data back to CPU. Async — works on
    /// desktop and browser.
    pub async fn f527(&self, p0: &t500) -> Result<Vec<f32>> {
        p0.f505(&self.s508).await
    }

    /// f528 = Tensor::buffer. Borrow the underlying t501.
    #[inline]
    pub fn f528(&self) -> &t501 {
        &self.s508
    }

    /// f529 = Tensor::reshape. Reshape to a new shape (same total elements, no
    /// data copy).
    pub fn f529(self, p0: &[u32]) -> Result<Self> {
        let v0: u32 = p0.iter().product();
        ensure!(
            self.s508.s507 == v0 as usize,
            "reshape: {} elements can't become shape {:?} ({})",
            self.s508.s507,
            p0,
            v0
        );
        ensure!(p0.len() <= 6, "max 6 dimensions");
        let mut v1 = [0u32; 6];
        v1[..p0.len()].copy_from_slice(p0);
        Ok(Self {
            s508: self.s508,
            s509: v1,
            s510: p0.len() as u8,
        })
    }

    /// f530 = Tensor::dim. Get a single dimension size.
    #[inline]
    pub fn f530(&self, p0: usize) -> u32 {
        self.s509[p0]
    }

    /// f531 = Tensor::matmul. A[m,k] × B[k,n] = C[m,n]. Both inputs must be 2D.
    pub fn f531(&self, p0: &t500, p1: &t502) -> Result<t502> {
        ensure!(
            self.f524() == 2,
            "matmul: self must be 2D, got {}D",
            self.f524()
        );
        ensure!(
            p1.f524() == 2,
            "matmul: other must be 2D, got {}D",
            p1.f524()
        );
        let v0 = self.f530(0); // m
        let v1 = self.f530(1); // k
        let v2 = p1.f530(1); // n
        ensure!(
            v1 == p1.f530(0),
            "matmul: inner dims must match: {}×{} vs {}×{}",
            v0,
            v1,
            p1.f530(0),
            v2
        );
        let v3 = p0.f580(&self.s508, &p1.s508, v0, v2, v1)?;
        t502::f521(v3, &[v0, v2])
    }

    /// f532 = Tensor::relu. Element-wise relu. Output has the same shape as input.
    pub fn f532(&self, p0: &t500) -> Result<t502> {
        let v0 = p0.f554(&self.s508)?;
        t502::f521(v0, self.f523())
    }

    /// f533 = Tensor::softmax. Softmax along the last dimension. Input must be 2D [rows, cols].
    pub fn f533(&self, p0: &t500) -> Result<t502> {
        ensure!(
            self.f524() == 2,
            "softmax: input must be 2D [rows, cols], got {}D",
            self.f524()
        );
        let v0 = p0.f620(&self.s508, self.f530(0), self.f530(1))?;
        t502::f521(v0, self.f523())
    }

    /// f534 = Tensor::mse_loss. mean((self − target)²). Returns a scalar tensor with shape [1].
    pub fn f534(&self, p0: &t500, p1: &t502) -> Result<t502> {
        ensure!(
            self.f525() == p1.f525(),
            "mse_loss: size mismatch: {} vs {}",
            self.f525(),
            p1.f525()
        );
        let v0 = p0.f622(&self.s508, &p1.s508)?;
        t502::f521(v0, &[1])
    }

    /// f535 = Tensor::conv2d. 2D convolution. `self` must be 4D [N,C,H,W].
    /// `p1` kernel must be 4D [out_c, in_c/groups, kH, kW].
    /// `p2` bias must be 1D [out_c] or None.
    pub fn f535(
        &self,
        p0: &t500,
        p1: &t502,
        p2: Option<&t502>,
        p3: (u32, u32),
        p4: (u32, u32),
        p5: (u32, u32),
        p6: u32,
    ) -> Result<t502> {
        ensure!(
            self.f524() == 4,
            "conv2d: input must be 4D [N,C,H,W], got {}D",
            self.f524()
        );
        ensure!(
            p1.f524() == 4,
            "conv2d: kernel must be 4D [out_c,in_c/g,kH,kW], got {}D",
            p1.f524()
        );
        let v0 = self.f530(0); // N (batch)
        let v1 = self.f530(1); // in_c
        let v2 = self.f530(2); // in_h
        let v3 = self.f530(3); // in_w
        let v4 = p1.f530(0); // out_c
        let v5 = p1.f530(2); // kH
        let v6 = p1.f530(3); // kW
        let v7 = p2.map(|v8| &v8.s508);
        let v9 = p0.f582(
            &self.s508, &p1.s508, v7, v0, v1, v2, v3, v4, v5, v6, p3, p4, p5, p6,
        )?;
        let v10 = (v2 + 2 * p4.0 - p5.0 * (v5 - 1) - 1) / p3.0 + 1;
        let v11 = (v3 + 2 * p4.1 - p5.1 * (v6 - 1) - 1) / p3.1 + 1;
        t502::f521(v9, &[v0, v4, v10, v11])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> &'static t500 {
        &crate::ops::TEST_DEV
    }

    #[test]
    fn f520_basic() {
        let v0 = t502::f520(dev(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(v0.f523(), &[2, 3]);
        assert_eq!(v0.f524(), 2);
        assert_eq!(v0.f525(), 6);
    }

    #[test]
    fn f526_readback() {
        let v0 = vec![1.0, 2.0, 3.0];
        let v1 = t502::f520(dev(), &v0, &[3]).unwrap();
        assert_eq!(v1.f526(dev()).unwrap(), v0);
    }

    #[test]
    fn f529_reshape() {
        let v0 = t502::f520(dev(), &[1.0; 12], &[3, 4]).unwrap();
        let v1 = v0.f529(&[2, 6]).unwrap();
        assert_eq!(v1.f523(), &[2, 6]);
        assert_eq!(v1.f525(), 12);
    }

    #[test]
    fn f529_mismatch() {
        let v0 = t502::f520(dev(), &[1.0; 12], &[3, 4]).unwrap();
        assert!(v0.f529(&[2, 5]).is_err());
    }

    #[test]
    fn f520_shape_mismatch() {
        assert!(t502::f520(dev(), &[1.0, 2.0, 3.0], &[2, 2]).is_err());
    }

    #[test]
    fn f520_4d() {
        // NCHW: batch=2, channels=3, height=4, width=5
        let v0 = t502::f520(dev(), &[0.0; 120], &[2, 3, 4, 5]).unwrap();
        assert_eq!(v0.f523(), &[2, 3, 4, 5]);
        assert_eq!(v0.f524(), 4);
        assert_eq!(v0.f525(), 120);
        assert_eq!(v0.f530(0), 2);
        assert_eq!(v0.f530(1), 3);
    }

    #[test]
    fn f522_zeros() {
        let v0 = t502::f522(dev(), &[3, 3]).unwrap();
        let v1 = v0.f526(dev()).unwrap();
        assert_eq!(v1, vec![0.0; 9]);
    }

    #[test]
    fn f520_scalar() {
        let v0 = t502::f520(dev(), &[42.0], &[1]).unwrap();
        assert_eq!(v0.f523(), &[1]);
        assert_eq!(v0.f526(dev()).unwrap(), vec![42.0]);
    }

    #[test]
    fn f521_basic() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v1 = t502::f521(v0, &[2, 3]).unwrap();
        assert_eq!(v1.f523(), &[2, 3]);
        assert_eq!(v1.f525(), 6);
        assert_eq!(v1.f526(dev()).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn f521_mismatch() {
        let v0 = dev().f502(&[1.0; 10]);
        assert!(t502::f521(v0, &[3, 4]).is_err());
    }

    #[test]
    fn f528_buffer_access() {
        let v0 = t502::f520(dev(), &[7.0, 8.0], &[2]).unwrap();
        let v1 = v0.f528();
        assert_eq!(v1.s507, 2);
        assert_eq!(dev().f504(v1).unwrap(), vec![7.0, 8.0]);
    }

    #[test]
    fn f530_all() {
        let v0 = t502::f520(dev(), &[0.0; 120], &[2, 3, 4, 5]).unwrap();
        assert_eq!(v0.f530(0), 2);
        assert_eq!(v0.f530(1), 3);
        assert_eq!(v0.f530(2), 4);
        assert_eq!(v0.f530(3), 5);
    }

    #[test]
    fn f520_6d_max() {
        let v0 = t502::f520(dev(), &[0.0; 1], &[1, 1, 1, 1, 1, 1]).unwrap();
        assert_eq!(v0.f524(), 6);
        assert_eq!(v0.f523(), &[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn f520_7d_exceeds_max() {
        assert!(t502::f520(dev(), &[0.0; 1], &[1, 1, 1, 1, 1, 1, 1]).is_err());
    }

    #[test]
    fn f529_flatten_unflatten() {
        let v0: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let v1 = t502::f520(dev(), &v0, &[2, 3, 4]).unwrap();
        let v2 = v1.f529(&[24]).unwrap();
        assert_eq!(v2.f523(), &[24]);
        let v3 = v2.f529(&[2, 3, 4]).unwrap();
        assert_eq!(v3.f523(), &[2, 3, 4]);
        assert_eq!(v3.f526(dev()).unwrap(), v0);
    }

    #[test]
    fn f529_7d_exceeds() {
        let v0 = t502::f520(dev(), &[0.0; 1], &[1]).unwrap();
        assert!(v0.f529(&[1, 1, 1, 1, 1, 1, 1]).is_err());
    }

    #[test]
    fn f522_odd_dim() {
        let v0 = t502::f522(dev(), &[7, 13]).unwrap();
        assert_eq!(v0.f525(), 91);
        let v1 = v0.f526(dev()).unwrap();
        assert!(v1.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f531_matmul_2x3_3x2() {
        // A = [[1,2,3],[4,5,6]] (2×3), B = [[1,0],[0,1],[1,1]] (3×2)
        // C[0] = [1+0+3, 0+2+3] = [4, 5]
        // C[1] = [4+0+6, 0+5+6] = [10, 11]
        let v0 = t502::f520(dev(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let v1 = t502::f520(dev(), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]).unwrap();
        let v2 = v0.f531(dev(), &v1).unwrap();
        assert_eq!(v2.f523(), &[2, 2]);
        let v3 = v2.f526(dev()).unwrap();
        crate::ops::f544(&v3, &[4.0, 5.0, 10.0, 11.0], 1e-5);
    }

    #[test]
    fn f531_matmul_shape_error() {
        let v0 = t502::f520(dev(), &[1.0; 6], &[2, 3]).unwrap();
        let v1 = t502::f520(dev(), &[1.0; 6], &[2, 3]).unwrap();
        assert!(v0.f531(dev(), &v1).is_err()); // inner dim mismatch
    }

    #[test]
    fn f532_relu_basic() {
        let v0 = t502::f520(dev(), &[-1.0, 0.0, 1.0, 2.0], &[2, 2]).unwrap();
        let v1 = v0.f532(dev()).unwrap();
        assert_eq!(v1.f523(), &[2, 2]);
        let v2 = v1.f526(dev()).unwrap();
        crate::ops::f544(&v2, &[0.0, 0.0, 1.0, 2.0], 1e-6);
    }

    #[test]
    fn f533_softmax_rows() {
        let v0 = t502::f520(dev(), &[1.0, 2.0, 0.0, 0.0], &[2, 2]).unwrap();
        let v1 = v0.f533(dev()).unwrap();
        assert_eq!(v1.f523(), &[2, 2]);
        let v2 = v1.f526(dev()).unwrap();
        // Row 0: softmax([1,2]) ≈ [0.269, 0.731]
        assert!((v2[0] + v2[1] - 1.0).abs() < 1e-5);
        // Row 1: softmax([0,0]) = [0.5, 0.5]
        crate::ops::f544(&v2[2..4], &[0.5, 0.5], 1e-5);
    }

    #[test]
    fn f534_mse_loss() {
        let v0 = t502::f520(dev(), &[1.0, 2.0, 3.0], &[3]).unwrap();
        let v1 = t502::f520(dev(), &[1.0, 2.0, 3.0], &[3]).unwrap();
        let v2 = v0.f534(dev(), &v1).unwrap();
        assert_eq!(v2.f523(), &[1]);
        let v3 = v2.f526(dev()).unwrap();
        crate::ops::f544(&v3, &[0.0], 1e-6);
    }

    #[test]
    fn f534_mse_nonzero() {
        let v0 = t502::f520(dev(), &[0.0, 0.0], &[2]).unwrap();
        let v1 = t502::f520(dev(), &[2.0, 2.0], &[2]).unwrap();
        let v2 = v0.f534(dev(), &v1).unwrap();
        // MSE = mean((0-2)^2, (0-2)^2) = 4.0
        let v3 = v2.f526(dev()).unwrap();
        crate::ops::f544(&v3, &[4.0], 1e-5);
    }

    #[test]
    fn f535_conv2d_identity() {
        // 1x1x3x3 input, 1x1x1x1 identity kernel (value=1), no bias, stride=1, pad=0
        let v0 = t502::f520(
            dev(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        )
        .unwrap();
        let v1 = t502::f520(dev(), &[1.0], &[1, 1, 1, 1]).unwrap();
        let v2 = v0
            .f535(dev(), &v1, None, (1, 1), (0, 0), (1, 1), 1)
            .unwrap();
        assert_eq!(v2.f523(), &[1, 1, 3, 3]);
        let v3 = v2.f526(dev()).unwrap();
        crate::ops::f544(&v3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-5);
    }
}
