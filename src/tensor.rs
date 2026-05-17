// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// t502 = Tensor: shaped view over a t501 (GpuBuffer). Tracks dimensions for op dispatch.

use crate::device::{t500, t501};
use anyhow::{ensure, Result};

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
            p2, v0, p1.len()
        );
        ensure!(p2.len() <= 6, "max 6 dimensions, got {}", p2.len());
        let v1 = p0.f502(p1);
        let mut v2 = [0u32; 6];
        v2[..p2.len()].copy_from_slice(p2);
        Ok(Self { s508: v1, s509: v2, s510: p2.len() as u8 })
    }

    /// f521 = Tensor::from_buf. Create a tensor from an existing t501 with the given shape.
    pub fn f521(p0: t501, p1: &[u32]) -> Result<Self> {
        let v0: u32 = p1.iter().product();
        ensure!(p0.s507 == v0 as usize, "buffer has {} elements, shape needs {}", p0.s507, v0);
        ensure!(p1.len() <= 6, "max 6 dimensions");
        let mut v1 = [0u32; 6];
        v1[..p1.len()].copy_from_slice(p1);
        Ok(Self { s508: p0, s509: v1, s510: p1.len() as u8 })
    }

    /// f522 = Tensor::zeros. Create a zero tensor with the given shape.
    pub fn f522(p0: &t500, p1: &[u32]) -> Result<Self> {
        let v0: u32 = p1.iter().product();
        ensure!(p1.len() <= 6, "max 6 dimensions");
        let v1 = p0.f503(v0 as usize);
        let mut v2 = [0u32; 6];
        v2[..p1.len()].copy_from_slice(p1);
        Ok(Self { s508: v1, s509: v2, s510: p1.len() as u8 })
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
            self.s508.s507, p0, v0
        );
        ensure!(p0.len() <= 6, "max 6 dimensions");
        let mut v1 = [0u32; 6];
        v1[..p0.len()].copy_from_slice(p0);
        Ok(Self { s508: self.s508, s509: v1, s510: p0.len() as u8 })
    }

    /// f530 = Tensor::dim. Get a single dimension size.
    #[inline]
    pub fn f530(&self, p0: usize) -> u32 {
        self.s509[p0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

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
}
