// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Tensor: shaped view over a GpuBuffer. Tracks dimensions for op dispatch.
// No autograd yet — that comes in Sprint 4.

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

/// GPU tensor with shape metadata. Wraps a GpuBuffer.
/// Shape is stored inline (max 6 dims covers batch x channel x D x H x W + extra).
pub struct Tensor {
    pub(crate) buf: GpuBuffer,
    dims: [u32; 6],
    ndim: u8,
}

impl Tensor {
    /// Create a tensor from data with the given shape.
    pub fn new(dev: &GpuDevice, data: &[f32], shape: &[u32]) -> Result<Self> {
        let numel: u32 = shape.iter().product();
        ensure!(
            data.len() == numel as usize,
            "shape {:?} needs {} elements, got {}",
            shape, numel, data.len()
        );
        ensure!(shape.len() <= 6, "max 6 dimensions, got {}", shape.len());
        let buf = dev.upload(data);
        let mut dims = [0u32; 6];
        dims[..shape.len()].copy_from_slice(shape);
        Ok(Self { buf, dims, ndim: shape.len() as u8 })
    }

    /// Create a tensor from an existing GpuBuffer with the given shape.
    pub fn from_buf(buf: GpuBuffer, shape: &[u32]) -> Result<Self> {
        let numel: u32 = shape.iter().product();
        ensure!(buf.len == numel as usize, "buffer has {} elements, shape needs {}", buf.len, numel);
        ensure!(shape.len() <= 6, "max 6 dimensions");
        let mut dims = [0u32; 6];
        dims[..shape.len()].copy_from_slice(shape);
        Ok(Self { buf, dims, ndim: shape.len() as u8 })
    }

    /// Create a zero tensor with the given shape.
    pub fn zeros(dev: &GpuDevice, shape: &[u32]) -> Result<Self> {
        let numel: u32 = shape.iter().product();
        ensure!(shape.len() <= 6, "max 6 dimensions");
        let buf = dev.alloc(numel as usize);
        let mut dims = [0u32; 6];
        dims[..shape.len()].copy_from_slice(shape);
        Ok(Self { buf, dims, ndim: shape.len() as u8 })
    }

    /// Shape as a slice.
    #[inline]
    pub fn shape(&self) -> &[u32] {
        &self.dims[..self.ndim as usize]
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.ndim as usize
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.buf.len
    }

    /// Read tensor data back to CPU.
    pub fn to_vec(&self, dev: &GpuDevice) -> Result<Vec<f32>> {
        dev.read(&self.buf)
    }

    /// Borrow the underlying GpuBuffer.
    #[inline]
    pub fn buffer(&self) -> &GpuBuffer {
        &self.buf
    }

    /// Reshape to a new shape (same total elements, no data copy).
    pub fn reshape(self, new_shape: &[u32]) -> Result<Self> {
        let numel: u32 = new_shape.iter().product();
        ensure!(
            self.buf.len == numel as usize,
            "reshape: {} elements can't become shape {:?} ({})",
            self.buf.len, new_shape, numel
        );
        ensure!(new_shape.len() <= 6, "max 6 dimensions");
        let mut dims = [0u32; 6];
        dims[..new_shape.len()].copy_from_slice(new_shape);
        Ok(Self { buf: self.buf, dims, ndim: new_shape.len() as u8 })
    }

    /// Get a single dimension size.
    #[inline]
    pub fn dim(&self, i: usize) -> u32 {
        self.dims[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(dev(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_tensor_readback() {
        let data = vec![1.0, 2.0, 3.0];
        let t = Tensor::new(dev(), &data, &[3]).unwrap();
        assert_eq!(t.to_vec(dev()).unwrap(), data);
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::new(dev(), &[1.0; 12], &[3, 4]).unwrap();
        let t2 = t.reshape(&[2, 6]).unwrap();
        assert_eq!(t2.shape(), &[2, 6]);
        assert_eq!(t2.numel(), 12);
    }

    #[test]
    fn test_tensor_reshape_mismatch() {
        let t = Tensor::new(dev(), &[1.0; 12], &[3, 4]).unwrap();
        assert!(t.reshape(&[2, 5]).is_err());
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        assert!(Tensor::new(dev(), &[1.0, 2.0, 3.0], &[2, 2]).is_err());
    }

    #[test]
    fn test_tensor_4d() {
        // NCHW: batch=2, channels=3, height=4, width=5
        let t = Tensor::new(dev(), &[0.0; 120], &[2, 3, 4, 5]).unwrap();
        assert_eq!(t.shape(), &[2, 3, 4, 5]);
        assert_eq!(t.ndim(), 4);
        assert_eq!(t.numel(), 120);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(dev(), &[3, 3]).unwrap();
        let data = t.to_vec(dev()).unwrap();
        assert_eq!(data, vec![0.0; 9]);
    }

    #[test]
    fn test_tensor_scalar() {
        let t = Tensor::new(dev(), &[42.0], &[1]).unwrap();
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.to_vec(dev()).unwrap(), vec![42.0]);
    }
}
