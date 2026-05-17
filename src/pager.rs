// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// Sprint 7 step 4: layer paging. t539 = LayerPager owns a persistent host-visible
// staging buffer and streams model tensors into VRAM one at a time.
// f768 = new, f769 = upload, f770 = page_layer.
//
// On discrete GPUs (AMD RX 5700 XT / RADV) MAP_WRITE | COPY_SRC maps to
// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT — the GPU DMA engine reads directly from
// this pinned region, eliminating per-upload malloc/free of staging memory.

use crate::device::{t500, t501};
use crate::safetensors::t538;
use anyhow::Result;
use std::collections::HashMap;

/// Default staging window: 512 MiB — large enough for any single tensor in a
/// 13B-class model (largest FFN weight ≈ 283 MiB at f32, hidden=5120, ff=13824).
pub const DEFAULT_STAGE_BYTES: usize = 512 * 1024 * 1024;

/// t539 = LayerPager. Owns one persistent host-visible (DMA-accessible) staging buffer.
/// Reused across every tensor upload — no per-layer buffer allocation.
/// Automatically chunks uploads that exceed the staging window.
pub struct t539 {
    /// s519 = staging. MAP_WRITE | COPY_SRC buffer in host-visible system RAM.
    s519: wgpu::Buffer,
    /// s520 = cap. Staging capacity in bytes.
    s520: usize,
}

impl t539 {
    /// f768 = LayerPager::new. Allocate `stage_bytes` of host-visible staging RAM.
    /// Tip: `DEFAULT_STAGE_BYTES` (512 MiB) fits any single 13B-class tensor in one chunk.
    pub fn f768(dev: &t500, stage_bytes: usize) -> Self {
        let s519 = dev.s500.create_buffer(&wgpu::BufferDescriptor {
            label: Some("any-gpu/pager/staging"),
            size: stage_bytes as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        t539 { s519, s520: stage_bytes }
    }

    /// f769 = LayerPager::upload. Copy a f32 slice into a freshly-allocated VRAM buffer
    /// through the persistent staging buffer. When `data` is larger than staging capacity,
    /// iterates chunks; each chunk polls GPU completion before reusing staging.
    pub fn f769(&self, dev: &t500, data: &[f32]) -> Result<t501> {
        let dst = dev.f503(data.len());
        let cap_elems = self.s520 / std::mem::size_of::<f32>();

        for (ci, chunk) in data.chunks(cap_elems).enumerate() {
            let byte_off = (ci * cap_elems * std::mem::size_of::<f32>()) as u64;
            let chunk_bytes: &[u8] = bytemuck::cast_slice(chunk);
            let chunk_len = chunk_bytes.len() as u64;

            let slice = self.s519.slice(..chunk_len);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Write, move |r| {
                let _ = tx.send(r);
            });
            dev.s500.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap()?;

            {
                let mut view = slice.get_mapped_range_mut();
                view.copy_from_slice(chunk_bytes);
            }
            self.s519.unmap();

            let mut enc = dev
                .s500
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&self.s519, 0, &dst.s505, byte_off, chunk_len);
            dev.s501.submit(Some(enc.finish()));

            // GPU must finish reading staging before the next chunk remaps it.
            dev.s500.poll(wgpu::Maintain::Wait);
        }

        Ok(dst)
    }

    /// f770 = LayerPager::page_layer. Upload a named set of tensors from a
    /// SafetensorsModel into VRAM. Returns a name→GpuBuffer map ready for compute
    /// dispatch. Drop the map to release all VRAM for those tensors.
    pub fn f770(
        &self,
        dev: &t500,
        model: &t538,
        names: &[&str],
    ) -> Result<HashMap<String, t501>> {
        let mut out = HashMap::with_capacity(names.len());
        for &name in names {
            let data = model
                .f764(name)
                .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in model", name))?;
            out.insert(name.to_owned(), self.f769(dev, data)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, Dtype, TensorView};

    fn dev() -> &'static crate::device::t500 {
        &crate::ops::TEST_DEV
    }

    fn to_bytes(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for &x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    // f768: staging buffer allocation does not panic.
    #[test]
    fn f768_creates_staging() {
        let _p = t539::f768(dev(), 128 * 4);
    }

    // f769: round-trip through staging buffer; staging larger than data (no chunking).
    #[test]
    fn f769_roundtrip() {
        let v0: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
        let p = t539::f768(dev(), 1024 * 4 * 4); // 16 KiB staging
        let buf = p.f769(dev(), &v0).unwrap();
        let v1 = dev().f504(&buf).unwrap();
        assert_eq!(v1.len(), v0.len());
        for (a, b) in v0.iter().zip(v1.iter()) {
            assert_eq!(a, b);
        }
    }

    // f769: chunked upload (staging = 32 f32, tensor = 1024 f32 → 32 chunks).
    // Validates every element including chunk-boundary alignment.
    #[test]
    fn f769_chunked_upload() {
        let v0: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let p = t539::f768(dev(), 32 * 4); // 128 bytes = 32 f32 per chunk
        let buf = p.f769(dev(), &v0).unwrap();
        let v1 = dev().f504(&buf).unwrap();
        assert_eq!(v1.len(), 1024);
        for (i, (&a, &b)) in v0.iter().zip(v1.iter()).enumerate() {
            assert_eq!(a, b, "chunk boundary error at element {i}");
        }
    }

    // f769: single-element upload.
    #[test]
    fn f769_single_element() {
        let p = t539::f768(dev(), 128 * 4);
        let buf = p.f769(dev(), &[3.14f32]).unwrap();
        let v = dev().f504(&buf).unwrap();
        assert!((v[0] - 3.14f32).abs() < 1e-6);
    }

    // f769: tensor size exactly equal to staging capacity (boundary case, no remainder).
    #[test]
    fn f769_exact_staging_fit() {
        let n = 64usize;
        let v0: Vec<f32> = (0..n).map(|i| i as f32 * -0.1).collect();
        let p = t539::f768(dev(), n * 4); // staging = exactly tensor size
        let buf = p.f769(dev(), &v0).unwrap();
        let v1 = dev().f504(&buf).unwrap();
        for (a, b) in v0.iter().zip(v1.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    // f770: uploads two named tensors from a SafetensorsModel into VRAM.
    #[test]
    fn f770_page_layer() {
        let w: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..4).map(|i| i as f32 * 10.0).collect();
        let w_bytes = to_bytes(&w);
        let b_bytes = to_bytes(&b);
        let tensors = vec![
            ("weight", TensorView::new(Dtype::F32, vec![4, 4], &w_bytes).unwrap()),
            ("bias",   TensorView::new(Dtype::F32, vec![4],    &b_bytes).unwrap()),
        ];
        let raw = serialize(tensors, &None).unwrap();
        let model = t538::f761(&raw).unwrap();
        let p = t539::f768(dev(), 1024 * 4);
        let layer = p.f770(dev(), &model, &["weight", "bias"]).unwrap();

        let w_back = dev().f504(&layer["weight"]).unwrap();
        let b_back = dev().f504(&layer["bias"]).unwrap();
        assert_eq!(w_back.len(), 16);
        assert_eq!(b_back.len(), 4);
        for (a, b_val) in w.iter().zip(w_back.iter()) {
            assert!((a - b_val).abs() < 1e-6);
        }
        for (a, b_val) in b.iter().zip(b_back.iter()) {
            assert!((a - b_val).abs() < 1e-6);
        }
    }

    // f770: unknown tensor name returns an error, does not panic.
    #[test]
    fn f770_missing_tensor_error() {
        let one = to_bytes(&[1.0f32]);
        let tensors = vec![("a", TensorView::new(Dtype::F32, vec![1], &one).unwrap())];
        let raw = serialize(tensors, &None).unwrap();
        let model = t538::f761(&raw).unwrap();
        let p = t539::f768(dev(), 128 * 4);
        assert!(p.f770(dev(), &model, &["nope"]).is_err());
    }
}
