// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// Sprint 7 step 4: layer paging. t539 = LayerPager owns a persistent host-visible
// staging buffer and streams model tensors into VRAM one at a time.
// f768 = new, f769 = upload (f32), f770 = page_layer (f32).
// f773 = upload_f16_raw (&[u16] → t540), f774 = page_layer_f16 (Sprint 7 step 5).
//
// On discrete GPUs (AMD RX 5700 XT / RADV) MAP_WRITE | COPY_SRC maps to
// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT — the GPU DMA engine reads directly from
// this pinned region, eliminating per-upload malloc/free of staging memory.

use crate::device::{t500, t501, t540};
use crate::safetensors::t538;
use anyhow::Result;
use half::f16 as F16;
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
        t539 {
            s519,
            s520: stage_bytes,
        }
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
    pub fn f770(&self, dev: &t500, model: &t538, names: &[&str]) -> Result<HashMap<String, t501>> {
        let mut out = HashMap::with_capacity(names.len());
        for &name in names {
            let data = model
                .f764(name)
                .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in model", name))?;
            out.insert(name.to_owned(), self.f769(dev, data)?);
        }
        Ok(out)
    }

    /// f773 = LayerPager::upload_f16_raw. Upload raw f16 bits (&[u16]) into a VRAM
    /// f16 buffer via the persistent staging buffer. Same chunked strategy as f769.
    /// The f16 elements are packed as u32 pairs (lower/upper 16 bits) for f772.
    pub fn f773(&self, dev: &t500, data: &[u16]) -> Result<t540> {
        let dst = dev.alloc_f16(data.len());
        let cap_elems = self.s520 / 2; // staging capacity in f16 elements (2 bytes each)

        for (ci, chunk) in data.chunks(cap_elems).enumerate() {
            let byte_off = (ci * cap_elems * 2) as u64;
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
            enc.copy_buffer_to_buffer(&self.s519, 0, &dst.s522, byte_off, chunk_len);
            dev.s501.submit(Some(enc.finish()));
            dev.s500.poll(wgpu::Maintain::Wait);
        }

        Ok(dst)
    }

    /// f774 = LayerPager::page_layer_f16. Upload named tensors from a SafetensorsModel
    /// into VRAM as f16. Model weights (stored as f32 in t538) are quantized f32→f16 on
    /// the CPU before staging, halving VRAM bandwidth during inference.
    /// Returns a name→t540 map. Expand to f32 for computation via f772.
    pub fn f774(&self, dev: &t500, model: &t538, names: &[&str]) -> Result<HashMap<String, t540>> {
        let mut out = HashMap::with_capacity(names.len());
        for &name in names {
            let f32_data = model
                .f764(name)
                .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in model", name))?;
            let f16_bits: Vec<u16> = f32_data
                .iter()
                .map(|&v| F16::from_f32(v).to_bits())
                .collect();
            out.insert(name.to_owned(), self.f773(dev, &f16_bits)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView, serialize};

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
            (
                "weight",
                TensorView::new(Dtype::F32, vec![4, 4], &w_bytes).unwrap(),
            ),
            (
                "bias",
                TensorView::new(Dtype::F32, vec![4], &b_bytes).unwrap(),
            ),
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

    // f773: upload f16 bits via staging, expand to f32 via f772, verify round-trip.
    #[test]
    fn f773_roundtrip_via_f772() {
        let f32_vals: Vec<f32> = vec![1.0, 2.0, -1.0, 0.5, 0.0, 100.0, -0.25, 0.125];
        let f16_bits: Vec<u16> = f32_vals
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let p = t539::f768(dev(), 1024 * 2);
        let buf = p.f773(dev(), &f16_bits).unwrap();
        assert_eq!(buf.s524, 8); // 8 elements, even
        assert_eq!(buf.s523, 16); // ceil(8/2)*4 = 16 bytes
        let expanded = dev().f772(&buf).unwrap();
        let back = dev().f504(&expanded).unwrap();
        assert_eq!(back.len(), 8);
        for (&a, &b) in f32_vals.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01, "f16 round-trip: {a} vs {b}");
        }
    }

    // f773: chunked f16 upload (staging = 16 f16 elements = 32 bytes, tensor = 256).
    // Staging must be a multiple of 4 bytes for correct u32 alignment at chunk boundaries.
    #[test]
    fn f773_chunked_f16_upload() {
        let f32_vals: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let f16_bits: Vec<u16> = f32_vals
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let p = t539::f768(dev(), 32 * 2); // 32 bytes = 16 f16 per chunk
        let buf = p.f773(dev(), &f16_bits).unwrap();
        let expanded = dev().f772(&buf).unwrap();
        let back = dev().f504(&expanded).unwrap();
        assert_eq!(back.len(), 256);
        for (i, (&a, &b)) in f32_vals.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.01,
                "chunk boundary error at {i}: {a} vs {b}"
            );
        }
    }

    // f774: upload f32 model tensors as f16 into VRAM via page_layer_f16.
    #[test]
    fn f774_page_layer_f16() {
        let w: Vec<f32> = vec![1.0, 2.0, 4.0, 8.0, 0.5, -0.5, 0.25, -0.25];
        let w_bytes = to_bytes(&w);
        let tensors = vec![(
            "weight",
            TensorView::new(Dtype::F32, vec![8], &w_bytes).unwrap(),
        )];
        let raw = serialize(tensors, &None).unwrap();
        let model = t538::f761(&raw).unwrap();

        let p = t539::f768(dev(), 1024 * 2);
        let layer = p.f774(dev(), &model, &["weight"]).unwrap();
        let buf = &layer["weight"];
        assert_eq!(buf.s524, 8);
        assert_eq!(buf.s523, 16); // ceil(8/2)*4 = 16 bytes

        let expanded = dev().f772(buf).unwrap();
        let back = dev().f504(&expanded).unwrap();
        assert_eq!(back.len(), 8);
        for (&a, &b) in w.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01, "f774 f32→f16→f32: {a} vs {b}");
        }
    }
}
