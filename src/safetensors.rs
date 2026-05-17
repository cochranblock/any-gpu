// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Sprint 7 step 3: safetensors loader. Reads .safetensors files (the HuggingFace
// standard format), dequantizes bf16/f16 blocks to f32, and exposes both the
// CPU-resident f32 weights and an on-demand GPU upload. Per the project's
// safetensors-only mandate: no GGUF, no PyTorch .bin, no ONNX.
//
// Public API:
//   t538 = SafetensorsModel — a loaded file, weights in RAM
//   f760 = load (path → t538, signature-aware via nanosign f746)
//   f761 = from_bytes (raw bytes → t538)
//   f762 = names, f763 = shape, f764 = data, f765 = upload
//   f766 = bf16_to_f32, f767 = f16_to_f32 (free fns; useful outside this module too)

use crate::device::{t500, t501};
use anyhow::{bail, Context, Result};
use safetensors::tensor::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

/// t538 = SafetensorsModel. Weights live in CPU RAM as f32; upload to GPU on demand.
/// Dequantization (bf16/f16 → f32) happens once at load time.
pub struct t538 {
    pub(crate) s517: HashMap<String, Vec<f32>>,
    pub(crate) s518: HashMap<String, Vec<u32>>,
}

impl t538 {
    /// f760 = SafetensorsModel::load. Read a safetensors file from disk and
    /// dequantize all tensors to f32 in RAM. Passes through `nanosign::f746`,
    /// so NanoSign-signed files are integrity-verified and unsigned files
    /// load as-is (matching the existing `load_verified` semantics).
    pub fn f760(p0: &Path) -> Result<Self> {
        let v0 = crate::nanosign::f746(p0)
            .with_context(|| format!("nanosign verify failed for {}", p0.display()))?;
        Self::f761(&v0)
    }

    /// f761 = SafetensorsModel::from_bytes. Parse raw safetensors bytes (header + blocks).
    /// Useful when bytes are already in memory — streamed download, embedded resource,
    /// or a NanoSign-verified buffer the caller already obtained.
    pub fn f761(p0: &[u8]) -> Result<Self> {
        let v0 = SafeTensors::deserialize(p0)
            .context("safetensors: failed to deserialize header/blocks")?;

        let mut v1: HashMap<String, Vec<f32>> = HashMap::new();
        let mut v2: HashMap<String, Vec<u32>> = HashMap::new();

        for (v3, v4) in v0.tensors() {
            let v5 = v4.shape();
            let v6 = v4.data();
            let v7: Vec<u32> = v5.iter().map(|&p| p as u32).collect();
            let v8: usize = v5.iter().product();

            let v9: Vec<f32> = match v4.dtype() {
                Dtype::F32 => {
                    // safetensors stores little-endian. Parse one f32 at a time
                    // — the byte slice from a freshly-deserialized header is not
                    // guaranteed to be 4-byte aligned, so a direct bytemuck cast
                    // can fail on strict-alignment targets.
                    if v6.len() != v8 * 4 {
                        bail!("safetensors: tensor `{v3}` size mismatch (got {} bytes, expected {})",
                            v6.len(), v8 * 4);
                    }
                    (0..v8)
                        .map(|v10| {
                            let v11 = v10 * 4;
                            f32::from_le_bytes([v6[v11], v6[v11+1], v6[v11+2], v6[v11+3]])
                        })
                        .collect()
                }
                Dtype::BF16 => {
                    if v6.len() != v8 * 2 {
                        bail!("safetensors: tensor `{v3}` bf16 size mismatch (got {} bytes, expected {})",
                            v6.len(), v8 * 2);
                    }
                    (0..v8)
                        .map(|v10| {
                            let v11 = v10 * 2;
                            f766(u16::from_le_bytes([v6[v11], v6[v11+1]]))
                        })
                        .collect()
                }
                Dtype::F16 => {
                    if v6.len() != v8 * 2 {
                        bail!("safetensors: tensor `{v3}` f16 size mismatch (got {} bytes, expected {})",
                            v6.len(), v8 * 2);
                    }
                    (0..v8)
                        .map(|v10| {
                            let v11 = v10 * 2;
                            f767(u16::from_le_bytes([v6[v11], v6[v11+1]]))
                        })
                        .collect()
                }
                dt => bail!(
                    "safetensors: tensor `{v3}` has unsupported dtype {:?} (Sprint 7 step 3 covers F32/BF16/F16 only)",
                    dt
                ),
            };

            v1.insert(v3.to_string(), v9);
            v2.insert(v3.to_string(), v7);
        }

        Ok(Self { s517: v1, s518: v2 })
    }

    /// f762 = SafetensorsModel::names. All tensor names in the loaded file.
    pub fn f762(&self) -> Vec<&str> {
        self.s517.keys().map(|p| p.as_str()).collect()
    }

    /// f763 = SafetensorsModel::shape. Shape of the named tensor, or None if absent.
    pub fn f763(&self, p0: &str) -> Option<&[u32]> {
        self.s518.get(p0).map(|p| p.as_slice())
    }

    /// f764 = SafetensorsModel::data. CPU-resident f32 data of the named tensor.
    pub fn f764(&self, p0: &str) -> Option<&[f32]> {
        self.s517.get(p0).map(|p| p.as_slice())
    }

    /// f765 = SafetensorsModel::upload. Upload the named tensor to GPU and return
    /// the resulting buffer. Caller pairs this with `f763` for shape.
    pub fn f765(&self, p0: &t500, p1: &str) -> Result<t501> {
        let v0 = self.f764(p1)
            .with_context(|| format!("safetensors: tensor `{p1}` not present in this model"))?;
        Ok(p0.f502(v0))
    }
}

/// f766 = bf16_to_f32. Converts a bfloat16 bit pattern (high 16 bits of an f32)
/// into the corresponding f32. Lossless — bf16 IS the top half of an f32.
pub fn f766(p0: u16) -> f32 {
    f32::from_bits((p0 as u32) << 16)
}

/// f767 = f16_to_f32. Converts an IEEE-754 half-precision (binary16) bit
/// pattern into the corresponding f32. Handles normals, denormals, zeros,
/// infinities, and NaNs.
pub fn f767(p0: u16) -> f32 {
    let v0 = ((p0 as u32) & 0x8000) << 16; // sign bit moved to position 31
    let v1 = (p0 as u32 >> 10) & 0x1f;     // 5-bit exponent
    let v2 = (p0 as u32) & 0x3ff;          // 10-bit mantissa

    let v3: u32 = if v1 == 0 {
        if v2 == 0 {
            0 // ±0 — sign added below
        } else {
            // Denormal — normalize: shift mantissa until the implicit leading 1 is at bit 10.
            let mut v4 = v2;
            let mut v5: i32 = -14;
            while (v4 & 0x400) == 0 {
                v4 <<= 1;
                v5 -= 1;
            }
            v4 &= 0x3ff;
            (((v5 + 127) as u32) << 23) | (v4 << 13)
        }
    } else if v1 == 31 {
        // Inf or NaN — preserve mantissa bits.
        (0xff << 23) | (v2 << 13)
    } else {
        // Normal — re-bias the exponent and pad the mantissa.
        (((v1 as i32 - 15 + 127) as u32) << 23) | (v2 << 13)
    };

    f32::from_bits(v0 | v3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::f544;
    use safetensors::tensor::{serialize, TensorView};

    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

    // --- f766 = bf16_to_f32 (hardcoded reference values) ---

    #[test]
    fn f766_known_values() {
        // bf16 is the top half of f32. Lossless conversion.
        assert_eq!(f766(0x0000), 0.0f32);
        assert_eq!(f766(0x8000), -0.0f32);
        assert_eq!(f766(0x3f80), 1.0f32);
        assert_eq!(f766(0xbf80), -1.0f32);
        assert_eq!(f766(0x4000), 2.0f32);
        assert_eq!(f766(0x3f00), 0.5f32);
        assert_eq!(f766(0x4040), 3.0f32);
    }

    #[test]
    fn f766_round_trip_via_truncation() {
        // f32 -> bf16 (drop low 16 mantissa bits) -> bf16_to_f32 should recover an
        // approximation within 1 ULP of bf16 (~0.4% relative error). For values
        // that fit exactly in bf16 (powers of 2 within range), it's lossless.
        for &v0 in &[1.0f32, 2.0, 4.0, 0.5, 0.25, -1.0, -2.0] {
            let v1 = (v0.to_bits() >> 16) as u16;
            let v2 = f766(v1);
            assert_eq!(v2, v0, "exact roundtrip failed for {v0}");
        }
    }

    // --- f767 = f16_to_f32 (hardcoded reference values from the IEEE spec) ---

    #[test]
    fn f767_zero_and_sign() {
        assert_eq!(f767(0x0000), 0.0f32);
        assert_eq!(f767(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn f767_known_normals() {
        // f16(1.0) = 0x3c00, f16(2.0) = 0x4000, f16(-1.0) = 0xbc00, f16(0.5) = 0x3800
        assert_eq!(f767(0x3c00), 1.0f32);
        assert_eq!(f767(0x4000), 2.0f32);
        assert_eq!(f767(0xbc00), -1.0f32);
        assert_eq!(f767(0x3800), 0.5f32);
        assert_eq!(f767(0x4200), 3.0f32);
    }

    #[test]
    fn f767_smallest_denormal() {
        // 0x0001 = 2^-24 ≈ 5.9604645e-8 (smallest positive subnormal in f16)
        let v0 = f767(0x0001);
        assert!((v0 - 5.9604645e-8).abs() < 1e-12, "got {v0}");
    }

    #[test]
    fn f767_largest_denormal() {
        // 0x03ff (mantissa all-ones, exp=0) = (1023 / 1024) * 2^-14
        let v0 = f767(0x03ff);
        let v1 = (1023.0_f32 / 1024.0) * (2.0_f32).powi(-14);
        assert!((v0 - v1).abs() < 1e-12, "got {v0} expected {v1}");
    }

    #[test]
    fn f767_infinity() {
        assert!(f767(0x7c00).is_infinite() && f767(0x7c00) > 0.0);
        assert!(f767(0xfc00).is_infinite() && f767(0xfc00) < 0.0);
    }

    #[test]
    fn f767_nan() {
        // 0x7e00 (exp=31, mantissa nonzero) — quiet NaN
        assert!(f767(0x7e00).is_nan());
    }

    // --- f761 = from_bytes ---

    fn build_f32_safetensors(p0: &[(&str, Vec<u32>, Vec<f32>)]) -> Vec<u8> {
        let mut v0: Vec<(String, Vec<u32>, Vec<f32>)> = Vec::new();
        for (v1, v2, v3) in p0 {
            v0.push((v1.to_string(), v2.clone(), v3.clone()));
        }
        // safetensors crate wants Vec<u8> for data — convert f32 vectors to little-endian bytes.
        let mut v4: Vec<(String, Vec<u32>, Vec<u8>)> = Vec::new();
        for (v1, v2, v3) in v0 {
            let mut v5 = Vec::with_capacity(v3.len() * 4);
            for v6 in &v3 {
                v5.extend_from_slice(&v6.to_le_bytes());
            }
            v4.push((v1, v2, v5));
        }
        // Build the serializable views borrowing from v4.
        let v7: Vec<(&str, TensorView)> = v4
            .iter()
            .map(|(v1, v2, v3)| {
                let v8: Vec<usize> = v2.iter().map(|&p| p as usize).collect();
                (v1.as_str(), TensorView::new(Dtype::F32, v8, v3.as_slice()).unwrap())
            })
            .collect();
        serialize(v7, &None).unwrap()
    }

    #[test]
    fn f761_single_f32_tensor() {
        let v0 = build_f32_safetensors(&[("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
        let v1 = t538::f761(&v0).unwrap();
        assert_eq!(v1.f763("w"), Some(&[2u32, 3][..]));
        assert_eq!(v1.f764("w"), Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..]));
    }

    #[test]
    fn f761_multi_f32_tensors() {
        let v0 = build_f32_safetensors(&[
            ("embed", vec![10, 4], vec![0.5; 40]),
            ("ln_w",  vec![4],     vec![1.0, 1.0, 1.0, 1.0]),
            ("ln_b",  vec![4],     vec![0.0, 0.0, 0.0, 0.0]),
        ]);
        let v1 = t538::f761(&v0).unwrap();
        let mut v2 = v1.f762();
        v2.sort();
        assert_eq!(v2, vec!["embed", "ln_b", "ln_w"]);
        assert_eq!(v1.f763("embed"), Some(&[10u32, 4][..]));
        assert_eq!(v1.f764("ln_w"), Some(&[1.0, 1.0, 1.0, 1.0][..]));
        assert_eq!(v1.f764("ln_b"), Some(&[0.0, 0.0, 0.0, 0.0][..]));
    }

    #[test]
    fn f761_bf16_dequantizes_to_f32() {
        // Build a safetensors file with a BF16 tensor containing known bit patterns,
        // then load and verify dequant matches f766 directly.
        let v0: Vec<u16> = vec![0x3f80, 0x4000, 0xbf80, 0x0000, 0x3f00]; // 1, 2, -1, 0, 0.5
        let mut v1 = Vec::with_capacity(v0.len() * 2);
        for &v2 in &v0 { v1.extend_from_slice(&v2.to_le_bytes()); }
        let v3 = vec![("w", TensorView::new(Dtype::BF16, vec![5], &v1).unwrap())];
        let v4 = serialize(v3, &None).unwrap();
        let v5 = t538::f761(&v4).unwrap();
        let v6 = v5.f764("w").unwrap();
        assert_eq!(v6, &[1.0, 2.0, -1.0, 0.0, 0.5]);
    }

    #[test]
    fn f761_f16_dequantizes_to_f32() {
        let v0: Vec<u16> = vec![0x3c00, 0x4000, 0xbc00, 0x0000, 0x3800]; // 1, 2, -1, 0, 0.5
        let mut v1 = Vec::with_capacity(v0.len() * 2);
        for &v2 in &v0 { v1.extend_from_slice(&v2.to_le_bytes()); }
        let v3 = vec![("w", TensorView::new(Dtype::F16, vec![5], &v1).unwrap())];
        let v4 = serialize(v3, &None).unwrap();
        let v5 = t538::f761(&v4).unwrap();
        let v6 = v5.f764("w").unwrap();
        assert_eq!(v6, &[1.0, 2.0, -1.0, 0.0, 0.5]);
    }

    #[test]
    fn f761_missing_tensor_returns_none() {
        let v0 = build_f32_safetensors(&[("w", vec![2], vec![1.0, 2.0])]);
        let v1 = t538::f761(&v0).unwrap();
        assert!(v1.f764("does_not_exist").is_none());
        assert!(v1.f763("does_not_exist").is_none());
    }

    #[test]
    fn f761_unsupported_dtype_errors() {
        // I32 isn't a transformer-weight dtype we handle in step 3.
        let v0: Vec<i32> = vec![1, 2, 3];
        let mut v1 = Vec::with_capacity(v0.len() * 4);
        for &v2 in &v0 { v1.extend_from_slice(&v2.to_le_bytes()); }
        let v3 = vec![("counts", TensorView::new(Dtype::I32, vec![3], &v1).unwrap())];
        let v4 = serialize(v3, &None).unwrap();
        let v5 = t538::f761(&v4);
        let v6 = match v5 {
            Err(p) => format!("{p}"),
            Ok(_) => panic!("I32 should fail load"),
        };
        assert!(v6.contains("unsupported dtype"), "error message should name the issue: {v6}");
    }

    // --- f765 = upload ---

    #[test]
    fn f765_uploads_to_gpu() {
        let v0 = build_f32_safetensors(&[("w", vec![4], vec![7.0, 8.0, 9.0, 10.0])]);
        let v1 = t538::f761(&v0).unwrap();
        let v2 = v1.f765(dev(), "w").unwrap();
        let v3 = dev().f504(&v2).unwrap();
        f544(&v3, &[7.0, 8.0, 9.0, 10.0], 1e-6);
    }

    #[test]
    fn f765_missing_tensor_errors() {
        let v0 = build_f32_safetensors(&[("w", vec![1], vec![1.0])]);
        let v1 = t538::f761(&v0).unwrap();
        assert!(v1.f765(dev(), "nope").is_err());
    }

    // --- f760 = load from disk (NanoSign-aware path) ---

    #[test]
    fn f760_disk_round_trip_unsigned() {
        // Write a vanilla (unsigned) safetensors file to a temp path, load via f760.
        // f746 (load_verified) treats unsigned files as Verified(content), so the
        // load succeeds and we can read the tensor back.
        let v0 = build_f32_safetensors(&[("layer.0.weight", vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])]);
        let v1 = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(v1.path(), &v0).unwrap();
        let v2 = t538::f760(v1.path()).unwrap();
        assert_eq!(v2.f763("layer.0.weight"), Some(&[2u32, 2][..]));
        f544(v2.f764("layer.0.weight").unwrap(), &[0.1, 0.2, 0.3, 0.4], 1e-6);
    }

    #[test]
    fn f760_disk_round_trip_nanosigned() {
        // Sign the safetensors bytes via nanosign f745, then load via f760.
        // f760 -> f746 verifies the signature, returns the inner safetensors bytes,
        // and we parse them normally.
        let v0 = build_f32_safetensors(&[("w", vec![3], vec![11.0, 22.0, 33.0])]);
        let v1 = tempfile::NamedTempFile::new().unwrap();
        crate::nanosign::f745(v1.path(), &v0).unwrap();
        let v2 = t538::f760(v1.path()).unwrap();
        f544(v2.f764("w").unwrap(), &[11.0, 22.0, 33.0], 1e-6);
    }

    #[test]
    fn f760_disk_tampered_signature_errors() {
        // Sign the file, then corrupt one byte mid-content. Loading must fail.
        let v0 = build_f32_safetensors(&[("w", vec![1], vec![1.0])]);
        let v1 = tempfile::NamedTempFile::new().unwrap();
        crate::nanosign::f745(v1.path(), &v0).unwrap();

        let mut v2 = std::fs::read(v1.path()).unwrap();
        // Flip a bit somewhere in the payload (not the NSIG trailer).
        // Payload is everything except the last 36 bytes (NSIG + 32-byte BLAKE3).
        let v3 = v2.len() - 50; // pick a stable offset inside payload
        v2[v3] ^= 0x01;
        std::fs::write(v1.path(), &v2).unwrap();

        let v4 = t538::f760(v1.path());
        assert!(v4.is_err(), "tampered signed file must fail to load");
    }
}
