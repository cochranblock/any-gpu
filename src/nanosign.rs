// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// NanoSign — 36-byte model file signing. NSIG + BLAKE3 hash.
// Spec: https://github.com/cochranblock/kova/blob/main/docs/NANOSIGN.md

use anyhow::{Context, Result};
use std::path::Path;

const MAGIC: &[u8; 4] = b"NSIG";
const SIG_LEN: usize = 4 + 32; // magic + BLAKE3 hash

#[derive(Debug, PartialEq)]
pub enum NanoSignResult {
    Verified(blake3::Hash),
    Failed { expected: [u8; 32], actual: blake3::Hash },
    Unsigned,
}

/// Sign a file: append NSIG + BLAKE3 hash (36 bytes).
pub fn sign(path: &Path) -> Result<blake3::Hash> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let hash = blake3::hash(&data);
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .with_context(|| format!("open for append {}", path.display()))?;
    std::io::Write::write_all(&mut f, MAGIC)?;
    std::io::Write::write_all(&mut f, hash.as_bytes())?;
    Ok(hash)
}

/// Verify a file's NanoSign signature.
pub fn verify(path: &Path) -> Result<NanoSignResult> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(verify_bytes(&data))
}

/// Verify NanoSign signature on in-memory bytes.
pub fn verify_bytes(data: &[u8]) -> NanoSignResult {
    if data.len() < SIG_LEN {
        return NanoSignResult::Unsigned;
    }
    let (payload, sig) = data.split_at(data.len() - SIG_LEN);
    if &sig[..4] != MAGIC {
        return NanoSignResult::Unsigned;
    }
    let mut expected = [0u8; 32];
    expected.copy_from_slice(&sig[4..]);
    let actual = blake3::hash(payload);
    if *actual.as_bytes() == expected {
        NanoSignResult::Verified(actual)
    } else {
        NanoSignResult::Failed { expected, actual }
    }
}

/// Sign in-memory bytes. Returns payload + 36-byte signature appended.
pub fn sign_bytes(data: &[u8]) -> Vec<u8> {
    let hash = blake3::hash(data);
    let mut out = Vec::with_capacity(data.len() + SIG_LEN);
    out.extend_from_slice(data);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(hash.as_bytes());
    out
}

/// Strip NanoSign signature from in-memory bytes. Returns payload without the 36-byte tail.
/// If unsigned, returns the data unchanged.
pub fn strip_bytes(data: &[u8]) -> &[u8] {
    if data.len() >= SIG_LEN && &data[data.len() - SIG_LEN..data.len() - 32] == MAGIC {
        &data[..data.len() - SIG_LEN]
    } else {
        data
    }
}

/// Save model weights with NanoSign. Writes data + NSIG + BLAKE3 hash.
pub fn save_signed(path: &Path, data: &[u8]) -> Result<blake3::Hash> {
    let hash = blake3::hash(data);
    let mut out = Vec::with_capacity(data.len() + SIG_LEN);
    out.extend_from_slice(data);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(hash.as_bytes());
    std::fs::write(path, &out).with_context(|| format!("write {}", path.display()))?;
    Ok(hash)
}

/// Load model weights with NanoSign verification. Returns payload (without signature).
/// Fails if signature is present but invalid (tampered). Unsigned files load with a warning.
pub fn load_verified(path: &Path) -> Result<Vec<u8>> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    match verify_bytes(&data) {
        NanoSignResult::Verified(_) => {
            Ok(data[..data.len() - SIG_LEN].to_vec())
        }
        NanoSignResult::Failed { expected, actual } => {
            anyhow::bail!(
                "NanoSign FAILED for {}: expected {}, got {} — file tampered or corrupted",
                path.display(),
                hex(&expected),
                actual.to_hex()
            );
        }
        NanoSignResult::Unsigned => {
            eprintln!("  nanosign: {} is unsigned (no NSIG marker)", path.display());
            Ok(data)
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_and_verify_bytes() {
        let payload = b"model weights go here";
        let signed = sign_bytes(payload);
        assert_eq!(signed.len(), payload.len() + 36);
        assert_eq!(&signed[signed.len() - 36..signed.len() - 32], b"NSIG");
        match verify_bytes(&signed) {
            NanoSignResult::Verified(hash) => {
                assert_eq!(hash, blake3::hash(payload));
            }
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    #[test]
    fn test_unsigned_detection() {
        let data = b"just some random data without a signature";
        assert_eq!(verify_bytes(data), NanoSignResult::Unsigned);
    }

    #[test]
    fn test_tamper_detection() {
        let mut signed = sign_bytes(b"original data");
        // Tamper with the payload
        signed[0] = b'X';
        match verify_bytes(&signed) {
            NanoSignResult::Failed { .. } => {} // expected
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn test_strip_bytes() {
        let payload = b"model data";
        let signed = sign_bytes(payload);
        let stripped = strip_bytes(&signed);
        assert_eq!(stripped, payload);
    }

    #[test]
    fn test_strip_unsigned() {
        let data = b"no signature here";
        assert_eq!(strip_bytes(data), data.as_slice());
    }

    #[test]
    fn test_too_short() {
        let data = b"short";
        assert_eq!(verify_bytes(data), NanoSignResult::Unsigned);
    }

    #[test]
    fn test_empty_payload() {
        let signed = sign_bytes(b"");
        match verify_bytes(&signed) {
            NanoSignResult::Verified(hash) => {
                assert_eq!(hash, blake3::hash(b""));
            }
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    #[test]
    fn test_file_roundtrip() {
        let dir = std::env::temp_dir().join("nanosign_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_model.weights");

        let payload = vec![42u8; 1024];
        let hash = save_signed(&path, &payload).unwrap();
        assert_eq!(hash, blake3::hash(&payload));

        let loaded = load_verified(&path).unwrap();
        assert_eq!(loaded, payload);

        // Tamper and verify rejection
        let mut raw = std::fs::read(&path).unwrap();
        raw[0] = 0xFF;
        std::fs::write(&path, &raw).unwrap();
        assert!(load_verified(&path).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_sign_bytes_deterministic() {
        let data = b"same data twice";
        let s1 = sign_bytes(data);
        let s2 = sign_bytes(data);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_verify_exactly_36_bytes() {
        // 36 bytes that happen to start with NSIG but aren't a valid signature
        let mut data = vec![0u8; 36];
        data[0..4].copy_from_slice(b"NSIG");
        // The "hash" is 32 zero bytes, which won't match blake3::hash(b"")
        match verify_bytes(&data) {
            NanoSignResult::Failed { .. } => {} // correct: magic present, hash wrong
            NanoSignResult::Verified(_) => panic!("should not verify with wrong hash"),
            NanoSignResult::Unsigned => panic!("should detect NSIG magic"),
        }
    }

    #[test]
    fn test_strip_bytes_no_false_positive() {
        // Data that contains "NSIG" but not at the right position
        let mut data = vec![0u8; 100];
        data[10..14].copy_from_slice(b"NSIG"); // NSIG in the middle, not at -36
        let stripped = strip_bytes(&data);
        assert_eq!(stripped.len(), 100); // should not strip
    }

    #[test]
    fn test_sign_and_strip_roundtrip() {
        let payload = b"round trip test data";
        let signed = sign_bytes(payload);
        let stripped = strip_bytes(&signed);
        assert_eq!(stripped, payload);
    }

    #[test]
    fn test_load_verified_unsigned_warning() {
        // Unsigned file should load successfully (with warning to stderr)
        let dir = std::env::temp_dir().join("nanosign_test_unsigned");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unsigned.bin");
        std::fs::write(&path, b"no signature").unwrap();
        let data = load_verified(&path).unwrap();
        assert_eq!(data, b"no signature");
        std::fs::remove_dir_all(&dir).ok();
    }
}
