// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// NanoSign — 36-byte model file signing. NSIG + BLAKE3 hash.
// Spec: https://github.com/cochranblock/kova/blob/main/docs/NANOSIGN.md
// t510=NanoSignResult. f740..f747 functions.

use anyhow::{Context, Result};
use std::path::Path;

const MAGIC: &[u8; 4] = b"NSIG";
const SIG_LEN: usize = 4 + 32; // magic + BLAKE3 hash

/// t510 = NanoSignResult. Variant names stay descriptive (Verified/Failed/Unsigned).
#[must_use]
#[derive(Debug, PartialEq)]
pub enum t510 {
    Verified(blake3::Hash),
    Failed { expected: [u8; 32], actual: blake3::Hash },
    Unsigned,
}

/// f740 = sign. Append NSIG + BLAKE3 hash (36 bytes) to a file in-place.
pub fn f740(p0: &Path) -> Result<blake3::Hash> {
    let v0 = std::fs::read(p0).with_context(|| format!("read {}", p0.display()))?;
    let v1 = blake3::hash(&v0);
    let mut v2 = std::fs::OpenOptions::new()
        .append(true)
        .open(p0)
        .with_context(|| format!("open for append {}", p0.display()))?;
    std::io::Write::write_all(&mut v2, MAGIC)?;
    std::io::Write::write_all(&mut v2, v1.as_bytes())?;
    Ok(v1)
}

/// f741 = verify. Verify a file's NanoSign signature.
pub fn f741(p0: &Path) -> Result<t510> {
    let v0 = std::fs::read(p0).with_context(|| format!("read {}", p0.display()))?;
    Ok(f742(&v0))
}

/// f742 = verify_bytes. Verify NanoSign signature on in-memory bytes.
pub fn f742(p0: &[u8]) -> t510 {
    if p0.len() < SIG_LEN {
        return t510::Unsigned;
    }
    let (v0, v1) = p0.split_at(p0.len() - SIG_LEN);
    if &v1[..4] != MAGIC {
        return t510::Unsigned;
    }
    let mut v2 = [0u8; 32];
    v2.copy_from_slice(&v1[4..]);
    let v3 = blake3::hash(v0);
    if *v3.as_bytes() == v2 {
        t510::Verified(v3)
    } else {
        t510::Failed { expected: v2, actual: v3 }
    }
}

/// f743 = sign_bytes. Sign in-memory bytes. Returns payload + 36-byte signature appended.
pub fn f743(p0: &[u8]) -> Vec<u8> {
    let v0 = blake3::hash(p0);
    let mut v1 = Vec::with_capacity(p0.len() + SIG_LEN);
    v1.extend_from_slice(p0);
    v1.extend_from_slice(MAGIC);
    v1.extend_from_slice(v0.as_bytes());
    v1
}

/// f744 = strip_bytes. Remove NanoSign signature from in-memory bytes. Returns
/// payload without the 36-byte tail. If unsigned, returns the data unchanged.
pub fn f744(p0: &[u8]) -> &[u8] {
    if p0.len() >= SIG_LEN && &p0[p0.len() - SIG_LEN..p0.len() - 32] == MAGIC {
        &p0[..p0.len() - SIG_LEN]
    } else {
        p0
    }
}

/// f745 = save_signed. Save model weights with NanoSign. Writes data + NSIG + BLAKE3 hash.
pub fn f745(p0: &Path, p1: &[u8]) -> Result<blake3::Hash> {
    let v0 = blake3::hash(p1);
    let mut v1 = Vec::with_capacity(p1.len() + SIG_LEN);
    v1.extend_from_slice(p1);
    v1.extend_from_slice(MAGIC);
    v1.extend_from_slice(v0.as_bytes());
    std::fs::write(p0, &v1).with_context(|| format!("write {}", p0.display()))?;
    Ok(v0)
}

/// f746 = load_verified. Load model weights with NanoSign verification. Returns
/// payload (without signature). Fails if signature is present but invalid
/// (tampered). Unsigned files load with a warning.
pub fn f746(p0: &Path) -> Result<Vec<u8>> {
    let v0 = std::fs::read(p0).with_context(|| format!("read {}", p0.display()))?;
    match f742(&v0) {
        t510::Verified(_) => {
            Ok(v0[..v0.len() - SIG_LEN].to_vec())
        }
        t510::Failed { expected, actual } => {
            anyhow::bail!(
                "NanoSign FAILED for {}: expected {}, got {} — file tampered or corrupted",
                p0.display(),
                f747(&expected),
                actual.to_hex()
            );
        }
        t510::Unsigned => {
            eprintln!("  nanosign: {} is unsigned (no NSIG marker)", p0.display());
            Ok(v0)
        }
    }
}

/// f747 = hex. Hex-format byte slice. Private helper.
fn f747(p0: &[u8]) -> String {
    p0.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f743_and_f742_roundtrip() {
        let v0 = b"model weights go here";
        let v1 = f743(v0);
        assert_eq!(v1.len(), v0.len() + 36);
        assert_eq!(&v1[v1.len() - 36..v1.len() - 32], b"NSIG");
        match f742(&v1) {
            t510::Verified(v2) => {
                assert_eq!(v2, blake3::hash(v0));
            }
            v3 => panic!("expected Verified, got {v3:?}"),
        }
    }

    #[test]
    fn f742_unsigned_detection() {
        let v0 = b"just some random data without a signature";
        assert_eq!(f742(v0), t510::Unsigned);
    }

    #[test]
    fn f742_tamper_detection() {
        let mut v0 = f743(b"original data");
        v0[0] = b'X';
        match f742(&v0) {
            t510::Failed { .. } => {}
            v1 => panic!("expected Failed, got {v1:?}"),
        }
    }

    #[test]
    fn f744_basic() {
        let v0 = b"model data";
        let v1 = f743(v0);
        let v2 = f744(&v1);
        assert_eq!(v2, v0);
    }

    #[test]
    fn f744_unsigned() {
        let v0 = b"no signature here";
        assert_eq!(f744(v0), v0.as_slice());
    }

    #[test]
    fn f742_too_short() {
        let v0 = b"short";
        assert_eq!(f742(v0), t510::Unsigned);
    }

    #[test]
    fn f743_empty_payload() {
        let v0 = f743(b"");
        match f742(&v0) {
            t510::Verified(v1) => {
                assert_eq!(v1, blake3::hash(b""));
            }
            v2 => panic!("expected Verified, got {v2:?}"),
        }
    }

    #[test]
    fn f745_and_f746_roundtrip() {
        let v0 = std::env::temp_dir().join("nanosign_test");
        std::fs::create_dir_all(&v0).unwrap();
        let v1 = v0.join("test_model.weights");

        let v2 = vec![42u8; 1024];
        let v3 = f745(&v1, &v2).unwrap();
        assert_eq!(v3, blake3::hash(&v2));

        let v4 = f746(&v1).unwrap();
        assert_eq!(v4, v2);

        // Tamper and verify rejection
        let mut v5 = std::fs::read(&v1).unwrap();
        v5[0] = 0xFF;
        std::fs::write(&v1, &v5).unwrap();
        assert!(f746(&v1).is_err());

        std::fs::remove_dir_all(&v0).ok();
    }

    #[test]
    fn f743_deterministic() {
        let v0 = b"same data twice";
        let v1 = f743(v0);
        let v2 = f743(v0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn f742_exactly_36_bytes() {
        let mut v0 = vec![0u8; 36];
        v0[0..4].copy_from_slice(b"NSIG");
        match f742(&v0) {
            t510::Failed { .. } => {}
            t510::Verified(_) => panic!("should not verify with wrong hash"),
            t510::Unsigned => panic!("should detect NSIG magic"),
        }
    }

    #[test]
    fn f744_no_false_positive() {
        let mut v0 = vec![0u8; 100];
        v0[10..14].copy_from_slice(b"NSIG");
        let v1 = f744(&v0);
        assert_eq!(v1.len(), 100);
    }

    #[test]
    fn f743_and_f744_roundtrip() {
        let v0 = b"round trip test data";
        let v1 = f743(v0);
        let v2 = f744(&v1);
        assert_eq!(v2, v0);
    }

    #[test]
    fn f746_unsigned_warning() {
        let v0 = std::env::temp_dir().join("nanosign_test_unsigned");
        std::fs::create_dir_all(&v0).unwrap();
        let v1 = v0.join("unsigned.bin");
        std::fs::write(&v1, b"no signature").unwrap();
        let v2 = f746(&v1).unwrap();
        assert_eq!(v2, b"no signature");
        std::fs::remove_dir_all(&v0).ok();
    }
}
