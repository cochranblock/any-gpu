// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//
// t544 = Tokenizer. Thin wrapper around the HuggingFace `tokenizers` crate.
// Supports any tokenizer.json (BPE, WordPiece, Unigram, WordLevel) produced
// by HuggingFace tokenizers, covering LLaMA, Mistral, Qwen, Phi, GPT2, etc.
//
// f775 = load, f776 = encode, f777 = decode, f778 = vocab_size, f779 = eos_id.

use anyhow::Result;

/// t544 = Tokenizer. Wraps a HuggingFace tokenizer loaded from `tokenizer.json`.
pub struct t544(tokenizers::Tokenizer);

impl t544 {
    /// f775 = Tokenizer::load. Load from a `tokenizer.json` file path.
    pub fn f775(path: &str) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
        Ok(Self(tok))
    }

    /// f775b = Tokenizer::from_bytes. Load tokenizer.json from raw bytes.
    pub fn f775b(bytes: &[u8]) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("tokenizer parse failed: {e}"))?;
        Ok(Self(tok))
    }

    /// f776 = Tokenizer::encode. Text → token IDs.
    /// `add_special`: whether to add BOS/EOS tokens per the tokenizer config.
    pub fn f776(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let enc = self.0.encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// f777 = Tokenizer::decode. Token IDs → text, skipping special tokens.
    pub fn f777(&self, ids: &[u32]) -> Result<String> {
        self.0.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
    }

    /// f778 = Tokenizer::vocab_size. Includes added special tokens.
    pub fn f778(&self) -> usize {
        self.0.get_vocab_size(true)
    }

    /// f779 = Tokenizer::eos_id. Returns the EOS token id if registered.
    /// Checks common EOS names across model families.
    pub fn f779(&self) -> Option<u32> {
        for name in &["</s>", "<eos>", "<|endoftext|>", "<|im_end|>", "[EOS]", "<EOS>"] {
            if let Some(id) = self.0.token_to_id(name) {
                return Some(id);
            }
        }
        None
    }

    /// Token id → string. Returns None if the id is out of vocabulary.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn make_tok() -> t544 {
        // WordLevel::builder().vocab() requires AHashMap; collect via from_iter.
        let pairs: Vec<(String, u32)> = vec![
            ("<unk>".into(), 0), ("</s>".into(), 1), ("hello".into(), 2),
            ("world".into(), 3), ("foo".into(), 4),
        ];
        let vocab = pairs.into_iter().collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace));
        t544(tok)
    }

    // f776: encode known words returns correct ids.
    #[test]
    fn f776_encode_known() {
        let tok = make_tok();
        let ids = tok.f776("hello world", false).unwrap();
        assert_eq!(ids, vec![2, 3]);
    }

    // f776: unknown word maps to <unk> (id 0).
    #[test]
    fn f776_unknown_word() {
        let tok = make_tok();
        let ids = tok.f776("unknown", false).unwrap();
        assert_eq!(ids, vec![0]);
    }

    // f777: decode round-trips through encode.
    #[test]
    fn f777_roundtrip() {
        let tok = make_tok();
        let ids = tok.f776("hello foo world", false).unwrap();
        let text = tok.f777(&ids).unwrap();
        assert!(text.contains("hello") && text.contains("foo") && text.contains("world"));
    }

    // f778: vocab_size returns at least 5 (our vocab has 5 entries).
    #[test]
    fn f778_vocab_size() {
        let tok = make_tok();
        assert!(tok.f778() >= 5);
    }

    // f779: eos_id finds </s> (id 1) in our vocab.
    #[test]
    fn f779_eos_id() {
        let tok = make_tok();
        assert_eq!(tok.f779(), Some(1));
    }

    // f775: non-existent path returns an error.
    #[test]
    fn f775_bad_path_errors() {
        assert!(t544::f775("/no/such/tokenizer.json").is_err());
    }
}
