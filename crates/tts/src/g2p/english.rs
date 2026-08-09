//! English G2P via misaki-rs.
//!
//! Built with the espeak fallback off, so nothing native enters the build;
//! the cost is that words outside the lexicon get spelled letter by letter.

use crate::error::TtsError;
use crate::g2p::G2p;
use misaki_rs::{language::Language, G2P as MisakiG2P};

pub struct EnglishG2p {
    inner: MisakiG2P,
}

impl EnglishG2p {
    pub fn new() -> Self {
        EnglishG2p {
            inner: MisakiG2P::new(Language::EnglishUS),
        }
    }
}

impl Default for EnglishG2p {
    fn default() -> Self {
        Self::new()
    }
}

impl G2p for EnglishG2p {
    fn phonemize(&self, text: &str) -> Result<String, TtsError> {
        let (phonemes, _tokens) = self
            .inner
            .g2p(text)
            .map_err(|e| TtsError::InferenceFailed(format!("g2p failed: {e:?}")))?;
        Ok(normalize(&phonemes))
    }
}

/// Bring misaki-rs output in line with the symbol set Kokoro was trained on.
///
/// misaki-rs joins the parts of a diphthong or affricate with U+200D, which
/// is a rendering hint rather than a phoneme and has no id in the table — so
/// left alone it is silently dropped, taking the sound with it. Upstream
/// misaki writes affricates as single ligature characters and diphthongs as
/// bare pairs, so this rewrites to that: `d‍ʒ` becomes `ʤ`, `t‍ʃ` becomes
/// `ʧ`, and any other joiner is simply removed.
fn normalize(phonemes: &str) -> String {
    const ZWJ: char = '\u{200d}';
    phonemes
        .replace("d\u{200d}ʒ", "ʤ")
        .replace("t\u{200d}ʃ", "ʧ")
        .replace(ZWJ, "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_phonemes_for_plain_english() {
        let g = EnglishG2p::new();
        let out = g.phonemize("hello world").unwrap();
        assert!(!out.is_empty(), "should produce phonemes");
        // Stress marks are part of Kokoro's symbol set and misaki emits them.
        assert!(
            out.chars().any(|c| c == 'ˈ'),
            "expected a stress mark in {out:?}"
        );
    }

    #[test]
    fn affricates_become_ligatures_and_stray_joiners_go() {
        assert_eq!(normalize("d\u{200d}ʒˈʌmps"), "ʤˈʌmps");
        assert_eq!(normalize("t\u{200d}ʃɛk"), "ʧɛk");
        assert_eq!(normalize("bɹˈa\u{200d}ʊn"), "bɹˈaʊn");
        assert!(!normalize("o\u{200d}ʊ").contains('\u{200d}'));
    }

    #[test]
    fn phonemes_tokenize_without_loss() {
        // Every symbol misaki emits must exist in the Kokoro table, or the
        // audio silently loses sounds.
        let g = EnglishG2p::new();
        let phonemes = g
            .phonemize("The quick brown fox jumps over the lazy dog.")
            .unwrap();
        let tokens = crate::vocab::tokenize(&phonemes);
        assert_eq!(
            tokens.len(),
            phonemes.chars().count() + 2,
            "some phoneme was dropped by the vocabulary: {phonemes:?}"
        );
    }
}
