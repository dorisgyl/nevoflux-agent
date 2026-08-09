//! Kokoro's phoneme table.
//!
//! The model was trained against one specific ordering of these symbols, so
//! the table is built by concatenation in exactly that order and an id is
//! simply a position. Reordering any of the four groups silently changes
//! every id after it, which is why the golden-vector test exists.

use std::collections::HashMap;
use std::sync::OnceLock;

const PAD: &str = "$";
const PUNCTUATION: &str = ";:,.!?¡¿—…\"«»“” ";
const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const LETTERS_IPA: &str = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

fn table() -> &'static HashMap<char, i64> {
    static TABLE: OnceLock<HashMap<char, i64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let symbols = format!("{PAD}{PUNCTUATION}{LETTERS}{LETTERS_IPA}");
        symbols
            .chars()
            .enumerate()
            .map(|(idx, c)| (c, idx as i64))
            .collect()
    })
}

/// Turn a phoneme string into model token ids, padded at both ends.
///
/// Characters absent from the table are dropped rather than substituted:
/// a wrong phoneme is audible, a missing one usually is not.
pub fn tokenize(phonemes: &str) -> Vec<i64> {
    let t = table();
    let mut out = Vec::with_capacity(phonemes.chars().count() + 2);
    out.push(0); // '$'
    out.extend(phonemes.chars().filter_map(|c| t.get(&c).copied()));
    out.push(0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vector: the phoneme string and its token ids are both taken
    /// from a known-good Kokoro implementation, so this pins our table to
    /// the one the model was trained with.
    #[test]
    fn golden_vector_matches() {
        let tokens = tokenize("həlˈoʊ, wˈɜːld!");
        assert_eq!(
            tokens,
            vec![0, 50, 83, 54, 156, 57, 135, 3, 16, 65, 156, 87, 158, 54, 46, 5, 0]
        );
    }

    #[test]
    fn pads_both_ends() {
        let tokens = tokenize("");
        assert_eq!(tokens, vec![0, 0], "empty input is still padded");
    }

    #[test]
    fn unknown_chars_are_dropped() {
        // A character outside the table must not shift every id after it.
        assert_eq!(tokenize("a\u{1F600}b"), tokenize("ab"));
    }
}
