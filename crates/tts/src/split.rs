//! Sentence splitting and token-budget packing.
//!
//! The model cannot speak more than MAX_TOKENS at once, so long text has to
//! be cut somewhere. Cutting mid-phrase is what produces the clicks people
//! blame on the model, so cuts only ever land on sentence terminators; a
//! sentence that is itself too long is refused rather than butchered.

use crate::error::TtsError;

const TERMINATORS: [char; 11] = ['.', '!', '?', ';', ':', '。', '！', '？', '；', '：', '\n'];

/// Break text into sentences, keeping the terminator on the sentence.
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        current.push(c);
        if TERMINATORS.contains(&c) {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

/// Greedily pack tokenized sentences into chunks of at most `budget` tokens.
pub fn pack(sentences: Vec<(String, Vec<i64>)>, budget: usize) -> Result<Vec<Vec<i64>>, TtsError> {
    let mut chunks: Vec<Vec<i64>> = Vec::new();
    let mut current: Vec<i64> = Vec::new();
    for (text, tokens) in sentences {
        if tokens.len() > budget {
            return Err(TtsError::TextTooLong(format!(
                "one sentence is {} tokens, over the {budget} limit — split it yourself: {:?}",
                tokens.len(),
                text.chars().take(40).collect::<String>()
            )));
        }
        if !current.is_empty() && current.len() + tokens.len() > budget {
            chunks.push(std::mem::take(&mut current));
        }
        current.extend(tokens);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_terminators_and_keeps_them() {
        let out = sentences("One. Two! Three?");
        assert_eq!(out, vec!["One.", "Two!", "Three?"]);
    }

    #[test]
    fn splits_on_cjk_punctuation_too() {
        let out = sentences("第一句。第二句！");
        assert_eq!(out, vec!["第一句。", "第二句！"]);
    }

    #[test]
    fn trailing_text_without_a_terminator_still_counts() {
        assert_eq!(sentences("No terminator here"), vec!["No terminator here"]);
    }

    #[test]
    fn blank_input_yields_nothing() {
        assert!(sentences("   \n  ").is_empty());
    }

    #[test]
    fn packs_greedily_up_to_the_budget() {
        let s = |n: usize| ("x".to_string(), vec![0i64; n]);
        // Budget 10: 4+4 fits, the next 4 starts a new chunk.
        let packed = pack(vec![s(4), s(4), s(4)], 10).unwrap();
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].len(), 8);
        assert_eq!(packed[1].len(), 4);
    }

    #[test]
    fn a_single_oversize_sentence_is_an_error_not_a_mid_phrase_cut() {
        let long = ("way too long".to_string(), vec![0i64; 20]);
        let err = pack(vec![long], 10).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, TtsError::TextTooLong(_)), "got: {msg}");
        // The caller needs the real number to re-split its own text.
        assert!(
            msg.contains("20"),
            "error should report the token count: {msg}"
        );
    }
}
