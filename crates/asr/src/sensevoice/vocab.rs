//! The token table, and turning token ids back into text.

use crate::error::AsrError;
use std::path::Path;

/// SentencePiece writes a word-initial space as U+2581.
const SPACE_MARK: char = '\u{2581}';

/// `tokens.txt`, one `<token> <id>` per line, indexed by id.
pub struct Vocab {
    tokens: Vec<String>,
}

impl Vocab {
    pub fn load(path: &Path) -> Result<Vocab, AsrError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| AsrError::ModelNotFound(format!("{}: {e}", path.display())))?;
        let mut tokens: Vec<String> = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                continue;
            }
            // Split from the right: the token itself may contain spaces.
            let (tok, id) = line.rsplit_once(' ').ok_or_else(|| {
                AsrError::ModelCorrupt(format!(
                    "{}:{}: expected `<token> <id>`, got {line:?}",
                    path.display(),
                    lineno + 1
                ))
            })?;
            let id: usize = id.trim().parse().map_err(|e| {
                AsrError::ModelCorrupt(format!(
                    "{}:{}: id {id:?} is not a number: {e}",
                    path.display(),
                    lineno + 1
                ))
            })?;
            if id >= tokens.len() {
                tokens.resize(id + 1, String::new());
            }
            tokens[id] = tok.to_string();
        }
        if tokens.is_empty() {
            return Err(AsrError::ModelCorrupt(format!(
                "{} is empty",
                path.display()
            )));
        }
        Ok(Vocab { tokens })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn get(&self, id: usize) -> &str {
        self.tokens.get(id).map(String::as_str).unwrap_or("")
    }

    /// Join token pieces into readable text.
    ///
    /// U+2581 marks a word-initial space, which is how the English side of
    /// this vocabulary encodes word boundaries; Chinese pieces carry no mark
    /// and must not gain spaces between them. Joining with plain spaces would
    /// space out every Chinese character, and joining with nothing would run
    /// English words together -- the mark is what distinguishes the two.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for id in ids {
            let piece = self.get(*id);
            if let Some(rest) = piece.strip_prefix(SPACE_MARK) {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(rest);
            } else {
                out.push_str(piece);
            }
        }
        out.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_vocab(lines: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(lines.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn loads_tokens_indexed_by_id() {
        let f = write_vocab("<unk> 0\n<s> 1\n</s> 2\n\u{2581}hello 3\n世 4\n");
        let v = Vocab::load(f.path()).unwrap();
        assert_eq!(v.len(), 5);
        assert_eq!(v.get(0), "<unk>");
        assert_eq!(v.get(4), "世");
    }

    #[test]
    fn out_of_range_ids_are_empty_not_a_panic() {
        let f = write_vocab("<unk> 0\n");
        let v = Vocab::load(f.path()).unwrap();
        assert_eq!(v.get(9999), "");
    }

    #[test]
    fn chinese_pieces_join_without_spaces() {
        let f = write_vocab("<unk> 0\n你 1\n好 2\n世 3\n界 4\n");
        let v = Vocab::load(f.path()).unwrap();
        assert_eq!(v.decode(&[1, 2, 3, 4]), "你好世界");
    }

    #[test]
    fn english_word_marks_become_spaces() {
        let f = write_vocab("<unk> 0\n\u{2581}hello 1\n\u{2581}world 2\n!\u{0020}3\n");
        let v = Vocab::load(f.path()).unwrap();
        assert_eq!(v.decode(&[1, 2]), "hello world");
    }

    #[test]
    fn subword_pieces_stay_glued_to_their_word() {
        let f = write_vocab("<unk> 0\n\u{2581}trans 1\ncri 2\nbe 3\n");
        let v = Vocab::load(f.path()).unwrap();
        assert_eq!(v.decode(&[1, 2, 3]), "transcribe");
    }

    #[test]
    fn mixed_chinese_and_english_keeps_both_conventions() {
        let f = write_vocab("<unk> 0\n今 1\n天 2\n\u{2581}OK 3\n吗 4\n");
        let v = Vocab::load(f.path()).unwrap();
        assert_eq!(v.decode(&[1, 2, 3, 4]), "今天 OK吗");
    }

    #[test]
    fn a_line_without_an_id_is_rejected() {
        let f = write_vocab("<unk>\n");
        assert!(matches!(
            Vocab::load(f.path()),
            Err(AsrError::ModelCorrupt(_))
        ));
    }

    #[test]
    fn a_non_numeric_id_is_rejected() {
        let f = write_vocab("<unk> zero\n");
        assert!(matches!(
            Vocab::load(f.path()),
            Err(AsrError::ModelCorrupt(_))
        ));
    }

    #[test]
    fn the_real_vocabulary_loads_with_the_size_the_model_states() {
        // vocab_size in the ONNX metadata is 25055; tokens.txt must agree, or
        // every id past the shorter of the two decodes to nothing.
        let Some(p) = crate::ort_env::default_model_dir() else {
            return;
        };
        let p = p.join("sensevoice-tokens.txt");
        if !p.exists() {
            return; // `just fetch-asr-models` has not run here
        }
        let v = Vocab::load(&p).unwrap();
        assert_eq!(v.len(), 25055);
        assert_eq!(v.get(0), "<unk>");
    }
}
