//! Kokoro's phoneme table.
//!
//! The model was trained against one specific ordering of these symbols, so
//! the table is built by concatenation in exactly that order and an id is
//! simply a position. Reordering any of the four groups silently changes
//! every id after it, which is why the golden-vector test exists.

use crate::error::TtsError;
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
    tokenize_with(table(), phonemes)
}

fn tokenize_with(t: &HashMap<char, i64>, phonemes: &str) -> Vec<i64> {
    let mut out = Vec::with_capacity(phonemes.chars().count() + 2);
    out.push(0); // '$'
    out.extend(phonemes.chars().filter_map(|c| t.get(&c).copied()));
    out.push(0);
    out
}

/// 一个发行版的音素表。
///
/// **必须与模型配套**,而且配不上是**静默**的:v1.1-zh 的词表把字母重新编了号,
/// 用 v1.0 的表去编码,注音符号全部落空、字母全部错位 —— 出来的不是报错,是一段
/// 半秒钟的杂音。实测就是这么发现的:39 个字和 7 个字都只合成出 0.5 秒。
pub struct Vocab {
    table: HashMap<char, i64>,
}

impl Vocab {
    /// 内置的 v1.0 英文表。
    pub fn builtin() -> Vocab {
        Vocab {
            table: table().clone(),
        }
    }

    /// 从模型自带的 `tokenizer.json` 读。
    ///
    /// 让模型带着自己的词表,比在代码里为每个发行版维护一张表可靠 —— 后者迟早
    /// 会漏掉一个新发行版,而漏掉的表现是杂音,不是编译错误。
    pub fn from_tokenizer_json(path: &std::path::Path) -> Result<Vocab, TtsError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| TtsError::ModelCorrupt(format!("{}: {e}", path.display())))?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| TtsError::ModelCorrupt(format!("{}: {e}", path.display())))?;
        let map = parsed
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                TtsError::ModelCorrupt(format!("{}: 没有 model.vocab", path.display()))
            })?;
        let mut table = HashMap::with_capacity(map.len());
        for (k, v) in map {
            let mut chars = k.chars();
            let (Some(c), None) = (chars.next(), chars.next()) else {
                continue; // 多字符条目不是音素
            };
            if let Some(id) = v.as_i64() {
                table.insert(c, id);
            }
        }
        if table.is_empty() {
            return Err(TtsError::ModelCorrupt(format!(
                "{}: 词表里一个单字符音素都没有",
                path.display()
            )));
        }
        Ok(Vocab { table })
    }

    pub fn tokenize(&self, phonemes: &str) -> Vec<i64> {
        tokenize_with(&self.table, phonemes)
    }

    /// 这张表不认识的音素,去重后按出现顺序给出。
    ///
    /// 丢弃本身是对的(见 [`tokenize`]),但**丢光了**是另一回事:那说明模型和
    /// 文本根本不是一套。这个方法存在的唯一理由,是让那句话能被说出来 ——
    /// 「你的词表里没有 ㄋ ㄧ ㄏ」远比半秒钟的空白有用。
    ///
    /// `limit` 是取样上限:诊断信息要能读,不是要完整。
    pub fn unknown_chars(&self, phonemes: &str, limit: usize) -> Vec<char> {
        let mut out: Vec<char> = Vec::new();
        for c in phonemes.chars() {
            if out.len() >= limit {
                break;
            }
            if !self.table.contains_key(&c) && !out.contains(&c) {
                out.push(c);
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
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

    /// 丢弃是对的,**丢光**要说得出是哪些符号。
    ///
    /// 这就是「中文没声音」那两轮排查缺的那句话:内置表(v1.0)里没有注音符号,
    /// 于是整句中文的每一个音素都落空 —— 而 [`tokenize`] 两端各补一个 0,拿到的
    /// 是 `[0, 0]` 而不是空表,所以「有没有内容」不能用 is_empty 判。
    #[test]
    fn the_builtin_table_reports_what_it_cannot_read() {
        let v = Vocab::builtin();
        let bopomofo = "ㄋㄧ2ㄏㄠ3";
        let unknown = v.unknown_chars(bopomofo, 8);
        assert!(unknown.contains(&'ㄋ'), "{unknown:?}");
        assert!(unknown.contains(&'ㄏ'), "{unknown:?}");
        // 丢光之后只剩两个填充,而不是空 —— 判据是长度,不是 is_empty。
        assert_eq!(v.tokenize(bopomofo).len(), 2);
    }

    #[test]
    fn a_table_that_understands_everything_reports_nothing() {
        let v = Vocab::builtin();
        assert!(v.unknown_chars("həlˈoʊ", 8).is_empty());
    }

    /// 取样要去重、要有上限 —— 诊断信息是给人读的。
    #[test]
    fn the_unknown_sample_is_deduped_and_bounded() {
        let v = Vocab::builtin();
        assert_eq!(v.unknown_chars("ㄋㄋㄋㄋ", 8), vec!['ㄋ']);
        assert_eq!(v.unknown_chars("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎ", 3).len(), 3);
    }
}
