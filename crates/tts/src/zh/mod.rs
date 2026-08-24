//! 中文的 grapheme-to-phoneme。
//!
//! Kokoro v1.1-zh 吃的是音素 id,不是汉字,所以中文那条链上必须有这一层。英文那
//! 半由 `misaki-rs` 负责(本来就在用),这里只管中文,以及中英交替时的切分。
//!
//! ## 正确性怎么保证
//!
//! 参考实现是 Python 的 `misaki[zh]`(它自己又源自 PaddleSpeech)。这里不靠感觉
//! 复刻:`fixtures/make_golden.py` 把参考实现在一份语料上的输出存成
//! `fixtures/zh_golden.json`,测试逐条比对。移植对不对是**断言出来的**,不是听
//! 出来的 —— 中文念错音,写代码的人自己往往听不出来。
//!
//! ## 已知与参考实现的差距
//!
//! 参考实现用 pypinyin 的大词库加短语覆盖来选多音字;这里用的是 Rust 的 `pinyin`
//! 库,按**词**查表、逐字兜底。两边的词典不是同一份,所以多音字上会有分歧。分歧
//! 有多大不靠估计,由 `tests::golden` 报出来的命中率说话。

#[cfg(test)]
mod golden;
pub mod sandhi;
pub mod syllable;

use jieba_rs::Jieba;
use pinyin::ToPinyin;
use std::sync::OnceLock;

/// 词表里的未知符。英文段落没有英文 G2P 可用时落到它。
pub const UNK: &str = "❓";

fn jieba() -> &'static Jieba {
    static JIEBA: OnceLock<Jieba> = OnceLock::new();
    JIEBA.get_or_init(Jieba::new)
}

/// 词组读音表。
///
/// `pinyin` 库是逐字查表的,而多音字的读音由**词**决定:「银行」的行念 hang2,
/// 「不行」的行念 xing2 —— 逐字查永远只能拿到其中一个,另一个就固定念错。
///
/// 表里只收「词组读音异于逐字默认」的条目(38k 条,0.85 MB),因为其余的条目
/// 抄过来对结果没有任何影响。数据从参考实现用的同一份词库导出。
const PHRASES: &str = include_str!("../../fixtures/zh_phrases.tsv");

fn phrases() -> &'static std::collections::HashMap<&'static str, Vec<&'static str>> {
    static M: OnceLock<std::collections::HashMap<&'static str, Vec<&'static str>>> =
        OnceLock::new();
    M.get_or_init(|| {
        PHRASES
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .filter_map(|l| {
                let (word, pys) = l.split_once('\t')?;
                Some((word, pys.split(' ').collect()))
            })
            .collect()
    })
}

/// 一个词的拼音(带数字声调)。
///
/// 先查词组表,查不到再逐字。顺序不能反:逐字读音在多音字上本来就有一半是错的,
/// 而词组表存在的全部意义就是纠正它。
fn word_pinyin(word: &str) -> Vec<String> {
    if let Some(pys) = phrases().get(word) {
        if pys.len() == word.chars().count() {
            return pys.iter().map(|s| s.to_string()).collect();
        }
    }
    word.to_pinyin()
        .map(|p| match p {
            Some(p) => p.with_tone_num_end().to_string(),
            // 非汉字(标点、拉丁字母)在这一层原样带过,由调用方处理。
            None => String::new(),
        })
        .collect()
}

/// 中文标点 → 参考实现用的那套 ASCII 标点。
///
/// 抄自参考实现的 `map_punctuation`。词表里只有 ASCII 那一套,中文标点直接喂进去
/// 会变成未知符,而停顿是靠标点表达的 —— 丢了它整段话会连成一口气。
pub fn map_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '、' | '，' => out.push_str(", "),
            '。' | '．' => out.push_str(". "),
            '！' => out.push_str("! "),
            '：' => out.push_str(": "),
            '；' => out.push_str("; "),
            '？' => out.push_str("? "),
            '«' | '《' | '「' | '【' => out.push_str(" “"),
            '»' | '》' | '」' | '】' => out.push_str("” "),
            '（' => out.push_str(" ("),
            '）' => out.push_str(") "),
            other => out.push(other),
        }
    }
    out.trim().to_string()
}

fn is_han(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// 把「不」「一」与后一个词合起来。
///
/// jieba 常把它们单独切出来,而变调规则要看后一个字的声调才能判 —— 分开之后
/// 「做/不/到」里的「不」永远看不到「到」是四声,于是念成 ㄅㄨ4。参考实现在变调
/// 之前也做同样的合并(`_merge_bu` / `_merge_yi`)。
fn merge_bu_yi(words: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let (w, pos) = &words[i];
        if (w == "不" || w == "一") && i + 1 < words.len() && words[i + 1].0.chars().any(is_han) {
            // 合并后的词性取后一个词的 —— 变调看的是它。
            out.push((format!("{w}{}", words[i + 1].0), words[i + 1].1.clone()));
            i += 2;
        } else {
            out.push((w.clone(), pos.clone()));
            i += 1;
        }
    }
    out
}

/// 把一段**纯中文**转成音素。
///
/// 词与词之间用 `/`,标点直接贴在前一个词后面再跟一个空格 —— 与参考实现的
/// 形状一致。标点是停顿,把它当成一个词用 `/` 隔开会让模型多出一个音节的犹豫。
fn phonemize_han(text: &str) -> String {
    // 带词性切分:轻声规则里有一半要看词性(叠字要 n/v/a,「们子」要 r/n,
    // 「上下」要 s/l/f)。不给词性,这些规则等于没写。
    let tagged: Vec<(String, String)> = jieba()
        .tag(text, false)
        .into_iter()
        .map(|t| (t.word.to_string(), t.tag.to_string()))
        .collect();
    let words = merge_bu_yi(tagged);
    let mut out = String::new();

    for (word, pos) in words {
        let word: &str = &word;
        if !word.chars().any(is_han) {
            let t = word.trim();
            if !t.is_empty() {
                out.push_str(t);
                out.push(' ');
            }
            continue;
        }
        let mut pinyins = word_pinyin(word);
        sandhi::apply_with_pos(word, &pos, &mut pinyins);
        sandhi::erhua(word, &pos, &mut pinyins);
        let mut buf = String::new();
        for py in &pinyins {
            match syllable::to_phonemes(py) {
                Some(p) => buf.push_str(&p),
                None => buf.push_str(UNK),
            }
        }
        if buf.is_empty() {
            continue;
        }
        if !out.is_empty() && !out.ends_with(' ') {
            out.push('/');
        }
        out.push_str(&buf);
    }
    out.trim().to_string()
}

/// 一段混合文本 → 音素串。
///
/// `english` 是英文段落的 G2P。给 `None` 时英文变成未知符 —— 与参考实现在没有
/// `en_callable` 时的行为一致,而且**是有意让它显形**:英文被静默丢掉的话,
/// 「CUDA 慢两倍」会被念成「慢两倍」,意思正好反过来。
pub fn phonemize(text: &str, english: Option<&dyn Fn(&str) -> String>) -> String {
    let text = map_punctuation(text);
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut buf_is_latin = false;

    let flush = |buf: &mut String, is_latin: bool, out: &mut Vec<String>| {
        let seg = buf.trim().to_string();
        buf.clear();
        if seg.is_empty() {
            return;
        }
        if is_latin {
            out.push(match english {
                Some(f) => f(&seg),
                None => UNK.to_string(),
            });
        } else {
            let p = phonemize_han(&seg);
            if !p.is_empty() {
                out.push(p);
            }
        }
    };

    for ch in text.chars() {
        let latin = ch.is_ascii_alphabetic();
        if !buf.is_empty() && latin != buf_is_latin {
            flush(&mut buf, buf_is_latin, &mut out);
        }
        buf_is_latin = latin;
        buf.push(ch);
    }
    flush(&mut buf, buf_is_latin, &mut out);
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 多音字由**词**决定,不是由字决定。
    ///
    /// 「银行」的行念 hang2,「不行」的行念 xing2 —— 逐字查表永远只能拿到其中
    /// 一个,另一个就固定念错,而念错的是一个常用词。
    #[test]
    fn a_polyphone_is_read_by_its_word() {
        assert_eq!(word_pinyin("银行"), ["yin2", "hang2"]);
        assert_eq!(word_pinyin("重新"), ["chong2", "xin1"]);
        assert_eq!(word_pinyin("重要"), ["zhong4", "yao4"]);
    }

    /// 词组表里没有的词,逐字兜底 —— 不能整个词丢掉。
    #[test]
    fn a_word_outside_the_table_still_gets_read() {
        let p = word_pinyin("天气");
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|s| !s.is_empty()), "{p:?}");
    }

    #[test]
    fn punctuation_becomes_the_ascii_set_the_vocab_has() {
        assert_eq!(map_punctuation("你好，世界。"), "你好, 世界.");
        assert_eq!(map_punctuation("真的吗？！"), "真的吗? !");
        assert_eq!(map_punctuation("他说「好」"), "他说 “好”");
    }

    #[test]
    fn a_plain_sentence_becomes_phonemes() {
        let p = phonemize("今天天气不错", None);
        assert!(p.contains('ㄊ'), "{p}");
        assert!(!p.contains(UNK), "纯中文不该出现未知符:{p}");
    }

    /// 英文段落没有 G2P 时要**显形**,不能静默消失。
    #[test]
    fn english_without_a_g2p_is_marked_not_dropped() {
        let p = phonemize("CUDA 比 CPU 慢", None);
        assert_eq!(p.matches(UNK).count(), 2, "{p}");
    }

    #[test]
    fn english_goes_through_the_callback_when_there_is_one() {
        let f = |s: &str| format!("<{s}>");
        let p = phonemize("CUDA 很慢", Some(&f));
        assert!(p.contains("<CUDA>"), "{p}");
    }
}
