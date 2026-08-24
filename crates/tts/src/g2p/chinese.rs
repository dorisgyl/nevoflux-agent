//! 中文的 G2P,接到 [`crate::g2p::G2p`] 上。
//!
//! 干活的是 [`crate::zh`];这里只做一件事:**把英文段落交给英文 G2P**。
//!
//! 这一步不是锦上添花。中文回答里夹英文词是常态(「CUDA 比 CPU 慢两倍」),而
//! 没有英文回退时它们会变成未知符 —— 模型念不出来,那句话的主语就没了。参考实现
//! 把这条叫 `en_callable`,同样是可选的;这里让它成为默认。

use super::{english::EnglishG2p, G2p};
use crate::error::TtsError;

pub struct ChineseG2p {
    english: EnglishG2p,
}

impl ChineseG2p {
    pub fn new() -> Self {
        Self {
            english: EnglishG2p::new(),
        }
    }
}

impl Default for ChineseG2p {
    fn default() -> Self {
        Self::new()
    }
}

impl G2p for ChineseG2p {
    fn phonemize(&self, text: &str) -> Result<String, TtsError> {
        // 英文那半失败时不整句失败:退回未知符,让这句话仍然念得出来。一个词念不
        // 出来是瑕疵,整句不出声是故障。
        let english = |seg: &str| match self.english.phonemize(seg) {
            Ok(p) => p,
            Err(_) => crate::zh::UNK.to_string(),
        };
        Ok(crate::zh::phonemize(text, Some(&english)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chinese_becomes_bopomofo() {
        let p = ChineseG2p::new().phonemize("你好").unwrap();
        assert_eq!(p, "ㄋㄧ2ㄏㄠ3");
    }

    /// 夹在中文里的英文要交给英文 G2P,而不是变成未知符 ——
    /// 「CUDA 比 CPU 慢」丢掉两个词之后就只剩「比 慢」。
    #[test]
    fn embedded_english_goes_through_the_english_stage() {
        let p = ChineseG2p::new().phonemize("CUDA 很慢").unwrap();
        assert!(!p.contains(crate::zh::UNK), "英文不该变成未知符:{p}");
        assert!(p.contains('ㄏ'), "中文那半还在:{p}");
    }
}
