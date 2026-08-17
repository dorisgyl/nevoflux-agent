//! 把一路模型输出拆成正文与口语稿(P3 / Q15、Q21、Q50)。
//!
//! 一次 LLM 调用产出两个视角:`<speak>` 内是送 TTS 的口语稿,其余是进 session
//! 消息的正文。拆流器在 token 流上实时分离两者。
//!
//! ## 三条硬约束
//!
//! 1. **标签会跨 delta 边界**。`<sp` 和 `eak>` 分两个 chunk 到达是常态,不是
//!    边缘情况 —— 按 delta 逐个匹配的实现会漏掉大部分标签。
//! 2. **标签本身永不进任何一路输出**。漏出去的话,正文里会出现 `<speak>`,
//!    而 history 里的正文是要被模型下一轮读到的(ADR-0004)—— 它会学着继续吐。
//! 3. **模型不吐 `<speak>` 时必须有兜底**,而且兜底需要一个**触发时机**。
//!    Q50 定了 `<speak>` 先出,于是「等太久」就是判据:超过阈值仍未见开标签,
//!    这一轮就没有口语稿。没有这个判据,兜底路径永远在等一个不会来的标签。
//!
//! ## 为什么按句子发,而不是等 `</speak>` 闭合
//!
//! 首字延迟是语音的第一体感。等闭合等于把整段生成时间加到 TTS 前面,而
//! `nevoflux-tts` 本就按句子分片合成 —— 这只是把既有的切分点前移。

/// 拆出来的一片。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    /// 一句完整的口语稿,可以立刻送 TTS。
    Spoken(String),
    /// 正文增量,原样进消息流。
    Prose(String),
    /// 判定这一轮没有口语稿。只会发一次。
    SpokenAbsent,
}

const OPEN: &str = "<speak>";
const CLOSE: &str = "</speak>";

/// 句末标点。与 `nevoflux-tts` 的切分点一致 —— 用同一套边界,免得同一句话在
/// 两处被切成不同的形状。
const SENTENCE_ENDS: [char; 8] = ['。', '!', '?', '.', '!', '?', ';', ';'];

pub struct SpeakSplitter {
    in_speak: bool,
    /// 可能是标签前缀的、还不能吐出去的字符。
    pending: String,
    /// 口语稿里尚未凑满一句的部分。
    sentence: String,
    /// `<speak>` 出现之前,正文侧已经过了多少字符。
    chars_before_speak: usize,
    /// 超过这个字符数仍未见开标签,判定本轮无口语稿。
    fallback_after: usize,
    seen_speak: bool,
    absent_emitted: bool,
}

impl SpeakSplitter {
    /// `fallback_after` 是 Q50 的「等太久」判据,按正文侧字符数计。
    pub fn new(fallback_after: usize) -> Self {
        Self {
            in_speak: false,
            pending: String::new(),
            sentence: String::new(),
            chars_before_speak: 0,
            fallback_after,
            seen_speak: false,
            absent_emitted: false,
        }
    }

    pub fn saw_speak(&self) -> bool {
        self.seen_speak
    }

    /// 喂一段 delta,拿回这一步产生的所有片。
    pub fn push(&mut self, delta: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        for ch in delta.chars() {
            self.feed(ch, &mut out);
        }
        self.flush_prose_if_absent(&mut out);
        out
    }

    /// 流结束:吐出未闭合的尾巴。
    ///
    /// 未闭合的 `<speak>` 会发生 —— 模型被打断、达到 token 上限。把攒了一半的
    /// 句子丢掉等于丢掉话尾,所以照发。
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        // pending 到这里已经不可能补全成标签了,当普通文本处理。
        let pending = std::mem::take(&mut self.pending);
        for ch in pending.chars() {
            self.emit_char(ch, &mut out);
        }
        let tail = std::mem::take(&mut self.sentence);
        if !tail.trim().is_empty() {
            out.push(Piece::Spoken(tail));
        }
        if !self.seen_speak && !self.absent_emitted {
            self.absent_emitted = true;
            out.push(Piece::SpokenAbsent);
        }
        out
    }

    fn feed(&mut self, ch: char, out: &mut Vec<Piece>) {
        if self.pending.is_empty() && ch != '<' {
            self.emit_char(ch, out);
            return;
        }

        self.pending.push(ch);
        if self.pending == OPEN {
            self.pending.clear();
            self.in_speak = true;
            self.seen_speak = true;
            return;
        }
        if self.pending == CLOSE {
            self.pending.clear();
            self.in_speak = false;
            // 闭合时把攒了一半的句子发掉,别等下一句。
            let tail = std::mem::take(&mut self.sentence);
            if !tail.trim().is_empty() {
                out.push(Piece::Spoken(tail));
            }
            return;
        }
        // 还可能补全成标签就继续攒;不可能了就整段当普通文本。
        if OPEN.starts_with(self.pending.as_str()) || CLOSE.starts_with(self.pending.as_str()) {
            return;
        }
        let stuck = std::mem::take(&mut self.pending);
        let mut chars = stuck.chars();
        // 第一个字符注定不是标签的一部分;剩下的可能是新标签的开头,
        // 所以重新走一遍状态机而不是直接吐出去。`<<speak>` 就靠这条。
        if let Some(first) = chars.next() {
            self.emit_char(first, out);
        }
        let rest: String = chars.collect();
        for c in rest.chars() {
            self.feed(c, out);
        }
    }

    fn emit_char(&mut self, ch: char, out: &mut Vec<Piece>) {
        if self.in_speak {
            self.sentence.push(ch);
            if SENTENCE_ENDS.contains(&ch) {
                let s = std::mem::take(&mut self.sentence);
                if !s.trim().is_empty() {
                    out.push(Piece::Spoken(s));
                }
            }
        } else {
            if !self.seen_speak {
                self.chars_before_speak += 1;
            }
            out.push(Piece::Prose(ch.to_string()));
        }
    }

    fn flush_prose_if_absent(&mut self, out: &mut Vec<Piece>) {
        if self.seen_speak || self.absent_emitted {
            return;
        }
        if self.chars_before_speak > self.fallback_after {
            self.absent_emitted = true;
            out.push(Piece::SpokenAbsent);
        }
    }
}

/// 把逐字的 `Prose` 合并成整段,便于断言与下游消费。
pub fn coalesce_prose(pieces: &[Piece]) -> String {
    let mut s = String::new();
    for p in pieces {
        if let Piece::Prose(t) = p {
            s.push_str(t);
        }
    }
    s
}

pub fn spoken(pieces: &[Piece]) -> Vec<String> {
    pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Spoken(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 按任意切分喂进去 —— 拆流器不该关心 delta 边界落在哪。
    fn run(chunks: &[&str], fallback_after: usize) -> Vec<Piece> {
        let mut sp = SpeakSplitter::new(fallback_after);
        let mut out = Vec::new();
        for c in chunks {
            out.extend(sp.push(c));
        }
        out.extend(sp.finish());
        out
    }

    #[test]
    fn separates_the_two_views() {
        let out = run(&["<speak>好的,这就去看。</speak>正文在这里。"], 1000);
        assert_eq!(spoken(&out), vec!["好的,这就去看。"]);
        assert_eq!(coalesce_prose(&out), "正文在这里。");
    }

    #[test]
    fn a_tag_split_across_deltas_still_matches() {
        // 这是常态,不是边缘情况。
        let out = run(&["<sp", "eak>", "好的。", "</spe", "ak>", "正文"], 1000);
        assert_eq!(spoken(&out), vec!["好的。"]);
        assert_eq!(coalesce_prose(&out), "正文");
    }

    #[test]
    fn tags_never_reach_either_output() {
        let out = run(&["<speak>话。</speak>文"], 1000);
        let prose = coalesce_prose(&out);
        assert!(!prose.contains('<'), "正文漏出了标签:{prose}");
        for s in spoken(&out) {
            assert!(!s.contains('<'), "口语稿漏出了标签:{s}");
        }
    }

    #[test]
    fn sentences_go_out_one_at_a_time_not_at_close() {
        // 等 `</speak>` 闭合就是把整段生成时间加到 TTS 前面。
        let mut sp = SpeakSplitter::new(1000);
        let first = sp.push("<speak>第一句。");
        assert_eq!(spoken(&first), vec!["第一句。"], "第一句该立刻可送 TTS");
        let second = sp.push("第二句。");
        assert_eq!(spoken(&second), vec!["第二句。"]);
    }

    #[test]
    fn an_unclosed_speak_still_yields_its_tail() {
        // 模型被打断或撞上 token 上限。丢掉攒了一半的句子等于丢掉话尾。
        let out = run(&["<speak>说到一半"], 1000);
        assert_eq!(spoken(&out), vec!["说到一半"]);
    }

    #[test]
    fn absence_of_speak_is_reported_once_and_early() {
        // Q50:`<speak>` 先出,所以「等太久」就是判据。没有它,兜底路径
        // 会一直等一个不会来的标签。
        let out = run(&["这是一段没有口语稿的正文,一直写下去"], 8);
        assert!(out.contains(&Piece::SpokenAbsent), "没有触发兜底");
        assert_eq!(
            out.iter().filter(|p| **p == Piece::SpokenAbsent).count(),
            1,
            "兜底只该报一次"
        );
    }

    #[test]
    fn absence_is_reported_at_finish_even_below_the_threshold() {
        let out = run(&["很短"], 1000);
        assert!(out.contains(&Piece::SpokenAbsent));
    }

    #[test]
    fn a_late_speak_does_not_trigger_the_fallback() {
        let mut sp = SpeakSplitter::new(1000);
        let out = sp.push("<speak>有口语稿。</speak>正文");
        assert!(!out.contains(&Piece::SpokenAbsent));
        assert!(sp.saw_speak());
        assert!(!sp.finish().contains(&Piece::SpokenAbsent));
    }

    #[test]
    fn a_lone_angle_bracket_is_ordinary_text() {
        let out = run(&["a < b 而且 5<6"], 1000);
        assert_eq!(coalesce_prose(&out), "a < b 而且 5<6");
    }

    #[test]
    fn a_near_miss_tag_is_ordinary_text() {
        let out = run(&["<speaker> 是个词"], 1000);
        assert_eq!(coalesce_prose(&out), "<speaker> 是个词");
        assert!(spoken(&out).is_empty());
    }

    #[test]
    fn a_doubled_bracket_still_finds_the_tag() {
        // `<<speak>` —— 第一个 `<` 是文本,第二个开始才是标签。回溯写错的话
        // 这里会把整个标签当文本吐出去。
        let out = run(&["<<speak>话。</speak>"], 1000);
        assert_eq!(coalesce_prose(&out), "<");
        assert_eq!(spoken(&out), vec!["话。"]);
    }

    #[test]
    fn english_sentence_ends_also_split() {
        let out = run(&["<speak>First one. Second one!</speak>"], 1000);
        assert_eq!(spoken(&out), vec!["First one.", " Second one!"]);
    }

    #[test]
    fn character_by_character_matches_whole_string() {
        // delta 边界不该改变结果 —— 一个字符一片是最坏的切法。
        let text = "<speak>甲。乙!</speak>丙";
        let whole = run(&[text], 1000);
        let per_char: Vec<&str> = text.split("").filter(|s| !s.is_empty()).collect();
        let split = run(&per_char, 1000);
        assert_eq!(spoken(&whole), spoken(&split));
        assert_eq!(coalesce_prose(&whole), coalesce_prose(&split));
    }
}
