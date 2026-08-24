//! 变调。
//!
//! 中文的声调不是查字典就完事:同一个字在不同的邻居旁边念不同的调。三条规则日常
//! 出现频率极高,漏掉哪一条都是**一听就出戏**:
//!
//! - 三声连读:`你好` 念 ㄋㄧ2ㄏㄠ3,不是 ㄋㄧ3ㄏㄠ3
//! - 「不」:`不错` 念 ㄅㄨ2,`不好` 念 ㄅㄨ4
//! - 「一」:`一下` 念 ㄧ2,`一天` 念 ㄧ4,`看一看` 念 ㄧ5
//!
//! 再加上轻声:`走走` 的第二个字要轻读,`桌子` 的「子」要轻读。这一层靠词表,
//! 词表从参考实现导出(`fixtures/zh_*_words.txt`),不手抄 —— 四百多个词手抄一定
//! 会错,而错了没人看得出来。
//!
//! 顺序照抄参考实现的 `modified_tone`:不 → 一 → 轻声 → 三声。顺序是有意义的,
//! 三声规则要在轻声之后跑,否则被改成轻声的字还会参与三声连读的判定。

use std::collections::HashSet;
use std::sync::OnceLock;

const NEUTRAL_WORDS: &str = include_str!("../../fixtures/zh_neutral_words.txt");
const NOT_NEUTRAL_WORDS: &str = include_str!("../../fixtures/zh_not_neutral_words.txt");

fn word_set(raw: &'static str) -> HashSet<&'static str> {
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

fn neutral() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| word_set(NEUTRAL_WORDS))
}

fn not_neutral() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| word_set(NOT_NEUTRAL_WORDS))
}

/// 取一个音节的声调数字。没有数字时算轻声。
fn tone_of(pinyin: &str) -> u8 {
    pinyin
        .chars()
        .last()
        .and_then(|c| c.to_digit(10))
        .unwrap_or(5) as u8
}

/// 换掉一个音节的声调。
fn set_tone(pinyin: &mut String, tone: u8) {
    if pinyin
        .chars()
        .last()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        pinyin.pop();
    }
    pinyin.push(char::from_digit(tone as u32, 10).unwrap_or('5'));
}

/// 末尾 n 个字组成的子串。词表里很多条目是两字词,而它可能出现在长词的末尾。
fn last_chars(word: &str, n: usize) -> String {
    let chars: Vec<char> = word.chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

/// 「不」的变调:后面是四声念二声;夹在中间(看不懂)念轻声。
fn bu(chars: &[char], pinyins: &mut [String]) {
    if chars.len() == 3 && chars[1] == '不' {
        set_tone(&mut pinyins[1], 5);
        return;
    }
    for i in 0..chars.len() {
        if chars[i] == '不' && i + 1 < pinyins.len() && tone_of(&pinyins[i + 1]) == 4 {
            set_tone(&mut pinyins[i], 2);
        }
    }
}

/// 「一」的变调。
fn yi(word: &str, chars: &[char], pinyins: &mut [String]) {
    // 念数字时不变调:「二一零」里的一还是一声。
    if chars.contains(&'一') && chars.iter().all(|c| *c == '一' || c.is_numeric()) {
        return;
    }
    // 夹在重叠动词中间读轻声:看一看。
    if chars.len() == 3 && chars[1] == '一' && chars[0] == chars[2] {
        set_tone(&mut pinyins[1], 5);
        return;
    }
    // 序数不变调:第一。
    if word.starts_with("第一") && pinyins.len() > 1 {
        set_tone(&mut pinyins[1], 1);
        return;
    }
    const PUNC: &str = "、：，；。？！“”‘’':,;.?!";
    for i in 0..chars.len() {
        if chars[i] != '一' || i + 1 >= pinyins.len() {
            continue;
        }
        let next = tone_of(&pinyins[i + 1]);
        if next == 4 || next == 5 {
            set_tone(&mut pinyins[i], 2);
        } else if !PUNC.contains(chars[i + 1]) {
            set_tone(&mut pinyins[i], 4);
        }
    }
}

/// 轻声。
///
/// 参考实现里这一段最长,因为汉语里「什么时候读轻声」本来就是一堆并列的习惯,
/// 没有统一规则。照抄,不归纳 —— 归纳出来的规则会在没覆盖到的地方悄悄念错。
fn neutral_tone(word: &str, pos: &str, chars: &[char], pinyins: &mut [String]) {
    if not_neutral().contains(word) {
        return;
    }
    // 名词、动词、形容词的叠字:奶奶、试试、走走。
    let head = pos.chars().next().unwrap_or(' ');
    if matches!(head, 'n' | 'v' | 'a') {
        for j in 1..chars.len() {
            if chars[j] == chars[j - 1] && j < pinyins.len() {
                set_tone(&mut pinyins[j], 5);
            }
        }
    }

    let last = *chars.last().unwrap_or(&' ');
    let n = pinyins.len();
    if n == 0 {
        return;
    }
    if "吧呢啊呐噻嘛吖嗨哦哒滴哩哟喽啰耶喔诶".contains(last) {
        set_tone(&mut pinyins[n - 1], 5);
    } else if "的地得".contains(last) {
        set_tone(&mut pinyins[n - 1], 5);
    } else if chars.len() == 1 && "了着过".contains(last) && matches!(pos, "ul" | "uz" | "ug") {
        set_tone(&mut pinyins[n - 1], 5);
    } else if chars.len() > 1 && "们子".contains(last) && matches!(pos, "r" | "n") {
        set_tone(&mut pinyins[n - 1], 5);
    } else if chars.len() > 1 && "上下".contains(last) && matches!(pos, "s" | "l" | "f") {
        set_tone(&mut pinyins[n - 1], 5);
    } else if chars.len() > 1
        && "来去".contains(last)
        && chars.len() >= 2
        && "上下进出回过起开".contains(chars[chars.len() - 2])
    {
        set_tone(&mut pinyins[n - 1], 5);
    } else if chars
        .iter()
        .position(|c| *c == '个')
        .filter(|ge| {
            // 「个」作量词才轻读:三个、几个、半个。
            //
            // 这里必须是**条件的一部分**,不能是「只要出现个字就进这一支」——
            // 第一版写成后者,于是「这个」「那个」进了这一支又不满足量词判定,
            // 什么都没做,再也到不了下面的习惯轻声词表。实测漏掉 5 处。
            (*ge >= 1
                && (chars[*ge - 1].is_numeric() || "几有两半多各整每做是".contains(chars[*ge - 1])))
                || word == "个"
        })
        .is_some_and(|ge| {
            if ge < pinyins.len() {
                set_tone(&mut pinyins[ge], 5);
            }
            true
        })
    {
    } else if neutral().contains(word) || neutral().contains(last_chars(word, 2).as_str()) {
        set_tone(&mut pinyins[n - 1], 5);
    }
}

/// 三声连读。两个三声挨着,前一个念二声。
fn three(chars: &[char], pinyins: &mut [String]) {
    let all_three = |s: &[String]| !s.is_empty() && s.iter().all(|p| tone_of(p) == 3);

    match chars.len() {
        2 if all_three(pinyins) => set_tone(&mut pinyins[0], 2),
        3 if all_three(pinyins) => {
            // 参考实现按词内切分决定改哪一个。这里取常见形态:前两个变二声。
            set_tone(&mut pinyins[0], 2);
            set_tone(&mut pinyins[1], 2);
        }
        4 => {
            // 成语按二二切开,各自判定。
            let (a, b) = pinyins.split_at_mut(2);
            if all_three(a) {
                set_tone(&mut a[0], 2);
            }
            if all_three(b) {
                set_tone(&mut b[0], 2);
            }
        }
        _ => {}
    }
}

const MUST_ERHUA: &str = include_str!("../../fixtures/zh_must_erhua.txt");
const NOT_ERHUA: &str = include_str!("../../fixtures/zh_not_erhua.txt");

fn must_erhua() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| word_set(MUST_ERHUA))
}

fn not_erhua() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| word_set(NOT_ERHUA))
}

/// 儿化。
///
/// 「一点儿」的儿不是一个独立音节,它把前一个字的韵母染成卷舌 —— 词表里记作在
/// 韵母和声调之间插一个 `R`(`ㄉ言R3`)。当成独立音节念出来,听起来像「一点·耳」。
///
/// 但不是所有带儿的词都儿化:「女儿」「幼儿」里的儿是实实在在的一个字。哪些不算
/// 靠词表,词表从参考实现导出。
///
/// 返回是否发生了合并(发生了的话最后一个音节要被丢掉)。
pub fn erhua(word: &str, pos: &str, pinyins: &mut Vec<String>) -> bool {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 2 || chars.len() != pinyins.len() || *chars.last().unwrap() != '儿' {
        return false;
    }
    // 形容词、简称、人名里的儿不儿化。
    if !must_erhua().contains(word)
        && (not_erhua().contains(word) || matches!(pos, "a" | "j" | "nr"))
    {
        return false;
    }
    let tail: String = chars[chars.len() - 2..].iter().collect();
    if not_erhua().contains(tail.as_str()) {
        return false;
    }
    let last = pinyins.last().cloned().unwrap_or_default();
    // 参考实现只在儿读 er2 / er5 时合并;er1 先被纠正成 er2。
    if !(last.starts_with("er") && matches!(tone_of(&last), 1 | 2 | 5)) {
        return false;
    }
    pinyins.pop();
    if let Some(prev) = pinyins.last_mut() {
        let tone = tone_of(prev);
        set_tone(prev, tone); // 归一化:确保结尾是数字
        prev.pop();
        prev.push('R');
        prev.push(char::from_digit(tone as u32, 10).unwrap_or('5'));
        return true;
    }
    false
}

/// 对一个词应用全部变调。`pos` 是 jieba 的词性,没有就传空串。
pub fn apply_with_pos(word: &str, pos: &str, pinyins: &mut [String]) {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() != pinyins.len() {
        // 长度对不上说明这个词里混了非汉字,变调规则的下标会错位 —— 宁可不改。
        return;
    }
    bu(&chars, pinyins);
    yi(word, &chars, pinyins);
    neutral_tone(word, pos, &chars, pinyins);
    three(&chars, pinyins);
}

/// 不带词性的版本。轻声那一层会少判几种,但三声与不/一照常。
pub fn apply(word: &str, pinyins: &mut [String]) {
    apply_with_pos(word, "", pinyins)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(word: &str, pos: &str, pys: &[&str]) -> Vec<String> {
        let mut v: Vec<String> = pys.iter().map(|s| s.to_string()).collect();
        apply_with_pos(word, pos, &mut v);
        v
    }

    /// 期望值来自参考实现在 fixtures/zh_golden.json 里的实际输出。
    #[test]
    fn two_third_tones_in_a_row() {
        assert_eq!(run("你好", "l", &["ni3", "hao3"]), ["ni2", "hao3"]);
        assert_eq!(run("老虎", "n", &["lao3", "hu3"]), ["lao2", "hu3"]);
    }

    #[test]
    fn bu_before_a_fourth_tone_becomes_second() {
        assert_eq!(run("不错", "a", &["bu4", "cuo4"]), ["bu2", "cuo4"]);
        assert_eq!(run("不好", "a", &["bu4", "hao3"]), ["bu4", "hao3"]);
    }

    /// 夹在中间的「不」读轻声:看不懂。
    #[test]
    fn bu_in_the_middle_goes_neutral() {
        assert_eq!(
            run("看不懂", "v", &["kan4", "bu4", "dong3"]),
            ["kan4", "bu5", "dong3"]
        );
    }

    #[test]
    fn yi_changes_with_what_follows() {
        assert_eq!(run("一下", "m", &["yi1", "xia4"]), ["yi2", "xia4"]);
        assert_eq!(run("一天", "m", &["yi1", "tian1"]), ["yi4", "tian1"]);
    }

    /// 重叠动词中间的「一」读轻声;序数「第一」不变。
    #[test]
    fn yi_between_reduplication_and_in_ordinals() {
        assert_eq!(
            run("看一看", "v", &["kan4", "yi1", "kan4"]),
            ["kan4", "yi5", "kan4"]
        );
        assert_eq!(run("第一", "m", &["di4", "yi1"]), ["di4", "yi1"]);
    }

    /// 叠字动词的第二个字轻读:走走。
    #[test]
    fn reduplicated_verbs_go_neutral() {
        assert_eq!(run("走走", "v", &["zou3", "zou3"]), ["zou3", "zou5"]);
    }

    /// 「的地得」恒轻声 —— 出现频率极高,错了每句话都听得出来。
    #[test]
    fn the_particles_are_always_neutral() {
        assert_eq!(run("的", "uj", &["de5"]), ["de5"]);
        assert_eq!(run("我的", "r", &["wo3", "de1"]), ["wo3", "de5"]);
    }

    /// 词表里的习惯轻声词。
    #[test]
    fn conventional_neutral_words_come_from_the_list() {
        assert_eq!(run("耳朵", "n", &["er3", "duo3"]), ["er3", "duo5"]);
        assert!(neutral().contains("耳朵"));
        assert!(neutral().len() > 400, "词表没加载全");
    }

    /// 儿化:儿不是独立音节,它把前一个字染成卷舌。
    #[test]
    fn erhua_folds_into_the_previous_syllable() {
        let mut v = vec!["dian3".to_string(), "er2".to_string()];
        assert!(erhua("点儿", "n", &mut v));
        assert_eq!(v, ["dianR3"]);
    }

    /// 但「女儿」的儿是实实在在一个字,不能吞掉。
    #[test]
    fn a_real_er_is_not_folded() {
        let mut v = vec!["nv3".to_string(), "er2".to_string()];
        assert!(!erhua("女儿", "n", &mut v));
        assert_eq!(v, ["nv3", "er2"]);
    }

    /// 长度对不上时宁可不改,也不要错位改调。
    #[test]
    fn a_length_mismatch_changes_nothing() {
        assert_eq!(run("你好a", "x", &["ni3", "hao3"]), ["ni3", "hao3"]);
    }
}
