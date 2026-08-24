//! 一个拼音音节 → Kokoro 的音素串。
//!
//! Kokoro v1.1-zh 的词表用的是**注音符号 + 数字声调**(`ㄋㄧ2`),韵母里没有对应
//! 注音符号的那些则借用汉字当记号(`uo` → `我`、`ian` → `言`)。这张表不是我们
//! 发明的,是模型词表定死的,所以照抄,不做「优化」。
//!
//! ## 为什么从整体拼音推导声母韵母,而不是查两张表
//!
//! 参考实现(misaki / PaddleSpeech)问的是 pypinyin 的 `Style.INITIALS` 与
//! `Style.FINALS_TONE3`。Rust 这边的拼音库只给整体形式,所以切分得自己做。好在
//! 这个切分是**完全确定**的,把 pypinyin 的行为逐条对照过:
//!
//! - `y` / `w` **不是**声母(`yi` → 声母空、韵母 `i`)
//! - 缩写形式要还原:`iu` → `iou`、`ui` → `uei`、`un` → `uen`
//! - `j/q/x` 后面的 `u` 实际是 `ü`(`ju` → `jv`)
//! - `zi/ci/si` 的 `i` 与 `zhi/chi/shi/ri` 的 `i` 是三个不同的韵母,词表里分别是
//!   `ㄧ`、`ㄭ`、`十` —— 混起来会让「资」和「知」念成同一个音

/// 声母表。**长的在前**:`zh` 必须先于 `z` 匹配,否则「知」会被切成 `z` + `hi`。
const INITIALS: [&str; 21] = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s",
];

/// 韵母 / 声母 → 词表里的记号。抄自模型自己的 `ZH_MAP`。
fn mark(part: &str) -> Option<&'static str> {
    Some(match part {
        // 声母
        "b" => "ㄅ",
        "p" => "ㄆ",
        "m" => "ㄇ",
        "f" => "ㄈ",
        "d" => "ㄉ",
        "t" => "ㄊ",
        "n" => "ㄋ",
        "l" => "ㄌ",
        "g" => "ㄍ",
        "k" => "ㄎ",
        "h" => "ㄏ",
        "j" => "ㄐ",
        "q" => "ㄑ",
        "x" => "ㄒ",
        "zh" => "ㄓ",
        "ch" => "ㄔ",
        "sh" => "ㄕ",
        "r" => "ㄖ",
        "z" => "ㄗ",
        "c" => "ㄘ",
        "s" => "ㄙ",
        // 韵母
        "a" => "ㄚ",
        "o" => "ㄛ",
        "e" => "ㄜ",
        "ie" => "ㄝ",
        "ai" => "ㄞ",
        "ei" => "ㄟ",
        "ao" => "ㄠ",
        "ou" => "ㄡ",
        "an" => "ㄢ",
        "en" => "ㄣ",
        "ang" => "ㄤ",
        "eng" => "ㄥ",
        "er" => "ㄦ",
        "i" => "ㄧ",
        "u" => "ㄨ",
        "v" => "ㄩ",
        // 「资」的 i 与「知」的 i:同一个字母,三个音
        "ii" => "ㄭ",
        "iii" => "十",
        "ve" => "月",
        "ia" => "压",
        "ian" => "言",
        "iang" => "阳",
        "iao" => "要",
        "in" => "阴",
        "ing" => "应",
        "iong" => "用",
        "iou" => "又",
        "ong" => "中",
        "ua" => "穵",
        "uai" => "外",
        "uan" => "万",
        "uang" => "王",
        "uei" => "为",
        "uen" => "文",
        "ueng" => "瓮",
        "uo" => "我",
        "van" => "元",
        "vn" => "云",
        _ => return None,
    })
}

/// 切开的音节:声母(可空)、韵母、声调。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Syllable<'a> {
    pub initial: &'a str,
    pub final_: String,
    pub tone: u8,
}

/// 把带数字声调的整体拼音切成声母 + 韵母 + 声调。
///
/// 输入形如 `zhuang1`、`lve4`、`er2`。轻声按 `5` 处理(与参考实现的
/// `neutral_tone_with_five` 一致)—— 没有声调数字时当轻声,而不是当第一声:
/// 一声和轻声在这套音素里是两个不同的记号,猜错会让所有轻声字听起来像重读。
pub fn split(pinyin: &str) -> Option<Syllable<'_>> {
    let (body, tone) = match pinyin.chars().last() {
        Some(c) if c.is_ascii_digit() => (
            &pinyin[..pinyin.len() - 1],
            c.to_digit(10).unwrap_or(5) as u8,
        ),
        _ => (pinyin, 5),
    };
    if body.is_empty() {
        return None;
    }

    let initial = INITIALS
        .iter()
        .find(|i| body.starts_with(**i))
        .copied()
        .unwrap_or("");
    let rest = &body[initial.len()..];
    Some(Syllable {
        initial,
        final_: normalize_final(initial, rest),
        tone,
    })
}

/// 把韵母的表面形式还原成词表用的那一套。
fn normalize_final(initial: &str, rest: &str) -> String {
    // 零声母:`y` / `w` 是书写用的填充,不是声母。
    let rest = match rest {
        // ü 系写作 yu-,还原成 v
        "yu" => "v",
        "yue" => "ve",
        "yuan" => "van",
        "yun" => "vn",
        // i 系
        "yi" => "i",
        "ya" => "ia",
        "ye" => "ie",
        "yao" => "iao",
        "you" => "iou",
        "yan" => "ian",
        "yin" => "in",
        "yang" => "iang",
        "ying" => "ing",
        "yong" => "iong",
        "y" => "i",
        // u 系
        "wu" => "u",
        "wa" => "ua",
        "wo" => "uo",
        "wai" => "uai",
        "wei" => "uei",
        "wan" => "uan",
        "wen" => "uen",
        "wang" => "uang",
        "weng" => "ueng",
        "w" => "u",
        other => other,
    };

    // 缩写还原。`liu` 的韵母是 `iou`,`gui` 是 `uei`,`lun` 是 `uen` ——
    // 书写省掉的中间那个元音在音素里是要念出来的。
    let rest = match rest {
        "iu" => "iou",
        "ui" => "uei",
        "un" if !matches!(initial, "j" | "q" | "x") => "uen",
        other => other,
    };

    // `j/q/x` 后面写作 u 的其实是 ü —— 但只限**直接跟在声母后**的那个 u。
    //
    // 第一版无条件替换第一个 u,于是 `jiu`(韵母 iou)被改成 `iov`,整个音节
    // 认不出来,变成未知符。「就」「秋」「休」这些高频字全中招。
    let owned = if matches!(initial, "j" | "q" | "x") && rest.starts_with('u') {
        rest.replacen('u', "v", 1)
    } else {
        rest.replace('ü', "v")
    };

    // 「资」`zi` 与「知」`zhi`:同写作 i,是两个不同的韵母。
    match (initial, owned.as_str()) {
        ("z" | "c" | "s", "i") => "ii".to_string(),
        ("zh" | "ch" | "sh" | "r", "i") => "iii".to_string(),
        _ => owned,
    }
}

/// 一个音节的音素串:声母记号 + 韵母记号 + 声调数字。
///
/// 认不出来时返回 `None` —— 交给调用方决定是丢掉还是记未知符。这里不猜:猜出来的
/// 音会变成一个长期存在、没人知道来源的口音错误。
pub fn to_phonemes(pinyin: &str) -> Option<String> {
    // 儿化记号 `R` 夹在韵母和声调之间(`dianR3`),先摘出来再切。
    let (pinyin, erhua) = match pinyin.find('R') {
        Some(i) => (format!("{}{}", &pinyin[..i], &pinyin[i + 1..]), true),
        None => (pinyin.to_string(), false),
    };
    let syl = split(&pinyin)?;
    let mut out = String::new();
    if !syl.initial.is_empty() {
        out.push_str(mark(syl.initial)?);
    }
    out.push_str(mark(&syl.final_)?);
    if erhua {
        out.push('R');
    }
    out.push(char::from_digit(syl.tone as u32, 10)?);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ph(p: &str) -> String {
        to_phonemes(p).unwrap_or_else(|| panic!("{p} 转不出来"))
    }

    /// 这些期望值全部来自参考实现的实际输出(见 fixtures/zh_golden.json)。
    #[test]
    fn ordinary_syllables() {
        assert_eq!(ph("ni3"), "ㄋㄧ3");
        assert_eq!(ph("hao3"), "ㄏㄠ3");
        assert_eq!(ph("guang1"), "ㄍ王1");
        assert_eq!(ph("shuang1"), "ㄕ王1");
    }

    /// 零声母:y / w 不是声母,是书写填充。
    #[test]
    fn y_and_w_are_not_initials() {
        assert_eq!(ph("yi1"), "ㄧ1");
        assert_eq!(ph("wu1"), "ㄨ1");
        assert_eq!(ph("wo3"), "我3");
        assert_eq!(ph("yong4"), "用4");
    }

    /// ü 系。写法有 yu / ju / lv 三种,音只有一个。
    #[test]
    fn the_u_umlaut_family_normalises() {
        assert_eq!(ph("yu2"), "ㄩ2");
        assert_eq!(ph("yue4"), "月4");
        assert_eq!(ph("yuan2"), "元2");
        assert_eq!(ph("yun2"), "云2");
        assert_eq!(ph("juan3"), "ㄐ元3");
        assert_eq!(ph("lve4"), "ㄌ月4");
        assert_eq!(ph("qu4"), "ㄑㄩ4");
    }

    /// 缩写还原:书写省掉的元音在音素里要念出来。
    #[test]
    fn written_abbreviations_expand() {
        assert_eq!(ph("liu4"), "ㄌ又4");
        assert_eq!(ph("gui1"), "ㄍ为1");
        assert_eq!(ph("lun4"), "ㄌ文4");
        // jun 的 un 是 ün,不是 uen
        assert_eq!(ph("jun1"), "ㄐ云1");
    }

    /// 「资」「知」「你」的 i 是三个不同的韵母。混起来是听得出来的错误。
    #[test]
    fn the_three_kinds_of_i() {
        assert_eq!(ph("zi1"), "ㄗㄭ1");
        assert_eq!(ph("zhi1"), "ㄓ十1");
        assert_eq!(ph("shi4"), "ㄕ十4");
        assert_eq!(ph("ri4"), "ㄖ十4");
        assert_eq!(ph("ni3"), "ㄋㄧ3");
    }

    /// 没有声调数字 = 轻声(5),不是一声。
    #[test]
    fn a_missing_tone_is_neutral_not_first() {
        assert_eq!(split("de").unwrap().tone, 5);
        assert_eq!(split("de5").unwrap().tone, 5);
        assert_eq!(split("de1").unwrap().tone, 1);
    }

    #[test]
    fn er_stands_alone() {
        assert_eq!(ph("er2"), "ㄦ2");
    }

    /// 认不出来就说认不出来,不猜。
    #[test]
    fn an_unknown_syllable_is_not_guessed() {
        assert_eq!(to_phonemes("qqq1"), None);
        assert_eq!(to_phonemes(""), None);
    }
}
