//! 把阿拉伯数字换成汉字,让它们能被念出来。
//!
//! 不做这一步的后果不是「念得不好」,是**整段消失**。数字既不是汉字也不是拉丁
//! 字母:`phonemize_han` 把它原样带过,而 v1.1-zh 的词表里 `0`–`9` 是**声调标记**
//! (`ㄋㄧ2` 里的那个 2),不是可读的字符 —— 于是它们在编码那一步被丢掉,一声不响。
//!
//! 实测过的后果:「RTF 是 0.51x」丢掉 `0`,「崩(80070057)」丢掉 `807`。技术对话里
//! 数字到处都是,所以**每一句**都跟着一条 "vocabulary dropped part of this text"。
//!
//! ## 读法不止一种,而选错了是听得出来的
//!
//! 同样一串数字,该怎么念取决于它是什么:
//!
//! * `2026 年` 是**年份** → 二零二六,不是两千零二十六
//! * `25 个` 是**数量** → 二十五,不是二五
//! * `80070057` 是**编号** → 逐位念,没人会把错误码读成「八千万零七万零五十七」
//! * `0.51` 是**小数** → 零点五一,小数点后一律逐位
//!
//! 这里做的是这四条,不是一个完整的文本正规化引擎(那需要单位、日期、货币、
//! 分数、区间……)。够用的判据:**没有一个数字应该被静默丢掉**,而念法尽量对。

/// 十个数字的汉字。
const DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

/// 逐位念。年份、编号、小数点之后都走这条。
fn digit_by_digit(s: &str) -> String {
    s.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| DIGITS[d as usize])
        .collect()
}

/// 按数值念:一百二十三、两万零五。
///
/// 只处理到「亿」为止。再大的数在口语里本来就会被拆开说,而这条路上更可能遇到
/// 的是一串编号 —— 那由调用方按长度判给 [`digit_by_digit`]。
fn by_value(n: u64) -> String {
    if n == 0 {
        return "零".to_string();
    }
    // 从小到大的节:个、万、亿。
    const SECTION: [&str; 3] = ["", "万", "亿"];
    let mut sections: Vec<u64> = Vec::new();
    let mut rest = n;
    while rest > 0 {
        sections.push(rest % 10_000);
        rest /= 10_000;
    }

    let mut out = String::new();
    for (i, &sec) in sections.iter().enumerate().rev() {
        if sec == 0 {
            // 空节要留一个「零」,但不能连着两个 —— 一千万零五,不是一千万零零五。
            if !out.is_empty() && !out.ends_with('零') {
                out.push('零');
            }
            continue;
        }
        // 一个非空节,但它前面有更高的节而它自己不足四位:中间要补「零」。
        // 一千万零五百 —— 那个「零」就是这么来的。
        if !out.is_empty() && sec < 1000 && !out.ends_with('零') {
            out.push('零');
        }
        out.push_str(&section_by_value(sec));
        out.push_str(SECTION[i.min(SECTION.len() - 1)]);
    }
    // 末尾的「零」没有意义:一千零 → 一千。
    while out.ends_with('零') {
        out.pop();
    }
    out
}

/// 四位以内。
fn section_by_value(n: u64) -> String {
    const UNIT: [&str; 4] = ["", "十", "百", "千"];
    let digits: Vec<u64> = {
        let mut v = Vec::new();
        let mut r = n;
        while r > 0 {
            v.push(r % 10);
            r /= 10;
        }
        v
    };
    let mut out = String::new();
    for (i, &d) in digits.iter().enumerate().rev() {
        if d == 0 {
            if !out.is_empty() && !out.ends_with('零') {
                out.push('零');
            }
            continue;
        }
        // 「一十五」在口语里是「十五」—— 但只在它是整个数的开头时。
        // 「二百一十五」的那个「一十」不能省。
        if !(i == 1 && d == 1 && out.is_empty()) {
            out.push(DIGITS[d as usize]);
        }
        out.push_str(UNIT[i]);
    }
    while out.ends_with('零') {
        out.pop();
    }
    out
}

/// 一串纯数字该怎么念,给定它后面紧跟着什么。
fn run_to_chinese(digits: &str, next: Option<char>) -> String {
    // 年份:四位数字后面跟着「年」。二零二六年,不是两千零二十六年。
    if digits.len() == 4 && next == Some('年') {
        return digit_by_digit(digits);
    }
    // 编号:长到没人会按数值念。错误码、ID、电话。
    if digits.len() >= 5 {
        return digit_by_digit(digits);
    }
    match digits.parse::<u64>() {
        Ok(n) => by_value(n),
        // 溢出等于「长得离谱」,那更是编号。
        Err(_) => digit_by_digit(digits),
    }
}

/// 把文本里的阿拉伯数字换成汉字。
///
/// 其余字符原样保留 —— 这一步只管数字,标点由 [`super::map_punctuation`] 管,
/// 拉丁字母由英文 G2P 管。
pub fn normalize(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;

    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        // 一段连续的数字。
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        let whole: String = chars[start..i].iter().collect();

        // 小数点:后面还是数字才算,「第 3 步。」里那个句号不是。
        let is_decimal = i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit();
        if is_decimal {
            let frac_start = i + 1;
            let mut j = frac_start;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let frac: String = chars[frac_start..j].iter().collect();
            // 整数部分按数值,小数部分逐位 —— 零点五一,不是零点五十一。
            out.push_str(&run_to_chinese(&whole, None));
            out.push('点');
            out.push_str(&digit_by_digit(&frac));
            i = j;
            continue;
        }

        out.push_str(&run_to_chinese(&whole, chars.get(i).copied()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 回归:数字被静默丢掉。
    ///
    /// v1.1-zh 的词表里 `0`–`9` 是声调标记,不是可读字符,所以原样带过去的数字
    /// 会在编码时消失。实测「RTF 是 0.51x」丢掉 `0`,「崩(80070057)」丢掉 `807`。
    #[test]
    fn digits_never_survive_as_digits() {
        for t in ["0.51", "80070057", "25", "2026年", "第3步"] {
            let got = normalize(t);
            assert!(
                !got.chars().any(|c| c.is_ascii_digit()),
                "{t} -> {got} 里还有阿拉伯数字"
            );
        }
    }

    #[test]
    fn a_quantity_is_read_by_value() {
        assert_eq!(normalize("25"), "二十五");
        assert_eq!(normalize("103"), "一百零三");
        assert_eq!(normalize("1000"), "一千");
        assert_eq!(normalize("1005"), "一千零五");
        // 「一十五」在开头要省成「十五」,但不在中间省。
        assert_eq!(normalize("15"), "十五");
        assert_eq!(normalize("215"), "二百一十五");
    }

    /// 年份逐位,不按数值 —— 二零二六年,不是两千零二十六年。
    #[test]
    fn a_year_is_read_digit_by_digit() {
        assert_eq!(normalize("2026年"), "二零二六年");
        // 但同样四位数,不跟「年」时按数值念。
        assert_eq!(normalize("2026个"), "两千零二十六个".replace("两", "二"));
    }

    /// 编号逐位。没人把错误码念成「八千万零七万零五十七」。
    #[test]
    fn a_long_code_is_read_digit_by_digit() {
        assert_eq!(normalize("80070057"), "八零零七零零五七");
    }

    /// 小数点之后一律逐位:零点五一,不是零点五十一。
    #[test]
    fn a_decimal_reads_its_fraction_digit_by_digit() {
        assert_eq!(normalize("0.51"), "零点五一");
        assert_eq!(normalize("1.25"), "一点二五");
    }

    /// 句号不是小数点 —— 后面得跟着数字才算。
    #[test]
    fn a_full_stop_is_not_a_decimal_point() {
        assert_eq!(normalize("第3步."), "第三步.");
    }

    /// 数字之外的东西一个都不该被动。
    #[test]
    fn everything_else_is_left_alone() {
        assert_eq!(normalize("好的，我来查。"), "好的，我来查。");
        assert_eq!(normalize("RTF 是 x 倍"), "RTF 是 x 倍");
    }
}
