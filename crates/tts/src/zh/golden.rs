//! 与参考实现的逐条比对。
//!
//! 这个文件存在的理由:**中文念错音,写代码的人自己听不出来。** 尤其是多音字和
//! 变调 —— 「银行」念成「银航」,母语者一秒就发现,而盯着代码的人永远发现不了。
//! 所以正确性必须是断言出来的。
//!
//! 黄金数据由 `fixtures/make_golden.py` 从 Python 的 `misaki[zh]` 生成,那是
//! Kokoro v1.1-zh 官方的前端。这里跑同样的语料,逐条对。
//!
//! ## 为什么门槛不是 100%
//!
//! 参考实现用 pypinyin 的大词库加短语覆盖选多音字,这里用的是 Rust 的 `pinyin`
//! 库 —— 两份词典不一样,多音字上必然有分歧。门槛按**实测**定,并且只能往上调:
//! 调低门槛等于把退步说成正常。差在哪由报告打出来,不猜。

use std::collections::BTreeMap;

const GOLDEN: &str = include_str!("../../fixtures/zh_golden.json");

#[derive(serde::Deserialize)]
struct Record {
    text: String,
    phonemes: String,
}

/// 一条记录的比对结果。
struct Diff {
    text: String,
    want: String,
    got: String,
}

/// 按「音节记号」切开,便于逐音节比对。
///
/// 音素串是 `ㄋㄧ2ㄏㄠ3` 这样连写的,直接比字符串只能得到「对/不对」;按声调数字
/// 切开就能知道**几个音节里错了几个**,而那才是能指导下一步的数字。
fn syllables(phonemes: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in phonemes.chars() {
        if ch == '/' || ch == ' ' {
            continue;
        }
        cur.push(ch);
        if ch.is_ascii_digit() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

fn compare() -> (usize, usize, Vec<Diff>) {
    let records: Vec<Record> = serde_json::from_str(GOLDEN).expect("黄金数据解析失败");
    let mut total = 0usize;
    let mut hit = 0usize;
    let mut diffs = Vec::new();

    for r in &records {
        let got = super::phonemize(&r.text, None);
        let want_syl = syllables(&r.phonemes);
        let got_syl = syllables(&got);
        total += want_syl.len();
        hit += want_syl
            .iter()
            .zip(got_syl.iter())
            .filter(|(a, b)| a == b)
            .count();
        if want_syl != got_syl {
            diffs.push(Diff {
                text: r.text.clone(),
                want: r.phonemes.clone(),
                got,
            });
        }
    }
    (hit, total, diffs)
}

#[test]
fn agreement_with_the_reference_implementation() {
    let (hit, total, diffs) = compare();
    let rate = hit as f32 / total as f32;

    // 差异按第一个不同的音节归类,好看出错误集中在哪一类上。
    let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
    for d in &diffs {
        let w = syllables(&d.want);
        let g = syllables(&d.got);
        let at = w.iter().zip(g.iter()).position(|(a, b)| a != b);
        let key = match at {
            Some(i) => format!("{} -> {}", w[i], g.get(i).cloned().unwrap_or_default()),
            None => "长度不同".to_string(),
        };
        *by_kind.entry(key).or_default() += 1;
    }

    println!("\n=== 与参考实现的一致率 ===");
    println!("音节 {hit}/{total} = {:.1}%", rate * 100.0);
    println!("有差异的句子 {}", diffs.len());
    println!("\n差异类型(前 15):");
    let mut kinds: Vec<_> = by_kind.into_iter().collect();
    kinds.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in kinds.iter().take(15) {
        println!("  {n:3}x  {k}");
    }
    println!("\n差异样例(前 8 句):");
    for d in diffs.iter().take(8) {
        println!("  原文: {}", d.text);
        println!("    参考: {}", d.want);
        println!("    本地: {}", d.got);
    }

    // 门槛按实测定(当前 95.6%),留一点余量给 jieba / pinyin 的版本漂移。
    // **只能往上调**:调低等于把退步说成正常。
    //
    // 剩下的差异主要是分词边界:jieba-rs 与 Python jieba 对同一句话的切分不完全
    // 一样,而变调的作用域是词 —— 「我很好」切成一个词还是三个,三声连读的结果
    // 就不同。这类差异要靠换分词器才能再往上,收益递减。
    assert!(
        rate >= 0.95,
        "与参考实现的音节一致率跌到 {:.1}%,低于门槛 95%",
        rate * 100.0
    );
}
