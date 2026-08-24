//! 把模型的回答变成能念出来的句子。
//!
//! 这里的立场是**念全**:模型写什么就念什么,只拿掉两样东西 —— 围栏代码块
//! (念出来是「反引号反引号反引号 r s」加一串标点,没人想听),以及 markdown
//! 自己的记号(星号、井号、表格竖线、链接的 URL 那一半)。记号不是内容,拿掉
//! 之后念出来仍然是同一句话;代码块是内容,但它不是**说**的那种内容。
//!
//! 早先这里是另一套:让模型在 `<speak>` 里另写一份口语稿,正文与口语稿分两路。
//! 那套要求模型把回答写两遍,而且它不守格式时整轮无声 —— 而模型不守格式是确定
//! 会发生的。直接念正文之后,「这一轮没有口语稿」这个失败模式从根上消失了。
//!
//! ## 流式
//!
//! 输入是 delta 不是整段,两样东西会跨 delta 边界:围栏记号本身(``` 可能拆成
//! 两个 delta),以及行首的块记号(`#`、`- `、`| `、`1. `)。所以行首要攒一小段
//! 再判定。攒的只是**记号字符**:遇到第一个普通字符就立刻判定并放行,普通段落
//! 因此不会为了等一个换行而卡住 —— 首字延迟是语音的第一体感,而一整段没有换行
//! 的长回答很常见。

/// 句末标点。与 `nevoflux-tts` 的切分点一致,免得同一句话在两处被切成不同的形状。
const SENTENCE_ENDS: [char; 8] = ['。', '！', '？', '.', '!', '?', '；', ';'];

/// 整轮下来一句都念不出来时说的话 —— 回答通篇是代码时会这样。
///
/// 存在的理由是「没什么可念」与「坏了」在用户耳朵里长得一模一样。
pub const NOTHING_TO_SAY_EN: &str = "The answer is on screen.";
pub const NOTHING_TO_SAY_ZH: &str = "回答我写在屏幕上了，你看一下。";

/// 行首还可能是块记号的那些字符。攒到它们之外的字符就判定。
fn is_marker_char(c: char) -> bool {
    matches!(
        c,
        '`' | '~' | '#' | '>' | '-' | '*' | '+' | '|' | ' ' | '\t' | '.' | '_' | ')'
    ) || c.is_ascii_digit()
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Inline {
    Text,
    /// `<...>` 里,整段丢掉。`<speak>` 这类残留标签也走这条。
    Tag,
    /// 链接的 `(...)` 那一半丢掉;`[...]` 里的文字照念。
    LinkUrl,
}

pub struct Speakable {
    in_fence: bool,
    /// 开围栏用的是哪个记号 —— ``` 开的要 ``` 关,不能被 ~~~ 关掉。
    fence_char: char,
    at_line_start: bool,
    prefix: String,
    /// 这一行整行不要(围栏记号行、分隔线、围栏里的每一行)。
    drop_line: bool,
    /// 这一行是表格行 —— 行尾要判断它是不是 `|---|---|` 那种分隔行。
    table_row: bool,
    buf: String,
    inline: Inline,
    /// `<` 之后还不知道是不是标签,先扣着。
    pending_lt: bool,
    /// `]` 之后还不知道后面跟不跟 `(`。
    pending_rb: bool,
    said_anything: bool,
    saw_cjk: bool,
}

impl Default for Speakable {
    fn default() -> Self {
        Self::new()
    }
}

impl Speakable {
    pub fn new() -> Self {
        Self {
            in_fence: false,
            fence_char: '`',
            at_line_start: true,
            prefix: String::new(),
            drop_line: false,
            table_row: false,
            buf: String::new(),
            inline: Inline::Text,
            pending_lt: false,
            pending_rb: false,
            said_anything: false,
            saw_cjk: false,
        }
    }

    /// 这一轮到目前为止念出过东西没有。通篇代码时是 false。
    pub fn said_anything(&self) -> bool {
        self.said_anything
    }

    /// 念过的内容里有没有中日韩文字。兜底那句要按它挑语言。
    pub fn saw_cjk(&self) -> bool {
        self.saw_cjk
    }

    /// 喂一段 delta,拿回这一步凑齐的句子。
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        let mut out = Vec::new();
        for ch in delta.chars() {
            self.feed(ch, &mut out);
        }
        out
    }

    /// 流结束:把攒了一半的东西吐出来。
    pub fn finish(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if self.at_line_start {
            let prefix = std::mem::take(&mut self.prefix);
            self.open_line(&prefix, &mut out);
        }
        self.end_line(&mut out);
        out
    }

    fn feed(&mut self, ch: char, out: &mut Vec<String>) {
        // 语言看的是**原始**回答,不是念出来的那部分:通篇代码的一轮一句都念不
        // 出来,而兜底那句仍然要挑对语言。
        if !self.saw_cjk && is_cjk(ch) {
            self.saw_cjk = true;
        }
        if ch == '\r' {
            return;
        }
        if ch == '\n' {
            if self.at_line_start {
                // 整行都是记号(或空行):没有正文进来过,先按块记号判定这一行 ——
                // 围栏的开关就落在这里。
                let prefix = std::mem::take(&mut self.prefix);
                self.open_line(&prefix, out);
            }
            self.end_line(out);
            return;
        }

        if self.at_line_start {
            if is_marker_char(ch) {
                self.prefix.push(ch);
                return;
            }
            let prefix = std::mem::take(&mut self.prefix);
            self.open_line(&prefix, out);
            if self.drop_line {
                return;
            }
            self.emit_char(ch, out);
            return;
        }

        if self.drop_line {
            return;
        }
        self.emit_char(ch, out);
    }

    /// 判定一行的开头,并把该丢的记号丢掉。
    ///
    /// 判不出来时什么都不丢 —— 「2024 年那次」的行首也是数字加空格,把它当成
    /// 列表记号吃掉的话,念出来就少了一年。
    fn open_line(&mut self, prefix: &str, out: &mut Vec<String>) {
        self.at_line_start = false;
        self.drop_line = false;
        self.table_row = false;
        let body = prefix.trim_start_matches([' ', '\t']);

        // 围栏。开与关都用这一行,行本身不念。
        let fence = if body.starts_with("```") {
            Some('`')
        } else if body.starts_with("~~~") {
            Some('~')
        } else {
            None
        };
        if let Some(c) = fence {
            if self.in_fence {
                if self.fence_char == c {
                    self.in_fence = false;
                }
            } else {
                self.in_fence = true;
                self.fence_char = c;
            }
            self.drop_line = true;
            return;
        }
        if self.in_fence {
            self.drop_line = true;
            return;
        }

        // 分隔线 `---` / `***` / `___`。
        //
        // 只有整行都是这个字符才算 —— 判定发生在换行处或第一个普通字符处,所以
        // `- 列表项` 到这里时 body 是 `- `,不会被误当成分隔线。
        let hr = body.chars().filter(|c| !c.is_whitespace()).count() >= 3
            && (body.chars().all(|c| c == '-' || c.is_whitespace())
                || body.chars().all(|c| c == '*' || c.is_whitespace())
                || body.chars().all(|c| c == '_' || c.is_whitespace()));
        if hr {
            self.drop_line = true;
            return;
        }

        // 表格行:竖线换成停顿,分隔行在行尾整行丢掉。
        if body.starts_with('|') {
            self.table_row = true;
            for c in body.chars() {
                if c == '|' {
                    self.push_text('，', out);
                } else {
                    self.push_text(c, out);
                }
            }
            return;
        }

        // 标题:井号与其后的空格不念。
        if body.starts_with('#') {
            let rest = body.trim_start_matches('#');
            if rest.is_empty() || rest.starts_with(' ') {
                self.push_str(rest.trim_start(), out);
                return;
            }
        }

        // 引用:`>` 不念。
        if let Some(rest) = body.strip_prefix('>') {
            self.push_str(rest.trim_start(), out);
            return;
        }

        // 列表:记号**加空格**才算记号。`**粗体**` 开头的行不是列表。
        for m in ['-', '*', '+'] {
            if let Some(rest) = body.strip_prefix(m) {
                if rest.starts_with(' ') {
                    self.push_str(rest.trim_start(), out);
                    return;
                }
            }
        }
        let digits: String = body.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            if let Some(rest) = body[digits.len()..].strip_prefix(['.', ')']) {
                if rest.starts_with(' ') {
                    self.push_str(rest.trim_start(), out);
                    return;
                }
            }
        }

        // 认不出来:原样念,只去掉缩进。
        self.push_str(body, out);
    }

    fn push_str(&mut self, s: &str, out: &mut Vec<String>) {
        for c in s.chars() {
            self.emit_char(c, out);
        }
    }

    /// 一行结束。攒着的话在这里发出去 —— 标题、列表项、表格行都没有句末标点,
    /// 不在行尾发就会和下一行黏成一句。
    fn end_line(&mut self, out: &mut Vec<String>) {
        if self.pending_lt {
            self.pending_lt = false;
            self.push_text('<', out);
        }
        self.pending_rb = false;
        // 标签与链接的 URL 那一半都不跨行。
        self.inline = Inline::Text;
        if self.table_row {
            if self.is_table_rule() {
                self.buf.clear();
            } else {
                self.tidy_table_row();
            }
        }
        self.flush(out);
        self.at_line_start = true;
        self.prefix.clear();
        self.drop_line = false;
        self.table_row = false;
    }

    /// 表格行的竖线换成了停顿,但首尾那两根换出来的是多余的,中间的空单元格
    /// 会换出连着的两个 —— 念出来是一串没有内容的停顿。
    fn tidy_table_row(&mut self) {
        let cells: Vec<&str> = self
            .buf
            .split('，')
            .map(|c| c.trim())
            .filter(|c| !c.is_empty())
            .collect();
        self.buf = cells.join("，");
    }

    /// `|---|:--:|` 这种分隔行,去掉竖线之后只剩横线和冒号。
    fn is_table_rule(&self) -> bool {
        !self.buf.is_empty()
            && self
                .buf
                .chars()
                .all(|c| matches!(c, '-' | ':' | '，' | ' ' | '='))
    }

    fn emit_char(&mut self, ch: char, out: &mut Vec<String>) {
        // `<` 之后要看下一个字符才知道是不是标签。
        if self.pending_lt {
            self.pending_lt = false;
            if ch.is_ascii_alphabetic() || ch == '/' || ch == '!' {
                self.inline = Inline::Tag;
                return;
            }
            self.push_text('<', out);
        }
        // `]` 之后紧跟 `(` 才是链接;否则那个方括号本来就是正文的一部分。
        if self.pending_rb {
            self.pending_rb = false;
            if ch == '(' {
                self.inline = Inline::LinkUrl;
                return;
            }
        }

        match self.inline {
            Inline::Tag => {
                if ch == '>' {
                    self.inline = Inline::Text;
                }
                return;
            }
            Inline::LinkUrl => {
                if ch == ')' {
                    self.inline = Inline::Text;
                }
                return;
            }
            Inline::Text => {}
        }

        match ch {
            // 表格的竖线是分隔,念出来该是一个停顿。行首那根在 `open_line` 里已经
            // 换过,行内的这些如果不换,「竖线」会被一个字一个字念出来。
            '|' if self.table_row => self.push_text('，', out),
            '<' => self.pending_lt = true,
            ']' => self.pending_rb = true,
            // 链接与图片的方括号不念,里面的文字念。
            '[' => {}
            // 强调与行内代码的记号不念,包着的字念。
            '*' | '`' | '~' => {}
            _ => self.push_text(ch, out),
        }
    }

    fn push_text(&mut self, ch: char, out: &mut Vec<String>) {
        // 行首空白不进句子,句中的连续空白压成一个。
        if ch.is_whitespace() {
            if self.buf.is_empty() || self.buf.ends_with(' ') {
                return;
            }
            self.buf.push(' ');
            return;
        }
        self.buf.push(ch);
        if SENTENCE_ENDS.contains(&ch) {
            self.flush(out);
        }
    }

    fn flush(&mut self, out: &mut Vec<String>) {
        let taken = std::mem::take(&mut self.buf);
        let s = taken.trim();
        // 只剩记号的残渣不值得念 —— 「一个点」念出来是噪音,不是内容。
        if s.is_empty() || !s.chars().any(|c| c.is_alphanumeric() || is_cjk(c)) {
            return;
        }
        self.said_anything = true;
        out.push(s.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 按任意切分喂进去 —— 过滤器不该关心 delta 边界落在哪。
    fn run(chunks: &[&str]) -> Vec<String> {
        let mut sp = Speakable::new();
        let mut out = Vec::new();
        for c in chunks {
            out.extend(sp.push(c));
        }
        out.extend(sp.finish());
        out
    }

    fn one(text: &str) -> Vec<String> {
        run(&[text])
    }

    /// 这是这次改动的核心要求:模型返回什么就念什么。
    #[test]
    fn plain_prose_is_spoken_as_written() {
        assert_eq!(
            one("好的，这就去看。第二句在这里。"),
            vec!["好的，这就去看。", "第二句在这里。"]
        );
    }

    #[test]
    fn a_fenced_block_is_skipped_but_the_prose_around_it_is_not() {
        let out = one("先说结论。\n\n```rust\nfn main() { println!(\"hi\"); }\n```\n\n就是这样。");
        assert_eq!(out, vec!["先说结论。", "就是这样。"]);
    }

    /// 围栏记号会被切在两个 delta 里,这是常态不是边缘情况。
    #[test]
    fn a_fence_split_across_deltas_still_matches() {
        let out = run(&["前面。\n``", "`js\nvar x = 1;\n``", "`\n后面。"]);
        assert_eq!(out, vec!["前面。", "后面。"]);
    }

    /// ``` 开的围栏不能被 ~~~ 关掉,否则代码会漏出来被念掉。
    #[test]
    fn a_different_fence_marker_does_not_close_the_block() {
        let out = one("前面。\n```\n~~~\nstill code\n```\n后面。");
        assert_eq!(out, vec!["前面。", "后面。"]);
    }

    #[test]
    fn markdown_marks_are_dropped_and_the_words_survive() {
        assert_eq!(one("## 标题"), vec!["标题"]);
        assert_eq!(one("- 第一项"), vec!["第一项"]);
        assert_eq!(one("1. 第一步"), vec!["第一步"]);
        assert_eq!(one("> 引用的话"), vec!["引用的话"]);
        assert_eq!(one("这是**重点**和`代码`。"), vec!["这是重点和代码。"]);
    }

    /// 链接念文字不念地址 —— 把 URL 念出来是这条链路上最难听的一种「完整」。
    #[test]
    fn a_link_speaks_its_text_not_its_url() {
        assert_eq!(
            one("见[官方文档](https://example.com/a/b?c=d)。"),
            vec!["见官方文档。"]
        );
    }

    #[test]
    fn a_bracket_that_is_not_a_link_stays() {
        assert_eq!(one("数组[0]是第一个。"), vec!["数组0是第一个。"]);
    }

    #[test]
    fn residual_tags_are_not_read_aloud() {
        // 早先那套要求模型写 `<speak>`;万一还有谁在写,标签不该被念出来。
        assert_eq!(one("<speak>你好。</speak>"), vec!["你好。"]);
    }

    #[test]
    fn a_less_than_sign_in_prose_survives() {
        assert_eq!(one("当 a < b 时成立。"), vec!["当 a < b 时成立。"]);
    }

    /// 表格念内容不念竖线,分隔行整行不念。
    #[test]
    fn a_table_speaks_its_cells() {
        let out = one("| 名称 | 值 |\n|---|---|\n| 上限 | 三十 |");
        assert_eq!(out, vec!["名称，值", "上限，三十"]);
    }

    #[test]
    fn a_horizontal_rule_is_not_a_list_item() {
        assert_eq!(one("上面。\n---\n下面。"), vec!["上面。", "下面。"]);
        assert_eq!(one("- 是列表"), vec!["是列表"]);
    }

    /// 标题与列表项没有句末标点,不在行尾发就会和下一行黏成一句。
    #[test]
    fn lines_without_punctuation_do_not_run_together() {
        assert_eq!(one("第一项\n第二项"), vec!["第一项", "第二项"]);
    }

    /// 通篇代码时一句都念不出来 —— 调用方据此播兜底那句,而不是静悄悄地什么都没有。
    #[test]
    fn an_answer_that_is_only_code_says_nothing() {
        let mut sp = Speakable::new();
        let out = sp.push("```py\nprint(1)\n```\n");
        assert!(out.is_empty());
        assert!(sp.finish().is_empty());
        assert!(!sp.said_anything());
    }

    #[test]
    fn language_is_taken_from_the_whole_answer_not_just_the_spoken_part() {
        let mut sp = Speakable::new();
        sp.push("Here is the answer.");
        assert!(!sp.saw_cjk());

        let mut sp = Speakable::new();
        sp.push("这是回答。");
        assert!(sp.saw_cjk());

        // 通篇代码:一句都念不出来,但兜底那句仍然该说中文。
        let mut sp = Speakable::new();
        sp.push("```py\n# 打印一下\nprint(1)\n```\n");
        assert!(!sp.said_anything());
        assert!(sp.saw_cjk(), "语言判定不该只看念出来的那部分");
    }

    /// 逐字喂与整段喂必须得到同一串句子。
    #[test]
    fn character_by_character_matches_whole_string() {
        let text =
            "## 标题\n\n先说**结论**。\n\n```rust\nlet x = 1;\n```\n\n见[文档](http://x/y)。\n";
        let whole = one(text);
        let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(|s| s.as_str()).collect();
        assert_eq!(whole, run(&refs));
    }
}
