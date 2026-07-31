//! 入口选择与调用代码生成：把用户脚本 + 请求拼成一段可执行源码。

/// 脚本对外暴露的入口。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryPoint {
    /// `def chat(request)` —— 完整契约入口。
    Chat,
    /// `def run(task)` —— 老入口，只收一个字符串。
    Run,
}

/// 判断脚本用哪个入口：定义了 `def chat(` 就用 `chat`，否则回退 `run`。
///
/// 刻意用**静态扫描**而非运行时探测：运行时 `try/except NameError` 会把
/// `chat()` 内部真实的 NameError 一并吞掉，静默降级到 `run()`，这类 bug
/// 极难定位。扫描逐行进行，跳过注释行，并要求 `chat` 后紧跟 `(`，
/// 以免 `chat_helper` / `mychat` 之类的名字误判。
pub fn detect_entry(code: &str) -> EntryPoint {
    for line in code.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("def ") else {
            continue;
        };
        let name = rest.trim_start();
        if let Some(after) = name.strip_prefix("chat") {
            if after.trim_start().starts_with('(') {
                return EntryPoint::Chat;
            }
        }
    }
    EntryPoint::Run
}

/// 把 JSON 值渲染成**合法的 Python 字面量**。
///
/// 不能直接用 `serde_json::to_string`：Monty 是 Python 解释器，JSON 的
/// `true` / `false` / `null` 在那里是未定义的名字，会当场炸。字符串则可以
/// 直接复用 JSON 的转义——两者的转义规则一致，所以字符串内容里的
/// `true` / `null` 不会被误改（这正是逐 token 渲染而非文本替换的原因）。
pub fn to_python_literal(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::Bool(true) => "True".to_string(),
        serde_json::Value::Bool(false) => "False".to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => render_string(s),
        // Collections break across lines. Monty's `CodeLoc::new` panics when a
        // column exceeds u16 (`exception_public.rs:343`), so any exception
        // raised on a line longer than 65535 chars crashes the interpreter
        // thread instead of surfacing as a script error — and a crashed runner
        // never sends a finish frame, leaving the HTTP client with an empty
        // answer and no explanation. A real request (system prompt + tool
        // definitions) is tens of KB, so rendering it on one line is not an
        // edge case. Newlines inside brackets are implicit continuations, so
        // this stays a single expression.
        serde_json::Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(to_python_literal).collect();
            format!("[{}]", inner.join(",\n"))
        }
        serde_json::Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    let key = serde_json::to_string(k).unwrap_or_else(|_| "\"\"".to_string());
                    format!("{key}: {}", to_python_literal(v))
                })
                .collect();
            format!("{{{}}}", inner.join(",\n"))
        }
    }
}

/// Longest string chunk emitted on one line. Well under Monty's u16 column
/// limit, with room for the escaping expansion of the worst-case input.
const STRING_CHUNK_CHARS: usize = 2000;

/// Render a JSON string as a Python expression, split across lines when long.
///
/// A single system prompt can exceed 65535 characters on its own, and Monty's
/// `CodeLoc::new` panics past a u16 column (`exception_public.rs:343`) — taking
/// the interpreter thread, and the request's finish frame, with it. Splitting
/// collections is not enough; the string itself has to be broken up.
///
/// Chunking happens on the **decoded** text and each chunk is escaped
/// separately, so a split can never land inside an escape sequence. Chunks are
/// joined with `+` rather than relying on adjacent-literal concatenation, which
/// is a Python nicety the sandbox need not implement.
fn render_string(s: &str) -> String {
    let encode = |part: &str| serde_json::to_string(part).unwrap_or_else(|_| "\"\"".to_string());
    if s.chars().count() <= STRING_CHUNK_CHARS {
        return encode(s);
    }
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    for (i, ch) in s.chars().enumerate() {
        current.push(ch);
        if (i + 1) % STRING_CHUNK_CHARS == 0 {
            parts.push(encode(&current));
            current.clear();
        }
    }
    if !current.is_empty() {
        parts.push(encode(&current));
    }
    format!("({})", parts.join(" +\n"))
}

/// 拼出交给执行器的完整源码：用户代码 + 一行入口调用。
///
/// 调用表达式放在末尾，其返回值即为 `CodeModeResult.result`
/// （与 `session.rs` 现有的 `run(...)` 拼法一致）。
pub fn build_invocation(
    user_code: &str,
    entry: EntryPoint,
    request: &serde_json::Value,
    task: &str,
) -> String {
    match entry {
        EntryPoint::Chat => {
            format!("{user_code}\n\nchat({})\n", to_python_literal(request))
        }
        EntryPoint::Run => {
            let literal = serde_json::to_string(task).unwrap_or_else(|_| "\"\"".to_string());
            format!("{user_code}\n\nrun({literal})\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_chat_entry() {
        let code = "def chat(request):\n    return {'content': 'hi'}\n";
        assert_eq!(detect_entry(code), EntryPoint::Chat);
    }

    #[test]
    fn falls_back_to_run_when_only_run_defined() {
        let code = "def run(task):\n    return 'hi'\n";
        assert_eq!(detect_entry(code), EntryPoint::Run);
    }

    #[test]
    fn chat_wins_when_both_defined() {
        let code =
            "def run(task):\n    return 'old'\n\ndef chat(request):\n    return {'content': 'new'}\n";
        assert_eq!(detect_entry(code), EntryPoint::Chat);
    }

    #[test]
    fn does_not_match_chat_in_comments_or_other_names() {
        // 注释里提到 chat、以及 chat_helper / mychat 这类名字都不算定义了入口
        let code = "# def chat(request) 这是注释\ndef chat_helper(x):\n    return x\ndef mychat(request):\n    return 1\ndef run(task):\n    return 'ok'\n";
        assert_eq!(detect_entry(code), EntryPoint::Run);
    }

    #[test]
    fn json_true_false_null_become_python_literals() {
        // Monty 是 Python：JSON 的 true/false/null 不是合法字面量
        let v = json!({"stream": true, "quiet": false, "tool_choice": null});
        let lit = to_python_literal(&v);
        assert!(lit.contains("True"), "got {lit}");
        assert!(lit.contains("False"), "got {lit}");
        assert!(lit.contains("None"), "got {lit}");
        assert!(!lit.contains("true"), "got {lit}");
        assert!(!lit.contains("null"), "got {lit}");
    }

    #[test]
    fn strings_containing_keywords_are_not_rewritten() {
        // 关键点：只替换 JSON 的字面量 token，不碰字符串内容
        let v = json!({"task": "把 true 和 null 写进文档"});
        let lit = to_python_literal(&v);
        assert!(lit.contains("把 true 和 null 写进文档"), "got {lit}");
    }

    #[test]
    fn nested_structures_render() {
        let v = json!({"messages": [{"role": "user", "content": "hi", "ok": true}], "n": 3});
        let lit = to_python_literal(&v);
        assert!(lit.starts_with('{') && lit.ends_with('}'), "got {lit}");
        assert!(lit.contains("[{"), "got {lit}");
        assert!(lit.contains("\"role\": \"user\""), "got {lit}");
        assert!(lit.contains("True"), "got {lit}");
    }

    #[test]
    fn quotes_and_newlines_survive() {
        let v = json!({"task": "他说\"你好\"\n换行"});
        let lit = to_python_literal(&v);
        // JSON 的转义规则与 Python 一致，直接复用
        assert!(lit.contains("\\\""), "got {lit}");
        assert!(lit.contains("\\n"), "got {lit}");
    }

    /// Monty panics when a column exceeds u16, so no generated line may come
    /// close to 65535 characters. A real request carries a multi-KB system
    /// prompt plus tool schemas; on one line that crashed the interpreter
    /// thread and the client saw an empty answer.
    #[test]
    fn long_requests_do_not_produce_overlong_lines() {
        let big_prompt = "x".repeat(8000);
        let tools: Vec<serde_json::Value> = (0..40)
            .map(|i| json!({"type": "function", "function": {"name": format!("tool_{i}"), "description": "y".repeat(500)}}))
            .collect();
        let req = json!({
            "messages": [{"role": "system", "content": big_prompt}],
            "tools": tools,
        });
        let out = build_invocation(
            "def chat(request):\n    return {}\n",
            EntryPoint::Chat,
            &req,
            "hi",
        );
        let longest = out.lines().map(|l| l.chars().count()).max().unwrap_or(0);
        assert!(longest < 60000, "longest line was {longest} chars");
        // The whole thing is still much larger than any single line.
        assert!(out.chars().count() > 20000);
    }

    /// A single string can blow the column limit on its own, so it is chunked
    /// too — and the chunks must reassemble to exactly the original text.
    #[test]
    fn very_long_strings_are_split_and_lossless() {
        let original: String = (0..90_000)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let lit = to_python_literal(&json!({"prompt": original.clone()}));
        let longest = lit.lines().map(|l| l.chars().count()).max().unwrap_or(0);
        assert!(longest < 60000, "longest line was {longest}");

        // Reassemble: every chunk is an independently escaped JSON string.
        let joined: String = lit
            .split(" +\n")
            .map(|p| {
                p.trim()
                    .trim_start_matches("{\"prompt\": (")
                    .trim_end_matches(")}")
            })
            .filter(|p| p.starts_with('"'))
            .map(|p| serde_json::from_str::<String>(p).expect("chunk must be a JSON string"))
            .collect();
        assert_eq!(joined, original);
    }

    #[test]
    fn strings_with_escapes_survive_chunking() {
        // A boundary must never land inside an escape sequence: chunking works
        // on decoded text and escapes each piece separately.
        let original: String = "他说\"你好\"\n".repeat(1000);
        let lit = to_python_literal(&json!(original.clone()));
        let joined: String = lit
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split(" +\n")
            .map(|p| serde_json::from_str::<String>(p.trim()).expect("chunk must be a JSON string"))
            .collect();
        assert_eq!(joined, original);
    }

    #[test]
    fn build_invocation_appends_chat_call() {
        let code = "def chat(request):\n    return {'content': 'hi'}\n";
        let req = json!({"task": "你好", "stream": false});
        let out = build_invocation(code, EntryPoint::Chat, &req, "你好");
        assert!(out.starts_with(code), "用户代码必须原样在前");
        assert!(out.contains("\nchat({"), "got {out}");
        assert!(out.contains("False"), "got {out}");
        assert!(out.trim_end().ends_with(')'), "got {out}");
    }

    #[test]
    fn build_invocation_appends_run_call_with_task_only() {
        let code = "def run(task):\n    return 'hi'\n";
        let req = json!({"task": "你好", "stream": false});
        let out = build_invocation(code, EntryPoint::Run, &req, "你好");
        assert!(out.contains("\nrun(\"你好\")"), "got {out}");
        // legacy 入口只拿到 task 字符串，看不到 request
        assert!(!out.contains("stream"), "got {out}");
    }
}
