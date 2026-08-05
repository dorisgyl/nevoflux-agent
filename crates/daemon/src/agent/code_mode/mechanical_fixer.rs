//! MechanicalFixer - Error-driven code fixes without LLM.
//!
//! When a runtime error matches a known pattern, applies a targeted code
//! transform instead of invoking an expensive LLM rewrite call.
//! Complements the auto_fixer (which runs pre-execution on source patterns)
//! by operating post-execution on error patterns.

/// Attempts to fix code based on a runtime error pattern.
///
/// Returns `Some(fixed_code)` if a mechanical fix was applied,
/// `None` if the error doesn't match any known pattern.
pub fn try_fix(
    code: &str,
    error_type: &str,
    error_msg: &str,
    _line: Option<usize>,
) -> Option<String> {
    // Try fixers in priority order; return first that produces different code.
    // `await`, `map`, `filter`, `sorted(key=/reverse=)` and `list.sort(key=)`
    // are native as of Monty v0.0.19 (see `monty_capabilities`), so the errors
    // their repairs keyed off no longer occur.
    let result = fix_max_min_kwargs(code, error_type, error_msg)
        .or_else(|| fix_name_error_reduce(code, error_type, error_msg))
        .or_else(|| fix_name_error_counter(code, error_type, error_msg))
        .or_else(|| fix_name_error_itertools(code, error_type, error_msg))
        // Bare names from `from X import Y` after import stripping:
        .or_else(|| fix_name_error_chain(code, error_type, error_msg))
        .or_else(|| fix_name_error_zip_longest(code, error_type, error_msg))
        .or_else(|| fix_name_error_ordered_dict(code, error_type, error_msg))
        .or_else(|| fix_name_error_deque(code, error_type, error_msg))
        .or_else(|| fix_name_error_islice(code, error_type, error_msg))
        // Helper functions lost during LLM rewrite cycles:
        .or_else(|| fix_name_error_shell_quote(code, error_type, error_msg))
        // open() is gone from here: Monty now surfaces it as an OsCall, which
        // the executor refuses outright, so there is no NameError left to match.
        // Tool name aliases: write→write_file, read→read_file
        .or_else(|| fix_name_error_tool_alias(code, error_type, error_msg))
        // General fallback: strip known module prefixes so the next retry
        // can match a specific name fixer (e.g. functools.reduce → reduce).
        .or_else(|| fix_module_prefix(code, error_type, error_msg));

    // Only return if the fix actually changed the code.
    result.filter(|fixed| fixed != code)
}

// ---------------------------------------------------------------------------
// Pattern 1: SyntaxError from leftover async/await
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Pattern 4: TypeError: sorted() got unexpected keyword argument
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Pattern 4b: TypeError: sort() got unexpected keyword argument (method call)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Pattern 4c: TypeError: max()/min() got unexpected keyword argument
// ---------------------------------------------------------------------------

fn fix_max_min_kwargs(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("TypeError") {
        return None;
    }
    if !error_msg.contains("keyword argument") && !error_msg.contains("unexpected") {
        return None;
    }
    if error_msg.contains("max") {
        return strip_func_keyword_args(code, "max");
    }
    if error_msg.contains("min") {
        return strip_func_keyword_args(code, "min");
    }
    None
}

// ---------------------------------------------------------------------------
// Pattern 5: NameError: name 'reduce' is not defined
// ---------------------------------------------------------------------------

const REDUCE_HELPER: &str = "\
def _reduce_fn(fn, iterable, initial=None):
    it = list(iterable)
    if initial is not None:
        acc = initial
        start = 0
    else:
        acc = it[0]
        start = 1
    for i in range(start, len(it)):
        acc = fn(acc, it[i])
    return acc
";

fn fix_name_error_reduce(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "reduce") {
        return None;
    }
    if !code.contains("reduce(") {
        return None;
    }
    let replaced = replace_function_calls(code, "reduce", "_reduce_fn");
    if replaced == code {
        return None;
    }
    Some(format!("{}\n{}", REDUCE_HELPER.trim(), replaced))
}

// ---------------------------------------------------------------------------
// Pattern 6: NameError: name 'Counter' is not defined
// ---------------------------------------------------------------------------

const COUNTER_HELPER: &str = "\
def _counter_fn(items):
    counts = {}
    for item in items:
        if item in counts:
            counts[item] = counts[item] + 1
        else:
            counts[item] = 1
    return counts
";

fn fix_name_error_counter(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "Counter") {
        return None;
    }
    if !code.contains("Counter(") {
        return None;
    }
    let replaced = replace_function_calls(code, "Counter", "_counter_fn");
    if replaced == code {
        return None;
    }
    Some(format!("{}\n{}", COUNTER_HELPER.trim(), replaced))
}

// ---------------------------------------------------------------------------
// Pattern 7: NameError: name 'itertools' is not defined
// ---------------------------------------------------------------------------

fn fix_name_error_itertools(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "itertools") {
        return None;
    }
    let mut fixed = code.to_string();
    let mut changed = false;

    // itertools.chain(a, b) → list(a) + list(b)
    while fixed.contains("itertools.chain(") {
        if let Some(start) = fixed.find("itertools.chain(") {
            let open = start + "itertools.chain(".len();
            if let Some(close) = find_matching_paren(&fixed, open - 1) {
                let args = &fixed[open..close];
                // Split on top-level comma
                let parts = split_top_level_args(args);
                let replacement = parts
                    .iter()
                    .map(|p| format!("list({})", p.trim()))
                    .collect::<Vec<_>>()
                    .join(" + ");
                fixed = format!("{}{}{}", &fixed[..start], replacement, &fixed[close + 1..]);
                changed = true;
            } else {
                break;
            }
        }
    }

    // itertools.islice(it, n) → list(it)[:n]
    while fixed.contains("itertools.islice(") {
        if let Some(start) = fixed.find("itertools.islice(") {
            let open = start + "itertools.islice(".len();
            if let Some(close) = find_matching_paren(&fixed, open - 1) {
                let args = &fixed[open..close];
                let parts = split_top_level_args(args);
                let replacement = if parts.len() == 2 {
                    format!("list({})[:{}]", parts[0].trim(), parts[1].trim())
                } else if parts.len() == 3 {
                    format!(
                        "list({})[{}:{}]",
                        parts[0].trim(),
                        parts[1].trim(),
                        parts[2].trim()
                    )
                } else {
                    break; // Can't handle this form
                };
                fixed = format!("{}{}{}", &fixed[..start], replacement, &fixed[close + 1..]);
                changed = true;
            } else {
                break;
            }
        }
    }

    // Other itertools.X( → strip prefix, leave as-is (will likely fail but
    // gives a more descriptive error than "itertools not defined")
    if !changed && fixed.contains("itertools.") {
        fixed = fixed.replace("itertools.", "");
        changed = true;
    }

    if changed {
        Some(fixed)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Pattern 8: NameError: name 'chain' is not defined (bare import)
// ---------------------------------------------------------------------------

fn fix_name_error_chain(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "chain") {
        return None;
    }
    if !code.contains("chain(") {
        return None;
    }
    // Rename to temp name to avoid false positives in the while loop
    let renamed = replace_function_calls(code, "chain", "__chain_tmp");
    if renamed == code {
        return None;
    }
    // Replace __chain_tmp(a, b, ...) → list(a) + list(b) + ...
    let mut fixed = renamed;
    let mut changed = false;

    while fixed.contains("__chain_tmp(") {
        if let Some(start) = fixed.find("__chain_tmp(") {
            let open = start + "__chain_tmp(".len();
            if let Some(close) = find_matching_paren(&fixed, open - 1) {
                let args = &fixed[open..close];
                let parts = split_top_level_args(args);
                let replacement = parts
                    .iter()
                    .map(|p| format!("list({})", p.trim()))
                    .collect::<Vec<_>>()
                    .join(" + ");
                fixed = format!("{}{}{}", &fixed[..start], replacement, &fixed[close + 1..]);
                changed = true;
            } else {
                break;
            }
        }
    }

    if changed {
        Some(fixed)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Pattern 9: NameError: name 'zip_longest' is not defined (bare import)
// ---------------------------------------------------------------------------

const ZIP_LONGEST_HELPER: &str = "\
def _zip_longest_fn(a, b, fillvalue=None):
    la = list(a)
    lb = list(b)
    ml = len(la)
    if len(lb) > ml:
        ml = len(lb)
    result = []
    for i in range(ml):
        va = la[i] if i < len(la) else fillvalue
        vb = lb[i] if i < len(lb) else fillvalue
        result.append([va, vb])
    return result
";

fn fix_name_error_zip_longest(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "zip_longest") {
        return None;
    }
    if !code.contains("zip_longest(") {
        return None;
    }
    let replaced = replace_function_calls(code, "zip_longest", "_zip_longest_fn");
    if replaced == code {
        return None;
    }
    Some(format!("{}\n{}", ZIP_LONGEST_HELPER.trim(), replaced))
}

// ---------------------------------------------------------------------------
// Pattern 10: NameError: name 'OrderedDict' is not defined (bare import)
// ---------------------------------------------------------------------------

fn fix_name_error_ordered_dict(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "OrderedDict") {
        return None;
    }
    if !code.contains("OrderedDict(") {
        return None;
    }
    // OrderedDict → dict (Python 3.7+ dicts preserve insertion order)
    let replaced = replace_function_calls(code, "OrderedDict", "dict");
    if replaced == code {
        return None;
    }
    Some(replaced)
}

// ---------------------------------------------------------------------------
// Pattern 11: NameError: name 'deque' is not defined (bare import)
// ---------------------------------------------------------------------------

fn fix_name_error_deque(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "deque") {
        return None;
    }
    if !code.contains("deque(") {
        return None;
    }
    // deque → list (loses appendleft/popleft but covers basic creation)
    let replaced = replace_function_calls(code, "deque", "list");
    if replaced == code {
        return None;
    }
    Some(replaced)
}

// ---------------------------------------------------------------------------
// Pattern 12: NameError: name 'islice' is not defined (bare import)
// ---------------------------------------------------------------------------

fn fix_name_error_islice(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "islice") {
        return None;
    }
    if !code.contains("islice(") {
        return None;
    }
    // Rename to temp to avoid false positives
    let renamed = replace_function_calls(code, "islice", "__islice_tmp");
    if renamed == code {
        return None;
    }
    // Replace __islice_tmp(it, n) → list(it)[:n]
    // Replace __islice_tmp(it, start, stop) → list(it)[start:stop]
    let mut fixed = renamed;
    let mut changed = false;

    while fixed.contains("__islice_tmp(") {
        if let Some(start) = fixed.find("__islice_tmp(") {
            let open = start + "__islice_tmp(".len();
            if let Some(close) = find_matching_paren(&fixed, open - 1) {
                let args = &fixed[open..close];
                let parts = split_top_level_args(args);
                let replacement = if parts.len() == 2 {
                    format!("list({})[:{}]", parts[0].trim(), parts[1].trim())
                } else if parts.len() == 3 {
                    format!(
                        "list({})[{}:{}]",
                        parts[0].trim(),
                        parts[1].trim(),
                        parts[2].trim()
                    )
                } else {
                    break; // Can't handle this form
                };
                fixed = format!("{}{}{}", &fixed[..start], replacement, &fixed[close + 1..]);
                changed = true;
            } else {
                break;
            }
        }
    }

    if changed {
        Some(fixed)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Pattern 13: NameError: name '_shell_quote' is not defined
// ---------------------------------------------------------------------------

const SHELL_QUOTE_HELPER: &str = "\
def _shell_quote(s):
    return \"'\" + str(s).replace(\"'\", \"'\\\\''\" ) + \"'\"
";

fn fix_name_error_shell_quote(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    if !extract_undefined_name(error_msg).is_some_and(|n| n == "_shell_quote") {
        return None;
    }
    // Don't check for _shell_quote( in code — it's called indirectly by
    // auto_fixer-generated helpers like _re_findall, _datetime_strptime, etc.
    // Just inject the definition at the top.
    if code.contains("def _shell_quote(") {
        return None; // Already defined
    }
    Some(format!("{}\n{}", SHELL_QUOTE_HELPER.trim(), code))
}

// ---------------------------------------------------------------------------
// Pattern 15: NameError: name 'open' is not defined
// Monty has no `open()` builtin; replace with write_file/read_file tools.
// ---------------------------------------------------------------------------

struct WithOpenBlock {
    path_expr: String,
    var_name: String,
    is_write: bool,
}

// ---------------------------------------------------------------------------
// Pattern 16: NameError for tool name aliases (write→write_file, read→read_file)
// LLMs use `write(path, content)` / `read(path)` because the system prompt
// says "Use `write` for new files" — but the actual Monty tool functions
// are `write_file` and `read_file`.
// ---------------------------------------------------------------------------

fn fix_name_error_tool_alias(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    let name = extract_undefined_name(error_msg)?;
    match name.as_str() {
        "write" if code.contains("write(") => {
            Some(replace_function_calls(code, "write", "write_file"))
        }
        "read" if code.contains("read(") => Some(replace_function_calls(code, "read", "read_file")),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Pattern 14: NameError for known module names (functools, collections)
// ---------------------------------------------------------------------------

fn fix_module_prefix(code: &str, error_type: &str, error_msg: &str) -> Option<String> {
    if !error_type.contains("NameError") {
        return None;
    }
    let name = extract_undefined_name(error_msg)?;
    // Strip module prefixes where the auto_fixer handles the underlying
    // patterns. After stripping, the next retry runs auto_fixer again which
    // rewrites the bare calls (e.g. functools.reduce → reduce → _reduce helper,
    // re.findall → _re_findall helper via run_command).
    //
    // Also handles LLM-rewritten code that re-introduces module prefixes
    // (the rewrite output does NOT pass through auto_fixer's rewrite_* phase).
    match name.as_str() {
        "functools" | "collections" | "re" | "json" | "math" | "os" | "datetime" | "random"
        | "time" => {
            let prefix = format!("{}.", name);
            if code.contains(&*prefix) {
                Some(code.replace(&*prefix, ""))
            } else {
                None
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the undefined name from a NameError message.
/// "name 'map' is not defined" → Some("map")
fn extract_undefined_name(msg: &str) -> Option<String> {
    let start = msg.find('\'')?;
    let rest = &msg[start + 1..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

/// Replace standalone function calls `old_name(` with `new_name(`.
/// Avoids replacing inside strings, method calls (`.old_name(`), or
/// names that are substrings of longer identifiers.
fn replace_function_calls(code: &str, old_name: &str, new_name: &str) -> String {
    let search = format!("{}(", old_name);
    let replace = format!("{}(", new_name);
    let mut result = String::with_capacity(code.len());
    let mut i = 0;
    let bytes = code.as_bytes();

    while i < code.len() {
        if code[i..].starts_with(&search) {
            // Check the character before: must not be alphanumeric, underscore, or dot
            let prev_ok = if i == 0 {
                true
            } else {
                let prev = bytes[i - 1];
                !prev.is_ascii_alphanumeric() && prev != b'_' && prev != b'.'
            };
            if prev_ok {
                result.push_str(&replace);
                i += search.len();
                continue;
            }
        }
        // Safe to index: we only advance by one byte for ASCII, which covers
        // all identifier chars and operators. Multi-byte UTF-8 chars are pushed
        // via the char.
        let ch = code[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }
    result
}

/// Split a comma-separated argument string at top-level commas only.
/// Respects nested parens, brackets, braces, and string literals.
fn split_top_level_args(args: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut in_string = false;
    let mut string_char = '"';
    let mut prev_char = '\0';
    let mut start = 0;

    for (i, ch) in args.char_indices() {
        if in_string {
            if ch == string_char && prev_char != '\\' {
                in_string = false;
            }
            prev_char = ch;
            continue;
        }
        match ch {
            '\'' | '"' => {
                in_string = true;
                string_char = ch;
            }
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&args[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        prev_char = ch;
    }
    parts.push(&args[start..]);
    parts
}

/// General-purpose keyword argument stripper for function calls.
/// Finds `func_name(` calls and removes keyword arguments (key=, reverse=, default=),
/// keeping only positional arguments.
fn strip_func_keyword_args(code: &str, func_name: &str) -> Option<String> {
    let search_pattern = format!("{}(", func_name);
    if !code.contains(&*search_pattern) {
        return None;
    }

    let mut result = String::new();
    let mut changed = false;
    let mut i = 0;

    while i < code.len() {
        if code[i..].starts_with(&search_pattern) {
            let open = i + search_pattern.len();
            if let Some(args_end) = find_matching_paren(code, open - 1) {
                let args_str = &code[open..args_end];
                if args_str.contains("key=")
                    || args_str.contains("reverse=")
                    || args_str.contains("default=")
                {
                    let positional = extract_positional_args(args_str);
                    result.push_str(&search_pattern);
                    result.push_str(positional.trim());
                    result.push(')');
                    i = args_end + 1;
                    changed = true;
                    continue;
                }
            }
        }
        let ch = code[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }

    if changed {
        Some(result)
    } else {
        None
    }
}

/// Find the matching closing paren for the open paren at `open_pos`.
fn find_matching_paren(code: &str, open_pos: usize) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut string_char = '"';
    let mut prev_char = '\0';

    for (offset, ch) in code[open_pos..].char_indices() {
        if in_string {
            // Only close string if quote is not escaped
            if ch == string_char && prev_char != '\\' {
                in_string = false;
            }
            prev_char = ch;
            continue;
        }
        match ch {
            '\'' | '"' => {
                in_string = true;
                string_char = ch;
            }
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open_pos + offset);
                }
            }
            _ => {}
        }
        prev_char = ch;
    }
    None
}

/// Extract all positional arguments from a function args string.
/// Stops at the first `, key=`, `, reverse=`, or `, default=` boundary.
fn extract_positional_args(args: &str) -> &str {
    let mut depth = 0;
    let mut in_string = false;
    let mut string_char = '"';

    for (i, ch) in args.char_indices() {
        if in_string {
            if ch == string_char {
                in_string = false;
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                in_string = true;
                string_char = ch;
            }
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                let rest = args[i + 1..].trim_start();
                if rest.starts_with("key=")
                    || rest.starts_with("reverse=")
                    || rest.starts_with("default=")
                {
                    return &args[..i];
                }
            }
            _ => {}
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- extract_undefined_name ---

    #[test]
    fn test_extract_name_from_error() {
        assert_eq!(
            extract_undefined_name("name 'map' is not defined"),
            Some("map".to_string())
        );
        assert_eq!(
            extract_undefined_name("name 'filter' is not defined"),
            Some("filter".to_string())
        );
        assert_eq!(extract_undefined_name("no quotes here"), None);
    }

    // --- fix_await_syntax ---

    #[test]
    fn test_no_fix_for_unrelated_syntax_error() {
        let code = "x = 1 +";
        let fixed = try_fix(code, "SyntaxError", "unexpected EOF", None);
        assert!(fixed.is_none());
    }

    // --- fix_name_error_map ---

    #[test]
    fn test_fix_map_no_false_positive() {
        // bitmap( should NOT be replaced
        let code = "result = bitmap(data)";
        let fixed = try_fix(code, "NameError", "name 'map' is not defined", None);
        // The code doesn't contain standalone `map(`, so no fix
        assert!(fixed.is_none());
    }

    // --- fix_name_error_filter ---

    // --- fix_sorted_kwargs ---

    #[test]
    fn test_fix_sorted_without_kwargs_unchanged() {
        let code = "result = sorted(items)";
        let fixed = try_fix(
            code,
            "TypeError",
            "sorted() got an unexpected keyword argument",
            None,
        );
        // No kwargs to strip, so no fix
        assert!(fixed.is_none());
    }

    // --- fix_name_error_reduce ---

    #[test]
    fn test_fix_reduce() {
        let code = "total = reduce(lambda a, b: a + b, numbers)";
        let fixed = try_fix(code, "NameError", "name 'reduce' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("def _reduce_fn("));
        assert!(fixed.contains("_reduce_fn(lambda a, b: a + b, numbers)"));
    }

    // --- replace_function_calls ---

    #[test]
    fn test_replace_skips_method_calls() {
        let result = replace_function_calls("obj.map(x)", "map", "_map_fn");
        assert_eq!(result, "obj.map(x)"); // dot prefix → skip
    }

    #[test]
    fn test_replace_skips_longer_names() {
        let result = replace_function_calls("bitmap(x)", "map", "_map_fn");
        assert_eq!(result, "bitmap(x)"); // part of longer name → skip
    }

    #[test]
    fn test_replace_at_line_start() {
        let result = replace_function_calls("map(fn, items)", "map", "_map_fn");
        assert_eq!(result, "_map_fn(fn, items)");
    }

    #[test]
    fn test_replace_after_equals() {
        let result = replace_function_calls("x = map(fn, items)", "map", "_map_fn");
        assert_eq!(result, "x = _map_fn(fn, items)");
    }

    #[test]
    fn test_replace_in_list() {
        let result = replace_function_calls("list(map(fn, items))", "map", "_map_fn");
        assert_eq!(result, "list(_map_fn(fn, items))");
    }

    // --- try_fix returns None for unknown errors ---

    #[test]
    fn test_no_fix_for_unknown_error() {
        let code = "x = 1 / 0";
        let fixed = try_fix(code, "ZeroDivisionError", "division by zero", None);
        assert!(fixed.is_none());
    }

    #[test]
    fn test_no_fix_for_attribute_error() {
        let code = "x.foo()";
        let fixed = try_fix(
            code,
            "AttributeError",
            "'int' object has no attribute 'foo'",
            None,
        );
        assert!(fixed.is_none());
    }

    // --- find_matching_paren ---

    #[test]
    fn test_find_matching_paren_simple() {
        assert_eq!(find_matching_paren("(abc)", 0), Some(4));
    }

    #[test]
    fn test_find_matching_paren_nested() {
        assert_eq!(find_matching_paren("(a(b)c)", 0), Some(6));
    }

    #[test]
    fn test_find_matching_paren_with_string() {
        assert_eq!(find_matching_paren("(a, ')')", 0), Some(7));
    }

    #[test]
    fn test_find_matching_paren_with_escaped_quote() {
        // The \' inside the string should not close it
        assert_eq!(find_matching_paren(r"(a, 'it\'s')", 0), Some(11));
    }

    // --- fix_await: async for / async with ---

    // --- fix_name_error_counter ---

    #[test]
    fn test_fix_counter() {
        let code = "counts = Counter(words)";
        let fixed = try_fix(code, "NameError", "name 'Counter' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("def _counter_fn("));
        assert!(fixed.contains("_counter_fn(words)"));
    }

    #[test]
    fn test_fix_counter_no_false_positive() {
        let code = "obj.Counter(x)";
        let fixed = try_fix(code, "NameError", "name 'Counter' is not defined", None);
        // Method call — should not be replaced
        assert!(fixed.is_none());
    }

    // --- fix_name_error_itertools ---

    #[test]
    fn test_fix_itertools_chain() {
        let code = "result = itertools.chain(a, b)";
        let fixed = try_fix(code, "NameError", "name 'itertools' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("list(a) + list(b)"));
        assert!(!fixed.contains("itertools"));
    }

    #[test]
    fn test_fix_itertools_islice() {
        let code = "result = itertools.islice(gen, 5)";
        let fixed = try_fix(code, "NameError", "name 'itertools' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("list(gen)[:5]"));
    }

    #[test]
    fn test_fix_itertools_other() {
        // Unknown itertools function — strip prefix as fallback
        let code = "result = itertools.product(a, b)";
        let fixed = try_fix(code, "NameError", "name 'itertools' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("product(a, b)"));
        assert!(!fixed.contains("itertools."));
    }

    // --- split_top_level_args ---

    #[test]
    fn test_split_args_simple() {
        let parts = split_top_level_args("a, b, c");
        assert_eq!(parts, vec!["a", " b", " c"]);
    }

    #[test]
    fn test_split_args_nested() {
        let parts = split_top_level_args("f(a, b), c");
        assert_eq!(parts, vec!["f(a, b)", " c"]);
    }

    // --- fix_sort_method_kwargs ---

    #[test]
    fn test_fix_sort_method_no_kwargs_unchanged() {
        let code = "items.sort()";
        let fixed = try_fix(
            code,
            "TypeError",
            "sort() got an unexpected keyword argument",
            None,
        );
        assert!(fixed.is_none());
    }

    // --- fix_max_min_kwargs ---

    #[test]
    fn test_fix_max_with_key() {
        let code = "result = max(items, key=lambda x: x[1])";
        let fixed = try_fix(
            code,
            "TypeError",
            "max() got an unexpected keyword argument 'key'",
            None,
        );
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "result = max(items)");
    }

    #[test]
    fn test_fix_min_with_key() {
        let code = "result = min(scores, key=lambda x: x['value'])";
        let fixed = try_fix(
            code,
            "TypeError",
            "min() got an unexpected keyword argument 'key'",
            None,
        );
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "result = min(scores)");
    }

    #[test]
    fn test_fix_max_with_default() {
        let code = "result = max(items, default=0)";
        let fixed = try_fix(
            code,
            "TypeError",
            "max() got an unexpected keyword argument 'default'",
            None,
        );
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "result = max(items)");
    }

    #[test]
    fn test_fix_max_multiple_positional_with_key() {
        // max(a, b, key=fn) → max(a, b)
        let code = "result = max(a, b, key=lambda x: abs(x))";
        let fixed = try_fix(
            code,
            "TypeError",
            "max() got an unexpected keyword argument 'key'",
            None,
        );
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "result = max(a, b)");
    }

    #[test]
    fn test_fix_min_without_kwargs_unchanged() {
        let code = "result = min(items)";
        let fixed = try_fix(
            code,
            "TypeError",
            "min() got an unexpected keyword argument",
            None,
        );
        assert!(fixed.is_none());
    }

    // --- fix_name_error_chain ---

    #[test]
    fn test_fix_bare_chain() {
        let code = "result = list(chain(a, b))";
        let fixed = try_fix(code, "NameError", "name 'chain' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("list(a) + list(b)"));
        assert!(!fixed.contains("chain"));
    }

    #[test]
    fn test_fix_bare_chain_three_args() {
        let code = "all_items = chain(x, y, z)";
        let fixed = try_fix(code, "NameError", "name 'chain' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("list(x) + list(y) + list(z)"));
    }

    #[test]
    fn test_fix_bare_chain_no_false_positive() {
        // blockchain( should NOT be replaced
        let code = "result = blockchain(data)";
        let fixed = try_fix(code, "NameError", "name 'chain' is not defined", None);
        assert!(fixed.is_none());
    }

    // --- fix_name_error_zip_longest ---

    #[test]
    fn test_fix_bare_zip_longest() {
        let code = "result = zip_longest(a, b)";
        let fixed = try_fix(code, "NameError", "name 'zip_longest' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("def _zip_longest_fn("));
        assert!(fixed.contains("_zip_longest_fn(a, b)"));
    }

    #[test]
    fn test_fix_bare_zip_longest_no_false_positive() {
        let code = "my_zip_longest(a, b)";
        let fixed = try_fix(code, "NameError", "name 'zip_longest' is not defined", None);
        assert!(fixed.is_none());
    }

    // --- fix_name_error_ordered_dict ---

    #[test]
    fn test_fix_bare_ordered_dict() {
        let code = "d = OrderedDict()";
        let fixed = try_fix(code, "NameError", "name 'OrderedDict' is not defined", None);
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "d = dict()");
    }

    #[test]
    fn test_fix_ordered_dict_with_args() {
        let code = "d = OrderedDict([(\"a\", 1), (\"b\", 2)])";
        let fixed = try_fix(code, "NameError", "name 'OrderedDict' is not defined", None);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("dict([(\"a\", 1), (\"b\", 2)])"));
    }

    // --- fix_name_error_deque ---

    #[test]
    fn test_fix_bare_deque() {
        let code = "q = deque([1, 2, 3])";
        let fixed = try_fix(code, "NameError", "name 'deque' is not defined", None);
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "q = list([1, 2, 3])");
    }

    #[test]
    fn test_fix_deque_empty() {
        let code = "q = deque()";
        let fixed = try_fix(code, "NameError", "name 'deque' is not defined", None);
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "q = list()");
    }

    // --- fix_name_error_islice ---

    #[test]
    fn test_fix_bare_islice_two_args() {
        let code = "result = islice(gen, 5)";
        let fixed = try_fix(code, "NameError", "name 'islice' is not defined", None);
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "result = list(gen)[:5]");
    }

    #[test]
    fn test_fix_bare_islice_three_args() {
        let code = "result = islice(gen, 2, 10)";
        let fixed = try_fix(code, "NameError", "name 'islice' is not defined", None);
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "result = list(gen)[2:10]");
    }

    #[test]
    fn test_fix_bare_islice_no_false_positive() {
        let code = "result = my_islice(data)";
        let fixed = try_fix(code, "NameError", "name 'islice' is not defined", None);
        assert!(fixed.is_none());
    }

    // --- fix_name_error_shell_quote ---

    #[test]
    fn test_fix_shell_quote_missing() {
        let code = "cmd = 'python3 -c ...' + _shell_quote(pattern)\nout = run_command(cmd)";
        let fixed = try_fix(
            code,
            "NameError",
            "name '_shell_quote' is not defined",
            None,
        );
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("def _shell_quote("));
        assert!(fixed.contains("_shell_quote(pattern)"));
    }

    #[test]
    fn test_fix_shell_quote_already_defined() {
        let code = "def _shell_quote(s):\n    return s\ncmd = _shell_quote(x)";
        let fixed = try_fix(
            code,
            "NameError",
            "name '_shell_quote' is not defined",
            None,
        );
        // Already defined — should not inject again
        assert!(fixed.is_none());
    }

    // --- fix_module_prefix ---

    #[test]
    fn test_fix_functools_prefix() {
        let code = "total = functools.reduce(lambda a, b: a + b, nums)";
        let fixed = try_fix(code, "NameError", "name 'functools' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("reduce(lambda a, b: a + b, nums)"));
        assert!(!fixed.contains("functools."));
    }

    #[test]
    fn test_fix_collections_prefix() {
        let code = "counts = collections.Counter(words)";
        let fixed = try_fix(code, "NameError", "name 'collections' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("Counter(words)"));
        assert!(!fixed.contains("collections."));
    }

    #[test]
    fn test_fix_re_prefix() {
        let code = "matches = re.findall(r'\\d+', text)";
        let fixed = try_fix(code, "NameError", "name 're' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("findall(r'\\d+', text)"));
        assert!(!fixed.contains("re."));
    }

    #[test]
    fn test_fix_json_prefix() {
        let code = "data = json.loads(text)";
        let fixed = try_fix(code, "NameError", "name 'json' is not defined", None);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("loads(text)"));
    }

    #[test]
    fn test_fix_math_prefix() {
        let code = "x = math.sqrt(16)";
        let fixed = try_fix(code, "NameError", "name 'math' is not defined", None);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("sqrt(16)"));
    }

    #[test]
    fn test_fix_datetime_prefix() {
        let code = "now = datetime.datetime.now()";
        let fixed = try_fix(code, "NameError", "name 'datetime' is not defined", None);
        assert!(fixed.is_some());
        // After stripping "datetime.", becomes "now = datetime.now()"
        // (second occurrence gets stripped too → "now = now()" — but auto_fixer
        // handles the rewrite on the next retry cycle)
        assert!(!fixed.unwrap().contains("datetime."));
    }

    #[test]
    fn test_fix_unknown_module_unchanged() {
        // Unknown modules should NOT be stripped (avoid breaking valid code)
        let code = "result = numpy.array([1, 2, 3])";
        let fixed = try_fix(code, "NameError", "name 'numpy' is not defined", None);
        assert!(fixed.is_none());
    }

    // --- fix_name_error_open ---

    #[test]
    fn test_fix_open_no_match_for_unrelated_error() {
        let code = "with open(\"/tmp/x.txt\", \"w\") as f:\n    f.write(\"hi\")";
        let fixed = try_fix(code, "NameError", "name 'foo' is not defined", None);
        assert!(fixed.is_none());
    }

    // --- fix_name_error_tool_alias ---

    #[test]
    fn test_fix_write_to_write_file() {
        let code = "content = \"hello\"\nresult = write(\"/tmp/test.txt\", content)";
        let fixed = try_fix(code, "NameError", "name 'write' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("write_file(\"/tmp/test.txt\", content)"));
        assert!(!fixed.contains("\nresult = write("));
    }

    #[test]
    fn test_fix_read_to_read_file() {
        let code = "data = read(\"/tmp/data.txt\")";
        let fixed = try_fix(code, "NameError", "name 'read' is not defined", None);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("read_file(\"/tmp/data.txt\")"));
    }

    #[test]
    fn test_fix_write_does_not_touch_write_file() {
        // Should not match if the code already uses write_file
        let code = "write_file(\"/tmp/test.txt\", content)";
        let fixed = try_fix(code, "NameError", "name 'write' is not defined", None);
        // write( is not present as a standalone call, only write_file(
        assert!(fixed.is_none());
    }

    #[test]
    fn test_fix_write_preserves_f_write() {
        // f.write() is a method call, should not be renamed
        let code = "f.write(content)\nresult = write(\"/tmp/x.txt\", data)";
        let fixed = try_fix(code, "NameError", "name 'write' is not defined", None);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("f.write(content)")); // method call preserved
        assert!(fixed.contains("write_file(\"/tmp/x.txt\", data)")); // standalone renamed
    }

    // --- strip_func_keyword_args ---

    #[test]
    fn test_strip_func_kwargs_preserves_complex_arg() {
        let fixed = strip_func_keyword_args("max([x for x in data], key=lambda x: x)", "max");
        assert!(fixed.is_some());
        assert_eq!(fixed.unwrap(), "max([x for x in data])");
    }

    // --- extract_positional_args ---

    #[test]
    fn test_extract_positional_with_default() {
        assert_eq!(extract_positional_args("items, default=0"), "items");
    }

    #[test]
    fn test_extract_positional_multiple_args() {
        assert_eq!(extract_positional_args("a, b, key=lambda x: x"), "a, b");
    }

    #[test]
    fn test_extract_positional_no_kwargs() {
        assert_eq!(extract_positional_args("a, b, c"), "a, b, c");
    }
}
