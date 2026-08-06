//! MontyAutoFixer - Mechanical code transforms for Monty compatibility.
//! Strips imports, decorators, typing annotations, and type:ignore comments.
//!
//! This is Layer 2 of the four-layer constraint pipeline. It applies
//! deterministic text transforms to fix common violations automatically,
//! with zero LLM cost and sub-millisecond execution.

/// Applies mechanical text transforms to Python code before it reaches
/// the Monty interpreter. All transforms are deterministic and preserve
/// the semantic meaning of code that Monty supports.
pub struct MontyAutoFixer;

impl MontyAutoFixer {
    /// Apply all mechanical transforms to the given Python code.
    ///
    /// Transforms applied (Phase 1-3):
    /// 1. Strip markdown artifacts (backticks, language tags, indented code fences)
    /// 2. Rewrite unsupported patterns:
    ///    - `sorted(key=...)` → `_keysort()` helper
    ///    - `map()`/`filter()` → `_map()`/`_filter()` helpers
    ///    - `math.*` → inline expressions or `_math_*` helpers
    ///    - `os.path.*` → `_path_*` helpers
    ///    - `json.dumps()`/`json.loads()` → `_json_dumps()`/`_json_loads()` helpers
    ///    - `functools.reduce()` → `_reduce()` helper
    ///    - `collections.Counter()` → `_counter()` helper
    ///    - `re.*` → `_re_*` helpers (via `run_command` + python3)
    ///    - `datetime.*` → `_datetime_*` helpers (via `run_command` + python3)
    ///    - `random.*` → `_random_*` helpers (via `run_command` + python3)
    /// 3. Per-line: strip imports, decorators, type annotations, type:ignore comments
    /// 4. Collapse excessive leading blank lines from stripped content
    pub fn fix(code: &str) -> String {
        // Phase 1: Strip markdown artifacts that LLMs sometimes include
        let code = Self::strip_markdown_artifacts(code);

        // Phase 2: Shims for what Monty still lacks. Each one is here because
        // `monty_capabilities` proves the real thing is missing — `sorted(key=)`,
        // `map`, `filter`, `math`, `json`, `re` and `datetime` all work natively
        // now, and their rewrites are gone.
        //
        // `os` is native but exposes no `path`, so that shim stays despite the
        // module existing.
        let code = Self::rewrite_os_path(&code);
        let code = Self::rewrite_reduce(&code);
        let code = Self::rewrite_counter(&code);
        // Phase 2b: Bash-bridged stdlib (requires the `run_command` tool, which
        // `AutomationPolicy` withholds when allow_shell is false). Only the
        // modules Monty still has no native version of.
        let code = Self::rewrite_random(&code);
        let code = Self::rewrite_time(&code);

        // Phase 3: Per-line transforms
        let mut result_lines: Vec<String> = Vec::new();

        for line in code.lines() {
            let trimmed = line.trim();

            // Strip the import only for modules this fixer shims. Monty binds
            // a module's names only once imported, so dropping `import json`
            // now breaks code that would otherwise run — while keeping
            // `from collections import Counter` would raise
            // ModuleNotFoundError even though the shim already bound the name.
            if Self::imports_shimmed_module(trimmed) {
                continue;
            }

            // Decorators are NOT stripped. Monty rejects them outright as of
            // v0.0.19, and letting that error through is the point: stripping
            // `@memoize` leaves code that parses, runs, and quietly does
            // something else. Measured on a real deployment — a decorated
            // function returned the undecorated result with nothing logged,
            // which is strictly worse than refusing to load.

            // Strip `# type: ignore` suffixes (with optional bracket annotations)
            if let Some(pos) = line.find("# type: ignore") {
                let before = line[..pos].trim_end();
                if before.is_empty() {
                    // The entire line is just a type: ignore comment, skip it
                    continue;
                }
                result_lines.push(before.to_string());
                continue;
            }

            // Strip type annotations from variable assignments: `x: int = 1` → `x = 1`
            let line = Self::strip_variable_annotation(line);

            result_lines.push(line);
        }

        // Remove leading blank lines that result from stripping
        let start = result_lines
            .iter()
            .position(|l| !l.trim().is_empty())
            .unwrap_or(result_lines.len());
        let trimmed_lines = &result_lines[start..];

        trimmed_lines.join("\n")
    }

    /// Modules this fixer shims, and whose imports therefore have to go.
    ///
    /// Deliberately *not* the complement of Monty's native module list: `os`
    /// is native (so `import os` stays) even though `os.path` is shimmed,
    /// because the shim rewrites the call site rather than the module.
    ///
    /// A module that is neither native nor shimmed — `itertools`,
    /// `subprocess`, `requests` — keeps its import on purpose, so it fails as
    /// a clear ModuleNotFoundError naming the module rather than as a
    /// NameError several lines later.
    const SHIMMED_MODULES: &'static [&'static str] =
        &["collections", "functools", "random", "time"];

    /// Whether `line` imports a module [`Self::SHIMMED_MODULES`] replaces.
    ///
    /// Handles `import X`, `import X as Y`, `import X.Y` and
    /// `from X import Y`.
    fn imports_shimmed_module(line: &str) -> bool {
        let module = if let Some(rest) = line.strip_prefix("import ") {
            rest.split_whitespace().next()
        } else if let Some(rest) = line.strip_prefix("from ") {
            if !line.contains(" import ") {
                return false;
            }
            rest.split_whitespace().next()
        } else {
            return false;
        };

        module
            .map(|m| m.split('.').next().unwrap_or(m))
            .is_some_and(|root| Self::SHIMMED_MODULES.contains(&root))
    }

    /// Strip markdown artifacts that LLMs commonly embed in generated code.
    ///
    /// Handles:
    /// - Leading/trailing backtick fences (```python, ```, ````python-exec`, etc.)
    /// - Lines that are just backticks
    /// - Inline backtick wrapping on individual lines
    fn strip_markdown_artifacts(code: &str) -> String {
        let mut lines: Vec<&str> = code.lines().collect();

        // Remove leading fence line if present (```python, ```python-exec, ```, etc.)
        if let Some(first) = lines.first() {
            let t = first.trim();
            if t.starts_with("```") {
                lines.remove(0);
            }
        }

        // Remove trailing fence line if present
        if let Some(last) = lines.last() {
            let t = last.trim();
            if t.starts_with("```") && !t.contains('=') && !t.contains('(') {
                lines.pop();
            }
        }

        // Remove any remaining lines that are only backticks (e.g., inner fences)
        lines
            .into_iter()
            .filter(|line| {
                let t = line.trim();
                // Keep the line unless it's ONLY backticks (3+)
                !(t.len() >= 3 && t.chars().all(|c| c == '`'))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Strip type annotations from simple variable assignments.
    ///
    /// `x: int = 1` → `x = 1`
    /// `result: list[str] = []` → `result = []`
    ///
    /// Does NOT touch function parameters (handled differently by Monty)
    /// or lines without `=` (bare annotations like `x: int`).
    fn strip_variable_annotation(line: &str) -> String {
        // Only process lines with both `:` and `=`
        let trimmed = line.trim();
        if !trimmed.contains(':') || !trimmed.contains('=') {
            return line.to_string();
        }

        // Skip function definitions
        if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
            return line.to_string();
        }

        // Skip dict literals and slices: `{"key": val}`, `x = a[1:2]`
        // Only match annotations at the top level (before any `=`)
        let eq_pos = match trimmed.find('=') {
            Some(p)
                if p > 0
                    && trimmed.as_bytes().get(p - 1) != Some(&b'!')
                    && trimmed.as_bytes().get(p + 1) != Some(&b'=') =>
            {
                p
            }
            _ => return line.to_string(),
        };

        let before_eq = &trimmed[..eq_pos];
        let after_eq = &trimmed[eq_pos..]; // includes `= value`

        // Check if `before_eq` has a `:` that looks like a type annotation
        // Pattern: `name: type` where name is an identifier
        if let Some(colon_pos) = before_eq.find(':') {
            let var_name = before_eq[..colon_pos].trim();
            // Validate it looks like a variable name (simple check)
            if !var_name.is_empty()
                && var_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                && !var_name.starts_with(|c: char| c.is_ascii_digit())
            {
                // Preserve leading whitespace
                let indent = line.len() - line.trim_start().len();
                let spaces = &line[..indent];
                return format!("{}{} {}", spaces, var_name, after_eq.trim_start());
            }
        }

        line.to_string()
    }

    /// Extract content inside balanced parentheses starting at `open_pos`.
    /// Returns (inner_content, close_paren_position).
    fn extract_balanced_parens(code: &str, open_pos: usize) -> Option<(String, usize)> {
        if code.as_bytes().get(open_pos) != Some(&b'(') {
            return None;
        }
        let mut depth = 1;
        let mut i = open_pos + 1;
        let bytes = code.as_bytes();

        while i < bytes.len() && depth > 0 {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        let inner = &code[open_pos + 1..i];
                        return Some((inner.to_string(), i));
                    }
                }
                b'"' | b'\'' => {
                    // Skip string literals
                    let quote = bytes[i];
                    i += 1;
                    while i < bytes.len() && bytes[i] != quote {
                        if bytes[i] == b'\\' {
                            i += 1; // skip escaped char
                        }
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Rewrite `os.path.join(...)` to a pure Python helper.
    ///
    /// Handles `os.path.join(a, b)`, `os.path.join(a, b, c)` etc.
    /// Also handles `os.path.basename(p)`, `os.path.dirname(p)`,
    /// `os.path.splitext(p)`, `os.path.exists(p)` (always returns True as best-effort).
    fn rewrite_os_path(code: &str) -> String {
        if !code.contains("os.path.") && !code.contains("os.sep") {
            return code.to_string();
        }

        let mut result = code.to_string();
        let mut need_join = false;
        let mut need_basename = false;
        let mut need_dirname = false;
        let mut need_splitext = false;

        // os.path.join(a, b, ...) → _path_join([a, b, ...])
        // We need to extract args and wrap them in a list
        while result.contains("os.path.join(") {
            let pos = match result.find("os.path.join(") {
                Some(p) => p,
                None => break,
            };
            let paren_start = pos + "os.path.join".len();
            if let Some((inner, end)) = Self::extract_balanced_parens(&result, paren_start) {
                need_join = true;
                let replacement = format!("_path_join([{}])", inner.trim());
                result = format!("{}{}{}", &result[..pos], replacement, &result[end + 1..]);
            } else {
                break;
            }
        }
        if result.contains("os.path.basename(") {
            need_basename = true;
            result = result.replace("os.path.basename(", "_path_basename(");
        }
        if result.contains("os.path.dirname(") {
            need_dirname = true;
            result = result.replace("os.path.dirname(", "_path_dirname(");
        }
        if result.contains("os.path.splitext(") {
            need_splitext = true;
            result = result.replace("os.path.splitext(", "_path_splitext(");
        }
        // os.path.exists → always True (best-effort, no filesystem access)
        if result.contains("os.path.exists(") {
            result = result.replace("os.path.exists(", "_path_exists(");
        }
        // os.path.sep → "/"
        result = result.replace("os.path.sep", "\"/\"");
        // os.sep → "/"
        result = result.replace("os.sep", "\"/\"");

        let mut helpers = String::new();
        if need_join {
            helpers.push_str(concat!(
                "def _path_join(parts):\n",
                "    result = \"\"\n",
                "    for p in parts:\n",
                "        if not result or p[0:1] == \"/\":\n",
                "            result = p\n",
                "        else:\n",
                "            if result[-1:] == \"/\":\n",
                "                result = result + p\n",
                "            else:\n",
                "                result = result + \"/\" + p\n",
                "    return result\n",
            ));
        }
        if need_basename {
            helpers.push_str(concat!(
                "def _path_basename(p):\n",
                "    idx = p.rfind(\"/\")\n",
                "    if idx < 0:\n",
                "        return p\n",
                "    return p[idx + 1:]\n",
            ));
        }
        if need_dirname {
            helpers.push_str(concat!(
                "def _path_dirname(p):\n",
                "    idx = p.rfind(\"/\")\n",
                "    if idx < 0:\n",
                "        return \"\"\n",
                "    return p[:idx]\n",
            ));
        }
        if need_splitext {
            helpers.push_str(concat!(
                "def _path_splitext(p):\n",
                "    idx = p.rfind(\".\")\n",
                "    slash = p.rfind(\"/\")\n",
                "    if idx < 0 or idx < slash:\n",
                "        return [p, \"\"]\n",
                "    return [p[:idx], p[idx:]]\n",
            ));
        }
        // _path_exists is a no-op stub
        if result.contains("_path_exists(") && !helpers.contains("_path_exists") {
            helpers.push_str("def _path_exists(p):\n    return True\n");
        }

        if helpers.is_empty() {
            result
        } else {
            format!("{}\n{}", helpers.trim(), result)
        }
    }

    /// Rewrite `reduce(func, iterable)` and `functools.reduce(func, iterable)`.
    ///
    /// Injects a `_reduce` helper and replaces calls.
    fn rewrite_reduce(code: &str) -> String {
        let has_reduce = Self::has_standalone_call(code, "reduce");
        let has_functools_reduce = code.contains("functools.reduce(");

        if !has_reduce && !has_functools_reduce {
            return code.to_string();
        }

        let helper = concat!(
            "def _reduce(fn, items, initial=None):\n",
            "    items = list(items)\n",
            "    if initial is not None:\n",
            "        acc = initial\n",
            "        start = 0\n",
            "    else:\n",
            "        acc = items[0]\n",
            "        start = 1\n",
            "    for i in range(start, len(items)):\n",
            "        acc = fn(acc, items[i])\n",
            "    return acc\n",
        );

        let mut result = code.to_string();

        // functools.reduce(...) → _reduce(...)
        if has_functools_reduce {
            result = result.replace("functools.reduce(", "_reduce(");
        }

        // standalone reduce(...) → _reduce(...)
        if has_reduce {
            result = Self::replace_standalone_call(&result, "reduce", "_reduce");
        }

        format!("{}\n{}", helper.trim(), result)
    }

    /// Rewrite `Counter(iterable)` and `collections.Counter(iterable)`.
    ///
    /// Injects a `_counter` helper and replaces calls.
    fn rewrite_counter(code: &str) -> String {
        let has_counter = Self::has_standalone_call(code, "Counter");
        let has_collections_counter = code.contains("collections.Counter(");

        if !has_counter && !has_collections_counter {
            return code.to_string();
        }

        let helper = concat!(
            "def _counter(items):\n",
            "    counts = {}\n",
            "    for item in items:\n",
            "        if item in counts:\n",
            "            counts[item] = counts[item] + 1\n",
            "        else:\n",
            "            counts[item] = 1\n",
            "    return counts\n",
        );

        let mut result = code.to_string();

        // collections.Counter(...) → _counter(...)
        if has_collections_counter {
            result = result.replace("collections.Counter(", "_counter(");
        }

        // standalone Counter(...) → _counter(...)
        if has_counter {
            result = Self::replace_standalone_call(&result, "Counter", "_counter");
        }

        format!("{}\n{}", helper.trim(), result)
    }

    /// Rewrite `random.*` calls to use `run_command`.
    ///
    /// Handles:
    /// - `random.randint(a, b)` → random integer
    /// - `random.choice(seq)` → random element (uses len)
    /// - `random.random()` → float 0..1
    /// - `random.shuffle(lst)` → shuffled in place
    /// - `random.sample(pop, k)` → k random elements
    fn rewrite_random(code: &str) -> String {
        if !code.contains("random.") {
            return code.to_string();
        }

        let needs_randint = code.contains("random.randint(");
        let needs_choice = code.contains("random.choice(");
        let needs_random = code.contains("random.random(");
        let needs_shuffle = code.contains("random.shuffle(");
        let needs_sample = code.contains("random.sample(");

        if !needs_randint && !needs_choice && !needs_random && !needs_shuffle && !needs_sample {
            return code.to_string();
        }

        let mut helpers = String::new();
        let mut result = code.to_string();

        if needs_randint {
            helpers.push_str(concat!(
                "def _random_randint(a, b):\n",
                "    cmd = 'python3 -c \"import random,sys; print(random.randint(int(sys.argv[1]),int(sys.argv[2])))\" '\n",
                "    cmd = cmd + str(a) + ' ' + str(b)\n",
                "    out = run_command(cmd)\n",
                "    return int(out.strip())\n",
            ));
            result = result.replace("random.randint(", "_random_randint(");
        }

        if needs_random {
            helpers.push_str(concat!(
                "def _random_random():\n",
                "    out = run_command('python3 -c \"import random; print(random.random())\"')\n",
                "    return float(out.strip())\n",
            ));
            result = result.replace("random.random()", "_random_random()");
        }

        if needs_choice {
            // For random.choice, we use the length to pick a random index
            helpers.push_str(concat!(
                "def _random_choice(seq):\n",
                "    items = list(seq)\n",
                "    idx = _random_randint(0, len(items) - 1)\n",
                "    return items[idx]\n",
            ));
            // Ensure randint helper is also available
            if !needs_randint {
                helpers.push_str(concat!(
                    "def _random_randint(a, b):\n",
                    "    cmd = 'python3 -c \"import random,sys; print(random.randint(int(sys.argv[1]),int(sys.argv[2])))\" '\n",
                    "    cmd = cmd + str(a) + ' ' + str(b)\n",
                    "    out = run_command(cmd)\n",
                    "    return int(out.strip())\n",
                ));
            }
            result = result.replace("random.choice(", "_random_choice(");
        }

        if needs_shuffle {
            // Shuffle in-place using Fisher-Yates with random.randint
            helpers.push_str(concat!(
                "def _random_shuffle(lst):\n",
                "    n = len(lst)\n",
                "    for i in range(n - 1, 0, -1):\n",
                "        j = _random_randint(0, i)\n",
                "        tmp = lst[i]\n",
                "        lst[i] = lst[j]\n",
                "        lst[j] = tmp\n",
            ));
            if !needs_randint && !needs_choice {
                helpers.push_str(concat!(
                    "def _random_randint(a, b):\n",
                    "    cmd = 'python3 -c \"import random,sys; print(random.randint(int(sys.argv[1]),int(sys.argv[2])))\" '\n",
                    "    cmd = cmd + str(a) + ' ' + str(b)\n",
                    "    out = run_command(cmd)\n",
                    "    return int(out.strip())\n",
                ));
            }
            result = result.replace("random.shuffle(", "_random_shuffle(");
        }

        if needs_sample {
            helpers.push_str(concat!(
                "def _random_sample(population, k):\n",
                "    items = list(population)\n",
                "    result = []\n",
                "    for _ in range(k):\n",
                "        idx = _random_randint(0, len(items) - 1)\n",
                "        result.append(items[idx])\n",
                "        items = items[:idx] + items[idx + 1:]\n",
                "    return result\n",
            ));
            if !needs_randint && !needs_choice && !needs_shuffle {
                helpers.push_str(concat!(
                    "def _random_randint(a, b):\n",
                    "    cmd = 'python3 -c \"import random,sys; print(random.randint(int(sys.argv[1]),int(sys.argv[2])))\" '\n",
                    "    cmd = cmd + str(a) + ' ' + str(b)\n",
                    "    out = run_command(cmd)\n",
                    "    return int(out.strip())\n",
                ));
            }
            result = result.replace("random.sample(", "_random_sample(");
        }

        format!("{}\n{}", helpers.trim(), result)
    }

    /// Rewrite `time.sleep(N)` → `run_command("sleep N")`.
    /// Also handles `time.time()` for elapsed-time patterns.
    fn rewrite_time(code: &str) -> String {
        if !code.contains("time.") {
            return code.to_string();
        }

        let mut helpers = String::new();
        let mut result = code.to_string();

        if code.contains("time.sleep(") {
            helpers.push_str(concat!(
                "def _time_sleep(seconds):\n",
                "    run_command(\"sleep \" + str(seconds))\n",
            ));
            result = result.replace("time.sleep(", "_time_sleep(");
        }

        if code.contains("time.time(") {
            helpers.push_str(concat!(
                "def _time_time():\n",
                "    out = run_command('python3 -c \"import time; print(time.time())\"')\n",
                "    return float(out.strip())\n",
            ));
            result = result.replace("time.time()", "_time_time()");
        }

        if helpers.is_empty() {
            return result;
        }

        format!("{}\n{}", helpers.trim(), result)
    }

    /// Check if `code` contains a standalone function call `name(` that is NOT
    /// a method call (`.name(`) or part of another identifier (`other_name(`).
    fn has_standalone_call(code: &str, name: &str) -> bool {
        let pattern = format!("{}(", name);
        let bytes = code.as_bytes();
        let pat_bytes = pattern.as_bytes();
        let mut i = 0;
        while i + pat_bytes.len() <= bytes.len() {
            if &bytes[i..i + pat_bytes.len()] == pat_bytes {
                // Check preceding char: must not be alphanumeric, `_`, or `.`
                if i == 0 || {
                    let prev = bytes[i - 1];
                    !prev.is_ascii_alphanumeric() && prev != b'_' && prev != b'.'
                } {
                    // Check we're not inside a string (simple heuristic: count quotes before)
                    let before = &code[..i];
                    let single_quotes = before.chars().filter(|&c| c == '\'').count();
                    let double_quotes = before.chars().filter(|&c| c == '"').count();
                    if single_quotes % 2 == 0 && double_quotes % 2 == 0 {
                        return true;
                    }
                }
            }
            i += 1;
        }
        false
    }

    /// Replace standalone calls of `name(` with `replacement(` in code.
    /// Does NOT replace method calls (`.name(`) or identifiers containing name.
    fn replace_standalone_call(code: &str, name: &str, replacement: &str) -> String {
        let pat = format!("{}(", name);
        let rep = format!("{}(", replacement);
        let bytes = code.as_bytes();
        let pat_bytes = pat.as_bytes();
        let mut result = String::new();
        let mut i = 0;

        while i < bytes.len() {
            if i + pat_bytes.len() <= bytes.len() && &bytes[i..i + pat_bytes.len()] == pat_bytes {
                let is_standalone = i == 0 || {
                    let prev = bytes[i - 1];
                    !prev.is_ascii_alphanumeric() && prev != b'_' && prev != b'.'
                };
                if is_standalone {
                    result.push_str(&rep);
                    i += pat_bytes.len();
                    continue;
                }
            }
            // Use char iteration to preserve multi-byte UTF-8 characters
            let ch = code[i..].chars().next().unwrap();
            result.push(ch);
            i += ch.len_utf8();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_strip_await_when_absent() {
        let code = "x = browser_eval_js(\"test\")\nprint(x)";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    #[test]
    /// Imports of natively-supported modules must survive: Monty binds those
    /// names only once imported, so stripping the line breaks working code.
    fn test_native_imports_are_kept() {
        let code = "import os\nimport sys\nx = 1\nfrom pathlib import Path\ny = 2";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    /// Imports of modules this fixer shims must still go — the shim already
    /// bound the name, and Monty has no such module to import.
    #[test]
    fn test_shimmed_imports_are_stripped() {
        for code in [
            "from collections import Counter\nx = 1",
            "import functools\nx = 1",
            "import random\nx = 1",
            "import time as t\nx = 1",
        ] {
            let fixed = MontyAutoFixer::fix(code);
            assert!(
                !fixed.contains("import"),
                "shimmed import should be stripped from {code:?}, got {fixed:?}"
            );
        }
    }

    /// A module that is neither native nor shimmed keeps its import, so it
    /// fails as a ModuleNotFoundError naming the module rather than as a
    /// NameError somewhere further down.
    #[test]
    fn test_unknown_imports_are_left_to_fail_loudly() {
        let fixed = MontyAutoFixer::fix("import itertools\nx = 1");
        assert!(fixed.contains("import itertools"), "got {fixed:?}");
    }

    #[test]
    fn test_strip_typing_annotations() {
        let code = "from typing import List, Dict\nx: List[int] = [1, 2]";
        let fixed = MontyAutoFixer::fix(code);
        // `typing` is native, so its import stays; the annotation still goes,
        // because Monty does not parse annotated assignments.
        assert_eq!(fixed, "from typing import List, Dict\nx = [1, 2]");
    }

    /// Decorators must reach Monty untouched so it can reject them.
    ///
    /// Stripping `@memoize` leaves code that parses and runs and quietly
    /// returns the undecorated result. Measured on a real deployment: a
    /// decorated function came back with the wrong value and nothing was
    /// logged. Monty refuses decorators outright, and that refusal is the
    /// useful outcome.
    #[test]
    fn test_decorators_are_not_stripped() {
        let code = "@memoize\ndef f():\n    return 1";
        assert_eq!(MontyAutoFixer::fix(code), code);
    }

    #[test]
    fn test_strip_type_ignore() {
        let code = "x = foo()  # type: ignore\ny = bar()  # type: ignore[no-return]";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, "x = foo()\ny = bar()");
    }

    #[test]
    fn test_preserves_normal_code() {
        let code = "def greet(name):\n    return f\"Hello, {name}!\"";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    #[test]
    fn test_mixed_code() {
        let code = "import functools\nfrom typing import List\n\ndef process(items):\n    x = compute()  # type: ignore\n    return x";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(
            fixed,
            "from typing import List\n\ndef process(items):\n    x = compute()\n    return x"
        );
    }

    #[test]
    fn test_empty_input() {
        let fixed = MontyAutoFixer::fix("");
        assert_eq!(fixed, "");
    }

    #[test]
    fn test_only_shimmed_imports_reduces_to_nothing() {
        let fixed = MontyAutoFixer::fix("import functools\nimport random");
        assert_eq!(fixed, "");
    }

    #[test]
    fn test_preserves_indentation() {
        let code = "def foo():\n    if True:\n        x = 1\n        y = 2";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    #[test]
    fn test_from_variable_not_stripped() {
        // `from_email = ...` should NOT be stripped (no ` import ` present)
        let code = "from_email = \"test@example.com\"\nresult = send(from_email)";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    // === Markdown artifact stripping tests ===

    #[test]
    fn test_strip_leading_code_fence() {
        let code = "```python\nx = 1\nprint(x)\n```";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, "x = 1\nprint(x)");
    }

    #[test]
    fn test_strip_python_exec_fence() {
        let code = "```python-exec\nx = 1\nprint(x)\n```";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, "x = 1\nprint(x)");
    }

    #[test]
    fn test_strip_backtick_only_lines() {
        let code = "````\nx = 1\n````";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, "x = 1");
    }

    #[test]
    fn test_no_strip_backticks_in_code() {
        // Backticks inside f-strings or dict keys should be preserved
        let code = "msg = f\"use `print` to debug\"\nprint(msg)";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    #[test]
    fn test_clean_code_unchanged() {
        let code = "x = 1\nfor i in range(10):\n    print(i)";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    // === Type annotation stripping tests ===

    #[test]
    fn test_strip_simple_type_annotation() {
        let code = "x: int = 1";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, "x = 1");
    }

    #[test]
    fn test_strip_complex_type_annotation() {
        let code = "results: list[str] = []";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, "results = []");
    }

    #[test]
    fn test_strip_annotation_preserves_indent() {
        let code = "    total: int = 0";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, "    total = 0");
    }

    #[test]
    fn test_no_strip_dict_literal() {
        let code = "d = {\"key\": \"value\"}";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    #[test]
    fn test_no_strip_slice() {
        let code = "x = items[1:3]";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    #[test]
    fn test_no_strip_def_annotation() {
        let code = "def foo(x: int) -> str:\n    return str(x)";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    // === sorted() keyword argument rewriting tests ===

    #[test]
    fn test_sorted_without_key_unchanged() {
        let code = "result = sorted(items)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("sorted(items)"));
        assert!(!fixed.contains("_keysort"));
    }

    // === .sort() method rewriting tests ===

    #[test]
    fn test_sort_method_without_key_unchanged() {
        let code = "items.sort()";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("items.sort()"));
        assert!(!fixed.contains("_keysort"));
    }

    // === map() rewriting tests ===

    #[test]
    fn test_no_replace_method_map() {
        // obj.map() should NOT be rewritten
        let code = "result = df.map(func)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("df.map(func)"));
        assert!(!fixed.contains("_map"));
    }

    #[test]
    fn test_no_replace_map_variable() {
        // map_data should NOT be rewritten
        let code = "map_data = {}\nhash_map(x)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("map_data = {}"));
        assert!(fixed.contains("hash_map(x)"));
        assert!(!fixed.contains("def _map("));
    }

    // === filter() rewriting tests ===

    #[test]
    fn test_no_replace_method_filter() {
        let code = "qs = queryset.filter(active=True)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("queryset.filter(active=True)"));
        assert!(!fixed.contains("_filter"));
    }

    // === Combined tests ===

    #[test]
    fn test_no_map_filter_when_absent() {
        let code = "x = sorted([3, 1, 2])\nprint(x)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(!fixed.contains("def _map("));
        assert!(!fixed.contains("def _filter("));
    }

    // === math.* rewriting tests ===

    #[test]
    fn test_math_no_rewrite_without_math_prefix() {
        let code = "x = sqrt(16)";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    // === os.path.* rewriting tests ===

    #[test]
    fn test_os_path_join_two_args() {
        let code = "import os\nresult = os.path.join(base, name)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_path_join([base, name])"));
        assert!(fixed.contains("def _path_join("));
        assert!(!fixed.contains("os.path.join"));
    }

    #[test]
    fn test_os_path_join_three_args() {
        let code = "import os\np = os.path.join(root, sub, file)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_path_join([root, sub, file])"));
    }

    #[test]
    fn test_os_path_basename() {
        let code = "import os\nname = os.path.basename(\"/foo/bar.txt\")";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_path_basename(\"/foo/bar.txt\")"));
        assert!(fixed.contains("def _path_basename("));
    }

    #[test]
    fn test_os_path_dirname() {
        let code = "import os\ndir = os.path.dirname(\"/foo/bar.txt\")";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_path_dirname(\"/foo/bar.txt\")"));
        assert!(fixed.contains("def _path_dirname("));
    }

    #[test]
    fn test_os_path_splitext() {
        let code = "import os\nparts = os.path.splitext(\"file.txt\")";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_path_splitext(\"file.txt\")"));
        assert!(fixed.contains("def _path_splitext("));
    }

    #[test]
    fn test_os_sep() {
        let code = "import os\nsep = os.sep";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("\"/\""));
    }

    #[test]
    fn test_os_path_no_rewrite_without_prefix() {
        let code = "result = path.join(a, b)";
        let fixed = MontyAutoFixer::fix(code);
        assert_eq!(fixed, code);
    }

    // === functools.reduce rewriting tests ===

    #[test]
    fn test_reduce_simple() {
        let code = "from functools import reduce\ntotal = reduce(lambda a, b: a + b, nums)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_reduce(lambda a, b: a + b, nums)"));
        assert!(fixed.contains("def _reduce("));
        assert!(!fixed.contains("import"));
    }

    #[test]
    fn test_reduce_with_initial() {
        let code = "from functools import reduce\ntotal = reduce(lambda a, b: a + b, nums, 0)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_reduce(lambda a, b: a + b, nums, 0)"));
    }

    #[test]
    fn test_functools_reduce_dotted() {
        let code = "import functools\ntotal = functools.reduce(lambda a, b: a + b, items)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_reduce(lambda a, b: a + b, items)"));
    }

    #[test]
    fn test_no_reduce_without_call() {
        let code = "x = 1\ny = x + 1";
        let fixed = MontyAutoFixer::fix(code);
        assert!(!fixed.contains("_reduce"));
    }

    // === collections.Counter rewriting tests ===

    #[test]
    fn test_counter_simple() {
        let code = "from collections import Counter\ncounts = Counter(words)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_counter(words)"));
        assert!(fixed.contains("def _counter("));
        assert!(!fixed.contains("import"));
    }

    #[test]
    fn test_collections_counter_dotted() {
        let code = "import collections\ncounts = collections.Counter(items)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_counter(items)"));
    }

    #[test]
    fn test_counter_no_rewrite_method() {
        let code = "obj.Counter(x)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(!fixed.contains("_counter"));
    }

    // === json.dumps/json.loads rewriting tests ===

    #[test]
    fn test_json_no_rewrite_without_prefix() {
        let code = "data = loads(text)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(!fixed.contains("_json_loads"));
    }

    // === re.* rewriting tests (bash-bridged) ===

    #[test]
    fn test_re_no_rewrite_without_prefix() {
        let code = "x = findall(pattern, text)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(!fixed.contains("_re_findall"));
    }

    // === datetime.* rewriting tests (bash-bridged) ===

    #[test]
    fn test_datetime_no_rewrite_without_prefix() {
        let code = "now = get_current_time()";
        let fixed = MontyAutoFixer::fix(code);
        assert!(!fixed.contains("_datetime"));
    }

    // === random.* rewriting tests (bash-bridged) ===

    #[test]
    fn test_random_randint() {
        let code = "import random\nn = random.randint(1, 10)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_random_randint(1, 10)"));
        assert!(fixed.contains("def _random_randint("));
        assert!(fixed.contains("run_command("));
    }

    #[test]
    fn test_random_random() {
        let code = "import random\nf = random.random()";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_random_random()"));
        assert!(fixed.contains("def _random_random("));
    }

    #[test]
    fn test_random_choice() {
        let code = "import random\nitem = random.choice(items)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_random_choice(items)"));
        assert!(fixed.contains("def _random_choice("));
        // choice depends on randint
        assert!(fixed.contains("def _random_randint("));
    }

    #[test]
    fn test_random_shuffle() {
        let code = "import random\nrandom.shuffle(items)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_random_shuffle(items)"));
        assert!(fixed.contains("def _random_shuffle("));
    }

    #[test]
    fn test_random_sample() {
        let code = "import random\nresult = random.sample(population, 3)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("_random_sample(population, 3)"));
        assert!(fixed.contains("def _random_sample("));
    }

    #[test]
    fn test_random_no_rewrite_without_prefix() {
        let code = "x = randint(1, 10)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(!fixed.contains("_random"));
    }

    #[test]
    fn test_fix_with_chinese_comments() {
        // This previously panicked: "byte index 7 is not a char boundary"
        let code = "# 基于当前页面收集的职位数据进行分析\njobs = []\nprint(len(jobs))";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("# 基于当前页面"));
        assert!(fixed.contains("print(len(jobs))"));
    }

    #[test]
    fn test_replace_standalone_call_with_multibyte() {
        // Ensure multi-byte chars are preserved, not corrupted via bytes[i] as char
        let code = "# 使用map处理\nresult = map(func, items)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(fixed.contains("# 使用map处理"));
    }
}
