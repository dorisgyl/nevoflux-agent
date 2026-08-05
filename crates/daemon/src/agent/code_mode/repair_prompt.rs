//! Repair prompt generator for LLM-assisted code fixing.
//!
//! When Layer 3 (linter) finds violations or Layer 4 (runtime) encounters
//! errors, this module generates structured prompts that guide the LLM
//! to produce corrected code within Monty's supported subset.

use super::linter::Violation;

/// A concise summary of Monty's supported Python syntax, included in
/// every repair prompt so the LLM has the full allow-list at hand.
pub const MONTY_SYNTAX_REMINDER: &str = "\
Monty supported syntax: variables, def, if/elif/else, for/while,
try/except/finally, break, continue, return, pass, del, assert, raise,
comprehensions, f-string, lambda, ternary, slice, unpack, walrus (:=).

Built-ins: len, range, sorted, enumerate, zip, sum,
min, max, abs, round, isinstance, type, print, int, str, float,
bool, list, dict, set, tuple, repr, any, all, reversed, chr, ord.

NOT supported:
- match/case, yield (and generators), decorators (@...)
- itertools, subprocess, requests

Supported, so do not rewrite these away:
- class, with, global/nonlocal, async/await
- sorted(key=, reverse=), list.sort(key=), map(), filter()
- imports of json, math, re, datetime, asyncio, os, sys, pathlib, typing \
(the name exists only once imported, so keep the import)

Other notes:
- os exists but has no os.path; write os.path.join / os.path.basename anyway \
and the runtime substitutes an equivalent, as it does for functools.reduce \
and collections.Counter
- random and time need shell access; avoid them if the task may run without it
- Tool calls that fail return {\"__tool_error\": true, \"error\": \"...\"}. \
Always check: if isinstance(result, dict) and result.get(\"__tool_error\"): handle error

Common patterns:
- match → if/elif/else chain
- yield → use list.append() to collect results
- decorator → call the wrapper explicitly";

/// Generates structured repair prompts for the LLM to fix Python code
/// that violates Monty constraints or fails at runtime.
pub struct RepairPrompt;

impl RepairPrompt {
    /// Generate a lint repair prompt from AST violations.
    ///
    /// The prompt includes the original code, a list of violations with
    /// line numbers and suggestions, the Monty syntax reminder, and an
    /// instruction to return only fixed code.
    pub fn from_violations(code: &str, violations: &[Violation]) -> String {
        let mut prompt = String::new();

        prompt.push_str("Your Python code has unsupported constructs:\n\n```python\n");
        prompt.push_str(code);
        prompt.push_str("\n```\n\n");

        prompt.push_str("Violations:\n");
        for v in violations {
            prompt.push_str(&format!(
                "- Line {}: `{}` is not supported. {}\n",
                v.line, v.construct, v.suggestion
            ));
        }

        prompt.push('\n');
        prompt.push_str(MONTY_SYNTAX_REMINDER);
        prompt.push_str("\n\nReturn only the fixed Python code, no explanation.\n");

        prompt
    }

    /// Generate a runtime error repair prompt.
    ///
    /// The prompt includes the original code, the error details (with
    /// optional line number), the Monty syntax reminder, and an
    /// instruction to return only fixed code.
    pub fn from_runtime_error(
        code: &str,
        error_type: &str,
        error_message: &str,
        line: Option<usize>,
        available_tools: &[String],
    ) -> String {
        let mut prompt = String::new();

        prompt.push_str("Your Python code encountered an error:\n\n```python\n");
        prompt.push_str(code);
        prompt.push_str("\n```\n\n");

        match line {
            Some(l) => {
                prompt.push_str(&format!(
                    "Error: {} at line {}: {}\n",
                    error_type, l, error_message
                ));
            }
            None => {
                prompt.push_str(&format!("Error: {}: {}\n", error_type, error_message));
            }
        }

        if !available_tools.is_empty() {
            prompt.push_str("\nAvailable pre-injected tool functions (ONLY these exist, do NOT call any other functions as tools): ");
            prompt.push_str(&available_tools.join(", "));
            prompt.push_str(".\nIf your code calls a function that is not in this list, replace it with equivalent logic using the available functions.\n");
        }

        prompt.push('\n');
        prompt.push_str(MONTY_SYNTAX_REMINDER);
        prompt.push_str("\n\nReturn only the fixed Python code, no explanation.\n");

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_repair_prompt() {
        let code = "class Item:\n    pass";
        let violations = vec![Violation {
            line: 1,
            construct: "class".to_string(),
            suggestion: "Use dict + factory function".to_string(),
        }];
        let prompt = RepairPrompt::from_violations(code, &violations);
        assert!(prompt.contains("class Item:"));
        assert!(prompt.contains("Line 1"));
        assert!(prompt.contains("dict + factory function"));
        assert!(prompt.contains("Return only the fixed Python code"));
        assert!(prompt.contains("Monty supported syntax"));
    }

    #[test]
    fn test_runtime_repair_prompt() {
        let code = "x = 1 / 0";
        let prompt = RepairPrompt::from_runtime_error(
            code,
            "ZeroDivisionError",
            "division by zero",
            Some(1),
            &[],
        );
        assert!(prompt.contains("x = 1 / 0"));
        assert!(prompt.contains("ZeroDivisionError"));
        assert!(prompt.contains("line 1"));
        assert!(prompt.contains("Return only the fixed Python code"));
    }

    #[test]
    fn test_runtime_error_no_line() {
        let tools = vec!["read_file".to_string(), "browser_get_markdown".to_string()];
        let prompt = RepairPrompt::from_runtime_error(
            "x = foo()",
            "NameError",
            "name 'foo' is not defined",
            None,
            &tools,
        );
        assert!(prompt.contains("NameError"));
        assert!(prompt.contains("name 'foo' is not defined"));
        assert!(!prompt.contains("at line"));
        assert!(prompt.contains("read_file, browser_get_markdown"));
        assert!(prompt.contains("ONLY these exist"));
    }

    #[test]
    fn test_multiple_violations() {
        let violations = vec![
            Violation {
                line: 1,
                construct: "import".to_string(),
                suggestion: "Remove".to_string(),
            },
            Violation {
                line: 3,
                construct: "class".to_string(),
                suggestion: "Use dict".to_string(),
            },
        ];
        let prompt =
            RepairPrompt::from_violations("import os\n\nclass Foo:\n    pass", &violations);
        assert!(prompt.contains("Line 1"));
        assert!(prompt.contains("Line 3"));
        assert!(prompt.contains("`import`"));
        assert!(prompt.contains("`class`"));
    }

    #[test]
    /// The repair prompt must name what Monty actually cannot do. Telling the
    /// model that a working construct is unsupported is worse than silence: it
    /// spends a retry rewriting code that already ran, and this prompt is only
    /// ever seen when something has already gone wrong.
    fn test_monty_limitations_in_prompt() {
        let prompt = RepairPrompt::from_runtime_error(
            "match x:\n    case 1:\n        pass",
            "NotImplementedError",
            "pattern matching (match statements)",
            Some(1),
            &[],
        );
        for unsupported in ["match/case", "yield", "decorators"] {
            assert!(prompt.contains(unsupported), "must name {unsupported}");
        }
        for supported in ["class", "with", "map()", "filter()"] {
            assert!(
                prompt.contains(supported),
                "must list {supported} as supported so it is not rewritten away"
            );
        }
        assert!(prompt.contains("__tool_error"));
    }
}
