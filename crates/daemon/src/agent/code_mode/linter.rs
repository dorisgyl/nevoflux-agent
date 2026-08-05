//! MontyLinter - Regex-based detection of unsupported Python constructs.
//!
//! This is Layer 3 of the four-layer constraint pipeline. It scans
//! Python code line-by-line to detect constructs that Monty cannot
//! execute, returning actionable suggestions for each violation.
//!
//! Detects `match`/`case` and `yield` — the two constructs Monty still cannot
//! parse. `class`, `with`, `import`, `global` and `nonlocal` were all detected
//! here until v0.0.19 made them work; flagging them now would be telling the
//! model to rewrite code that runs. See `super::monty_capabilities` for what
//! is actually supported.

/// A single violation found by the linter, with a concrete suggestion
/// for how to rewrite the offending construct.
#[derive(Debug, Clone)]
pub struct Violation {
    /// 1-based line number where the violation was found.
    pub line: usize,
    /// Name of the unsupported construct (e.g. "class", "import").
    pub construct: String,
    /// Concrete alternative suggestion for the user/LLM.
    pub suggestion: String,
}

/// Scans Python source code for constructs unsupported by the Monty
/// interpreter. Uses simple string matching on trimmed lines — no
/// external parser dependencies required.
pub struct MontyLinter;

impl MontyLinter {
    /// Check the given Python source for unsupported constructs.
    ///
    /// Returns a list of [`Violation`]s, one per detected issue. An empty
    /// vec means the code is clean (as far as the linter can tell).
    pub fn check(code: &str) -> Vec<Violation> {
        let mut violations = Vec::new();

        for (idx, line) in code.lines().enumerate() {
            let line_num = idx + 1;
            let trimmed = line.trim();

            // Skip blank lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // match statements: `match <expr>:`
            if trimmed.starts_with("match ") && trimmed.contains(':') {
                violations.push(Violation {
                    line: line_num,
                    construct: "match".to_string(),
                    suggestion: "Use if/elif/else chain instead".to_string(),
                });
            }

            // yield / yield from
            // Note: may false-positive inside string literals; acceptable for this layer.
            if trimmed == "yield"
                || trimmed.starts_with("yield ")
                || trimmed.contains(" yield ")
                || trimmed.ends_with(" yield")
            {
                violations.push(Violation {
                    line: line_num,
                    construct: "yield".to_string(),
                    suggestion: "Use list.append() to collect results".to_string(),
                });
            }
        }

        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detects_match() {
        let code = "match x:\n    case 1:\n        pass";
        let violations = MontyLinter::check(code);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].suggestion.contains("if/elif"));
    }

    #[test]
    fn test_allows_async_await() {
        // async/await are now supported for orchestrate parallel execution
        let code = "async def fetch():\n    await get()";
        let violations = MontyLinter::check(code);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_passes_valid_code() {
        let code = "def greet(name):\n    return f\"Hello, {name}!\"\n\nfor i in range(10):\n    print(greet(str(i)))";
        let violations = MontyLinter::check(code);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_detects_yield() {
        let code = "def gen():\n    yield 1\n    yield from [2, 3]";
        let violations = MontyLinter::check(code);
        assert!(violations.len() >= 1);
    }

    #[test]
    fn test_empty_input() {
        let violations = MontyLinter::check("");
        assert!(violations.is_empty());
    }

    #[test]
    fn test_comments_ignored() {
        let code = "# class Foo:\n# import os\ndef f():\n    pass";
        let violations = MontyLinter::check(code);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_from_variable_not_import() {
        // `from_email = ...` has no ` import ` so should not trigger
        let code = "from_email = \"test@example.com\"";
        let violations = MontyLinter::check(code);
        assert!(violations.is_empty());
    }

    /// Line numbers are 1-based and point at the offending line.
    #[test]
    fn test_line_numbers_correct() {
        let violations = MontyLinter::check("x = 1\ny = 2\nmatch x:\nz = 3");
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].line, 3);
    }

    /// The linter exists to name what Monty cannot parse. Flagging something
    /// it runs fine tells the model to rewrite working code, which is worse
    /// than saying nothing — so every construct here is one
    /// `monty_capabilities` records as unsupported.
    #[test]
    fn test_supported_constructs_are_not_flagged() {
        for code in [
            "class Foo:\n    pass",
            "with open_thing() as f:\n    pass",
            "import json",
            "from typing import List",
            "def f():\n    global x\n    x = 1",
            "def outer():\n    v = 1\n    def inner():\n        nonlocal v",
            "async def f():\n    return await g()",
        ] {
            assert!(
                MontyLinter::check(code).is_empty(),
                "must not flag supported code: {code:?}"
            );
        }
    }

    #[test]
    fn test_multiple_violations() {
        let violations = MontyLinter::check("match x:\n    case 1:\n        yield 1");
        assert_eq!(violations.len(), 2, "match + yield");
    }
}
