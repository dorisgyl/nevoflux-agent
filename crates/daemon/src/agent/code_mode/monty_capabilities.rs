//! What the pinned Monty interpreter can actually run.
//!
//! Every workaround in [`super::auto_fixer`], [`super::mechanical_fixer`] and
//! [`super::linter`] exists because some Python construct was unsupported at
//! the version we pinned. Those workarounds are cheap to add and expensive to
//! remove, because "does Monty support X now?" is otherwise answered by
//! reading a changelog and hoping.
//!
//! This module answers it by execution. Each capability is probed against the
//! **raw** interpreter — no auto-fixing, no mechanical repair, no LLM rewrite —
//! so the result reflects Monty itself rather than the scaffolding we wrap it
//! in. The `supported` column records what the currently pinned version does;
//! when an upgrade flips a `false` to `true`, that flip is the evidence that
//! the corresponding workaround is dead code and the matching claim in
//! `crates/builtin-wasm/prompts/agent.md` is now a lie.
//!
//! Keeping this honest matters most for `/loop`: loop iterations reach Monty
//! through the `orchestrate` tool, which has **no LLM repair path** (see the
//! "No LLM retry in orchestrate tool mode" branch in [`super::executor`]), so
//! the mechanical layer is the only safety net an unattended iteration gets.

use monty::{LimitedTracker, MontyRun, PrintWriter, ResourceLimits, RunProgress};

/// Limits for a capability probe: generous enough that nothing here trips them,
/// tight enough that a runaway probe fails the suite instead of hanging it.
fn probe_limits() -> ResourceLimits {
    ResourceLimits {
        max_allocations: Some(100_000),
        max_duration: Some(std::time::Duration::from_secs(5)),
        max_memory: Some(64 * 1024 * 1024),
        gc_interval: Some(10_000),
        max_recursion_depth: Some(100),
    }
}

/// Run `code` on the raw interpreter and return whatever it printed.
///
/// Deliberately passes no external function names: a capability probe must not
/// be able to accidentally succeed by calling out to a tool. `Err` carries
/// `"ExcType: message"` for both parse and runtime failures, since for the
/// purpose of "can Monty do this" the distinction does not matter.
pub fn probe(code: &str) -> Result<String, String> {
    let runner =
        MontyRun::new(code.to_string(), "capability.py", vec![], vec![]).map_err(|exc| {
            format!(
                "{}: {}",
                exc.exc_type(),
                exc.message().unwrap_or("parse error")
            )
        })?;

    let mut writer = PrintWriter::Collect(String::new());
    let progress = runner
        .start(vec![], LimitedTracker::new(probe_limits()), &mut writer)
        .map_err(|exc| {
            format!(
                "{}: {}",
                exc.exc_type(),
                exc.message().unwrap_or("runtime error")
            )
        })?;

    match progress {
        RunProgress::Complete(_) => Ok(match writer {
            PrintWriter::Collect(s) => s,
            _ => String::new(),
        }),
        // A probe that suspends wanted something this module refuses to give
        // it (a tool call, an OS call). That is a failed probe, not a pass.
        other => Err(format!("suspended instead of completing: {other:?}")),
    }
}

/// One language or builtin capability, and whether the pinned Monty has it.
pub struct Capability {
    /// Short name, matching the vocabulary used in `agent.md`.
    pub name: &'static str,
    /// Probe source. Must print `ok` on success so a silent no-op cannot pass.
    pub code: &'static str,
    /// What the **currently pinned** Monty does. Update only alongside a
    /// version bump, and treat every `false` → `true` flip as a TODO to delete
    /// the workaround named in `workaround`.
    pub supported: bool,
    /// Where the workaround for this capability lives, or `None` if the
    /// capability is unsupported and simply documented as such.
    pub workaround: Option<&'static str>,
}

/// The capability matrix. Ordered to match the claims in `agent.md`.
pub const CAPABILITIES: &[Capability] = &[
    // ---- agent.md:182 "NOT supported" ----
    Capability {
        name: "class",
        code: "class P:\n    def __init__(self, n):\n        self.n = n\n    def hi(self):\n        return self.n\nprint(P('ok').hi())\n",
        supported: false,
        workaround: Some("linter::detects class"),
    },
    Capability {
        name: "match/case",
        code: "def f(x):\n    match x:\n        case 1:\n            return 'ok'\n        case _:\n            return 'no'\nprint(f(1))\n",
        supported: false,
        workaround: Some("linter::detects match"),
    },
    Capability {
        name: "import",
        code: "import json\nprint('ok' if json.dumps({'a': 1}) else 'no')\n",
        supported: false,
        workaround: Some("auto_fixer phase 3 strips imports"),
    },
    // Probes `with` for *parse* support only. There is deliberately no context
    // manager in the body: with `class` and `open()` both unsupported, nothing
    // in the language can produce one, so a realistic probe would fail for the
    // wrong reason and hide the answer to the question actually being asked.
    Capability {
        name: "with",
        code: "with 1 as x:\n    print('ok')\n",
        supported: false,
        workaround: Some("linter::detects with"),
    },
    // Must actually `await`, not merely define an `async def`: Monty parsed
    // async definitions long before it could run one, so a define-only probe
    // reports support that isn't there. This is the capability `/loop` leans
    // on hardest — `orchestrate` renders every parallel-safe tool as
    // `async def` (see builtin-wasm/src/agent.rs).
    Capability {
        name: "async/await",
        code: "async def f():\n    return 'ok'\nprint(await f())\n",
        supported: true,
        workaround: Some("auto_fixer phase 1b strips async/await; mechanical_fixer::fix_await_syntax"),
    },
    Capability {
        name: "yield",
        code: "def g():\n    yield 'ok'\nfor v in g():\n    print(v)\n",
        supported: false,
        workaround: Some("linter::detects yield"),
    },
    Capability {
        name: "decorator",
        code: "def d(f):\n    return f\n@d\ndef g():\n    return 'ok'\nprint(g())\n",
        supported: true,
        workaround: Some("auto_fixer phase 3 strips decorators"),
    },
    // ---- agent.md:183 "Builtin limitations" ----
    Capability {
        name: "sorted(key=)",
        code: "print('ok' if sorted([(2, 'b'), (1, 'a')], key=lambda p: p[0])[0][1] == 'a' else 'no')\n",
        supported: true,
        workaround: Some("auto_fixer phase 2 rewrite; mechanical_fixer::fix_sorted_kwargs"),
    },
    Capability {
        name: "sorted(reverse=)",
        code: "print('ok' if sorted([1, 3, 2], reverse=True)[0] == 3 else 'no')\n",
        supported: true,
        workaround: Some("auto_fixer phase 2 rewrite; mechanical_fixer::fix_sorted_kwargs"),
    },
    Capability {
        name: "map()",
        code: "print('ok' if list(map(lambda x: x + 1, [1, 2]))[1] == 3 else 'no')\n",
        supported: true,
        workaround: Some("auto_fixer phase 2 rewrite; mechanical_fixer::fix_name_error_map"),
    },
    Capability {
        name: "filter()",
        code: "print('ok' if list(filter(lambda x: x > 1, [1, 2, 3]))[0] == 2 else 'no')\n",
        supported: false,
        workaround: Some("auto_fixer phase 2 rewrite; mechanical_fixer::fix_name_error_filter"),
    },
    Capability {
        name: "open()",
        code: "f = open('/dev/null', 'r')\nprint('ok')\n",
        supported: false,
        workaround: Some("mechanical_fixer::fix_name_error_open + try_parse_with_open"),
    },
    // ---- agent.md:184-186 "no imports needed" — pure helpers ----
    Capability {
        name: "json (injected helper)",
        code: "print('ok' if json.dumps({'a': 1}) == '{\"a\": 1}' else 'no')\n",
        supported: false,
        workaround: Some("auto_fixer injects a pure-Python json shim"),
    },
    // ---- Loop-critical: these currently route through run_command + python3,
    // which `AutomationPolicy::tool_allowlist` drops when allow_shell is false
    // (the headless default). Native support removes that dependency. ----
    Capability {
        name: "re (native, no shell)",
        code: "import re\nprint('ok' if re.findall(r'\\d+', 'a1b22')[1] == '22' else 'no')\n",
        supported: false,
        workaround: Some("auto_fixer phase 2b bridges re.* via run_command + python3"),
    },
    Capability {
        name: "datetime (native, no shell)",
        code: "import datetime\nprint('ok' if datetime.datetime(2026, 1, 2).year == 2026 else 'no')\n",
        supported: false,
        workaround: Some("auto_fixer phase 2b bridges datetime.* via run_command + python3"),
    },
    Capability {
        name: "random (native, no shell)",
        code: "import random\nprint('ok' if random.randint(1, 1) == 1 else 'no')\n",
        supported: false,
        workaround: Some("auto_fixer phase 2b bridges random.* via run_command + python3"),
    },
    Capability {
        name: "time (native, no shell)",
        code: "import time\nprint('ok' if time.time() > 0 else 'no')\n",
        supported: false,
        workaround: Some("auto_fixer phase 2b bridges time.* via run_command + python3"),
    },
    // ---- agent.md:187 "truly unavailable" ----
    Capability {
        name: "itertools",
        code: "import itertools\nprint('ok' if len(list(itertools.chain([1], [2]))) == 2 else 'no')\n",
        supported: false,
        workaround: None,
    },
    // ---- Baseline sanity: things agent.md:181 claims already work. If one of
    // these ever fails, the probe harness is broken, not Monty. ----
    Capability {
        name: "comprehension + f-string + lambda",
        code: "xs = [x * 2 for x in range(3)]\nadd = lambda a, b: a + b\nprint(f\"{'ok' if add(xs[2], 0) == 4 else 'no'}\")\n",
        supported: true,
        workaround: None,
    },
    Capability {
        name: "try/except",
        code: "try:\n    raise ValueError('x')\nexcept ValueError:\n    print('ok')\n",
        supported: true,
        workaround: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Probe every capability and assert reality matches the recorded matrix.
    ///
    /// Runs the whole table before failing so an upgrade produces one complete
    /// before/after picture rather than stopping at the first surprise — the
    /// point of this test is the diff, and a diff needs every row.
    #[test]
    fn matrix_matches_reality() {
        let mut newly_supported = Vec::new();
        let mut regressed = Vec::new();
        let mut report = String::new();

        for cap in CAPABILITIES {
            let actual = probe(cap.code);
            let works = matches!(&actual, Ok(out) if out.trim() == "ok");

            report.push_str(&format!(
                "  {:<34} recorded={:<5} actual={:<5} {}\n",
                cap.name,
                cap.supported,
                works,
                match &actual {
                    Ok(out) if works => String::new(),
                    Ok(out) => format!("(printed {:?})", out.trim()),
                    Err(e) => format!("({e})"),
                }
            ));

            match (cap.supported, works) {
                (false, true) => newly_supported.push(cap),
                (true, false) => regressed.push(cap.name),
                _ => {}
            }
        }

        println!("Monty capability matrix:\n{report}");

        assert!(
            regressed.is_empty(),
            "capabilities recorded as supported no longer work: {regressed:?}\n{report}"
        );

        assert!(
            newly_supported.is_empty(),
            "Monty gained capabilities the matrix still records as missing.\n\
             Flip `supported` to true for each, then remove the workaround it \
             names and the matching claim in crates/builtin-wasm/prompts/agent.md:\n{}\n{report}",
            newly_supported
                .iter()
                .map(|c| format!(
                    "  - {} → {}",
                    c.name,
                    c.workaround.unwrap_or("(no workaround)")
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// The probe must not be able to pass by doing nothing: a capability whose
    /// construct fails to parse has to surface as `Err`, not as empty output
    /// that some future refactor might mistake for success.
    #[test]
    fn probe_reports_failure_rather_than_silence() {
        let err = probe("this is not python\n").unwrap_err();
        assert!(!err.is_empty(), "a parse failure must carry a reason");
        assert!(
            probe("print('ok')\n").unwrap().trim() == "ok",
            "the probe must be able to observe output at all"
        );
    }
}
