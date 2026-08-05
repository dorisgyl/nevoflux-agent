//! Custom tools loaded from Python scripts.
//!
//! A script declares its tools with `describe()` and implements each one as a
//! function named after the tool's short name. The file stem becomes a
//! namespace prefix (`jira.py` exporting `search` → `jira__search`), so script
//! tools can never collide with the built-in `browser_*` / `computer_*` names.
//!
//! `describe()` runs at scan time only — `tools/list` is a hot path and must
//! not execute user code.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use nevoflux_mcp::ToolDefinition;

/// How long `describe()` may run. It should only return a static declaration;
/// anything slower is doing work it shouldn't, and we would rather skip it
/// loudly than stall startup.
const DESCRIBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Separator between a script's namespace and its tool's short name.
const NAMESPACE_SEP: &str = "__";

/// Longest a namespace may be.
const MAX_STEM_LEN: usize = 32;

/// One tool exported by one script.
#[derive(Debug, Clone)]
pub struct ScriptTool {
    /// Name on the wire, e.g. `jira__search`.
    pub full_name: String,
    /// Namespace, i.e. the file stem.
    pub stem: String,
    /// Function to call inside the script.
    pub short_name: String,
    pub definition: ToolDefinition,
    pub source_path: PathBuf,
}

/// Something that was not loaded, and why.
///
/// Never silently dropped: this is surfaced in logs, reload results, and
/// `GET /admin/scripts`. A tool that vanishes without explanation is the
/// hardest kind of failure to diagnose.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SkipReport {
    pub path: String,
    pub reason: String,
}

/// An immutable view of the script directories, swapped atomically on reload.
#[derive(Debug, Default)]
pub struct Snapshot {
    pub tools: Vec<ScriptTool>,
    pub skipped: Vec<SkipReport>,
}

/// The namespace a script file contributes, or `None` when the file name
/// cannot be one.
///
/// Restricted to the same alphabet as the admin API's `<name>` so a file
/// dropped on disk and a file uploaded over HTTP behave identically.
pub fn stem_of(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    if stem.len() > MAX_STEM_LEN {
        return None;
    }
    let mut chars = stem.chars();
    let first = chars.next()?;
    if !first.is_ascii_lowercase() {
        return None;
    }
    if !chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
        return None;
    }
    Some(stem.to_string())
}

fn kind_of(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "list",
        serde_json::Value::Object(_) => "object",
    }
}

/// Turn one script's `describe()` return value into tools, collecting a reason
/// for everything rejected.
pub fn parse_describe_output(
    stem: &str,
    value: &serde_json::Value,
) -> (Vec<ScriptTool>, Vec<SkipReport>) {
    let path = format!("{stem}.py");
    let entries: Vec<&serde_json::Value> = match value {
        serde_json::Value::Array(items) => items.iter().collect(),
        serde_json::Value::Object(_) => vec![value],
        other => {
            return (
                Vec::new(),
                vec![SkipReport {
                    path,
                    reason: format!(
                        "describe() must return a list or a dict, got {}",
                        kind_of(other)
                    ),
                }],
            )
        }
    };

    let mut tools: Vec<ScriptTool> = Vec::new();
    let mut skipped = Vec::new();
    for entry in entries {
        let Some(short) = entry.get("name").and_then(|n| n.as_str()) else {
            skipped.push(SkipReport {
                path: path.clone(),
                reason: "declaration is missing 'name'".to_string(),
            });
            continue;
        };
        if tools.iter().any(|t| t.short_name == short) {
            skipped.push(SkipReport {
                path: path.clone(),
                reason: format!("duplicate tool name '{short}' in one script; keeping the first"),
            });
            continue;
        }
        let input_schema = match entry.get("inputSchema") {
            None => serde_json::json!({"type": "object", "properties": {}}),
            Some(v) if v.is_object() => v.clone(),
            Some(v) => {
                skipped.push(SkipReport {
                    path: path.clone(),
                    reason: format!(
                        "'{short}': inputSchema must be an object, got {}",
                        kind_of(v)
                    ),
                });
                continue;
            }
        };
        let full_name = format!("{stem}{NAMESPACE_SEP}{short}");
        tools.push(ScriptTool {
            full_name: full_name.clone(),
            stem: stem.to_string(),
            short_name: short.to_string(),
            definition: ToolDefinition {
                name: full_name,
                description: entry
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or_default()
                    .to_string(),
                input_schema,
            },
            source_path: PathBuf::from(&path),
        });
    }
    (tools, skipped)
}

/// Script directories from `NEVOFLUX_MCP_TOOLS`.
///
/// Separator is the platform's (`:` on Unix, `;` on Windows) via
/// `std::env::split_paths`. Unset means no script tools at all, which is the
/// safe default for a container that did not opt in.
pub fn script_dirs_from_env() -> Vec<PathBuf> {
    match std::env::var_os("NEVOFLUX_MCP_TOOLS") {
        Some(raw) => std::env::split_paths(&raw)
            .filter(|p| !p.as_os_str().is_empty())
            .collect(),
        None => Vec::new(),
    }
}

/// Execute `describe()` in `code` and return its value.
///
/// The declaration is read from the script's trailing expression, which is
/// what `CodeModeResult::result` carries.
///
/// Requires a multi-threaded tokio runtime: the interpreter runs under
/// `block_in_place`, which panics on a current-thread runtime.
pub async fn run_describe(code: &str, stem: &str) -> Result<serde_json::Value, String> {
    if !code.contains("def describe") {
        return Err(format!("{stem}.py defines no describe()"));
    }
    let program = format!("{code}\n\ndescribe()\n");
    let outcome = crate::agent::code_mode::execute_python_simple_with_timeout(
        &program,
        None,
        Some(DESCRIBE_TIMEOUT),
        None,
    );
    if !outcome.success {
        return Err(outcome
            .error
            .unwrap_or_else(|| "describe() failed with no error message".to_string()));
    }
    outcome
        .result
        .ok_or_else(|| "describe() returned no value".to_string())
}

/// A browser context for script tools to drive, when one can exist.
///
/// Script tools are meant to compose NevoFlux's own capabilities — a
/// `google__search` that drives the live browser is the point of the feature,
/// not an extra. Without a context the interpreter only pre-injects the
/// non-browser builtins, and a script calling `web_search` dies with a bare
/// `NameError` that says nothing about why.
///
/// Resolved per call rather than held: the browser registry changes as
/// browsers connect and disconnect, and a context captured at startup would
/// address a browser that has since gone.
fn script_browser_context() -> Option<crate::wasm::services::BrowserContext> {
    let template = crate::automation::CURRENT_SERVICES_TEMPLATE.get()?;
    let browsers = crate::registry::CURRENT_BROWSER_REGISTRY.get()?;
    // Same routing rule as the built-in tools: an MCP caller is not itself a
    // browser, so the identity comes from the registry.
    let entry = browsers.single().ok()?;
    let mut services = template.clone();
    services.proxy_id = entry.proxy_id;
    services.client_identity = entry.client_identity;
    services.browser_context()
}

/// Marker used to recognise a whole-directory failure in [`ScriptSource::reload`].
const DIR_FAILURE_PREFIX: &str = "cannot read script directory";

/// Scan `dirs` in order and build a snapshot.
///
/// Earlier directories win a stem collision, matching the admin API, which
/// writes to the first directory.
pub async fn build_snapshot(dirs: &[PathBuf]) -> Snapshot {
    let mut snapshot = Snapshot::default();
    let mut claimed: Vec<String> = Vec::new();

    for dir in dirs {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                snapshot.skipped.push(SkipReport {
                    path: dir.display().to_string(),
                    reason: format!("{DIR_FAILURE_PREFIX}: {e}"),
                });
                continue;
            }
        };
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("py"))
            .collect();
        // Deterministic order so a stem collision always resolves the same way.
        paths.sort();

        for path in paths {
            let Some(stem) = stem_of(&path) else {
                snapshot.skipped.push(SkipReport {
                    path: path.display().to_string(),
                    reason: "file name is not a valid namespace (want ^[a-z][a-z0-9_]{0,31}$)"
                        .to_string(),
                });
                continue;
            };
            if claimed.contains(&stem) {
                snapshot.skipped.push(SkipReport {
                    path: path.display().to_string(),
                    reason: format!("namespace '{stem}' already claimed by an earlier directory"),
                });
                continue;
            }
            let code = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    snapshot.skipped.push(SkipReport {
                        path: path.display().to_string(),
                        reason: format!("cannot read script: {e}"),
                    });
                    continue;
                }
            };
            match run_describe(&code, &stem).await {
                Ok(value) => {
                    let (mut tools, mut skipped) = parse_describe_output(&stem, &value);
                    for tool in &mut tools {
                        tool.source_path = path.clone();
                    }
                    for skip in &mut skipped {
                        skip.path = path.display().to_string();
                    }
                    if !tools.is_empty() {
                        claimed.push(stem);
                    }
                    snapshot.tools.append(&mut tools);
                    snapshot.skipped.append(&mut skipped);
                }
                Err(reason) => snapshot.skipped.push(SkipReport {
                    path: path.display().to_string(),
                    reason,
                }),
            }
        }
    }
    snapshot
}

/// Outcome of one reload, returned to the caller and logged.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ReloadReport {
    /// Tools in the snapshot now serving.
    pub loaded: usize,
    /// Everything that did not load, with a reason.
    pub skipped: Vec<SkipReport>,
    /// Set when the snapshot could not be built at all; the previous snapshot
    /// is still serving.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Script-backed tools, hot-swappable.
///
/// The snapshot is behind an `RwLock<Arc<_>>` so a reload swaps a pointer:
/// calls already in flight keep the snapshot they started with and finish
/// normally.
pub struct ScriptSource {
    dirs: Vec<PathBuf>,
    snapshot: RwLock<Arc<Snapshot>>,
}

impl ScriptSource {
    /// Build over `dirs` with an empty snapshot; call [`Self::reload`] to load.
    pub fn new(dirs: Vec<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            dirs,
            snapshot: RwLock::new(Arc::new(Snapshot::default())),
        })
    }

    /// Build from `NEVOFLUX_MCP_TOOLS`.
    pub fn from_env() -> Arc<Self> {
        Self::new(script_dirs_from_env())
    }

    /// Directories this source reads, in priority order.
    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    /// Current snapshot.
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.snapshot.read().unwrap().clone()
    }

    /// Rescan and swap in a new snapshot.
    ///
    /// When *every* configured directory failed to read, the previous snapshot
    /// stays: a transient unmount must not empty a live endpoint.
    pub async fn reload(&self) -> ReloadReport {
        let next = build_snapshot(&self.dirs).await;
        let dirs_failed = next
            .skipped
            .iter()
            .filter(|s| s.reason.starts_with(DIR_FAILURE_PREFIX))
            .count();
        if !self.dirs.is_empty() && dirs_failed == self.dirs.len() {
            let error = next
                .skipped
                .iter()
                .map(|s| format!("{}: {}", s.path, s.reason))
                .collect::<Vec<_>>()
                .join("; ");
            tracing::error!(%error, "script reload failed; keeping the previous snapshot");
            return ReloadReport {
                loaded: self.snapshot().tools.len(),
                skipped: next.skipped,
                error: Some(error),
            };
        }

        let report = ReloadReport {
            loaded: next.tools.len(),
            skipped: next.skipped.clone(),
            error: None,
        };
        for skip in &report.skipped {
            tracing::warn!(path = %skip.path, reason = %skip.reason, "script tool skipped");
        }
        tracing::info!(
            loaded = report.loaded,
            skipped = report.skipped.len(),
            "script tools loaded"
        );
        *self.snapshot.write().unwrap() = Arc::new(next);
        report
    }

    /// Rescan a single script and splice the result into the snapshot.
    ///
    /// Everything belonging to another stem is carried over untouched, so
    /// deploying one script does not pay to re-run every other script's
    /// `describe()`. A stem whose file is gone simply loses its tools — that
    /// is how a delete takes effect.
    pub async fn reload_one(&self, stem: &str) -> ReloadReport {
        let current = self.snapshot();
        let marker = format!("{stem}.py");
        let mut tools: Vec<ScriptTool> = current
            .tools
            .iter()
            .filter(|t| t.stem != stem)
            .cloned()
            .collect();
        let mut skipped: Vec<SkipReport> = current
            .skipped
            .iter()
            .filter(|s| !s.path.ends_with(&marker))
            .cloned()
            .collect();

        for dir in &self.dirs {
            let path = dir.join(&marker);
            if !path.is_file() {
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(code) => match run_describe(&code, stem).await {
                    Ok(value) => {
                        let (mut fresh, mut fresh_skips) = parse_describe_output(stem, &value);
                        for tool in &mut fresh {
                            tool.source_path = path.clone();
                        }
                        for skip in &mut fresh_skips {
                            skip.path = path.display().to_string();
                        }
                        tools.append(&mut fresh);
                        skipped.append(&mut fresh_skips);
                    }
                    Err(reason) => skipped.push(SkipReport {
                        path: path.display().to_string(),
                        reason,
                    }),
                },
                Err(e) => skipped.push(SkipReport {
                    path: path.display().to_string(),
                    reason: format!("cannot read script: {e}"),
                }),
            }
            // Earlier directories win, same as a full scan.
            break;
        }

        for skip in &skipped {
            tracing::warn!(path = %skip.path, reason = %skip.reason, "script tool skipped");
        }
        let report = ReloadReport {
            loaded: tools.len(),
            skipped: skipped.clone(),
            error: None,
        };
        *self.snapshot.write().unwrap() = Arc::new(Snapshot { tools, skipped });
        report
    }
}

#[async_trait::async_trait]
impl crate::mcp_service::source::ToolSource for ScriptSource {
    fn tools(&self) -> Vec<ToolDefinition> {
        self.snapshot()
            .tools
            .iter()
            .map(|t| t.definition.clone())
            .collect()
    }

    async fn call(&self, name: &str, arguments: &serde_json::Value) -> Result<String, String> {
        let snapshot = self.snapshot();
        let Some(tool) = snapshot.tools.iter().find(|t| t.full_name == name) else {
            return Err(format!("Unknown tool: {name}"));
        };
        let code = std::fs::read_to_string(&tool.source_path)
            .map_err(|e| format!("cannot read {}: {e}", tool.source_path.display()))?;

        // Prefer the function named after the tool; fall back to a generic
        // `call(tool, arguments)` for scripts that dispatch themselves.
        let args_literal = crate::script_backend::to_python_literal(arguments);
        let invocation = if code.contains(&format!("def {}", tool.short_name)) {
            format!("{}({})", tool.short_name, args_literal)
        } else if code.contains("def call") {
            format!(
                "call({}, {})",
                crate::script_backend::to_python_literal(&serde_json::Value::String(
                    tool.short_name.clone()
                )),
                args_literal
            )
        } else {
            return Err(format!(
                "{} declares '{}' but defines neither def {}() nor def call()",
                tool.source_path.display(),
                tool.short_name,
                tool.short_name
            ));
        };

        let program = format!("{code}\n\n{invocation}\n");
        let outcome = crate::agent::code_mode::execute_python_simple_with_timeout(
            &program,
            script_browser_context(),
            None,
            None,
        );
        if !outcome.success {
            return Err(outcome
                .error
                .unwrap_or_else(|| format!("{name} failed with no error message")));
        }
        Ok(match outcome.result {
            Some(serde_json::Value::String(s)) => s,
            Some(v) => v.to_string(),
            None => outcome.output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_service::source::ToolSource;

    fn decl(name: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "description": format!("does {name}"),
            "inputSchema": {"type": "object", "properties": {}}
        })
    }

    #[test]
    fn list_of_declarations_becomes_prefixed_tools() {
        let (tools, skipped) =
            parse_describe_output("jira", &serde_json::json!([decl("search"), decl("create")]));
        assert!(skipped.is_empty(), "{skipped:?}");
        let names: Vec<&str> = tools.iter().map(|t| t.full_name.as_str()).collect();
        assert_eq!(names, vec!["jira__search", "jira__create"]);
        assert_eq!(tools[0].short_name, "search");
        assert_eq!(tools[0].definition.description, "does search");
    }

    /// A single dict is accepted so one-tool scripts need no boilerplate.
    #[test]
    fn a_single_declaration_is_accepted() {
        let (tools, skipped) = parse_describe_output("price", &decl("watch"));
        assert!(skipped.is_empty());
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].full_name, "price__watch");
    }

    #[test]
    fn declarations_without_a_name_are_skipped_with_a_reason() {
        let bad = serde_json::json!({"description": "x", "inputSchema": {"type": "object"}});
        let (tools, skipped) = parse_describe_output("x", &serde_json::json!([bad, decl("ok")]));
        assert_eq!(tools.len(), 1);
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].reason.contains("name"), "{}", skipped[0].reason);
    }

    #[test]
    fn non_object_input_schema_is_skipped() {
        let bad = serde_json::json!({"name": "b", "description": "x", "inputSchema": "nope"});
        let (tools, skipped) = parse_describe_output("x", &serde_json::json!([bad]));
        assert!(tools.is_empty());
        assert!(
            skipped[0].reason.contains("inputSchema"),
            "{}",
            skipped[0].reason
        );
    }

    #[test]
    fn a_missing_input_schema_defaults_to_an_empty_object_schema() {
        let minimal = serde_json::json!({"name": "m", "description": "x"});
        let (tools, skipped) = parse_describe_output("x", &serde_json::json!([minimal]));
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(tools[0].definition.input_schema["type"], "object");
    }

    #[test]
    fn duplicate_short_names_within_one_script_keep_the_first() {
        let (tools, skipped) =
            parse_describe_output("x", &serde_json::json!([decl("a"), decl("a")]));
        assert_eq!(tools.len(), 1);
        assert_eq!(skipped.len(), 1);
        assert!(
            skipped[0].reason.contains("duplicate"),
            "{}",
            skipped[0].reason
        );
    }

    #[test]
    fn a_non_list_non_object_return_is_reported() {
        let (tools, skipped) = parse_describe_output("x", &serde_json::json!("nope"));
        assert!(tools.is_empty());
        assert_eq!(skipped.len(), 1);
    }

    #[test]
    fn stem_is_derived_from_the_file_name() {
        assert_eq!(stem_of(Path::new("/opt/tools/jira.py")).unwrap(), "jira");
        assert!(stem_of(Path::new("/opt/tools/Jira.py")).is_none());
        assert!(stem_of(Path::new("/opt/tools/1bad.py")).is_none());
        assert!(stem_of(Path::new("/opt/tools/with-dash.py")).is_none());
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nevoflux_scriptsrc_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn build_snapshot_loads_every_valid_script() {
        let dir = tmp_dir("load");
        std::fs::write(
            dir.join("jira.py"),
            "def describe():\n    return [{\"name\": \"search\", \"description\": \"s\"}]\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("price.py"),
            "def describe():\n    return {\"name\": \"watch\", \"description\": \"w\"}\n",
        )
        .unwrap();

        let snap = build_snapshot(&[dir.clone()]).await;
        let mut names: Vec<String> = snap.tools.iter().map(|t| t.full_name.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["jira__search", "price__watch"]);
        assert!(snap.skipped.is_empty(), "{:?}", snap.skipped);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// One broken script must not take the endpoint down with it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_script_that_raises_is_skipped_and_the_others_load() {
        let dir = tmp_dir("raises");
        std::fs::write(
            dir.join("bad.py"),
            "def describe():\n    return undefined_name\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("good.py"),
            "def describe():\n    return [{\"name\": \"ok\", \"description\": \"o\"}]\n",
        )
        .unwrap();

        let snap = build_snapshot(&[dir.clone()]).await;
        assert_eq!(snap.tools.len(), 1);
        assert_eq!(snap.tools[0].full_name, "good__ok");
        assert_eq!(snap.skipped.len(), 1);
        assert!(snap.skipped[0].path.contains("bad"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_script_without_describe_is_skipped() {
        let dir = tmp_dir("nodescribe");
        std::fs::write(dir.join("plain.py"), "def other():\n    return 1\n").unwrap();
        let snap = build_snapshot(&[dir.clone()]).await;
        assert!(snap.tools.is_empty());
        assert_eq!(snap.skipped.len(), 1);
        assert!(
            snap.skipped[0].reason.contains("describe"),
            "{}",
            snap.skipped[0].reason
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_file_name_that_cannot_be_a_namespace_is_skipped() {
        let dir = tmp_dir("badname");
        std::fs::write(dir.join("With-Dash.py"), "def describe():\n    return []\n").unwrap();
        let snap = build_snapshot(&[dir.clone()]).await;
        assert!(snap.tools.is_empty());
        assert_eq!(snap.skipped.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Earlier directories win, matching the admin API writing to the first.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_first_directory_wins_on_a_stem_collision() {
        let a = tmp_dir("cola");
        let b = tmp_dir("colb");
        std::fs::write(
            a.join("dup.py"),
            "def describe():\n    return [{\"name\": \"from_a\", \"description\": \"a\"}]\n",
        )
        .unwrap();
        std::fs::write(
            b.join("dup.py"),
            "def describe():\n    return [{\"name\": \"from_b\", \"description\": \"b\"}]\n",
        )
        .unwrap();

        let snap = build_snapshot(&[a.clone(), b.clone()]).await;
        assert_eq!(snap.tools.len(), 1);
        assert_eq!(snap.tools[0].full_name, "dup__from_a");
        assert_eq!(snap.skipped.len(), 1);
        std::fs::remove_dir_all(&a).ok();
        std::fs::remove_dir_all(&b).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn calls_the_function_named_after_the_tool() {
        let dir = tmp_dir("dispatch");
        std::fs::write(
            dir.join("m.py"),
            "def describe():\n    return [{\"name\": \"echo\", \"description\": \"e\"}]\n\
             \ndef echo(arguments):\n    return arguments[\"text\"]\n",
        )
        .unwrap();
        let src = ScriptSource::new(vec![dir.clone()]);
        src.reload().await;
        let got = src
            .call("m__echo", &serde_json::json!({"text": "hi"}))
            .await
            .unwrap();
        assert!(got.contains("hi"), "got: {got}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The escape hatch for scripts that generate their tools dynamically.
    #[tokio::test(flavor = "multi_thread")]
    async fn falls_back_to_call_when_no_function_matches() {
        let dir = tmp_dir("fallback");
        std::fs::write(
            dir.join("m.py"),
            "def describe():\n    return [{\"name\": \"any\", \"description\": \"a\"}]\n\
             \ndef call(tool, arguments):\n    return tool\n",
        )
        .unwrap();
        let src = ScriptSource::new(vec![dir.clone()]);
        src.reload().await;
        let got = src.call("m__any", &serde_json::json!({})).await.unwrap();
        assert!(got.contains("any"), "got: {got}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_raising_tool_is_a_tool_level_error() {
        let dir = tmp_dir("raise");
        std::fs::write(
            dir.join("m.py"),
            "def describe():\n    return [{\"name\": \"boom\", \"description\": \"b\"}]\n\
             \ndef boom(arguments):\n    return missing_name\n",
        )
        .unwrap();
        let src = ScriptSource::new(vec![dir.clone()]);
        src.reload().await;
        assert!(src.call("m__boom", &serde_json::json!({})).await.is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn calling_an_unknown_tool_names_it() {
        let src = ScriptSource::new(vec![]);
        let err = src
            .call("nope__nope", &serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err.contains("nope__nope"), "got: {err}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reload_picks_up_a_new_script() {
        let dir = tmp_dir("reload");
        let src = ScriptSource::new(vec![dir.clone()]);
        src.reload().await;
        assert!(src.tools().is_empty());

        std::fs::write(
            dir.join("late.py"),
            "def describe():\n    return [{\"name\": \"t\", \"description\": \"t\"}]\n",
        )
        .unwrap();
        let report = src.reload().await;
        assert_eq!(report.loaded, 1);
        assert_eq!(src.tools().len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reload_one_touches_only_its_own_stem() {
        let dir = tmp_dir("reload_one");
        for stem in ["a", "b"] {
            std::fs::write(
                dir.join(format!("{stem}.py")),
                "def describe():\n    return [{\"name\": \"t\", \"description\": \"t\"}]\n",
            )
            .unwrap();
        }
        let src = ScriptSource::new(vec![dir.clone()]);
        src.reload().await;
        assert_eq!(src.tools().len(), 2);

        std::fs::write(
            dir.join("a.py"),
            "def describe():\n    return [{\"name\": \"t\", \"description\": \"t\"}, \
             {\"name\": \"u\", \"description\": \"u\"}]\n",
        )
        .unwrap();
        let report = src.reload_one("a").await;
        assert_eq!(report.loaded, 3, "a now exports two tools, b still one");

        let names: Vec<String> = src.tools().into_iter().map(|t| t.name).collect();
        assert!(names.contains(&"a__t".to_string()));
        assert!(names.contains(&"a__u".to_string()));
        assert!(names.contains(&"b__t".to_string()), "b must be untouched");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Reloading a name whose file is gone is how a delete takes effect.
    #[tokio::test(flavor = "multi_thread")]
    async fn reload_one_on_a_missing_file_removes_its_tools() {
        let dir = tmp_dir("reload_one_gone");
        std::fs::write(
            dir.join("a.py"),
            "def describe():\n    return [{\"name\": \"t\", \"description\": \"t\"}]\n",
        )
        .unwrap();
        let src = ScriptSource::new(vec![dir.clone()]);
        src.reload().await;
        assert_eq!(src.tools().len(), 1);

        std::fs::remove_file(dir.join("a.py")).unwrap();
        let report = src.reload_one("a").await;
        assert_eq!(report.loaded, 0);
        assert!(src.tools().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The regression this guards: script tools used to run with no browser
    /// context, so a script calling `web_search` died with a bare `NameError`
    /// and no hint that the capability was simply absent. Composing NevoFlux's
    /// own tools is the point of script tools, not an extra.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_browser_context_puts_browser_tools_in_scope() {
        use crate::wasm::services::BrowserContext;

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let ctx = BrowserContext {
            sender: tx,
            proxy_id: "p".to_string(),
            client_identity: Vec::new(),
            session_id: "s".to_string(),
            asset_server: None,
            recording_collector: None,
            recordings_dir: std::path::PathBuf::new(),
        };

        // Merely naming the function is enough: an absent tool is a NameError
        // at lookup, so this separates "not injected" from "call failed".
        let without = crate::agent::code_mode::execute_python_simple_with_timeout(
            "web_search\n",
            None,
            None,
            None,
        );
        assert!(
            without.error.unwrap_or_default().contains("web_search"),
            "without a context web_search must be unknown"
        );

        let with = crate::agent::code_mode::execute_python_simple_with_timeout(
            "web_search\n",
            Some(ctx),
            None,
            None,
        );
        assert!(
            with.success,
            "with a context web_search must resolve, got: {:?}",
            with.error
        );
    }

    /// Never leave the endpoint empty because a reload went wrong.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_reload_keeps_the_previous_snapshot() {
        let dir = tmp_dir("keepold");
        std::fs::write(
            dir.join("m.py"),
            "def describe():\n    return [{\"name\": \"t\", \"description\": \"t\"}]\n",
        )
        .unwrap();
        let src = ScriptSource::new(vec![dir.clone()]);
        src.reload().await;
        assert_eq!(src.tools().len(), 1);

        // Every configured directory disappearing is a whole-snapshot failure.
        std::fs::remove_dir_all(&dir).ok();
        let report = src.reload().await;
        assert!(report.error.is_some(), "expected a reload error");
        assert_eq!(src.tools().len(), 1, "old snapshot must survive");
    }
}
