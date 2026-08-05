//! CodeModeExecutor - Monty execution loop with external function routing.
//! Runs auto-fix -> lint -> execute -> retry pipeline.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use monty::{MontyRun, RunProgress};
use monty_types::CompileOptions;
use monty_types::{
    ExtFunctionResult, LimitedTracker, MontyObject, NameLookupResult, PrintWriter, ResourceLimits,
};

use super::auto_fixer::MontyAutoFixer;
use super::linter::MontyLinter;
use super::mechanical_fixer;
use super::repair_prompt::RepairPrompt;

/// Maximum number of retries (rewrite attempts) before giving up.
const MAX_RETRIES: u32 = 2;

/// Result of a Code Mode execution.
#[derive(Debug)]
pub struct CodeModeResult {
    /// Final output from print() statements during execution.
    pub output: String,
    /// Final expression value as JSON (the last expression in the script).
    /// `None` when the script has no trailing expression or ends with a statement.
    pub result: Option<serde_json::Value>,
    /// Tool call results collected during execution.
    pub tool_results: Vec<ToolCallResult>,
    /// Whether execution completed successfully.
    pub success: bool,
    /// Error message if execution failed.
    pub error: Option<String>,
    /// Number of retries used (0 = first attempt succeeded).
    pub retries: u32,
}

impl CodeModeResult {
    /// Create a successful result.
    pub fn success(output: String) -> Self {
        Self {
            output,
            result: None,
            tool_results: Vec::new(),
            success: true,
            error: None,
            retries: 0,
        }
    }

    /// Create a failed result with an error message.
    pub fn fail(error: impl Into<String>) -> Self {
        Self {
            output: String::new(),
            result: None,
            tool_results: Vec::new(),
            success: false,
            error: Some(error.into()),
            retries: 0,
        }
    }

    /// Create a failed result that includes partial output.
    pub fn fail_with_output(output: String, error: impl Into<String>) -> Self {
        Self {
            output,
            result: None,
            tool_results: Vec::new(),
            success: false,
            error: Some(error.into()),
            retries: 0,
        }
    }

    /// Set the final expression result.
    pub fn with_result(mut self, value: serde_json::Value) -> Self {
        self.result = Some(value);
        self
    }

    /// Set the retry count.
    pub fn with_retries(mut self, retries: u32) -> Self {
        self.retries = retries;
        self
    }

    /// Set tool call results.
    pub fn with_tool_results(mut self, tool_results: Vec<ToolCallResult>) -> Self {
        self.tool_results = tool_results;
        self
    }

    /// Format the result as a JSON string matching the design spec:
    /// `{"output": "...", "result": ..., "success": true, "error": null}`
    pub fn to_json_string(&self) -> String {
        serde_json::json!({
            "output": self.output,
            "result": self.result,
            "success": self.success,
            "error": self.error,
        })
        .to_string()
    }
}

/// Default wall-clock budget for a single Code Mode execution.
///
/// Recorded flows frequently wait on a slow remote response (e.g. an LLM
/// streaming its reply), so this must comfortably exceed a typical
/// navigate-fill-send-wait round trip. Individual flows can still raise it
/// further via their `flow.json` `timeout_ms`; see
/// [`execute_python_simple_with_timeout`].
pub const DEFAULT_MAX_DURATION: Duration = Duration::from_secs(180);

/// Wall-clock budget for the `orchestrate` tool's Code Mode scripts. These are
/// long-running agent-authored orchestrations that legitimately wait on many
/// slow tool calls (browser automation in headless, remote fetches, LLM
/// sub-calls), so they get a very generous 24-hour cap rather than the 180s
/// default. The timeout is a runaway backstop, not a normal completion bound.
pub const ORCHESTRATE_MAX_DURATION: Duration = Duration::from_secs(24 * 60 * 60);

/// Whether a Monty runtime error is the wall-clock timeout (`TimeoutError` /
/// "time limit exceeded"). A timeout is NOT a code defect — rewriting the code
/// (mechanical fix or LLM rewrite) cannot fix it, and a rewritten version would
/// just time out again. It also produces a misleading "LLM rewrite failed: No
/// LLM retry in orchestrate tool mode" error on the orchestrate path (where
/// `llm_rewrite` is a stub). So a timeout must be returned directly, not routed
/// through the repair path.
pub fn is_timeout_error(error_type: &str, error_msg: &str) -> bool {
    error_type == "TimeoutError" || error_msg.contains("time limit exceeded")
}

/// Default resource limits for Monty execution.
fn default_resource_limits() -> ResourceLimits {
    ResourceLimits {
        // v0.0.19 dropped `max_allocations`. `max_memory` still bounds a
        // runaway allocator and `max_duration` still bounds a runaway loop, so
        // nothing is unguarded — but an allocation-count ceiling used to trip
        // sooner than either, which mattered most for /loop, where a runaway
        // iteration reruns on a schedule with nobody watching.
        max_duration: Some(DEFAULT_MAX_DURATION),
        max_memory: Some(64 * 1024 * 1024), // 64MB
        gc_interval: Some(10_000),
        max_recursion_depth: Some(100),
    }
}

/// A tool call made during Python execution.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub result: serde_json::Value,
}

/// Convert a `MontyObject` to a `serde_json::Value`.
fn monty_object_to_json(obj: &MontyObject) -> serde_json::Value {
    match obj {
        MontyObject::None => serde_json::Value::Null,
        MontyObject::Bool(b) => serde_json::Value::Bool(*b),
        MontyObject::Int(i) => serde_json::json!(*i),
        MontyObject::BigInt(bi) => {
            // Try to fit into i64 first, otherwise use string representation
            if let Ok(i) = i64::try_from(bi) {
                serde_json::json!(i)
            } else {
                serde_json::Value::String(bi.to_string())
            }
        }
        MontyObject::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        MontyObject::String(s) => serde_json::Value::String(s.clone()),
        MontyObject::List(items) => {
            serde_json::Value::Array(items.iter().map(monty_object_to_json).collect())
        }
        MontyObject::Tuple(items) => {
            serde_json::Value::Array(items.iter().map(monty_object_to_json).collect())
        }
        MontyObject::Dict(pairs) => {
            let mut map = serde_json::Map::new();
            for (k, v) in pairs {
                let key = match k {
                    MontyObject::String(s) => s.clone(),
                    other => other.to_string(),
                };
                map.insert(key, monty_object_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        MontyObject::Bytes(b) => {
            serde_json::Value::Array(b.iter().map(|byte| serde_json::json!(*byte)).collect())
        }
        MontyObject::Set(items) | MontyObject::FrozenSet(items) => {
            serde_json::Value::Array(items.iter().map(monty_object_to_json).collect())
        }
        MontyObject::NamedTuple { values, .. } => {
            serde_json::Value::Array(values.iter().map(monty_object_to_json).collect())
        }
        MontyObject::Path(p) => serde_json::Value::String(p.clone()),
        // For all other variants, use debug/repr formatting
        other => serde_json::Value::String(format!("{other}")),
    }
}

/// Convert a `serde_json::Value` to a `MontyObject`.
fn json_to_monty_object(val: &serde_json::Value) -> MontyObject {
    match val {
        serde_json::Value::Null => MontyObject::None,
        serde_json::Value::Bool(b) => MontyObject::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                MontyObject::Int(i)
            } else if let Some(f) = n.as_f64() {
                MontyObject::Float(f)
            } else {
                // u64 that doesn't fit in i64
                MontyObject::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => MontyObject::String(s.clone()),
        serde_json::Value::Array(arr) => {
            MontyObject::List(arr.iter().map(json_to_monty_object).collect())
        }
        serde_json::Value::Object(map) => {
            let pairs: Vec<(MontyObject, MontyObject)> = map
                .iter()
                .map(|(k, v)| (MontyObject::String(k.clone()), json_to_monty_object(v)))
                .collect();
            MontyObject::dict(pairs)
        }
    }
}

/// Core execution engine for Agent Code Mode.
///
/// Orchestrates the four-layer constraint pipeline:
/// 1. Prompt constraint (handled externally by system prompt)
/// 2. `MontyAutoFixer` - mechanical text transforms
/// 3. `MontyLinter` - regex-based violation detection
/// 4. Monty interpreter execution with external function routing
///
/// On lint violations or runtime errors, the executor can request the LLM
/// to rewrite the code, up to `MAX_RETRIES` times.
#[derive(Default)]
pub struct CodeModeExecutor {
    /// Optional override for the wall-clock execution budget. `None` uses
    /// [`DEFAULT_MAX_DURATION`]. Recorded flows thread their `flow.json`
    /// `timeout_ms` here so a legitimately long browser wait isn't aborted by
    /// the default budget.
    max_duration: Option<Duration>,
    /// Optional cooperative cancellation flag. When set to `true` (e.g. the
    /// user interrupts the session), the executor aborts at the next tool-call
    /// boundary instead of running to the (possibly 24h) wall-clock cap.
    /// Wired from `HostServices::interrupt_flag` on the orchestrate path.
    cancel_flag: Option<Arc<AtomicBool>>,
}

impl CodeModeExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the wall-clock execution budget (default [`DEFAULT_MAX_DURATION`]).
    pub fn with_max_duration(mut self, max_duration: Duration) -> Self {
        self.max_duration = Some(max_duration);
        self
    }

    /// Attach a cooperative cancellation flag (e.g. the session's interrupt
    /// flag). Checked at each tool-call boundary during execution.
    pub fn with_cancel_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel_flag = Some(flag);
        self
    }

    /// Whether cancellation has been requested.
    fn is_cancelled(&self) -> bool {
        self.cancel_flag
            .as_ref()
            .map(|f| f.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Resolve the resource limits for this execution, applying the
    /// `max_duration` override (if any) on top of the module defaults.
    fn resource_limits(&self) -> ResourceLimits {
        let mut limits = default_resource_limits();
        if let Some(max_duration) = self.max_duration {
            limits.max_duration = Some(max_duration);
        }
        limits
    }

    /// Execute Python code through the full pipeline.
    ///
    /// Pipeline: auto-fix -> lint -> monty execute -> retry on error
    ///
    /// # Arguments
    /// * `code` - Raw Python code from LLM
    /// * `external_function_names` - Names of tool functions available to the code
    /// * `tool_executor` - Async callback to execute tool calls via ToolRegistry
    /// * `llm_rewrite` - Async callback to ask LLM to rewrite code given a repair prompt
    pub async fn execute<F, R>(
        &self,
        code: &str,
        external_function_names: &[String],
        tool_executor: F,
        llm_rewrite: R,
    ) -> CodeModeResult
    where
        F: Fn(
                &str,
                serde_json::Value,
            )
                -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send>>
            + Send
            + Sync,
        R: Fn(&str) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> + Send + Sync,
    {
        let mut current_code = code.to_string();
        let mut retries: u32 = 0;
        // E: Tool result cache — survives across retries so re-executed code
        // reuses results from previous tool calls with identical arguments.
        //
        // The key carries an OCCURRENCE INDEX (`…#3` = the third call with these
        // arguments in this attempt), and the counter resets at the top of every
        // attempt. Without it the key was just (name, args), which is only
        // correct if a script never calls the same tool twice the same way — and
        // a polling script does nothing else:
        //
        //   * `browser_get_markdown(tab_id=5)` in a poll loop executed ONCE and
        //     every later round was handed that first snapshot, so the loop
        //     watched a frozen page for its whole budget;
        //   * the sleep (`browser_wait_for` on a fixed selector) was cached too,
        //     so the pacing collapsed and the budget burned in seconds;
        //   * `emit_progress` is dispatched through this same path, so repeated
        //     identical progress lines silently vanished — the logs lied about
        //     what the script was doing, which is the worst part.
        //
        // With the index, a REPLAY after an auto-fix still reuses results
        // positionally (the intent this cache was added for), while a genuinely
        // new call inside one attempt always executes.
        let mut tool_cache: std::collections::HashMap<String, serde_json::Value> =
            std::collections::HashMap::new();

        loop {
            // Per-attempt occurrence counters; see the note on `tool_cache`.
            let mut call_occurrences: std::collections::HashMap<String, u32> =
                std::collections::HashMap::new();
            // Layer 2: Auto-fix mechanical transforms
            let auto_fixed = MontyAutoFixer::fix(&current_code);
            if auto_fixed != current_code {
                tracing::debug!(
                    "Code Mode: auto_fixer modified code (retry {}), first 300 chars: {:?}",
                    retries,
                    &auto_fixed[..auto_fixed.floor_char_boundary(300)]
                );
            }

            // Layer 3: Lint for unsupported constructs
            let violations = MontyLinter::check(&auto_fixed);
            if !violations.is_empty() {
                if retries >= MAX_RETRIES {
                    let violation_msgs: Vec<String> = violations
                        .iter()
                        .map(|v| format!("Line {}: `{}` - {}", v.line, v.construct, v.suggestion))
                        .collect();
                    return CodeModeResult::fail(format!(
                        "Code has unsupported constructs after {} retries: {}",
                        retries,
                        violation_msgs.join("; ")
                    ))
                    .with_retries(retries);
                }

                let repair_prompt = RepairPrompt::from_violations(&auto_fixed, &violations);
                match llm_rewrite(&repair_prompt).await {
                    Ok(rewritten) => {
                        current_code = rewritten;
                        retries += 1;
                        continue;
                    }
                    Err(e) => {
                        return CodeModeResult::fail(format!(
                            "LLM rewrite failed for lint violations: {e}"
                        ))
                        .with_retries(retries);
                    }
                }
            }

            // Layer 4: Execute with Monty interpreter
            let runner = match MontyRun::new(
                auto_fixed.clone(),
                "code_mode.py",
                vec![],
                CompileOptions::default(),
            ) {
                Ok(runner) => runner,
                Err(exc) => {
                    let error_msg = exc.message().unwrap_or("parse error").to_string();
                    let error_type = format!("{}", exc.exc_type());
                    tracing::warn!(
                        "Code Mode: parse error (retry {}/{}): {}: {}, first 200 chars: {:?}",
                        retries,
                        MAX_RETRIES,
                        error_type,
                        error_msg,
                        &auto_fixed[..auto_fixed.floor_char_boundary(200)]
                    );

                    if retries < MAX_RETRIES {
                        let line = exc.traceback().last().map(|f| f.start.line as usize);

                        // B: Try mechanical fix before expensive LLM rewrite
                        if let Some(fixed) =
                            mechanical_fixer::try_fix(&auto_fixed, &error_type, &error_msg, line)
                        {
                            tracing::info!(
                                "Code Mode: mechanical fix applied for parse error: {}: {}",
                                error_type,
                                error_msg
                            );
                            current_code = fixed;
                            retries += 1;
                            continue;
                        }

                        tracing::debug!(
                            "Code Mode: mechanical_fixer returned None for parse error: {}: {} (line={:?})",
                            error_type,
                            error_msg,
                            line
                        );

                        let repair_prompt = RepairPrompt::from_runtime_error(
                            &auto_fixed,
                            &error_type,
                            &error_msg,
                            line,
                            external_function_names,
                        );
                        tracing::info!("Code Mode: requesting LLM rewrite for parse error");
                        match llm_rewrite(&repair_prompt).await {
                            Ok(rewritten) => {
                                tracing::info!(
                                    "Code Mode: LLM rewrite succeeded ({} bytes)",
                                    rewritten.len()
                                );
                                current_code = rewritten;
                                retries += 1;
                                continue;
                            }
                            Err(e) => {
                                tracing::error!("Code Mode: LLM rewrite failed: {}", e);
                                return CodeModeResult::fail(format!(
                                    "Parse error and LLM rewrite failed: {error_type}: {error_msg} (rewrite error: {e})"
                                ))
                                .with_retries(retries);
                            }
                        }
                    }

                    return CodeModeResult::fail(format!("{error_type}: {error_msg}"))
                        .with_retries(retries);
                }
            };

            let resource_tracker = LimitedTracker::new(self.resource_limits());
            let mut output = String::new();
            let mut tool_results: Vec<ToolCallResult> = Vec::new();
            // Track resolved external call results keyed by call_id for ResolveFutures.
            // When a FunctionCall is processed synchronously via state.run(), we also
            // store the result here so that if ResolveFutures fires (e.g. from
            // asyncio.gather edge cases), we can provide the already-computed values.
            let mut pending_results: Vec<(u32, MontyObject)> = Vec::new();

            let start_result = runner.start(
                vec![],
                resource_tracker,
                PrintWriter::CollectString(&mut output, None),
            );

            let mut progress = match start_result {
                Ok(p) => p,
                Err(exc) => {
                    let error_msg = exc.message().unwrap_or("runtime error").to_string();
                    let error_type = format!("{}", exc.exc_type());
                    tracing::warn!(
                        "Code Mode: start error (retry {}/{}): {}: {}, first 200 chars: {:?}",
                        retries,
                        MAX_RETRIES,
                        error_type,
                        error_msg,
                        &auto_fixed[..auto_fixed.floor_char_boundary(200)]
                    );

                    // A timeout is not fixable by rewriting — return it directly.
                    if is_timeout_error(&error_type, &error_msg) {
                        return CodeModeResult::fail_with_output(
                            output.clone(),
                            format!("{error_type}: {error_msg}"),
                        )
                        .with_tool_results(tool_results)
                        .with_retries(retries);
                    }

                    if retries < MAX_RETRIES {
                        let line = exc.traceback().last().map(|f| f.start.line as usize);

                        // B: Try mechanical fix first
                        if let Some(fixed) =
                            mechanical_fixer::try_fix(&auto_fixed, &error_type, &error_msg, line)
                        {
                            tracing::info!(
                                "Code Mode: mechanical fix applied for start error: {}: {}",
                                error_type,
                                error_msg
                            );
                            current_code = fixed;
                            retries += 1;
                            continue;
                        }

                        tracing::debug!(
                            "Code Mode: mechanical_fixer returned None for start error: {}: {} (line={:?})",
                            error_type,
                            error_msg,
                            line
                        );

                        let repair_prompt = RepairPrompt::from_runtime_error(
                            &auto_fixed,
                            &error_type,
                            &error_msg,
                            line,
                            external_function_names,
                        );
                        tracing::info!("Code Mode: requesting LLM rewrite for start error");
                        match llm_rewrite(&repair_prompt).await {
                            Ok(rewritten) => {
                                tracing::info!(
                                    "Code Mode: LLM rewrite succeeded for start error ({} bytes)",
                                    rewritten.len()
                                );
                                current_code = rewritten;
                                retries += 1;
                                continue;
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Code Mode: LLM rewrite failed for start error: {}",
                                    e
                                );
                                return CodeModeResult::fail_with_output(
                                    output.clone(),
                                    format!("Runtime error and LLM rewrite failed: {error_type}: {error_msg} (rewrite error: {e})"),
                                )
                                .with_tool_results(tool_results)
                                .with_retries(retries);
                            }
                        }
                    }

                    return CodeModeResult::fail_with_output(
                        output.clone(),
                        format!("{error_type}: {error_msg}"),
                    )
                    .with_tool_results(tool_results)
                    .with_retries(retries);
                }
            };

            // Execution loop: handle function calls until completion
            loop {
                // Cooperative cancellation: if the session was interrupted (user
                // hit stop), abort at this tool-call boundary rather than running
                // to the wall-clock cap (which may be 24h for orchestrate).
                if self.is_cancelled() {
                    tracing::info!("Code Mode: cancelled by interrupt");
                    return CodeModeResult::fail_with_output(
                        output.clone(),
                        "Cancelled by user interrupt",
                    )
                    .with_tool_results(tool_results)
                    .with_retries(retries);
                }

                match progress {
                    RunProgress::FunctionCall(call) => {
                        // Cloned rather than borrowed: `call` is consumed by
                        // `resume` below, and a tool call dwarfs the copy.
                        let function_name = call.function_name.clone();
                        let args = call.args.clone();
                        let kwargs = call.kwargs.clone();
                        let call_id = call.call_id;

                        // Monty offers *every* unresolved callee to the host,
                        // not just the ones we advertised, so the catalogue has
                        // to be enforced here rather than at construction time.
                        // Without this an invented name reaches the tool
                        // executor and comes back as a `__tool_error` dict that
                        // the script then treats as data; `NotFound` instead
                        // raises the NameError the caller (and the mechanical
                        // fixer) expects.
                        if !external_function_names.contains(&function_name) {
                            match call.resume(
                                ExtFunctionResult::NotFound(function_name.clone()),
                                PrintWriter::CollectString(&mut output, None),
                            ) {
                                Ok(next) => {
                                    progress = next;
                                    continue;
                                }
                                Err(exc) => {
                                    let error_msg =
                                        exc.message().unwrap_or("runtime error").to_string();
                                    let error_type = format!("{}", exc.exc_type());
                                    return CodeModeResult::fail_with_output(
                                        output.clone(),
                                        format!("{error_type}: {error_msg}"),
                                    )
                                    .with_tool_results(tool_results)
                                    .with_retries(retries);
                                }
                            }
                        }

                        // Build arguments JSON, merging positional args and kwargs.
                        // When kwargs are present, use a special envelope so
                        // positional_to_named_auto can map positional args by
                        // index (using param names) and merge kwargs on top.
                        let arguments = if kwargs.is_empty() {
                            let json_args: Vec<serde_json::Value> =
                                args.iter().map(monty_object_to_json).collect();
                            serde_json::Value::Array(json_args)
                        } else {
                            let positional: Vec<serde_json::Value> =
                                args.iter().map(monty_object_to_json).collect();
                            let mut kw_obj = serde_json::Map::new();
                            for (key, val) in &kwargs {
                                let key_str = match monty_object_to_json(key) {
                                    serde_json::Value::String(s) => s,
                                    other => other.to_string(),
                                };
                                kw_obj.insert(key_str, monty_object_to_json(val));
                            }
                            serde_json::json!({
                                "__positional": positional,
                                "__kwargs": kw_obj
                            })
                        };

                        // E: Check tool cache before executing
                        let call_signature = format!(
                            "{}:{}",
                            function_name,
                            serde_json::to_string(&arguments).unwrap_or_default()
                        );
                        let occurrence = {
                            let n = call_occurrences.entry(call_signature.clone()).or_insert(0);
                            *n += 1;
                            *n
                        };
                        let cache_key = format!("{call_signature}#{occurrence}");

                        let (result_json, resume_value): (serde_json::Value, ExtFunctionResult) =
                            if let Some(cached) = tool_cache.get(&cache_key) {
                                tracing::debug!(
                                    "Code Mode: tool cache hit for {} (key len={})",
                                    function_name,
                                    cache_key.len()
                                );
                                let return_obj = json_to_monty_object(cached);
                                (cached.clone(), ExtFunctionResult::Return(return_obj))
                            } else {
                                // Execute the tool
                                let tool_result =
                                    tool_executor(&function_name, arguments.clone()).await;

                                let (rj, rv) = match tool_result {
                                    Ok(result_val) => {
                                        let return_obj = json_to_monty_object(&result_val);
                                        (result_val, ExtFunctionResult::Return(return_obj))
                                    }
                                    Err(e) => {
                                        // Return error as a dict rather than
                                        // ExtFunctionResult::Error, because Monty may not
                                        // properly raise Python exceptions from
                                        // ExtFunctionResult::Error.
                                        let error_val = serde_json::json!({
                                            "__tool_error": true,
                                            "error": format!("{function_name}: {e}"),
                                        });
                                        let return_obj = json_to_monty_object(&error_val);
                                        (error_val, ExtFunctionResult::Return(return_obj))
                                    }
                                };
                                // Cache successful results (skip error dicts)
                                if !rj
                                    .as_object()
                                    .is_some_and(|o| o.contains_key("__tool_error"))
                                {
                                    tool_cache.insert(cache_key, rj.clone());
                                }
                                (rj, rv)
                            };

                        tool_results.push(ToolCallResult {
                            tool_name: function_name,
                            arguments,
                            result: result_json,
                        });

                        let pending_obj = match &resume_value {
                            ExtFunctionResult::Return(obj) => obj.clone(),
                            _ => MontyObject::None,
                        };
                        pending_results.push((call_id, pending_obj));

                        // Resume execution with the result
                        match call
                            .resume(resume_value, PrintWriter::CollectString(&mut output, None))
                        {
                            Ok(next) => {
                                progress = next;
                            }
                            Err(exc) => {
                                let error_msg =
                                    exc.message().unwrap_or("runtime error").to_string();
                                let error_type = format!("{}", exc.exc_type());
                                tracing::warn!(
                                    "Code Mode: post-tool-call error (retry {}/{}): {}: {}, first 200 chars: {:?}",
                                    retries,
                                    MAX_RETRIES,
                                    error_type,
                                    error_msg,
                                    &auto_fixed[..auto_fixed.floor_char_boundary(200)]
                                );

                                // A timeout is not fixable by rewriting — return
                                // it directly (also avoids the misleading
                                // "No LLM retry in orchestrate tool mode" error).
                                if is_timeout_error(&error_type, &error_msg) {
                                    return CodeModeResult::fail_with_output(
                                        output.clone(),
                                        format!("{error_type}: {error_msg}"),
                                    )
                                    .with_tool_results(tool_results)
                                    .with_retries(retries);
                                }

                                if retries < MAX_RETRIES {
                                    let line =
                                        exc.traceback().last().map(|f| f.start.line as usize);

                                    // B: Try mechanical fix first
                                    if let Some(fixed) = mechanical_fixer::try_fix(
                                        &auto_fixed,
                                        &error_type,
                                        &error_msg,
                                        line,
                                    ) {
                                        tracing::info!(
                                            "Code Mode: mechanical fix applied after tool call: {}: {}",
                                            error_type,
                                            error_msg
                                        );
                                        current_code = fixed;
                                        retries += 1;
                                        break;
                                    }

                                    tracing::debug!(
                                        "Code Mode: mechanical_fixer returned None for post-tool-call error: {}: {} (line={:?})",
                                        error_type,
                                        error_msg,
                                        line
                                    );

                                    let repair_prompt = RepairPrompt::from_runtime_error(
                                        &auto_fixed,
                                        &error_type,
                                        &error_msg,
                                        line,
                                        external_function_names,
                                    );
                                    tracing::info!("Code Mode: requesting LLM rewrite for post-tool-call error");
                                    match llm_rewrite(&repair_prompt).await {
                                        Ok(rewritten) => {
                                            tracing::info!(
                                                "Code Mode: LLM rewrite succeeded for post-tool-call error ({} bytes)",
                                                rewritten.len()
                                            );
                                            current_code = rewritten;
                                            retries += 1;
                                            break; // break inner loop to restart outer loop
                                        }
                                        Err(e) => {
                                            tracing::error!("Code Mode: LLM rewrite failed for post-tool-call error: {}", e);
                                            return CodeModeResult::fail_with_output(
                                                output.clone(),
                                                format!("Runtime error after tool call and LLM rewrite failed: {error_type}: {error_msg} (rewrite error: {e})"),
                                            )
                                            .with_tool_results(tool_results)
                                            .with_retries(retries);
                                        }
                                    }
                                }

                                return CodeModeResult::fail_with_output(
                                    output.clone(),
                                    format!("{error_type}: {error_msg}"),
                                )
                                .with_tool_results(tool_results)
                                .with_retries(retries);
                            }
                        }
                    }
                    RunProgress::Complete(value) => {
                        let final_value = monty_object_to_json(&value);
                        let mut result = CodeModeResult::success(output.clone())
                            .with_tool_results(tool_results)
                            .with_retries(retries);
                        // Capture non-None final expressions as the result
                        if !final_value.is_null() {
                            result = result.with_result(final_value);
                        }
                        return result;
                    }
                    RunProgress::OsCall(_) => {
                        return CodeModeResult::fail_with_output(
                            output.clone(),
                            "OS calls are not permitted in sandboxed execution",
                        )
                        .with_tool_results(tool_results)
                        .with_retries(retries);
                    }
                    // Monty resolves undefined names by asking rather than by
                    // being told upfront, so the tool catalogue is applied here
                    // instead of at `MontyRun::new`. Answering `Undefined` for
                    // an unknown name is what produces the familiar
                    // "name 'x' is not defined" NameError that the mechanical
                    // fixer keys off, so the allow-list stays exactly as tight
                    // as the catalogue the caller passed in.
                    RunProgress::NameLookup(lookup) => {
                        let resolved = if external_function_names.contains(&lookup.name) {
                            NameLookupResult::Value(MontyObject::Function {
                                name: lookup.name.clone(),
                                docstring: None,
                            })
                        } else {
                            NameLookupResult::Undefined
                        };
                        match lookup.resume(resolved, PrintWriter::CollectString(&mut output, None))
                        {
                            Ok(next) => {
                                progress = next;
                            }
                            Err(exc) => {
                                return CodeModeResult::fail_with_output(
                                    output.clone(),
                                    format!(
                                        "{}: {}",
                                        exc.exc_type(),
                                        exc.message().unwrap_or("runtime error")
                                    ),
                                )
                                .with_tool_results(tool_results)
                                .with_retries(retries);
                            }
                        }
                    }
                    RunProgress::ResolveFutures(future_state) => {
                        // Sequential dispatch: resolve all pending futures using
                        // results that were already computed during FunctionCall
                        // handling.  For any call_id without a stored result, we
                        // fall back to MontyObject::None.
                        let results: Vec<(u32, ExtFunctionResult)> = future_state
                            .pending_call_ids()
                            .iter()
                            .map(|&cid| {
                                let value = pending_results
                                    .iter()
                                    .find(|(id, _)| *id == cid)
                                    .map(|(_, v)| v.clone())
                                    .unwrap_or_else(|| {
                                        tracing::warn!(
                                            "Code Mode: ResolveFutures has unknown call_id {cid}, \
                                             resolving with None"
                                        );
                                        MontyObject::None
                                    });
                                (cid, ExtFunctionResult::Return(value))
                            })
                            .collect();

                        tracing::debug!(
                            "Code Mode: resolving {} pending futures sequentially",
                            results.len()
                        );

                        match future_state
                            .resume(results, PrintWriter::CollectString(&mut output, None))
                        {
                            Ok(next) => {
                                progress = next;
                            }
                            Err(exc) => {
                                let error_msg =
                                    exc.message().unwrap_or("runtime error").to_string();
                                let error_type = format!("{}", exc.exc_type());
                                tracing::warn!(
                                    "Code Mode: post-futures error (retry {}/{}): {}: {}, first 200 chars: {:?}",
                                    retries,
                                    MAX_RETRIES,
                                    error_type,
                                    error_msg,
                                    &auto_fixed[..auto_fixed.floor_char_boundary(200)]
                                );

                                // A timeout is not fixable by rewriting — return
                                // it directly.
                                if is_timeout_error(&error_type, &error_msg) {
                                    return CodeModeResult::fail_with_output(
                                        output.clone(),
                                        format!("{error_type}: {error_msg}"),
                                    )
                                    .with_tool_results(tool_results)
                                    .with_retries(retries);
                                }

                                if retries < MAX_RETRIES {
                                    let line =
                                        exc.traceback().last().map(|f| f.start.line as usize);

                                    // B: Try mechanical fix first
                                    if let Some(fixed) = mechanical_fixer::try_fix(
                                        &auto_fixed,
                                        &error_type,
                                        &error_msg,
                                        line,
                                    ) {
                                        tracing::info!(
                                            "Code Mode: mechanical fix applied after futures: {}: {}",
                                            error_type,
                                            error_msg
                                        );
                                        current_code = fixed;
                                        retries += 1;
                                        break;
                                    }

                                    tracing::debug!(
                                        "Code Mode: mechanical_fixer returned None for post-futures error: {}: {} (line={:?})",
                                        error_type,
                                        error_msg,
                                        line
                                    );

                                    let repair_prompt = RepairPrompt::from_runtime_error(
                                        &auto_fixed,
                                        &error_type,
                                        &error_msg,
                                        line,
                                        external_function_names,
                                    );
                                    tracing::info!(
                                        "Code Mode: requesting LLM rewrite for post-futures error"
                                    );
                                    match llm_rewrite(&repair_prompt).await {
                                        Ok(rewritten) => {
                                            tracing::info!(
                                                "Code Mode: LLM rewrite succeeded for post-futures error ({} bytes)",
                                                rewritten.len()
                                            );
                                            current_code = rewritten;
                                            retries += 1;
                                            break; // break inner loop to restart outer
                                        }
                                        Err(e) => {
                                            tracing::error!("Code Mode: LLM rewrite failed for post-futures error: {}", e);
                                            return CodeModeResult::fail_with_output(
                                                output.clone(),
                                                format!(
                                                    "Runtime error resolving futures and LLM \
                                                     rewrite failed: {error_type}: {error_msg} \
                                                     (rewrite error: {e})"
                                                ),
                                            )
                                            .with_tool_results(tool_results)
                                            .with_retries(retries);
                                        }
                                    }
                                }

                                return CodeModeResult::fail_with_output(
                                    output.clone(),
                                    format!("{error_type}: {error_msg}"),
                                )
                                .with_tool_results(tool_results)
                                .with_retries(retries);
                            }
                        }
                    }
                }
            }
            // If we broke out of the inner loop (retry after tool call error),
            // continue the outer loop.
        }
    }
}

use crate::agent::tools::ToolRegistry;
use crate::wasm::services::BrowserContext;
use std::collections::HashMap;
use std::sync::Arc;

/// Create a shared ToolRegistry and tool executor callback for `execute_python_simple`.
///
/// `param_mappings` maps tool names to ordered parameter name lists, used to
/// convert positional args (JSON arrays from Monty) to named args (JSON objects
/// for ToolRegistry).  When a tool is not present in the map, extra positional
/// arguments are assigned generic names (`arg0`, `arg1`, ...).
///
/// `tools_config` optionally restricts which tools can be executed at runtime.
/// When set, the executor guard checks the allowlist before dispatching.
/// 取第一个位置实参并转成字符串。伪工具（`emit_text` / `emit_progress`）
/// 没有 param mapping，实参以数组形式到达；非字符串按 JSON 文本处理。
fn first_string_arg(args: &serde_json::Value) -> String {
    let first = match args {
        serde_json::Value::Array(items) => items.first().cloned(),
        serde_json::Value::Object(map) => map.values().next().cloned(),
        other => Some(other.clone()),
    };
    match first {
        Some(serde_json::Value::String(s)) => s,
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

fn build_registry_and_executor(
    browser_ctx: Option<BrowserContext>,
    param_mappings: HashMap<String, Vec<String>>,
    tools_config: Option<nevoflux_protocol::subagent::ToolsConfig>,
    sink: Option<crate::script_backend::DeltaSink>,
) -> (
    Vec<String>,
    impl Fn(
        &str,
        serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send>>,
) {
    let registry = match browser_ctx {
        Some(ctx) => ToolRegistry::with_browser(ctx),
        None => ToolRegistry::new(),
    };

    // Use caller-provided param_mappings, or auto-generate from registry hints.
    let effective_mappings = if param_mappings.is_empty() {
        registry.param_mappings()
    } else {
        param_mappings
    };

    let shared_registry = Arc::new(registry);
    let mut external_names: Vec<String> = shared_registry
        .tool_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    // 伪工具：脚本的增量出口。无条件暴露——没有 sink 时它们是空操作，
    // 这样同一份脚本在带/不带增量通道的两种调用下都不会因“函数未定义”而崩。
    external_names.push("emit_text".to_string());
    external_names.push("emit_progress".to_string());

    let param_cache = Arc::new(effective_mappings);
    let tools_config = Arc::new(tools_config);

    let sink = sink.map(std::sync::Arc::new);

    let tool_executor = move |name: &str, args: serde_json::Value| {
        let name = name.to_string();
        let args = args.clone();
        let registry = shared_registry.clone();
        let mappings = param_cache.clone();
        let tools_config = tools_config.clone();
        let sink = sink.clone();
        let fut: Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send>> =
            Box::pin(async move {
                // 伪工具：脚本的增量出口。拦在这里而不是进注册表，因为它不是
                // 浏览器能力，也不该出现在 LLM 的工具清单里。
                if name == "emit_text" || name == "emit_progress" {
                    let text = first_string_arg(&args);
                    if let Some(sink) = sink.as_ref() {
                        if name == "emit_text" {
                            sink.text(text);
                        } else {
                            sink.progress(text);
                        }
                    }
                    return Ok(serde_json::json!({ "ok": true }));
                }

                // Executor guard: check tool allowlist if configured
                match tools_config.as_ref() {
                    Some(nevoflux_protocol::subagent::ToolsConfig::None) => {
                        return Err(format!(
                            "Tool '{}' is not available: all tools are disabled",
                            name
                        ));
                    }
                    Some(nevoflux_protocol::subagent::ToolsConfig::Allow(ref allowlist)) => {
                        if !nevoflux_protocol::subagent::is_tool_allowed(allowlist, &name) {
                            return Err(format!(
                                "Tool '{}' is not available: not in the allowed tool list",
                                name
                            ));
                        }
                    }
                    None => {} // inherit: allow all
                }

                let param_names = mappings.get(&name).cloned().unwrap_or_default();
                let named_args = positional_to_named_auto(&param_names, &args);
                let call = crate::agent::abi::PendingToolCall {
                    id: format!("code-mode-{}", uuid_simple()),
                    name: name.clone(),
                    arguments: named_args,
                };
                let result = registry.execute(&call).await;

                // Auto-inject wait_for_stable after navigation actions so that
                // SPA pages have time to render before the next tool call reads
                // page content. This mirrors the Agent-mode behaviour in
                // auto_snapshot_after_action().
                if matches!(
                    name.as_str(),
                    "browser_navigate" | "browser_go_back" | "browser_go_forward"
                ) {
                    let wait_call = crate::agent::abi::PendingToolCall {
                        id: format!("code-mode-wait-{}", uuid_simple()),
                        name: "browser_wait_for_stable".to_string(),
                        arguments: serde_json::json!({
                            "strategy": "navigation",
                            "max_wait": 3000
                        }),
                    };
                    // Best-effort: ignore wait errors (page may already be stable)
                    let _ = registry.execute(&wait_call).await;
                }

                if let Some(error) = result.error {
                    Err(error)
                } else {
                    let content = result.content.unwrap_or_default();
                    match serde_json::from_str::<serde_json::Value>(&content) {
                        Ok(val) => Ok(val),
                        Err(_) => Ok(serde_json::Value::String(content)),
                    }
                }
            });
        fut
    };

    (external_names, tool_executor)
}

/// Execute Python code through Monty with optional tool support.
///
/// When `browser_ctx` is provided, browser and web tools are available.
/// This is the entry point for the `orchestrate` tool call.
///
/// Delegates to `CodeModeExecutor::execute()` with a no-op LLM rewrite callback.
pub fn execute_python_simple(code: &str, browser_ctx: Option<BrowserContext>) -> CodeModeResult {
    execute_python_simple_with_timeout(code, browser_ctx, None, None)
}

/// Like [`execute_python_simple`], but with an explicit wall-clock budget and
/// optional cooperative cancellation flag.
///
/// `max_duration` overrides [`DEFAULT_MAX_DURATION`]; pass `None` to use the
/// default. Recorded flows pass their `flow.json` `timeout_ms` here so a step
/// that legitimately waits on a slow remote response (e.g. an LLM streaming a
/// reply) isn't aborted mid-flow. `cancel_flag` (e.g. the session interrupt
/// flag) aborts execution at the next tool-call boundary when set.
pub fn execute_python_simple_with_timeout(
    code: &str,
    browser_ctx: Option<BrowserContext>,
    max_duration: Option<Duration>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> CodeModeResult {
    let runtime = tokio::runtime::Handle::current();
    // Build param mappings from registry tool definitions.
    // Empty mappings = positional args use generic names (arg0, arg1, ...).
    // When SignatureCache is wired in from the caller, real mappings will be provided.
    let (external_names, tool_executor) =
        build_registry_and_executor(browser_ctx, HashMap::new(), None, None);
    let mut executor = CodeModeExecutor::new();
    if let Some(max_duration) = max_duration {
        executor = executor.with_max_duration(max_duration);
    }
    if let Some(flag) = cancel_flag {
        executor = executor.with_cancel_flag(flag);
    }

    let llm_rewrite =
        |_prompt: &str| -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> {
            Box::pin(async { Err("No LLM retry in orchestrate tool mode".to_string()) })
        };

    tokio::task::block_in_place(|| {
        runtime.block_on(async {
            executor
                .execute(code, &external_names, tool_executor, llm_rewrite)
                .await
        })
    })
}

/// 同 [`execute_python_simple_with_timeout`]，但把脚本的 `emit_text` /
/// `emit_progress` 接到 `sink` 上。脚本后端（[`crate::script_backend`]）用这个入口。
pub fn execute_python_with_sink(
    code: &str,
    browser_ctx: Option<BrowserContext>,
    sink: Option<crate::script_backend::DeltaSink>,
    max_duration: Option<Duration>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> CodeModeResult {
    let runtime = tokio::runtime::Handle::current();
    let (external_names, tool_executor) =
        build_registry_and_executor(browser_ctx, HashMap::new(), None, sink);
    let mut executor = CodeModeExecutor::new();
    if let Some(max_duration) = max_duration {
        executor = executor.with_max_duration(max_duration);
    }
    if let Some(flag) = cancel_flag {
        executor = executor.with_cancel_flag(flag);
    }

    let llm_rewrite =
        |_prompt: &str| -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> {
            Box::pin(async { Err("No LLM retry in script backend mode".to_string()) })
        };

    tokio::task::block_in_place(|| {
        runtime.block_on(async {
            executor
                .execute(code, &external_names, tool_executor, llm_rewrite)
                .await
        })
    })
}

/// Execute Python code with LLM-powered error recovery.
///
/// Like [`execute_python_simple`], but uses a real LLM call to rewrite code
/// when the linter finds violations or the runtime encounters errors.
/// The LLM receives a repair prompt and returns corrected Python code.
///
/// # Arguments
/// * `code` - Raw Python code from the LLM
/// * `browser_ctx` - Optional browser context for browser/web tool access
/// * `provider` - LLM provider type (Anthropic, OpenAI, etc.)
/// * `api_key` - API key for the provider
/// * `model` - Model name to use for the rewrite call
pub fn execute_python_with_llm(
    code: &str,
    browser_ctx: Option<BrowserContext>,
    provider: nevoflux_llm::ProviderType,
    api_key: String,
    model: String,
    base_url: Option<String>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> CodeModeResult {
    let runtime = tokio::runtime::Handle::current();
    let (external_names, tool_executor) =
        build_registry_and_executor(browser_ctx, HashMap::new(), None, None);
    // orchestrate scripts are long-running by design — give them the 24h budget.
    let mut executor = CodeModeExecutor::new().with_max_duration(ORCHESTRATE_MAX_DURATION);
    if let Some(flag) = cancel_flag {
        executor = executor.with_cancel_flag(flag);
    }

    let llm_rewrite =
        move |prompt: &str| -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> {
            let prompt = prompt.to_string();
            let api_key = api_key.clone();
            let model = model.clone();
            let base_url = base_url.clone();
            Box::pin(async move {
                let request = crate::wasm::llm::LlmChatRequest {
                    messages: vec![crate::wasm::llm::LlmMessage::user(&prompt)],
                    system: Some(
                        "You are a Python code repair assistant. \
                     Fix the code according to the error description. \
                     Return ONLY the corrected Python code inside a ```python fence. \
                     Do not include any explanation outside the fence."
                            .to_string(),
                    ),
                    temperature: Some(0.0),
                    max_tokens: Some(4096),
                    tools: None,
                };

                let response = crate::wasm::llm::execute_llm_chat(
                    provider,
                    &api_key,
                    &model,
                    request,
                    base_url.as_deref(),
                )
                .await
                .map_err(|e| format!("LLM rewrite call failed: {e}"))?;

                // Extract Python code from the response (handles ```python, ```py, ``` fences)
                let text = response.content;
                if let Some(code) = crate::agent::runner::extract_any_python_block(&text) {
                    Ok(code)
                } else {
                    // If no fence found, use the raw response as code
                    // (the LLM may have returned bare code without fences)
                    Ok(text)
                }
            })
        };

    tokio::task::block_in_place(|| {
        runtime.block_on(async {
            executor
                .execute(code, &external_names, tool_executor, llm_rewrite)
                .await
        })
    })
}

/// Convert positional args to named args using auto-generated parameter mapping.
/// If args is already a plain object, pass through unchanged.
///
/// Also handles the `{"__positional": [...], "__kwargs": {...}}` envelope
/// produced when Monty delivers both positional and keyword arguments:
/// positional args are mapped by index using `param_names`, then kwargs
/// are merged on top (kwargs override positional).
fn positional_to_named_auto(param_names: &[String], args: &serde_json::Value) -> serde_json::Value {
    // Special envelope: positional + kwargs from Monty FunctionCall
    if let Some(obj) = args.as_object() {
        if obj.contains_key("__positional") || obj.contains_key("__kwargs") {
            let mut result = serde_json::Map::new();
            // Map positional args by index
            if let Some(positional) = obj.get("__positional").and_then(|v| v.as_array()) {
                for (i, val) in positional.iter().enumerate() {
                    let key = if i < param_names.len() {
                        param_names[i].clone()
                    } else {
                        format!("arg{}", i)
                    };
                    result.insert(key, val.clone());
                }
            }
            // Merge kwargs (override positional)
            if let Some(kwargs) = obj.get("__kwargs").and_then(|v| v.as_object()) {
                for (k, v) in kwargs {
                    result.insert(k.clone(), v.clone());
                }
            }
            return serde_json::Value::Object(result);
        }
        // Plain object — pass through
        return args.clone();
    }
    // Array — map by index
    let arr = match args.as_array() {
        Some(a) => a,
        None => return serde_json::json!({}),
    };
    let mut obj = serde_json::Map::new();
    for (i, val) in arr.iter().enumerate() {
        let key = if i < param_names.len() {
            param_names[i].clone()
        } else {
            format!("arg{}", i)
        };
        obj.insert(key, val.clone());
    }
    serde_json::Value::Object(obj)
}

/// Generate a simple unique ID (timestamp + nanos).
fn uuid_simple() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}-{:x}", now.as_millis(), now.subsec_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_timeout_error() {
        // The Monty wall-clock timeout, by type and by message.
        assert!(is_timeout_error(
            "TimeoutError",
            "time limit exceeded: 32.1s > 30s"
        ));
        assert!(is_timeout_error("SomeType", "time limit exceeded: ..."));
        // Ordinary code defects must still route through the repair path.
        assert!(!is_timeout_error("TypeError", "unsupported operand type"));
        assert!(!is_timeout_error("NameError", "name 'x' is not defined"));
        assert!(!is_timeout_error("KeyError", "__tool_error"));
    }

    #[test]
    fn test_orchestrate_max_duration_is_24h() {
        assert_eq!(ORCHESTRATE_MAX_DURATION, Duration::from_secs(86_400));
        // execute_python_with_llm applies this budget, not the 180s default.
        let executor = CodeModeExecutor::new().with_max_duration(ORCHESTRATE_MAX_DURATION);
        assert_eq!(
            executor.resource_limits().max_duration,
            Some(ORCHESTRATE_MAX_DURATION)
        );
    }

    #[test]
    fn test_default_max_duration_is_180s() {
        assert_eq!(DEFAULT_MAX_DURATION, Duration::from_secs(180));
        // The default limits must carry the module default.
        let limits = default_resource_limits();
        assert_eq!(limits.max_duration, Some(DEFAULT_MAX_DURATION));
    }

    #[test]
    fn test_executor_uses_default_duration_without_override() {
        let limits = CodeModeExecutor::new().resource_limits();
        assert_eq!(limits.max_duration, Some(DEFAULT_MAX_DURATION));
    }

    #[test]
    fn test_with_max_duration_overrides_default() {
        // A flow declaring a 10-minute budget must win over the default, so a
        // long wait-for-reply step isn't aborted mid-flow.
        let limits = CodeModeExecutor::new()
            .with_max_duration(Duration::from_secs(600))
            .resource_limits();
        assert_eq!(limits.max_duration, Some(Duration::from_secs(600)));
        // Other limits stay at their defaults.
        let defaults = default_resource_limits();
        assert_eq!(limits.max_memory, defaults.max_memory);
        assert_eq!(limits.max_recursion_depth, defaults.max_recursion_depth);
    }

    #[test]
    fn test_monty_object_to_json() {
        assert_eq!(
            monty_object_to_json(&MontyObject::None),
            serde_json::Value::Null
        );
        assert_eq!(
            monty_object_to_json(&MontyObject::Int(42)),
            serde_json::json!(42)
        );
        assert_eq!(
            monty_object_to_json(&MontyObject::Bool(true)),
            serde_json::json!(true)
        );
        assert_eq!(
            monty_object_to_json(&MontyObject::String("hello".to_string())),
            serde_json::json!("hello")
        );
        assert_eq!(
            monty_object_to_json(&MontyObject::Float(2.5)),
            serde_json::json!(2.5)
        );
        assert_eq!(
            monty_object_to_json(&MontyObject::List(vec![
                MontyObject::Int(1),
                MontyObject::Int(2),
            ])),
            serde_json::json!([1, 2])
        );
    }

    #[test]
    fn test_json_to_monty_object() {
        assert!(matches!(
            json_to_monty_object(&serde_json::json!(null)),
            MontyObject::None
        ));
        assert!(matches!(
            json_to_monty_object(&serde_json::json!(42)),
            MontyObject::Int(42)
        ));
        assert!(matches!(
            json_to_monty_object(&serde_json::json!(true)),
            MontyObject::Bool(true)
        ));
        match json_to_monty_object(&serde_json::json!("hello")) {
            MontyObject::String(s) => assert_eq!(s, "hello"),
            other => panic!("Expected String, got {:?}", other),
        }
        match json_to_monty_object(&serde_json::json!(2.5)) {
            MontyObject::Float(f) => assert!((f - 2.5).abs() < f64::EPSILON),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_json_to_monty_object_dict() {
        let json = serde_json::json!({"name": "test", "value": 42});
        let obj = json_to_monty_object(&json);
        // Should convert to Dict, not String
        match &obj {
            MontyObject::Dict(_) => {}
            other => panic!("Expected Dict, got {:?}", other),
        }
        // Round-trip: Dict → JSON → verify keys
        let back = monty_object_to_json(&obj);
        assert_eq!(back.get("name").unwrap(), "test");
        assert_eq!(back.get("value").unwrap(), 42);
    }

    #[test]
    fn test_json_to_monty_object_nested_dict() {
        let json = serde_json::json!([{"id": "e1", "tag": "a"}, {"id": "e2", "tag": "div"}]);
        let obj = json_to_monty_object(&json);
        // Should be a list of dicts
        match &obj {
            MontyObject::List(items) => {
                assert_eq!(items.len(), 2);
                match &items[0] {
                    MontyObject::Dict(_) => {}
                    other => panic!("Expected Dict element, got {:?}", other),
                }
            }
            other => panic!("Expected List, got {:?}", other),
        }
    }

    #[test]
    fn test_monty_object_to_json_tuple() {
        let tuple = MontyObject::Tuple(vec![MontyObject::Int(1), MontyObject::String("a".into())]);
        assert_eq!(monty_object_to_json(&tuple), serde_json::json!([1, "a"]));
    }

    #[test]
    fn test_monty_object_to_json_nan() {
        // NaN cannot be represented in JSON, should map to null
        assert_eq!(
            monty_object_to_json(&MontyObject::Float(f64::NAN)),
            serde_json::Value::Null
        );
    }

    #[test]
    fn test_auto_fix_applied() {
        // Verify that auto-fixer runs: code with `import os` should have it stripped
        let code = "import os\nx = 1 + 2\nprint(x)";
        let fixed = MontyAutoFixer::fix(code);
        assert!(!fixed.contains("import os"));
        assert!(fixed.contains("x = 1 + 2"));
    }

    #[tokio::test]
    async fn test_simple_execution() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "x = 1 + 2\nprint(x)",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("ok")) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert!(
            result.output.contains('3'),
            "Expected output to contain '3', got: {:?}",
            result.output
        );
        assert_eq!(result.retries, 0);
    }

    #[tokio::test]
    async fn test_lint_violation_triggers_rewrite() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "class Foo:\n    pass",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("ok")) }),
                |_prompt| {
                    Box::pin(async {
                        // Return valid code without class
                        Ok("x = {\"type\": \"Foo\"}\nprint(x)".to_string())
                    })
                },
            )
            .await;
        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert!(result.retries >= 1);
    }

    #[tokio::test]
    async fn test_external_function_call() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "result = web_search(\"test\")\nprint(result)",
                &["web_search".to_string()],
                |name, _args| {
                    let name = name.to_string();
                    Box::pin(async move {
                        assert_eq!(name, "web_search");
                        Ok(serde_json::json!("search results"))
                    })
                },
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].tool_name, "web_search");
    }

    #[tokio::test]
    async fn test_max_retries_exceeded() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "class Foo:\n    pass",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("ok")) }),
                |_prompt| {
                    Box::pin(async {
                        // Always return code with class (never fixes it)
                        Ok("class Bar:\n    pass".to_string())
                    })
                },
            )
            .await;
        assert!(!result.success);
        assert!(result.error.is_some());
        assert_eq!(result.retries, MAX_RETRIES);
    }

    #[tokio::test]
    async fn test_import_auto_stripped_before_execution() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "import os\nimport sys\nx = 10\nprint(x)",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("ok")) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert!(result.output.contains("10"));
        assert_eq!(result.retries, 0);
    }

    #[tokio::test]
    async fn test_multiple_tool_calls() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "a = tool_a(\"x\")\nb = tool_b(\"y\")\nprint(a, b)",
                &["tool_a".to_string(), "tool_b".to_string()],
                |name, _args| {
                    let name = name.to_string();
                    Box::pin(async move { Ok(serde_json::json!(format!("result_{}", name))) })
                },
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert_eq!(result.tool_results.len(), 2);
        assert_eq!(result.tool_results[0].tool_name, "tool_a");
        assert_eq!(result.tool_results[1].tool_name, "tool_b");
    }

    #[tokio::test]
    async fn test_empty_code() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("ok")) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        // Empty code should either succeed with no output or fail gracefully
        // (depends on Monty behavior with empty input)
        // Just verify it doesn't panic
        assert!(result.output.is_empty() || result.success || result.error.is_some());
    }

    #[tokio::test]
    async fn test_async_external_call() {
        // Verifies that calling an external function in non-async context still works
        // (goes through FunctionCall path, ResolveFutures should not fire).
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "result = fetch(\"https://example.com\")\nprint(result)",
                &["fetch".to_string()],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("page content")) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].tool_name, "fetch");
        assert!(
            result.output.contains("page content"),
            "Expected output to contain 'page content', got: {:?}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_async_gather_rewritten_to_sequential() {
        // Verify that async gather code triggers lint → rewrite → sequential execution.
        // The LLM rewrite callback converts async code into sequential calls.
        use std::sync::{Arc, Mutex};

        let call_order = Arc::new(Mutex::new(Vec::new()));
        let call_order_clone = call_order.clone();

        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                // Initial code uses async (triggers lint violation)
                "import asyncio\n\
                 async def main():\n\
                     a, b, c = await asyncio.gather(api_a(), api_b(), api_c())\n\
                     return [a, b, c]\n\
                 results = await main()\n\
                 print(results)",
                &[
                    "api_a".to_string(),
                    "api_b".to_string(),
                    "api_c".to_string(),
                ],
                move |name, _args| {
                    let name = name.to_string();
                    let order = call_order_clone.clone();
                    Box::pin(async move {
                        order.lock().unwrap().push(name.clone());
                        Ok(serde_json::json!(format!("result_{}", name)))
                    })
                },
                |_prompt| {
                    // Rewrite async code as sequential calls
                    Box::pin(async {
                        Ok("a = api_a()\nb = api_b()\nc = api_c()\n\
                            results = [a, b, c]\nprint(results)"
                            .to_string())
                    })
                },
            )
            .await;
        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert!(result.retries >= 1, "Should have retried at least once");
        // All three tools should have been called sequentially
        assert_eq!(result.tool_results.len(), 3);

        let order = call_order.lock().unwrap();
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], "api_a");
        assert_eq!(order[1], "api_b");
        assert_eq!(order[2], "api_c");
    }

    #[test]
    fn test_positional_to_named_auto() {
        let mapping = vec!["selector".to_string(), "button".to_string()];
        let args = serde_json::json!(["#submit", "right"]);
        let named = positional_to_named_auto(&mapping, &args);
        assert_eq!(
            named,
            serde_json::json!({"selector": "#submit", "button": "right"})
        );
    }

    #[test]
    fn test_positional_to_named_auto_object_passthrough() {
        let mapping = vec!["url".to_string()];
        let args = serde_json::json!({"url": "https://example.com"});
        let named = positional_to_named_auto(&mapping, &args);
        assert_eq!(named, serde_json::json!({"url": "https://example.com"}));
    }

    #[test]
    fn test_positional_to_named_auto_overflow() {
        let mapping = vec!["url".to_string()];
        let args = serde_json::json!(["https://example.com", "extra"]);
        let named = positional_to_named_auto(&mapping, &args);
        assert_eq!(
            named,
            serde_json::json!({"url": "https://example.com", "arg1": "extra"})
        );
    }

    #[test]
    fn test_positional_to_named_auto_empty() {
        let mapping: Vec<String> = vec![];
        let args = serde_json::json!(null);
        let named = positional_to_named_auto(&mapping, &args);
        assert_eq!(named, serde_json::json!({}));
    }

    #[test]
    fn test_positional_to_named_auto_kwargs_only() {
        // Pure kwargs: web_fetch(url="https://example.com")
        let mapping = vec!["url".to_string()];
        let args = serde_json::json!({
            "__positional": [],
            "__kwargs": {"url": "https://example.com"}
        });
        let named = positional_to_named_auto(&mapping, &args);
        assert_eq!(named, serde_json::json!({"url": "https://example.com"}));
    }

    #[test]
    fn test_positional_to_named_auto_mixed_positional_kwargs() {
        // Mixed: browser_click("#btn", button="right")
        let mapping = vec!["selector".to_string(), "button".to_string()];
        let args = serde_json::json!({
            "__positional": ["#btn"],
            "__kwargs": {"button": "right"}
        });
        let named = positional_to_named_auto(&mapping, &args);
        assert_eq!(
            named,
            serde_json::json!({"selector": "#btn", "button": "right"})
        );
    }

    #[test]
    fn test_positional_to_named_auto_kwargs_override_positional() {
        // kwargs should override positional when both specify the same param
        let mapping = vec!["url".to_string()];
        let args = serde_json::json!({
            "__positional": ["https://old.com"],
            "__kwargs": {"url": "https://new.com"}
        });
        let named = positional_to_named_auto(&mapping, &args);
        assert_eq!(named, serde_json::json!({"url": "https://new.com"}));
    }

    // ---- End-to-end integration tests ----

    #[tokio::test]
    async fn test_cancel_flag_aborts_execution() {
        // Interrupt already requested before the (long-running) script runs.
        let flag = Arc::new(AtomicBool::new(true));
        let executor = CodeModeExecutor::new().with_cancel_flag(flag);
        let result = executor
            .execute(
                "a = fetch(\"https://a.com\")\nprint(a)\na",
                &["fetch".to_string()],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("x")) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(!result.success, "cancelled run must not succeed");
        assert!(
            result.error.as_deref().unwrap_or("").contains("Cancelled"),
            "expected a cancellation error, got: {:?}",
            result.error
        );
        // Aborted at the tool-call boundary — the tool never ran.
        assert!(result.tool_results.is_empty());
    }

    #[tokio::test]
    async fn test_no_cancel_flag_runs_normally() {
        // Without a flag (or with it false), execution proceeds as usual.
        let executor = CodeModeExecutor::new().with_cancel_flag(Arc::new(AtomicBool::new(false)));
        let result = executor
            .execute(
                "x = 1 + 1\nprint(x)\nx",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!(null)) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(result.success, "got: {:?}", result.error);
    }

    #[tokio::test]
    async fn test_orchestrate_full_pipeline() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                r#"
a = fetch("https://a.com")
b = fetch("https://b.com")
combined = a + " | " + b
print(combined)
combined
"#,
                &["fetch".to_string()],
                |_name, args| {
                    Box::pin(async move {
                        let url = args
                            .as_array()
                            .and_then(|a| a.first())
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        Ok(serde_json::json!(format!("content from {}", url)))
                    })
                },
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;

        assert!(result.success, "Expected success, got: {:?}", result.error);
        assert_eq!(result.tool_results.len(), 2);
        assert!(
            result.output.contains("content from"),
            "Expected output to contain 'content from', got: {:?}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_orchestrate_auto_fix_import() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "import json\nx = 42\nprint(x)",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("ok")) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;

        assert!(result.success, "Expected success, got: {:?}", result.error);
        assert!(
            result.output.contains("42"),
            "Expected output to contain '42', got: {:?}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_orchestrate_tool_result_in_computation() {
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "items = search(\"rust programming\")\ncount = len(items)\nprint(\"Found \" + str(count) + \" results\")",
                &["search".to_string()],
                |_name, _args| {
                    Box::pin(async {
                        Ok(serde_json::json!(["result1", "result2", "result3"]))
                    })
                },
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;

        assert!(result.success, "Expected success, got: {:?}", result.error);
        assert!(
            result.output.contains("Found 3 results"),
            "Expected output to contain 'Found 3 results', got: {:?}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_final_expression_captured_as_result() {
        // §3.7: Final expression value should be captured in `result` field
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "x = 40 + 2\nx",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("ok")) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(result.success, "Expected success, got: {:?}", result.error);
        assert_eq!(
            result.result,
            Some(serde_json::json!(42)),
            "Final expression should be captured as result"
        );
    }

    #[tokio::test]
    async fn test_no_final_expression_result_is_none() {
        // When code ends with a statement (not an expression), result should be None
        let executor = CodeModeExecutor::new();
        let result = executor
            .execute(
                "x = 42\nprint(x)",
                &[],
                |_name, _args| Box::pin(async { Ok(serde_json::json!("ok")) }),
                |_prompt| Box::pin(async { Err("no rewrite".to_string()) }),
            )
            .await;
        assert!(result.success, "Expected success, got: {:?}", result.error);
        // print() returns None, which is filtered out
        assert!(
            result.result.is_none(),
            "Result should be None when code has no trailing expression, got: {:?}",
            result.result
        );
    }

    #[test]
    fn test_to_json_string_format() {
        // §3.7: Return format must be {"output", "result", "success", "error"}
        let result =
            CodeModeResult::success("hello world".to_string()).with_result(serde_json::json!(42));
        let json_str = result.to_json_string();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["output"], "hello world");
        assert_eq!(parsed["result"], 42);
        assert_eq!(parsed["success"], true);
        assert!(parsed["error"].is_null());
    }

    #[test]
    fn test_to_json_string_error_format() {
        let result = CodeModeResult::fail("something broke");
        let json_str = result.to_json_string();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["output"], "");
        assert!(parsed["result"].is_null());
        assert_eq!(parsed["success"], false);
        assert_eq!(parsed["error"], "something broke");
    }
}

#[cfg(test)]
mod sink_tests {
    use super::*;
    use crate::script_backend::{Delta, DeltaSink};

    /// emit 通路端到端：脚本 → 主机函数 → channel。`browser_ctx = None`，
    /// 所以这条测试不依赖 Xvfb/浏览器。
    #[tokio::test(flavor = "multi_thread")]
    async fn emit_text_reaches_the_sink() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let code = "emit_progress(\"开始\")\nemit_text(\"你\")\nemit_text(\"好\")\n\"done\"\n";
        let result = execute_python_with_sink(
            code,
            None,
            Some(DeltaSink::new(tx)),
            Some(Duration::from_secs(10)),
            None,
        );
        assert!(result.success, "script failed: {:?}", result.error);
        assert_eq!(rx.recv().await.unwrap(), Delta::Progress("开始".into()));
        assert_eq!(rx.recv().await.unwrap(), Delta::Text("你".into()));
        assert_eq!(rx.recv().await.unwrap(), Delta::Text("好".into()));
    }

    /// 回归：轮询脚本反复发出**参数完全相同**的调用。工具结果缓存原先按
    /// (名字, 参数) 记忆且永不失效，于是第二条起就再也不执行——`emit_progress`
    /// 走的是同一条派发路径，所以相同的进度行会凭空消失，日志因此撒谎，
    /// 掩盖了「整个轮询在读第一轮那张冻结页面」这件事。
    ///
    /// 用 emit 而不是浏览器工具来钉这个行为：它不依赖 Xvfb，而且缓存命中与否
    /// 直接体现为 channel 上收到几条。
    #[tokio::test(flavor = "multi_thread")]
    async fn repeated_identical_calls_are_not_swallowed_by_the_tool_cache() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let code = "for i in range(3):\n    emit_progress(\"tick\")\n\"done\"\n";
        let result = execute_python_with_sink(
            code,
            None,
            Some(DeltaSink::new(tx)),
            Some(Duration::from_secs(10)),
            None,
        );
        assert!(result.success, "script failed: {:?}", result.error);
        for i in 0..3 {
            assert_eq!(
                rx.recv().await.unwrap(),
                Delta::Progress("tick".into()),
                "progress note {i} was swallowed by the tool cache"
            );
        }
    }

    /// 没有 sink 时 emit 是空操作，脚本不应因未定义函数而失败。
    #[tokio::test(flavor = "multi_thread")]
    async fn emit_without_sink_is_a_no_op() {
        let result = execute_python_with_sink(
            "emit_text(\"忽略\")\n\"ok\"\n",
            None,
            None,
            Some(Duration::from_secs(10)),
            None,
        );
        assert!(result.success, "script failed: {:?}", result.error);
    }
}
