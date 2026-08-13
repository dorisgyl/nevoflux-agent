//! MCP tool executor for ACP bridge mode.
//!
//! Routes tool calls to the appropriate NevoFlux subsystem: browser tools,
//! computer tools, memory tools, knowledge tools, skill/tool search,
//! artifact creation, and external MCP servers.

use nevoflux_computer::{
    ClickType, ComputerController, Key, KeyCombination, KeyOrChar, KeyboardController, MouseButton,
    MouseController, Point, Region, ScreenshotProvider, ScrollDirection,
};
use nevoflux_llm::providers::acp::mcp_bridge::{
    McpToolBridge, PendingArtifact, PermissionRequest, PermissionResponse, ToolCallRequest,
};
use nevoflux_llm::EmbedKind;
use nevoflux_protocol::BrowserToolAction;
use nevoflux_storage::{CreateKnowledgeParams, KnowledgeRepository};
use std::sync::Arc;
use tokio::sync::mpsc;

use super::services::{BrowserContext, BrowserRequest, HostServices};

/// Run the tool executor loop, dispatching each incoming request to the
/// appropriate tool category.
pub async fn run_tool_executor(
    mut rx: mpsc::Receiver<ToolCallRequest>,
    services: HostServices,
    tool_bridge: Arc<McpToolBridge>,
) {
    use nevoflux_llm::providers::acp::mcp_bridge::ToolCallRecord;

    while let Some(req) = rx.recv().await {
        let start = std::time::Instant::now();
        // Logged on entry, not just on completion. This loop is strictly
        // serial behind a capacity-1 queue, so a tool that never returns
        // leaves a silent gap in the log with no record of what is holding it
        // — which is exactly how a 24-minute stall once became impossible to
        // attribute after the fact.
        tracing::debug!(tool = %req.name, "MCP tool dispatch start");
        let result = execute_mcp_tool(&req.name, &req.arguments, &services, &tool_bridge).await;
        let duration_ms = start.elapsed().as_millis() as u64;
        match &result {
            Ok(_) => tracing::info!(tool = %req.name, ms = duration_ms, "MCP tool dispatch ok"),
            Err(e) => {
                tracing::warn!(tool = %req.name, ms = duration_ms, error = %e, "MCP tool dispatch failed")
            }
        }

        // Log tool call for sidebar display
        let call_id = format!("mcp-{}-{}", req.name, start.elapsed().as_nanos());
        tool_bridge.log_tool_call(ToolCallRecord {
            id: call_id,
            name: req.name.clone(),
            arguments: req.arguments.clone(),
            result: result.as_ref().ok().cloned(),
            error: result.as_ref().err().cloned(),
            duration_ms,
        });

        let _ = req.result_tx.send(result);
    }
}

/// Handle permission requests by showing a dialog in the sidebar via browser_ask_user.
///
/// When `is_iteration` is true (the call site is a /loop iteration), all
/// requests are auto-approved without a dialog — the loop's
/// `allowed_tool_classes` already gates which tools the LLM is told about,
/// and the iteration has no sidebar to display dialogs to anyway.
pub async fn run_permission_handler(
    mut rx: mpsc::Receiver<PermissionRequest>,
    browser_ctx: BrowserContext,
    is_iteration: bool,
) {
    while let Some(req) = rx.recv().await {
        if is_iteration {
            tracing::debug!(
                "iteration auto-approve for {} (proxy_id empty, no sidebar)",
                req.tool_name
            );
            let _ = req.result_tx.send(PermissionResponse::AllowOnce);
            continue;
        }

        let description = describe_tool_action(&req.tool_name, &req.arguments_summary);
        let question = format!(
            "AI wants to perform an action:\n\n{}\n\nDo you want to allow this?",
            description
        );

        let options = vec![
            "Allow".to_string(),
            "Always allow this type of action".to_string(),
            "Deny".to_string(),
        ];

        // Use browser_ask_user to show dialog in sidebar
        let response = execute_ask_user(&question, &description, &options, &browser_ctx).await;

        let decision = match response.as_deref() {
            Some("Allow") => PermissionResponse::AllowOnce,
            Some("Always allow this type of action") => PermissionResponse::AllowAlways,
            Some("Deny") => PermissionResponse::Reject,
            _ => {
                // Timeout or error — default to reject (safer than allowing)
                tracing::warn!(
                    "Permission dialog failed or timed out for {}, defaulting to Reject",
                    req.tool_name
                );
                PermissionResponse::Reject
            }
        };

        let _ = req.result_tx.send(decision);
    }
}

/// Describe a tool action in natural language for the permission dialog.
pub fn describe_tool_action(tool_name: &str, args_summary: &str) -> String {
    // args_summary may be JSON (from MCP path) or a raw string (from agent_host path).
    // Parse as JSON; if it fails, use the raw string as fallback for field lookups.
    let args: serde_json::Value = serde_json::from_str(args_summary).unwrap_or_default();
    let raw = args_summary;

    match tool_name {
        // Browser navigation
        "browser_navigate" => {
            let url = args
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or("a webpage");
            format!("Navigate to: {}", url)
        }
        "browser_go_back" => "Go back to the previous page".to_string(),
        "browser_go_forward" => "Go forward to the next page".to_string(),

        // Browser interaction
        "browser_click" | "browser_click_by_id" => {
            let target = args
                .get("element_id")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("selector").and_then(|v| v.as_str()))
                .unwrap_or("an element");
            format!("Click on element '{}'", target)
        }
        "browser_type" | "browser_type_by_id" => {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or(raw);
            let short_text = if text.len() > 50 {
                &text[..text.floor_char_boundary(50)]
            } else {
                text
            };
            format!("Type text: \"{}\"", short_text)
        }
        "browser_fill" | "browser_fill_by_id" => {
            let value = args.get("value").and_then(|v| v.as_str()).unwrap_or(raw);
            let short_val = if value.len() > 50 {
                &value[..value.floor_char_boundary(50)]
            } else {
                value
            };
            format!("Fill a form field with: \"{}\"", short_val)
        }
        "browser_key_press" => {
            let key = args.get("key").and_then(|v| v.as_str()).unwrap_or(raw);
            format!("Press key: {}", key)
        }
        "browser_eval_js" => "Execute JavaScript code on the current page".to_string(),
        "browser_edit_artifact" => "Edit the Canvas content".to_string(),

        // Create
        "create_artifact" => {
            let title = args
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("Untitled");
            format!("Create a Canvas artifact: \"{}\"", title)
        }

        // Computer control
        n if n.starts_with("computer_") => match n {
            "computer_mouse_move" => "Move the mouse cursor".to_string(),
            "computer_mouse_click" | "computer_click" => "Click the mouse".to_string(),
            "computer_mouse_down" => "Press mouse button down".to_string(),
            "computer_mouse_up" => "Release mouse button".to_string(),
            "computer_mouse_drag" | "computer_drag" => "Drag with the mouse".to_string(),
            "computer_type_text" => {
                let text = args.get("text").and_then(|v| v.as_str()).unwrap_or(raw);
                let short = if text.len() > 50 {
                    &text[..text.floor_char_boundary(50)]
                } else {
                    text
                };
                format!("Type on keyboard: \"{}\"", short)
            }
            "computer_key_press" | "computer_key" => {
                let key = args.get("key").and_then(|v| v.as_str()).unwrap_or(raw);
                format!("Press keyboard key: {}", key)
            }
            "computer_hold_key" => {
                let key = args.get("key").and_then(|v| v.as_str()).unwrap_or(raw);
                format!("Hold keyboard key: {}", key)
            }
            _ => format!(
                "Control the computer ({})",
                n.strip_prefix("computer_").unwrap_or(n)
            ),
        },

        // Memory write
        "notify_user" => "Send you a notification".to_string(),
        "memory_create" => "Save new information to memory".to_string(),
        "memory_update" => "Update existing memory entry".to_string(),
        "memory_delete" => "Delete a memory entry".to_string(),
        "knowledge_teach" => "Learn new knowledge from you".to_string(),

        // File operations
        "write_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(raw);
            format!("Write to file: {}", path)
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(raw);
            format!("Edit file: {}", path)
        }
        "run_command" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or(raw);
            let short = if cmd.len() > 80 {
                &cmd[..cmd.floor_char_boundary(80)]
            } else {
                cmd
            };
            format!("Run command: {}", short)
        }

        // Subagent
        "subagent_spawn" => {
            let task = args.get("task").and_then(|v| v.as_str()).unwrap_or(raw);
            let short = if task.len() > 80 {
                &task[..task.floor_char_boundary(80)]
            } else {
                task
            };
            format!("Start a sub-agent: \"{}\"", short)
        }

        // Default
        _ => {
            format!("Perform action: {}", tool_name)
        }
    }
}

/// Show a question dialog in the sidebar via browser_ask_user action.
async fn execute_ask_user(
    question: &str,
    description: &str,
    options: &[String],
    browser_ctx: &BrowserContext,
) -> Option<String> {
    use tokio::sync::oneshot;

    let (response_tx, response_rx) = oneshot::channel();

    let request = BrowserRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        // Scoped, so the gateway for this session projects it to the portal.
        // The ACP/MCP bridge raises the same dialog as `check_tool_permission`
        // and was the path still sending a blank session — a phone saw the
        // dialog for one and not the other, with nothing to distinguish them.
        session_id: browser_ctx.session_id.clone(),
        tab_id: None,
        action: BrowserToolAction::AskUser,
        params: serde_json::json!({
            "question": question,
            // The action alone, so a client can render the question without
            // parsing the prose around it. Mirrors agent_host.
            "description": description,
            "options": options,
            "allow_custom": false,
            "timeout_ms": 86400000
        }),
        timeout_ms: 86_400_000, // 24 hours — wait for user decision
        client_identity: browser_ctx.client_identity.clone(),
        proxy_id: browser_ctx.proxy_id.clone(),
    };

    if browser_ctx
        .sender
        .send((request, response_tx))
        .await
        .is_err()
    {
        return None;
    }

    match response_rx.await {
        Ok(response) if response.success => {
            // browser_ask_user returns {"answer": "user's selection"} in result
            response
                .result
                .as_ref()
                .and_then(|v| v.get("answer").and_then(|a| a.as_str()).map(String::from))
        }
        _ => None,
    }
}

// ============================================================================
// Top-level dispatcher
// ============================================================================

/// Persist an ACP-bridge tool's result as a `tool_result` message, mirroring
/// `DaemonHostFunctions::record_tool_result` (agent_host.rs) for the
/// direct-API path. Without this, `/loop` verify checks and `/goal`
/// programmatic checks — which read `messages` where
/// `content_type='tool_result'` via `MessageRepository::list_recent_tool_results`
/// — always find nothing for ACP-bridge providers (claude-code, kimi-agent,
/// ...), because their tool calls execute here in `execute_mcp_tool` rather
/// than on the `Agent::run` / `DaemonHostFunctions` path.
///
/// Best effort: skips silently when there is no session_id, the content is
/// empty/whitespace, or the tool is the meta `think` tool (reasoning, not an
/// observation) — same skip conditions as `record_tool_result`. Persist
/// failures are logged, not propagated, so a storage hiccup never fails the
/// tool call itself.
pub(crate) fn record_acp_tool_result(services: &HostServices, tool_name: &str, content: &str) {
    if content.trim().is_empty() || tool_name == "think" {
        return;
    }
    if services.session_id.is_empty() {
        return;
    }
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("tool_name".to_string(), serde_json::json!(tool_name));
    metadata.insert("tool_id".to_string(), serde_json::json!(""));
    let params = nevoflux_storage::CreateMessageParams::new(
        &services.session_id,
        nevoflux_storage::models::MessageRole::Assistant,
        content,
    )
    .with_content_type(nevoflux_storage::models::ContentType::ToolResult)
    .with_metadata(metadata);
    if let Err(e) =
        nevoflux_storage::repositories::MessageRepository::new(&services.database).create(params)
    {
        tracing::warn!(tool = tool_name, error = %e, "record_acp_tool_result: persist failed");
    }
}

/// Execute a single MCP tool call, routing to the correct category.
///
/// Thin wrapper around [`execute_mcp_tool_inner`] that records a successful
/// result to the `messages` table via [`record_acp_tool_result`] before
/// returning it unchanged — see that function's docs for why. Errors are not
/// recorded (a failed tool call is not an observation to match against).
pub async fn execute_mcp_tool(
    name: &str,
    arguments: &serde_json::Value,
    services: &HostServices,
    tool_bridge: &Arc<McpToolBridge>,
) -> Result<String, String> {
    let result = execute_mcp_tool_inner(name, arguments, services, tool_bridge).await;
    if let Ok(ref content) = result {
        record_acp_tool_result(services, name, content);
    }
    result
}

/// Actual dispatch logic for [`execute_mcp_tool`]. Split out so the public
/// entry point can wrap every return path with a single result-recording
/// call instead of touching each of the many `return` sites below.
async fn execute_mcp_tool_inner(
    name: &str,
    arguments: &serde_json::Value,
    services: &HostServices,
    tool_bridge: &Arc<McpToolBridge>,
) -> Result<String, String> {
    // /loop iteration gate: reject tools that are forbidden inside iterations
    // (spec §10.2). MCP tools are registered server-side and visible to the
    // ACP provider regardless of the builtin-wasm Agent's tools_config filter,
    // so the gate has to live here too.
    if services.is_iteration {
        if matches!(name, "browser_ask_user" | "ask_user") {
            return Err("ask_user is forbidden inside /loop iterations \
                 (sidebar may be closed; nobody to answer). \
                 Use loop_scratchpad_set to persist state instead."
                .to_string());
        }
        if name == "loop_create" {
            return Err(
                "loop_create is forbidden inside /loop iterations (no nested loops)".to_string(),
            );
        }
        if name == "schedule_create" {
            return Err("schedule_create is not available inside unattended runs".to_string());
        }
        if name == "goal_set" {
            return Err("goal_set is not available inside unattended runs".to_string());
        }
    }

    // 0'. gbrain knowledge-base tools (M3-4).
    //
    // The agent calls these as `brain_*`. We translate to the gbrain-
    // side tool name (no prefix) and forward through the running
    // supervisor. If the brain subsystem isn't up (services.brain_supervisor
    // is None), surface a clear error rather than falling through to the
    // generic "unknown tool" branch — the agent gets a more actionable
    // hint about what's misconfigured.
    if name.starts_with("brain_") {
        let Some(def) = crate::brain_tools::lookup_by_nevoflux_name(name) else {
            return Err(format!(
                "brain tool {name} is not in the gbrain tool catalog; see crates/daemon/src/brain_tools.rs (resources/gbrain-tools.json)"
            ));
        };
        // M4-2.5: read through the hot-reloadable slot accessor so a
        // post-wizard `install_brain` is immediately visible without a
        // daemon restart.
        let supervisor = services.brain_supervisor().ok_or_else(|| {
            format!(
                "brain tool {name} requires the gbrain subsystem, but it is not running \
                 (check knowledge_base.brain.enabled in config, and that bun + gbrain CLI \
                 are installed)"
            )
        })?;
        return crate::brain_tools::invoke_brain_tool(
            &supervisor,
            &def.gbrain_name,
            arguments.clone(),
        )
        .await;
    }

    // 1. Browser tools
    if let Some(action) = tool_name_to_browser_action(name) {
        let browser_ctx = services
            .browser_context()
            .ok_or_else(|| "browser not available".to_string())?;
        return execute_browser_tool(action, arguments, &browser_ctx).await;
    }

    // 1'. Record & Replay tools (start_recording / stop_recording).
    //
    // Daemon-orchestrated: they mint the recording_id, write the JSONL header
    // through the RecordingCollector on the BrowserContext, and dispatch a
    // recording_start/recording_stop BrowserRequest. They are intentionally NOT
    // in `tool_name_to_browser_action` (which only maps single-shot browser
    // actions), so they need their own arm. ACP-bridge providers (claude-code,
    // gemini-cli, ...) reach this branch; the direct-API path mirrors it via
    // `agent_host::tool_call_dynamic`. Both delegate to the same helper.
    if name == "start_recording" || name == "stop_recording" {
        let browser_ctx = services
            .browser_context()
            .ok_or_else(|| "browser not available".to_string())?;
        return crate::agent::tools::dispatch_recording_tool(name, arguments, browser_ctx)
            .await
            .map_err(|e| e.to_string());
    }

    // 1''. Recorded-flow tools (run_flow / list_flows / report_flow_repair).
    //
    // Daemon-orchestrated: flow packages live in the user skills dirs and
    // run_flow executes the flow script via the code-mode executor. Browser
    // context is optional (list_flows / report_flow_repair need none); the
    // direct-API path mirrors this in `agent_host::tool_call_dynamic`.
    if name == "run_flow" || name == "list_flows" || name == "report_flow_repair" {
        let browser_ctx = services.browser_context();
        return crate::agent::flows::dispatch_flow_tool(name, arguments, browser_ctx)
            .await
            .map_err(|e| e.to_string());
    }

    // 2. Computer Use tools
    if name.starts_with("computer_") {
        return execute_computer_tool(name, arguments, services).await;
    }

    // `orchestrate` runs LLM-authored Python through Code Mode. The built-in
    // agent intercepts it inside its own loop (agent_host.rs), so an ACP agent
    // was offered the tool and got "unknown tool: orchestrate" on calling it.
    //
    // No LLM rewrite here, deliberately: orchestrate has never had one — the
    // built-in path also refuses ("No LLM retry in orchestrate tool mode") —
    // so the mechanical fixer is the whole safety net either way, and this
    // path behaves the same as the one it mirrors.
    if name == "orchestrate" {
        let code = arguments
            .get("code")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        if code.trim().is_empty() {
            return Err("orchestrate: missing required argument 'code'".to_string());
        }
        let outcome = crate::agent::code_mode::execute_python_simple_with_timeout(
            &code,
            services.browser_context(),
            Some(crate::agent::code_mode::ORCHESTRATE_MAX_DURATION),
            Some(services.interrupt_flag.clone()),
        );
        return if outcome.success {
            Ok(outcome.to_json_string())
        } else {
            Err(outcome
                .error
                .unwrap_or_else(|| "orchestrate failed with no error message".to_string()))
        };
    }

    // 3. Special tools (no external deps)
    match name {
        "think" => return Ok("Thought recorded.".to_string()),
        "create_plan" => {
            return Ok("Plan submitted for review.".to_string());
        }
        "create_artifact" => {
            return execute_create_artifact(arguments, tool_bridge).await;
        }
        _ => {}
    }

    // 3'. Canvas video tools (P2)
    //
    // ACP-style providers (claude-code, gemini-cli, kimi, openclaw) see
    // canvas_create_composition / canvas_render_video via the MCP HTTP
    // bridge and dispatch them through this executor. The corresponding
    // direct-API-provider path lives in `DaemonHostFunctions::canvas_video_*`
    // plus the builtin-wasm Agent::execute_tool arm.
    match name {
        "canvas_create_composition"
        | "canvas_render_video"
        | "canvas_lint_composition"
        | "canvas_apply_design_md"
        | "canvas_create_from_visual_identity"
        | "canvas_attach_asset"
        | "canvas_inspect_layout" => {
            return execute_canvas_video_tool(name, arguments, services).await;
        }
        _ => {}
    }

    // 4. Memory tools
    match name {
        "memory_search" => return execute_memory_search(arguments, services),
        "memory_create" => return execute_memory_create(arguments, services),
        "memory_update" => return execute_memory_update(arguments, services),
        "memory_delete" => return execute_memory_delete(arguments, services),
        "memory_view" => return execute_memory_view(arguments, services),
        "knowledge_teach" => return execute_knowledge_teach(arguments, services),
        _ => {}
    }

    // 4'. TTS subsystem tools (P5b).
    //
    // ACP-style providers see `tts_synthesize_api` via the MCP HTTP bridge
    // and dispatch through this executor. The direct-API-provider path
    // lives in `DaemonHostFunctions::tts_synthesize_api`.
    if name == "tts_synthesize_api" {
        return execute_tts_synthesize_api(arguments, services).await;
    }
    if name == "tts_voices" {
        return execute_tts_voices(services).await;
    }
    if name == "play_local_file" {
        return execute_play_local_file(arguments, services);
    }
    if name == "tts_synthesize_local" {
        return execute_tts_synthesize_local(arguments, services).await;
    }
    if name == "tts_transcribe" {
        return execute_tts_transcribe(arguments, services).await;
    }

    // 5. Skill/tool search
    match name {
        "skill_load" => return execute_skill_load(arguments, services).await,
        "tool_search" => return execute_tool_search(arguments, services).await,
        _ => {}
    }

    // 5'. Dynamic tool dispatch for ACP/MCP-HTTP-bridge providers
    // (claude-code, gemini-cli, ...). These external agents reach the 83
    // gbrain tools ONLY through `tool_search` + `tool_call_dynamic` — the
    // individual `brain_*` tools are not advertised over the bridge. The
    // in-daemon WASM agent routes `tool_call_dynamic` via
    // `AgentHost::tool_call_dynamic` (agent_host.rs); mirror that here by
    // unwrapping the inner tool and re-dispatching through THIS executor,
    // which already handles `brain_*` (via the gbrain supervisor) and
    // external MCP tools. Without this arm the bridge falls through to
    // "unknown tool: tool_call_dynamic".
    if name == "tool_call_dynamic" {
        let inner_name = arguments
            .get("tool_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "tool_call_dynamic: missing required 'tool_name'".to_string())?
            .to_string();
        if inner_name == "tool_call_dynamic" {
            return Err("tool_call_dynamic cannot call itself".to_string());
        }
        // The schema declares `arguments` as a JSON *string*; tolerate an
        // already-parsed object (some bridges pass objects) and an empty value.
        let inner_args: serde_json::Value = match arguments.get("arguments") {
            Some(serde_json::Value::String(s)) if !s.trim().is_empty() => {
                serde_json::from_str(s)
                    .map_err(|e| format!("tool_call_dynamic: 'arguments' is not valid JSON: {e}"))?
            }
            Some(serde_json::Value::String(_)) | None => serde_json::json!({}),
            Some(other) => other.clone(),
        };
        return Box::pin(execute_mcp_tool(
            &inner_name,
            &inner_args,
            services,
            tool_bridge,
        ))
        .await;
    }

    // 6. Subagent tools
    match name {
        "subagent_spawn" | "subagent_status" | "subagent_wait" | "subagent_wait_all"
        | "subagent_kill" | "subagent_list" => {
            return execute_subagent_tool(name, arguments, services).await;
        }
        _ => {}
    }

    // 6'. /loop skill tools (spec §10).
    //
    // Dispatched via `crate::loops::execute_loop_tool`. ACP-bridge providers
    // (claude-code, gemini-cli, kimi, openclaw) reach this branch through
    // the MCP HTTP bridge; direct-API providers reach the same dispatcher
    // here too because builtin-wasm's `Agent::execute_tool` arm has no
    // direct access to `LoopManager` (see Phase 9 scope correction).
    //
    // Requires `services.loop_manager` to be set at daemon startup
    // (Phase 23 wires this; until then tool calls surface a clear error).
    if matches!(
        name,
        "loop_create"
            | "loop_list"
            | "loop_cancel"
            | "loop_scratchpad_get"
            | "loop_scratchpad_set"
            | "loop_evolve"
            | "loop_proposal_respond"
    ) {
        let mgr = match services.loop_manager.as_ref() {
            Some(m) => m,
            None => {
                return Err(
                    "/loop tools are not available — daemon was started without a LoopManager"
                        .to_string(),
                );
            }
        };
        tracing::info!(
            tool = %name,
            is_iteration = services.is_iteration,
            iteration_loop_id = ?services.iteration_loop_id,
            session_id = %services.session_id,
            "loop_* MCP dispatch ctx"
        );
        let ctx = crate::loops::ToolCallContext {
            session_id: services.session_id.clone(),
            // When the call originates inside an /loop iteration (claude-code
            // via ACP/MCP HTTP), `services.is_iteration` is true and the
            // iteration's loop_id is set on the per-iteration HostServices
            // clone (see IterationExecutor::execute). The main-session tool
            // path leaves both at default so context gating works correctly.
            is_iteration: services.is_iteration,
            own_loop_id: services
                .iteration_loop_id
                .as_ref()
                .map(|id| crate::loops::LoopId(id.clone())),
        };
        // Gate G / 附录 C 缺口 1+2: bound a newly created loop's capability by
        // the session's execution tier on the ACP/MCP path too (mirrors the
        // native `agent_host::tool_loop_create` guard). Only applies to
        // creation; other loop_* tools are unaffected.
        if name == "loop_create" {
            let tier = crate::agent_host::resolve_execution_tier(services);
            crate::loops::tools::guard_loop_create(arguments, tier)?;
        }
        let result = crate::loops::execute_loop_tool(
            name,
            arguments,
            &ctx,
            mgr.as_ref(),
            services.database.as_ref(),
            Some(services),
        )
        .await;
        return match result {
            Ok(v) => Ok(serde_json::to_string(&v).unwrap_or_default()),
            Err(e) => Err(e),
        };
    }

    // 6''. /schedule skill tools (Task 1.6).
    //
    // Dispatched via `crate::schedules::execute_schedule_tool`. ACP-bridge
    // providers (claude-code, gemini-cli, kimi, openclaw) reach this branch
    // through the MCP HTTP bridge; direct-API providers reach the same
    // dispatcher through `agent_host.rs`'s `tool_schedule_*` methods, which
    // call the identical `execute_schedule_tool` dispatcher — only the
    // surface differs, mirroring the /loop wiring above.
    //
    // Requires `services.schedule_manager` to be set at daemon startup;
    // until then tool calls surface a clear error.
    // notify_user — publish a user-facing notification on the EventBus.
    // ACP-bridge providers reach this here; direct-API providers reach the
    // identical `crate::notify::publish_user_notification` via
    // `agent_host::tool_notify_user`. Only the surface differs.
    if name == "notify_user" {
        let bus = match services.event_bus.as_ref() {
            Some(b) => b,
            None => {
                return Err(
                    "notify_user is not available — daemon was started without an EventBus"
                        .to_string(),
                );
            }
        };
        let message = arguments
            .get("message")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| "notify_user requires a non-empty `message`".to_string())?;
        let title = arguments.get("title").and_then(|v| v.as_str());
        crate::notify::publish_user_notification(bus, title, message, "notify_user", None).await;
        return Ok(serde_json::json!({ "notified": true }).to_string());
    }

    if matches!(
        name,
        "schedule_create"
            | "schedule_list"
            | "schedule_cancel"
            | "schedule_pause"
            | "schedule_resume"
            | "schedule_run_now"
            | "schedule_runs"
    ) {
        let mgr = match services.schedule_manager.as_ref() {
            Some(m) => m,
            None => {
                return Err(
                    "/schedule tools are not available — daemon was started without a ScheduleManager"
                        .to_string(),
                );
            }
        };
        let ctx = crate::schedules::ScheduleToolContext {
            session_id: services.session_id.clone(),
            is_unattended: services.is_iteration,
        };
        let result =
            crate::schedules::execute_schedule_tool(name, arguments, &ctx, mgr.as_ref()).await;
        return match result {
            Ok(v) => Ok(serde_json::to_string(&v).unwrap_or_default()),
            Err(e) => Err(e),
        };
    }

    // 6'''. /goal skill tools (Task 2.3).
    //
    // Dispatched via `crate::goals::execute_goal_tool`. ACP-bridge providers
    // (claude-code, gemini-cli, kimi, openclaw) reach this branch through
    // the MCP HTTP bridge; direct-API providers reach the same dispatcher
    // through `agent_host.rs`'s `tool_goal_*` methods, which call the
    // identical `execute_goal_tool` dispatcher — only the surface differs,
    // mirroring the /schedule wiring above. `goal_set` is additionally
    // rejected earlier by the /loop iteration gate when
    // `services.is_iteration` is true (unattended runs must not hijack the
    // session goal — P3 composition passes goal config explicitly instead).
    //
    // Requires `services.goal_manager` to be set at daemon startup (Task
    // 2.4 wires boot); until then tool calls surface a clear error.
    if matches!(name, "goal_set" | "goal_status" | "goal_clear") {
        let mgr = match services.goal_manager.as_ref() {
            Some(m) => m,
            None => {
                return Err(
                    "/goal tools are not available — daemon was started without a GoalManager"
                        .to_string(),
                );
            }
        };
        let result = crate::goals::execute_goal_tool(
            name,
            arguments,
            &services.session_id,
            services.is_iteration,
            mgr.as_ref(),
        )
        .await;
        return match result {
            Ok(v) => Ok(serde_json::to_string(&v).unwrap_or_default()),
            Err(e) => Err(e),
        };
    }

    // 7. External MCP tools (via McpManager)
    if let Some(ref mcp_manager) = services.mcp_manager {
        if let Ok(result) = execute_mcp_manager_tool(name, arguments, mcp_manager).await {
            return Ok(result);
        }
    }

    Err(format!("unknown tool: {name}"))
}

// ============================================================================
// 1. Browser tools
// ============================================================================

/// Map tool name to BrowserToolAction.
fn tool_name_to_browser_action(name: &str) -> Option<BrowserToolAction> {
    // Strip either `browser_` or `canvas_` prefix. Most tools use the
    // `browser_` namespace; `canvas_extract_visual_identity` lives in the
    // `canvas_` family for product-grouping reasons (Mode 3 is a canvas
    // workflow) but its dispatch goes through the same browser-tool
    // bridge — so we also accept the `canvas_` prefix here.
    let key = name
        .strip_prefix("browser_")
        .or_else(|| name.strip_prefix("canvas_"))
        .unwrap_or(name);
    match key {
        "navigate" => Some(BrowserToolAction::Navigate),
        "activate_tab" => Some(BrowserToolAction::ActivateTab),
        "go_back" => Some(BrowserToolAction::GoBack),
        "go_forward" => Some(BrowserToolAction::GoForward),
        "click" => Some(BrowserToolAction::Click),
        "click_by_id" => Some(BrowserToolAction::ClickById),
        "type" => Some(BrowserToolAction::Type),
        "type_by_id" => Some(BrowserToolAction::TypeById),
        "fill" => Some(BrowserToolAction::Fill),
        "fill_by_id" => Some(BrowserToolAction::FillById),
        "get_content" => Some(BrowserToolAction::GetContent),
        "get_markdown" => Some(BrowserToolAction::GetMarkdown),
        "screenshot" => Some(BrowserToolAction::Screenshot),
        "snapshot" => Some(BrowserToolAction::Snapshot),
        "eval_js" => Some(BrowserToolAction::EvalJs),
        // `canvas_eval` strips the `canvas_` prefix to bare `eval` (distinct
        // from `eval_js`); run JS inside a canvas artifact's iframe (spec §6).
        "eval" => Some(BrowserToolAction::CanvasEval),
        "wait_for" => Some(BrowserToolAction::WaitFor),
        "wait_for_stable" => Some(BrowserToolAction::WaitForStable),
        "scroll" => Some(BrowserToolAction::Scroll),
        "get_element" => Some(BrowserToolAction::GetElement),
        "get_elements" => Some(BrowserToolAction::GetElements),
        "query_all" => Some(BrowserToolAction::QueryAll),
        "get_tabs" | "list_tabs" => Some(BrowserToolAction::ListTabs),
        "query_tabs" => Some(BrowserToolAction::QueryTabs),
        "key_press" => Some(BrowserToolAction::KeyPress),
        "read_artifact" => Some(BrowserToolAction::ReadArtifact),
        "edit_artifact" => Some(BrowserToolAction::EditArtifact),
        // canvas_extract_visual_identity is dispatched as a browser tool
        // (the action runs in the extension: open tab, run extractor, close).
        // Caller may pass `canvas_extract_visual_identity` (preferred,
        // matches the agent-facing tool name) or the bare suffix.
        "extract_visual_identity" => Some(BrowserToolAction::ExtractVisualIdentity),
        "ask_user" => Some(BrowserToolAction::AskUser),
        "web_search" => Some(BrowserToolAction::WebSearch),
        // Both spellings reach the same action. `web_fetch` is the name the
        // agent is given (and the one `protocol::READ_ONLY_TOOLS` classifies);
        // `fetch_page` is what the dispatch was originally written against.
        // Only accepting the second meant the advertised name resolved to
        // nothing and fell through to "unknown tool".
        "fetch_page" | "web_fetch" => Some(BrowserToolAction::WebFetch),
        // Browser input strategy engine (PR #2 + #2.5)
        "input" => Some(BrowserToolAction::Input),
        "probe" => Some(BrowserToolAction::Probe),
        // File upload (PR #5)
        "upload_file" => Some(BrowserToolAction::UploadFile),
        _ => None,
    }
}

/// Execute a browser tool via BrowserContext channel.
async fn execute_browser_tool(
    action: BrowserToolAction,
    arguments: &serde_json::Value,
    browser_ctx: &BrowserContext,
) -> Result<String, String> {
    use tokio::sync::oneshot;

    // Intercept PR #2 orchestrated tools (browser_input, browser_probe) before
    // the standard single-call dispatch. These run multi-step pipelines in the
    // daemon (probe → decide → execute → verify) rather than forwarding a
    // single request to the browser extension.
    if matches!(action, BrowserToolAction::Input | BrowserToolAction::Probe) {
        return execute_browser_input_orchestrated(action, arguments, browser_ctx).await;
    }

    // Intercept PR #5 upload tool — orchestrated like browser_input.
    if matches!(action, BrowserToolAction::UploadFile) {
        return execute_browser_upload_orchestrated(arguments, browser_ctx).await;
    }

    // Routing tab_id: top-level for most tools; for ExtractVisualIdentity
    // it lives at `target.tab_id` (nested) per the spec's two-mode shape
    // (URL mode → no tab_id; TabId mode → target.tab_id present).
    let tab_id = arguments
        .get("tab_id")
        .and_then(|v| v.as_i64())
        .or_else(|| {
            if matches!(action, BrowserToolAction::ExtractVisualIdentity) {
                arguments
                    .get("target")
                    .and_then(|t| t.get("tab_id"))
                    .and_then(|v| v.as_i64())
            } else {
                None
            }
        });

    // Remove tab_id from params — it's a routing field on BrowserRequest,
    // not a tool parameter. Firefox WebExtension schema rejects unknown properties.
    let params = if let Some(obj) = arguments.as_object() {
        let mut clean = obj.clone();
        clean.remove("tab_id");
        serde_json::Value::Object(clean)
    } else {
        arguments.clone()
    };

    let (response_tx, response_rx) = oneshot::channel();

    let request = BrowserRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        // Every browser request belongs to the session that asked for it.
        // Blank here meant `ask_user` — the agent's own question, which comes
        // through this generic path rather than `execute_ask_user` — was
        // fanned out to the remote gateway and then dropped by its session
        // filter: the sidebar raised the dialog, the phone saw nothing.
        session_id: browser_ctx.session_id.clone(),
        tab_id,
        action,
        params,
        timeout_ms: 30_000,
        client_identity: browser_ctx.client_identity.clone(),
        proxy_id: browser_ctx.proxy_id.clone(),
    };

    browser_ctx
        .sender
        .send((request, response_tx))
        .await
        .map_err(|e| format!("browser_sender failed: {e}"))?;

    let response = response_rx
        .await
        .map_err(|_| "browser response channel closed".to_string())?;

    if response.success {
        if action == BrowserToolAction::Screenshot {
            return Ok(screenshot_result(&response.result, &browser_ctx.session_id));
        }
        Ok(response
            .result
            .map(|v| serde_json::to_string(&v).unwrap_or_default())
            .unwrap_or_default())
    } else {
        Err(response
            .error
            .map(|e| format!("{}: {}", e.code, e.message))
            .unwrap_or_else(|| "unknown tool error".to_string()))
    }
}

/// The base64 a screenshot response carries, whatever key it used.
///
/// The extension has answered with a bare string, with `data`, and with
/// `screenshot` at different times; taking whichever is present beats
/// depending on one.
fn screenshot_payload(result: &Option<serde_json::Value>) -> Option<String> {
    let value = result.as_ref()?;
    if let Some(s) = value.as_str() {
        return (!s.is_empty()).then(|| s.to_string());
    }
    let obj = value.as_object()?;
    for key in ["screenshot", "data", "data_url", "image", "base64"] {
        if let Some(serde_json::Value::String(s)) = obj.get(key) {
            if !s.is_empty() {
                return Some(s.clone());
            }
        }
    }
    None
}

/// Turn a screenshot response into what the model should be given.
///
/// It used to be given the bytes. This path returned the extension's reply
/// verbatim, and that reply puts the picture under `data` — while the vision
/// lift in `wasm::llm` looks for `screenshot`, so it never fired here and
/// twenty thousand characters of base64 went into the model's context as
/// text. What came back out was that text re-emitted token by token, with a
/// couple of thousand characters missing from the middle: the phone got a
/// JPEG whose length was not a multiple of four and which no browser would
/// decode.
///
/// So: the bytes go to the asset store and the model gets an id, plus the
/// markdown to write with it and a sentence telling it not to write bytes.
/// The payload is also re-keyed to `screenshot`, which is what makes the
/// vision lift fire — the model still sees the picture, it just no longer
/// holds a transcript of it.
fn screenshot_result(result: &Option<serde_json::Value>, session_id: &str) -> String {
    let Some(payload) = screenshot_payload(result) else {
        return result
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_default();
    };

    let asset = store_screenshot(&payload, session_id);
    let mut out = serde_json::Map::new();
    out.insert("success".into(), serde_json::Value::Bool(true));
    out.insert("screenshot".into(), serde_json::Value::String(payload));
    if let Some(id) = asset {
        out.insert("asset_id".into(), serde_json::Value::String(id.clone()));
        out.insert(
            "show_with".into(),
            serde_json::Value::String(format!("![screenshot](nevo-asset:{id})")),
        );
        out.insert(
            "note".into(),
            serde_json::Value::String(
                "To show this to the user, write the markdown in show_with. Never write image                  bytes or a data: URL — anything you produce will be discarded."
                    .into(),
            ),
        );
    }
    serde_json::to_string(&serde_json::Value::Object(out)).unwrap_or_default()
}

/// Put a screenshot where a portal can fetch it. `None` when none is watching.
fn store_screenshot(raw: &str, session_id: &str) -> Option<String> {
    use base64::Engine;
    let (mime, b64) = match raw.strip_prefix("data:") {
        Some(rest) => {
            let (header, payload) = rest.split_once(',')?;
            (
                header.split(';').next().unwrap_or("image/png").to_string(),
                payload,
            )
        }
        None => ("image/jpeg".to_string(), raw),
    };
    let clean: String = b64.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(clean)
        .ok()?;
    let ext = mime.rsplit('/').next().unwrap_or("png");
    let offer = crate::remote::asset::put_for_session(
        session_id,
        &bytes,
        &format!("screenshot.{ext}"),
        &mime,
    )?;
    tracing::info!(
        target: "remote",
        id = %offer.id, bytes = offer.size, session = %session_id,
        "screenshot offered as an asset instead of inline bytes"
    );
    Some(offer.id)
}

/// Dispatch PR #2 orchestrated browser tools (browser_input / browser_probe).
///
/// Unlike single-call browser tools, these run a multi-step pipeline inside
/// the daemon: probe the target, run the pure `decide()` strategy function,
/// execute the chosen plan via Actor methods, and verify the result. The
/// WASM guest only sees one tool call return — the daemon expands it into
/// the full orchestration internally.
///
/// Called from two paths:
/// 1. `execute_browser_tool` when dispatching via the standard MCP executor
///    loop (for tools that come through mcp_tool_executor::execute_mcp_tool)
/// 2. `agent_host::tool_call_dynamic` when the WASM guest dispatches via the
///    generic tool_call_dynamic host function (the default path for PR #2
///    tools in Agent / Browser mode)
pub async fn execute_browser_input_orchestrated(
    action: BrowserToolAction,
    arguments: &serde_json::Value,
    browser_ctx: &BrowserContext,
) -> Result<String, String> {
    use crate::agent::browser_input::bridge::RealBrowserBridge;
    use crate::agent::browser_input::{run_browser_input, run_browser_probe, InputMode};
    use std::sync::Arc;

    // Wrap the BrowserContext in an Arc so RealBrowserBridge can hold onto
    // it; the bridge only needs read access so a clone is sufficient.
    let bridge = RealBrowserBridge::new(Arc::new(browser_ctx.clone()));

    match action {
        BrowserToolAction::Input => {
            let selector = arguments
                .get("selector")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "browser_input: selector required".to_string())?;
            let text = arguments
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "browser_input: text required".to_string())?;
            let mode = arguments
                .get("mode")
                .and_then(|v| v.as_str())
                .map(|s| {
                    if s == "type" {
                        InputMode::Type
                    } else {
                        InputMode::Fill
                    }
                })
                .unwrap_or(InputMode::Fill);
            let verify_enabled = arguments
                .get("verify")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let tab_id = arguments.get("tab_id").and_then(|v| v.as_i64());

            let adapter_registry =
                crate::agent::browser_input::AdapterRegistry::load_standard(None, None);
            let result = run_browser_input(
                &bridge,
                &adapter_registry,
                selector,
                text,
                mode,
                tab_id,
                verify_enabled,
            )
            .await
            .map_err(|e| e.to_string())?;
            Ok(serde_json::to_string(&result).unwrap_or_default())
        }
        BrowserToolAction::Probe => {
            let selector = arguments
                .get("selector")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "browser_probe: selector required".to_string())?;
            let tab_id = arguments.get("tab_id").and_then(|v| v.as_i64());

            let fingerprint = run_browser_probe(&bridge, selector, tab_id)
                .await
                .map_err(|e| e.to_string())?;
            Ok(serde_json::to_string(&fingerprint).unwrap_or_default())
        }
        _ => unreachable!("execute_browser_input_orchestrated called with non-orchestrated action"),
    }
}

/// Execute `browser_upload_file` in WASM/MCP mode.
///
/// Mirrors the logic in `BrowserTool::run_browser_upload_from_args` (tools.rs)
/// but operates on the BrowserContext available in the MCP executor path.
async fn execute_browser_upload_orchestrated(
    arguments: &serde_json::Value,
    browser_ctx: &BrowserContext,
) -> Result<String, String> {
    use crate::agent::browser_input::upload::{
        check_file_size, check_sensitive_path, detect_mime, validate_workspace_path,
        DEFAULT_MAX_SIZE, TOKEN_TTL,
    };
    use std::path::PathBuf;

    let selector = arguments
        .get("selector")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "browser_upload_file: selector required".to_string())?;
    let file_path_str = arguments
        .get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "browser_upload_file: file_path required".to_string())?;
    let tab_id = arguments.get("tab_id").and_then(|v| v.as_i64());

    // Resolve workspace directory (supports custom workspace_dir parameter).
    let workspace_dir = match arguments.get("workspace_dir").and_then(|v| v.as_str()) {
        Some(dir) => PathBuf::from(dir),
        None => dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("nevoflux")
            .join("workspace"),
    };

    if !workspace_dir.exists() {
        std::fs::create_dir_all(&workspace_dir)
            .map_err(|e| format!("browser_upload_file: cannot create workspace dir: {e}"))?;
    }

    let canonical = validate_workspace_path(std::path::Path::new(file_path_str), &workspace_dir)
        .map_err(|e| e.to_string())?;

    check_sensitive_path(&canonical).map_err(|e| e.to_string())?;

    let size = check_file_size(&canonical, DEFAULT_MAX_SIZE).map_err(|e| e.to_string())?;

    let mime_type = detect_mime(&canonical).map_err(|e| e.to_string())?;

    let file_name = canonical
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());

    let asset_server = browser_ctx.asset_server.as_ref().ok_or_else(|| {
        "browser_upload_file: AssetServer is not running on this daemon".to_string()
    })?;
    let file_url =
        asset_server.register_download(canonical, mime_type.clone(), file_name.clone(), TOKEN_TTL);

    // Send to Actor via BrowserContext channel.
    let params = serde_json::json!({
        "selector": selector,
        "fileUrl": file_url,
        "fileName": file_name,
        "mimeType": mime_type,
    });

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    let request = BrowserRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        session_id: String::new(),
        tab_id,
        action: BrowserToolAction::UploadFile,
        params,
        timeout_ms: 120_000,
        client_identity: browser_ctx.client_identity.clone(),
        proxy_id: browser_ctx.proxy_id.clone(),
    };

    browser_ctx
        .sender
        .send((request, response_tx))
        .await
        .map_err(|_| "browser_upload_file: channel closed".to_string())?;

    let response = tokio::time::timeout(std::time::Duration::from_secs(120), response_rx)
        .await
        .map_err(|_| "browser_upload_file: request timed out".to_string())?
        .map_err(|_| "browser_upload_file: response channel closed".to_string())?;

    if response.success {
        Ok(serde_json::json!({
            "success": true,
            "file_name": file_name,
            "mime_type": mime_type,
            "size": size,
        })
        .to_string())
    } else {
        let msg = response
            .error
            .map(|e| e.message)
            .unwrap_or_else(|| "Upload failed".to_string());
        Err(msg)
    }
}

// ============================================================================
// 2. Computer Use tools
// ============================================================================

/// Execute a computer control tool via ComputerController.
async fn execute_computer_tool(
    name: &str,
    arguments: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let controller = services
        .computer_controller
        .as_ref()
        .ok_or_else(|| "computer controller not available".to_string())?;

    match name {
        "computer_screenshot" => {
            let screenshot = if let Some(region) = arguments.get("region") {
                let x = region.get("x").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let y = region.get("y").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let width = region.get("width").and_then(|v| v.as_u64()).unwrap_or(100) as u32;
                let height = region.get("height").and_then(|v| v.as_u64()).unwrap_or(100) as u32;
                controller
                    .capture_region(Region::new(x, y, width, height))
                    .await
                    .map_err(|e| format!("screenshot failed: {e}"))?
            } else if let Some(display_id) = arguments.get("display_id").and_then(|v| v.as_u64()) {
                controller
                    .capture_display(display_id as u32)
                    .await
                    .map_err(|e| format!("screenshot failed: {e}"))?
            } else {
                controller
                    .capture_screen()
                    .await
                    .map_err(|e| format!("screenshot failed: {e}"))?
            };
            // Return the serialized screenshot (contains base64 data)
            serde_json::to_string(&screenshot).map_err(|e| format!("serialize failed: {e}"))
        }
        "computer_mouse_move" => {
            let x = arguments
                .get("x")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| "missing 'x' argument".to_string())? as i32;
            let y = arguments
                .get("y")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| "missing 'y' argument".to_string())? as i32;
            controller
                .move_to(Point::new(x, y))
                .await
                .map_err(|e| format!("mouse move failed: {e}"))?;
            Ok(format!("Moved mouse to ({x}, {y})"))
        }
        "computer_mouse_click" => {
            let button = parse_mouse_button(arguments);
            let click_type = parse_click_type(arguments);

            if let (Some(x), Some(y)) = (
                arguments.get("x").and_then(|v| v.as_i64()),
                arguments.get("y").and_then(|v| v.as_i64()),
            ) {
                controller
                    .click_at(Point::new(x as i32, y as i32), button, click_type)
                    .await
                    .map_err(|e| format!("mouse click failed: {e}"))?;
                Ok(format!("Clicked {click_type:?} {button:?} at ({x}, {y})"))
            } else {
                controller
                    .click(button, click_type)
                    .await
                    .map_err(|e| format!("mouse click failed: {e}"))?;
                Ok(format!("Clicked {click_type:?} {button:?}"))
            }
        }
        "computer_mouse_scroll" => {
            let direction = arguments
                .get("direction")
                .and_then(|v| v.as_str())
                .map(|s| match s {
                    "down" => ScrollDirection::Down,
                    "left" => ScrollDirection::Left,
                    "right" => ScrollDirection::Right,
                    _ => ScrollDirection::Up,
                })
                .unwrap_or(ScrollDirection::Down);
            let amount = arguments
                .get("amount")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as u32;
            controller
                .scroll(direction, amount)
                .await
                .map_err(|e| format!("mouse scroll failed: {e}"))?;
            Ok(format!("Scrolled {direction:?} by {amount}"))
        }
        "computer_mouse_drag" => {
            let from_x = arguments
                .get("from_x")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| "missing 'from_x'".to_string())? as i32;
            let from_y = arguments
                .get("from_y")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| "missing 'from_y'".to_string())? as i32;
            let to_x = arguments
                .get("to_x")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| "missing 'to_x'".to_string())? as i32;
            let to_y = arguments
                .get("to_y")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| "missing 'to_y'".to_string())? as i32;
            let button = parse_mouse_button(arguments);
            controller
                .drag(Point::new(from_x, from_y), Point::new(to_x, to_y), button)
                .await
                .map_err(|e| format!("mouse drag failed: {e}"))?;
            Ok(format!(
                "Dragged from ({from_x}, {from_y}) to ({to_x}, {to_y})"
            ))
        }
        "computer_mouse_position" => {
            let pos = controller
                .get_position()
                .await
                .map_err(|e| format!("get position failed: {e}"))?;
            serde_json::to_string(&pos).map_err(|e| format!("serialize failed: {e}"))
        }
        "computer_type_text" => {
            let text = arguments
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing 'text' argument".to_string())?;
            controller
                .type_text(text)
                .await
                .map_err(|e| format!("type text failed: {e}"))?;
            Ok(format!("Typed {} characters", text.len()))
        }
        "computer_press_key" => {
            let key_or_char = if let Some(key_str) = arguments.get("key").and_then(|v| v.as_str()) {
                parse_key_string(key_str)?
            } else if let Some(char_str) = arguments.get("char").and_then(|v| v.as_str()) {
                if char_str.len() != 1 {
                    return Err("char must be a single character".to_string());
                }
                KeyOrChar::Char(char_str.chars().next().unwrap())
            } else {
                return Err("missing 'key' or 'char' argument".to_string());
            };

            let mut combination = KeyCombination {
                key: key_or_char,
                modifiers: Vec::new(),
            };

            if let Some(modifiers) = arguments.get("modifiers").and_then(|v| v.as_array()) {
                for modifier in modifiers {
                    if let Some(mod_str) = modifier.as_str() {
                        match mod_str.to_lowercase().as_str() {
                            "shift" => combination = combination.with_shift(),
                            "ctrl" | "control" => combination = combination.with_ctrl(),
                            "alt" => combination = combination.with_alt(),
                            "meta" | "cmd" | "command" | "win" | "windows" => {
                                combination = combination.with_meta()
                            }
                            _ => {}
                        }
                    }
                }
            }

            controller
                .press_key(combination.clone())
                .await
                .map_err(|e| format!("press key failed: {e}"))?;
            Ok(format!("Pressed key combination: {combination:?}"))
        }
        "computer_get_displays" => {
            let displays = controller
                .get_displays()
                .await
                .map_err(|e| format!("get displays failed: {e}"))?;
            serde_json::to_string(&displays).map_err(|e| format!("serialize failed: {e}"))
        }
        _ => Err(format!("unknown computer tool: {name}")),
    }
}

/// Parse mouse button from arguments (default: left).
fn parse_mouse_button(arguments: &serde_json::Value) -> MouseButton {
    arguments
        .get("button")
        .and_then(|v| v.as_str())
        .map(|s| match s {
            "right" => MouseButton::Right,
            "middle" => MouseButton::Middle,
            _ => MouseButton::Left,
        })
        .unwrap_or(MouseButton::Left)
}

/// Parse click type from arguments (default: single).
fn parse_click_type(arguments: &serde_json::Value) -> ClickType {
    arguments
        .get("click_type")
        .and_then(|v| v.as_str())
        .map(|s| match s {
            "double" => ClickType::Double,
            "triple" => ClickType::Triple,
            _ => ClickType::Single,
        })
        .unwrap_or(ClickType::Single)
}

/// Parse a key string into KeyOrChar.
fn parse_key_string(key_str: &str) -> Result<KeyOrChar, String> {
    let key = match key_str.to_lowercase().as_str() {
        "shift" => Key::Shift,
        "ctrl" | "control" => Key::Control,
        "alt" => Key::Alt,
        "meta" | "cmd" | "command" | "win" | "windows" => Key::Meta,
        "f1" => Key::F1,
        "f2" => Key::F2,
        "f3" => Key::F3,
        "f4" => Key::F4,
        "f5" => Key::F5,
        "f6" => Key::F6,
        "f7" => Key::F7,
        "f8" => Key::F8,
        "f9" => Key::F9,
        "f10" => Key::F10,
        "f11" => Key::F11,
        "f12" => Key::F12,
        "escape" | "esc" => Key::Escape,
        "tab" => Key::Tab,
        "capslock" | "caps_lock" => Key::CapsLock,
        "space" => Key::Space,
        "enter" | "return" => Key::Enter,
        "backspace" => Key::Backspace,
        "delete" | "del" => Key::Delete,
        "insert" | "ins" => Key::Insert,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" | "page_up" => Key::PageUp,
        "pagedown" | "page_down" => Key::PageDown,
        "up" | "arrowup" | "arrow_up" => Key::ArrowUp,
        "down" | "arrowdown" | "arrow_down" => Key::ArrowDown,
        "left" | "arrowleft" | "arrow_left" => Key::ArrowLeft,
        "right" | "arrowright" | "arrow_right" => Key::ArrowRight,
        "printscreen" | "print_screen" => Key::PrintScreen,
        "scrolllock" | "scroll_lock" => Key::ScrollLock,
        "pause" => Key::Pause,
        "numlock" | "num_lock" => Key::NumLock,
        s if s.len() == 1 => {
            return Ok(KeyOrChar::Char(s.chars().next().unwrap()));
        }
        _ => {
            return Err(format!("unknown key: {key_str}"));
        }
    };
    Ok(KeyOrChar::Key(key))
}

// ============================================================================
// 3. Special tools: create_artifact
// ============================================================================

/// Execute `create_artifact` by persisting to the database.
///
/// In the normal WASM agent flow, artifacts are rendered by the sidebar.
/// In ACP bridge mode we persist the artifact and return its metadata.
/// The sidebar picks up artifacts via the existing session artifact API.
async fn execute_create_artifact(
    arguments: &serde_json::Value,
    tool_bridge: &Arc<McpToolBridge>,
) -> Result<String, String> {
    let title = arguments
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Untitled")
        .to_string();
    let content_type = arguments
        .get("content_type")
        .and_then(|v| v.as_str())
        .unwrap_or("text/html")
        .to_string();
    let content = arguments
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let description = arguments
        .get("description")
        .and_then(|v| v.as_str())
        .map(String::from);
    let entry = arguments
        .get("entry")
        .and_then(|v| v.as_str())
        .map(String::from);
    let files: Option<std::collections::HashMap<String, String>> =
        arguments.get("files").and_then(|f| {
            if let Some(obj) = f.as_object() {
                Some(
                    obj.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect(),
                )
            } else if let Some(s) = f.as_str() {
                serde_json::from_str(s).ok()
            } else {
                None
            }
        });

    // Generate unique artifact ID
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let id = format!("art-{}-1", ts);

    // Store as pending artifact — server will pick it up after prompt completes
    // and call send_artifact_stream to push to sidebar.
    tool_bridge.push_artifact(PendingArtifact {
        id: id.clone(),
        title: title.clone(),
        content_type,
        description,
        content,
        files,
        entry,
    });

    Ok(serde_json::json!({
        "id": id,
        "title": title,
        "status": "created"
    })
    .to_string())
}

// ============================================================================
// 3'. Canvas video tools (P2)
// ============================================================================

/// Execute `canvas_create_composition` / `canvas_render_video` via the shared
/// CanvasVideoService on `HostServices`.
///
/// Non-blocking: `canvas_render_video` returns a `job_id` immediately; the
/// render loop emits progress and terminal events on the EventBus channel
/// `jobs.render.{job_id}`, which the sidebar consumes (see P2 design §3.1).
///
/// Mirrors the contract of `DaemonHostFunctions::canvas_video_create_composition`
/// / `canvas_video_render_start` (agent_host.rs) so both dispatch paths
/// (builtin-wasm's Agent loop and the ACP MCP bridge) share the same service
/// and response shape.
async fn execute_canvas_video_tool(
    name: &str,
    arguments: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let svc = services
        .canvas_video_service
        .as_ref()
        .ok_or_else(|| "canvas_video service not wired into HostServices".to_string())?;

    match name {
        "canvas_create_composition" => {
            // Use the shared strict parser so the LLM-facing dispatch path
            // gets the same `html`-rejection gate as the in-process tool
            // executor (canvas_video::tool::CanvasCreateCompositionTool).
            // Without this, the MCP/ACP path silently accepted hallucinated
            // html submissions and meta.origin.template ended up null.
            let mut req =
                crate::canvas_video::tool::parse_create_composition_args_strict(arguments)
                    .map_err(|e| e.to_string())?;
            // Inject the current session_id from HostServices when the LLM
            // didn't supply one (the LLM tool schema doesn't expose
            // session_id, so req.session_id is always None here). Without
            // this, the artifact row gets created with session_id=NULL and
            // ContentStore mirror writes have to fall back to update_files;
            // we'd rather have a proper FK link from the start so listing /
            // session-scoped queries see the artifact.
            if req.session_id.is_none() && !services.session_id.is_empty() {
                req.session_id = Some(services.session_id.clone());
            }
            let resp = svc
                .create_composition(req)
                .await
                .map_err(|e| e.to_string())?;
            // Auto-open the canvas tab so the user immediately sees the
            // composition they just asked the agent to create. Without
            // this broadcast, canvas_create_composition is a silent SQL
            // insert from the user's perspective — the artifact exists
            // but no UI surface shows it until lint/render runs and the
            // sidebar's artifact card appears (or the user manually opens
            // it from the canvas list). Mirrors the pattern used by
            // canvas_render_video → canvas_video_open_render_tab.
            //
            // The canvas page self-hydrates from the daemon (content_store.
            // load → artifact.get fallback) so the extension only needs to
            // open the tab; no upfront ContentStore population needed.
            if let Some(tx) = services.broadcast_tx.as_ref() {
                let payload = serde_json::json!({
                    "type": "canvas_video_open_canvas_tab",
                    "payload": { "artifact_id": resp.artifact_id }
                });
                let env = nevoflux_protocol::DaemonEnvelope::broadcast(
                    nevoflux_protocol::Channel::Chat,
                    payload,
                );
                let _ = tx.send((b"*".to_vec(), env)).await;
            }
            serde_json::to_string(&resp)
                .map_err(|e| format!("serialize canvas_create_composition response: {}", e))
        }
        "canvas_render_video" => {
            let req: nevoflux_protocol::canvas_video::RenderStartRequest =
                serde_json::from_value(arguments.clone())
                    .map_err(|e| format!("invalid canvas_render_video args: {}", e))?;
            let resp = svc.render_start(req).await.map_err(|e| e.to_string())?;

            // Broadcast canvas_video_open_render_tab to all connected
            // proxies so the extension A4 handler opens
            // nevoflux://render/{job_id}. The TCP-proxy code path in
            // server.rs has its own broadcast for
            // canvas_video_render_start — this one covers the MCP/ACP
            // LLM tool-call path which never flows through that handler.
            if let Some(tx) = services.broadcast_tx.as_ref() {
                let payload = serde_json::json!({
                    "type": "canvas_video_open_render_tab",
                    "payload": { "job_id": resp.job_id }
                });
                let env = nevoflux_protocol::DaemonEnvelope::broadcast(
                    nevoflux_protocol::Channel::Chat,
                    payload,
                );
                let _ = tx.send((b"*".to_vec(), env)).await;
            }

            serde_json::to_string(&resp)
                .map_err(|e| format!("serialize canvas_render_video response: {}", e))
        }
        "canvas_lint_composition" => {
            let req: nevoflux_protocol::canvas_video::LintCompositionRequest =
                serde_json::from_value(arguments.clone())
                    .map_err(|e| format!("invalid canvas_lint_composition args: {}", e))?;
            let report = svc
                .lint_composition(&req.composition_id)
                .await
                .map_err(|e| e.to_string())?;
            serde_json::to_string(&report)
                .map_err(|e| format!("serialize canvas_lint_composition response: {}", e))
        }
        "canvas_apply_design_md" => {
            let req: nevoflux_protocol::canvas_video::ApplyDesignMdRequest =
                serde_json::from_value(arguments.clone())
                    .map_err(|e| format!("invalid canvas_apply_design_md args: {}", e))?;
            svc.apply_design_md(&req.composition_id)
                .await
                .map_err(|e| e.to_string())?;
            let resp = nevoflux_protocol::canvas_video::ApplyDesignMdResponse {
                composition_id: req.composition_id,
            };
            serde_json::to_string(&resp)
                .map_err(|e| format!("serialize canvas_apply_design_md response: {}", e))
        }
        "canvas_attach_asset" => {
            let req: nevoflux_protocol::canvas_video::AttachAssetRequest =
                serde_json::from_value(arguments.clone())
                    .map_err(|e| format!("invalid canvas_attach_asset args: {}", e))?;
            let resolved = resolve_attach_asset_payload(&req)
                .await
                .map_err(|e| format!("canvas_attach_asset: {e}"))?;
            let composition_id = req.composition_id.clone();
            let path = svc
                .attach_asset(
                    &req.composition_id,
                    &resolved.name,
                    &resolved.mime_type,
                    &resolved.payload_b64,
                    resolved.size_bytes,
                )
                .await
                .map_err(|e| e.to_string())?;
            // Notify the extension that the composition's binary side
            // changed (assets are NOT in artifacts.files / ContentStore
            // anymore — moved to composition_assets in migration 016).
            // Without this, an open Canvas Editor tab keeps rendering
            // the pre-attach HTML and the image only shows up after a
            // manual reload. Background.js handles the broadcast by
            // re-fetching the artifact via system_command artifact.get,
            // whose response hydrates ContentStore → fires the canvas
            // page subscriber → re-render → fresh URL-rewritten HTML.
            if let Some(tx) = services.broadcast_tx.as_ref() {
                let payload = serde_json::json!({
                    "type": "canvas_video_artifact_changed",
                    "payload": { "artifact_id": composition_id }
                });
                let env = nevoflux_protocol::DaemonEnvelope::broadcast(
                    nevoflux_protocol::Channel::Chat,
                    payload,
                );
                let _ = tx.send((b"*".to_vec(), env)).await;
            }
            let resp = nevoflux_protocol::canvas_video::AttachAssetResponse {
                path,
                mime_type: resolved.mime_type,
                size_bytes: resolved.size_bytes,
            };
            serde_json::to_string(&resp)
                .map_err(|e| format!("serialize canvas_attach_asset response: {}", e))
        }
        "canvas_inspect_layout" => {
            let req: nevoflux_protocol::canvas_video::InspectLayoutRequest =
                serde_json::from_value(arguments.clone())
                    .map_err(|e| format!("invalid canvas_inspect_layout args: {}", e))?;
            let frames = req.frames.unwrap_or(8);
            let report = svc
                .inspect_layout(&req.composition_id, frames, &req.at)
                .await
                .map_err(|e| e.to_string())?;
            let resp = nevoflux_protocol::canvas_video::InspectLayoutResponse { report };
            serde_json::to_string(&resp)
                .map_err(|e| format!("serialize canvas_inspect_layout response: {}", e))
        }
        "canvas_create_from_visual_identity" => {
            let mut req: nevoflux_protocol::canvas_video::CreateFromVisualIdentityRequest =
                serde_json::from_value(arguments.clone()).map_err(|e| {
                    format!("invalid canvas_create_from_visual_identity args: {}", e)
                })?;
            // Inject session_id from HostServices when LLM didn't supply one
            // — same rationale as canvas_create_composition (artifact's
            // session_id FK must be populated for ContentStore mirror to
            // hit the artifacts table).
            if req.session_id.is_none() && !services.session_id.is_empty() {
                req.session_id = Some(services.session_id.clone());
            }
            let resp = svc
                .create_from_visual_identity(req)
                .await
                .map_err(|e| e.to_string())?;
            // Auto-open canvas tab — mirror canvas_create_composition.
            if let Some(tx) = services.broadcast_tx.as_ref() {
                let payload = serde_json::json!({
                    "type": "canvas_video_open_canvas_tab",
                    "payload": { "artifact_id": resp.artifact_id }
                });
                let env = nevoflux_protocol::DaemonEnvelope::broadcast(
                    nevoflux_protocol::Channel::Chat,
                    payload,
                );
                let _ = tx.send((b"*".to_vec(), env)).await;
            }
            serde_json::to_string(&resp).map_err(|e| {
                format!(
                    "serialize canvas_create_from_visual_identity response: {}",
                    e
                )
            })
        }
        other => Err(format!(
            "execute_canvas_video_tool called with unexpected name: {}",
            other
        )),
    }
}

// ============================================================================
// 4. Memory tools
// ============================================================================

fn execute_memory_search(
    args: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let query = args["query"].as_str().unwrap_or("");
    let limit = args["limit"].as_u64().unwrap_or(10) as usize;
    let results = services
        .database
        .memory()
        .search_fts(query, limit)
        .map_err(|e| format!("memory search failed: {e}"))?;
    serde_json::to_string_pretty(&results).map_err(|e| format!("serialize failed: {e}"))
}

fn execute_memory_create(
    args: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let content = args["content"].as_str().unwrap_or("");
    let metadata = args
        .get("metadata")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let category = args["category"]
        .as_str()
        .or_else(|| metadata.get("category").and_then(|v| v.as_str()))
        .unwrap_or("user_preference");
    let domain = args["domain"]
        .as_str()
        .or_else(|| metadata.get("domain").and_then(|v| v.as_str()));

    tracing::debug!(
        category = category,
        domain = ?domain,
        content_len = content.len(),
        "knowledge_teach(via memory_create): creating knowledge entry"
    );

    let start = std::time::Instant::now();

    // Dedup: check if similar hot knowledge already exists
    if let Some(existing_id) = find_similar_hot_knowledge(services, content) {
        tracing::debug!(
            existing_id = %existing_id,
            "memory_create: skipping duplicate"
        );
        return Ok(serde_json::json!({"id": existing_id, "status": "already_exists"}).to_string());
    }

    let summary = if content.len() > 120 {
        let boundary = content.floor_char_boundary(117);
        format!("{}...", &content[..boundary])
    } else {
        content.to_string()
    };

    // Create knowledge entry directly (same as knowledge_teach path)
    let params = nevoflux_storage::CreateKnowledgeParams {
        category: category.to_string(),
        domain: domain.map(|d| d.to_string()),
        summary: summary.clone(),
        details: content.to_string(),
        source_type: Some("manual".to_string()),
        priority: Some("high".to_string()),
        tags: Some("[\"user_taught\"]".to_string()),
        privacy_level: Some("internal".to_string()),
        ..Default::default()
    };

    let knowledge_repo = nevoflux_storage::KnowledgeRepository::new(&services.database);
    let entry = knowledge_repo
        .create(params)
        .map_err(|e| format!("memory create failed: {e}"))?;
    let id = entry.id.clone();

    knowledge_repo
        .update_status(&id, "validated")
        .map_err(|e| format!("status update failed: {e}"))?;

    let hot_summary = summary;
    knowledge_repo
        .mark_hot(&id, &hot_summary)
        .map_err(|e| format!("mark_hot failed: {e}"))?;

    // Suppress auto-extraction this turn (same as agent_host.rs path)
    if let Some(ref extractor) = services.session_extractor {
        extractor.mark_manual_create();
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        id = %id,
        category = category,
        duration_ms = duration_ms,
        "Knowledge taught and marked hot (via memory_create)"
    );

    Ok(serde_json::json!({"id": id, "status": "created"}).to_string())
}

/// Check if content is similar to an existing hot knowledge entry (cosine > 0.92).
fn find_similar_hot_knowledge(services: &HostServices, content: &str) -> Option<String> {
    use crate::wasm::services::get_embedding;

    let provider = get_embedding(&services.embedding)?;
    let content_owned = content.to_string();

    // Passage: the candidate content is being deduped against existing stored
    // hot knowledge passage embeddings; both sides of the cosine comparison
    // should use the indexing prefix.
    let query_emb = match tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            provider
                .embed_kind(EmbedKind::Passage, &content_owned)
                .await
        })
    }) {
        Ok(emb) => emb,
        Err(_) => return None,
    };

    let knowledge_repo = KnowledgeRepository::new(&services.database);
    let hot_entries = knowledge_repo.list_hot().ok()?;

    for entry in &hot_entries {
        if let Some(ref entry_emb) = entry.embedding {
            let sim = nevoflux_storage::cosine_similarity(&query_emb, entry_emb);
            if sim > 0.92 {
                return Some(entry.id.clone());
            }
        }
    }

    None
}

fn execute_memory_update(
    args: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let id = args["id"]
        .as_str()
        .ok_or_else(|| "missing 'id' argument".to_string())?;
    let content = args["content"]
        .as_str()
        .ok_or_else(|| "missing 'content' argument".to_string())?;

    if id.starts_with("K-") {
        // Knowledge table entry
        let summary = if content.len() > 120 {
            let boundary = content.floor_char_boundary(117);
            format!("{}...", &content[..boundary])
        } else {
            content.to_string()
        };
        KnowledgeRepository::new(&services.database)
            .update_content(id, content, &summary)
            .map_err(|e| format!("knowledge update failed: {e}"))?;
    } else {
        // Legacy memory_chunks table
        services
            .database
            .memory()
            .update(id, content)
            .map_err(|e| format!("memory update failed: {e}"))?;
    }

    Ok(serde_json::json!({"id": id, "status": "updated"}).to_string())
}

fn execute_memory_delete(
    args: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let id = args["id"]
        .as_str()
        .ok_or_else(|| "missing 'id' argument".to_string())?;

    if id.starts_with("K-") {
        KnowledgeRepository::new(&services.database)
            .delete(id)
            .map_err(|e| format!("knowledge delete failed: {e}"))?;
    } else {
        services
            .database
            .memory()
            .delete(id)
            .map_err(|e| format!("memory delete failed: {e}"))?;
    }

    Ok(serde_json::json!({"id": id, "status": "deleted"}).to_string())
}

fn execute_memory_view(
    args: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(20) as usize;
    let knowledge_repo = nevoflux_storage::KnowledgeRepository::new(&services.database);
    let hot_entries = knowledge_repo
        .list_hot()
        .map_err(|e| format!("memory view failed: {e}"))?;
    let entries: Vec<serde_json::Value> = hot_entries
        .into_iter()
        .take(limit)
        .map(|e| {
            serde_json::json!({
                "id": e.id,
                "category": e.category,
                "summary": e.hot_summary.unwrap_or(e.summary),
                "domain": e.domain,
                "created_at": e.created_at,
            })
        })
        .collect();
    serde_json::to_string_pretty(&entries).map_err(|e| format!("serialize failed: {e}"))
}

// ============================================================================
// Knowledge teach
// ============================================================================

fn execute_knowledge_teach(
    args: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let category = args["category"]
        .as_str()
        .unwrap_or("user_preference")
        .to_string();
    let summary = args["summary"].as_str().unwrap_or("").to_string();
    let details = args["details"].as_str().unwrap_or("").to_string();
    let domain = args
        .get("domain")
        .and_then(|v| v.as_str())
        .map(String::from);

    tracing::debug!(
        category = %category,
        summary_len = summary.len(),
        domain = ?domain,
        "knowledge_teach: creating knowledge entry"
    );

    let start = std::time::Instant::now();

    let params = CreateKnowledgeParams {
        category: category.clone(),
        summary: summary.clone(),
        details,
        domain: domain.clone(),
        ..Default::default()
    };

    let repo = KnowledgeRepository::new(&services.database);
    let entry = repo
        .create(params)
        .map_err(|e| format!("knowledge teach failed: {e}"))?;

    let duration_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        id = %entry.id,
        category = %category,
        duration_ms = duration_ms,
        "Knowledge taught (via knowledge_teach)"
    );

    Ok(serde_json::json!({"id": entry.id, "status": "taught"}).to_string())
}

// ============================================================================
// 4'. TTS subsystem (P5b)
// ============================================================================

/// MCP/ACP dispatch arm for `tts_synthesize_api`. Reads `[tts.elevenlabs]`
/// from the `tts_config` plumbed onto `HostServices` at server boot. When
/// `composition_id` is provided in the request, the synthesized MP3 is
/// written into the artifact's files map as `narration.mp3`.
async fn execute_tts_synthesize_api(
    arguments: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let req: nevoflux_protocol::tts::SynthesizeRequest = serde_json::from_value(arguments.clone())
        .map_err(|e| format!("invalid tts_synthesize_api args: {e}"))?;
    let cfg = services
        .tts_config
        .as_ref()
        .map(|c| c.elevenlabs.clone())
        .unwrap_or_default();

    let mut resp = crate::tts::synthesize_api(&cfg, &req)
        .await
        .map_err(|e| e.to_string())?;

    // Optional composition file write (mirrors agent_host's direct-API path).
    if let Some(comp_id) = req.composition_id.as_deref() {
        use nevoflux_storage::repositories::ArtifactRepository;
        let repo = ArtifactRepository::new(&services.database);
        match repo.get(comp_id) {
            Ok(Some(record)) => {
                let mut files = record.files.unwrap_or_default();
                files.insert("narration.mp3".to_string(), resp.audio_b64.clone());
                let entry = record.entry.unwrap_or_else(|| "index.html".to_string());
                let content = files
                    .get(&entry)
                    .cloned()
                    .unwrap_or_else(|| record.content.clone());
                if let Err(e) = repo.update_files(comp_id, &files, &content) {
                    tracing::warn!(
                        "tts_synthesize_api: failed to write narration.mp3 into {}: {}",
                        comp_id,
                        e
                    );
                } else {
                    resp.wrote_to_files = Some("narration.mp3".into());
                }
            }
            Ok(None) => tracing::warn!(
                "tts_synthesize_api: composition_id {} not found; returning audio only",
                comp_id
            ),
            Err(e) => tracing::warn!("tts_synthesize_api: artifact get for {}: {}", comp_id, e),
        }
    }

    serde_json::to_string(&resp).map_err(|e| format!("serialize tts_synthesize_api response: {e}"))
}

/// MCP/ACP dispatch arm for `play_local_file`.
///
/// No permission call of its own: this path is gated before dispatch by
/// `McpBridge::request_permission`, which classifies the name through the same
/// `execution_tier` table the native gate uses. Asking twice would show the
/// reader two dialogs for one act.
fn execute_play_local_file(
    arguments: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let path_arg = arguments
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "play_local_file: `path` is required".to_string())?;
    let (path, mime) = crate::remote::local_media::resolve(path_arg)?;
    let out = crate::remote::local_media::offer(&services.session_id, &path, mime)?;
    serde_json::to_string(&out).map_err(|e| format!("serialize play_local_file response: {e}"))
}

/// MCP/ACP dispatch arm for `tts_synthesize_local` (P5b-2). Reads
/// `[tts.kokoro]` config; until ONNX inference lands the call returns
/// a clear ConfigMissing pointing at setup steps.
async fn execute_tts_synthesize_local(
    arguments: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let req: nevoflux_protocol::tts::SynthesizeRequest = serde_json::from_value(arguments.clone())
        .map_err(|e| format!("invalid tts_synthesize_local args: {e}"))?;
    let cfg = services
        .tts_config
        .as_ref()
        .map(|c| c.kokoro.clone())
        .unwrap_or_default();
    let mut resp = crate::tts::synthesize_local(&cfg, &req, Some(&services.session_id))
        .await
        .map_err(|e| e.to_string())?;

    // Same composition write the API arm does, with Kokoro's extension.
    // Without this the local path cannot feed the video pipeline at all.
    if let Some(comp_id) = req.composition_id.as_deref() {
        use nevoflux_storage::repositories::ArtifactRepository;
        let repo = ArtifactRepository::new(&services.database);
        match repo.get(comp_id) {
            Ok(Some(record)) => {
                let mut files = record.files.unwrap_or_default();
                files.insert("narration.wav".to_string(), resp.audio_b64.clone());
                let entry = record.entry.unwrap_or_else(|| "index.html".to_string());
                let content = files
                    .get(&entry)
                    .cloned()
                    .unwrap_or_else(|| record.content.clone());
                if let Err(e) = repo.update_files(comp_id, &files, &content) {
                    tracing::warn!(
                        "tts_synthesize_local: failed to write narration.wav into {}: {}",
                        comp_id,
                        e
                    );
                } else {
                    resp.wrote_to_files = Some("narration.wav".into());
                }
            }
            Ok(None) => tracing::warn!(
                "tts_synthesize_local: composition_id {} not found; returning audio only",
                comp_id
            ),
            Err(e) => tracing::warn!("tts_synthesize_local: artifact get for {}: {}", comp_id, e),
        }
    }

    let mut out = serde_json::to_value(&resp)
        .map_err(|e| format!("serialize tts_synthesize_local response: {e}"))?;
    crate::tts::strip_delivered_audio(&mut out, req.composition_id.is_some());
    serde_json::to_string(&out).map_err(|e| format!("serialize tts_synthesize_local response: {e}"))
}

/// MCP/ACP dispatch arm for `tts_voices`. Reports what the configured voice
/// bank holds; takes no arguments.
async fn execute_tts_voices(services: &HostServices) -> Result<String, String> {
    let cfg = services
        .tts_config
        .as_ref()
        .map(|c| c.kokoro.clone())
        .unwrap_or_default();
    let voices = crate::tts::list_voices(&cfg)
        .await
        .map_err(|e| e.to_string())?;
    serde_json::to_string(&serde_json::json!({ "voices": voices }))
        .map_err(|e| format!("serialize tts_voices response: {e}"))
}

/// MCP/ACP dispatch arm for `tts_transcribe`. Routing, audio decode and the
/// engines themselves all live in `crate::tts::asr`; this arm only supplies
/// the config and the database the artifact path needs.
async fn execute_tts_transcribe(
    arguments: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let req: nevoflux_protocol::tts::TranscribeRequest = serde_json::from_value(arguments.clone())
        .map_err(|e| format!("invalid tts_transcribe args: {e}"))?;
    let cfg = services.tts_config.clone().unwrap_or_default();
    let resp = crate::tts::transcribe(&cfg, &req, Some(&services.database))
        .await
        .map_err(|e| e.to_string())?;
    serde_json::to_string(&resp).map_err(|e| format!("serialize tts_transcribe response: {e}"))
}

// ============================================================================
// 5. Skill load & tool search
// ============================================================================

async fn execute_skill_load(
    args: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let name = args["name"]
        .as_str()
        .ok_or_else(|| "missing 'name' argument".to_string())?;
    // Lazily rescan once on a miss so a skill created this session (e.g. by
    // skill-creator) is loadable without a daemon restart.
    services.ensure_skill_or_reload(name).await;
    let skills = services.skills.read().await;
    match skills.get(name) {
        Some(skill) => {
            let mut result = String::new();

            // Inject base_path instruction so Claude Code can read auxiliary files
            if let Some(ref file_path) = skill.file_path {
                if let Some(base_dir) = file_path.parent() {
                    let base_path = base_dir.display();
                    // List available files in skill directory
                    let available_files: Vec<String> = std::fs::read_dir(base_dir)
                        .ok()
                        .map(|entries| {
                            entries
                                .filter_map(|e| e.ok())
                                .filter(|e| e.path().is_file() && e.file_name() != "SKILL.md")
                                .map(|e| e.file_name().to_string_lossy().to_string())
                                .collect()
                        })
                        .unwrap_or_default();

                    result.push_str(&format!(
                        "[Skill directory: {base_path}]\n\
                         [To read any file referenced below, use the Read tool with full path: {base_path}/<filename>]\n"
                    ));
                    if !available_files.is_empty() {
                        result.push_str("[Available files: ");
                        result.push_str(&available_files.join(", "));
                        result.push_str("]\n");
                    }
                    result.push('\n');
                }
            }

            result.push_str(&skill.content);
            Ok(result)
        }
        None => Err(format!("skill not found: {name}")),
    }
}

async fn execute_tool_search(
    args: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let query = args["query"].as_str().unwrap_or("");
    let max_results = args["max_results"].as_u64().unwrap_or(5) as usize;

    let index = services
        .tool_search
        .as_ref()
        .ok_or_else(|| "tool search not available".to_string())?;
    let index = index.read().await;
    let results = index.search(query);

    // Limit to max_results and serialize tool definitions
    let tools: Vec<&nevoflux_mcp::ToolDefinition> =
        results.iter().take(max_results).map(|r| &r.tool).collect();
    serde_json::to_string_pretty(&tools).map_err(|e| format!("serialize failed: {e}"))
}

// ============================================================================
// 6. External MCP tools
// ============================================================================

async fn execute_mcp_manager_tool(
    name: &str,
    arguments: &serde_json::Value,
    mcp_manager: &nevoflux_mcp::McpManager,
) -> Result<String, String> {
    let tool_result = mcp_manager
        .call_tool_any(name, arguments.clone())
        .await
        .map_err(|e| format!("MCP tool call failed: {e}"))?;

    // Concatenate text content from the ToolResult
    let mut text_parts = Vec::new();
    for content in &tool_result.content {
        match content {
            nevoflux_mcp::ToolResultContent::Text { text } => {
                text_parts.push(text.clone());
            }
            _ => {
                // Serialize non-text content as JSON
                if let Ok(json) = serde_json::to_string(content) {
                    text_parts.push(json);
                }
            }
        }
    }

    if tool_result.is_error {
        Err(text_parts.join("\n"))
    } else {
        Ok(text_parts.join("\n"))
    }
}

// ============================================================================
// 6. Subagent tools
// ============================================================================

async fn execute_subagent_tool(
    name: &str,
    arguments: &serde_json::Value,
    services: &HostServices,
) -> Result<String, String> {
    let executor = services
        .subagent_executor
        .as_ref()
        .ok_or_else(|| "subagent executor not available".to_string())?;

    match name {
        "subagent_spawn" => {
            let task = arguments["task"]
                .as_str()
                .or_else(|| arguments.as_str()) // task may be the entire argument
                .unwrap_or("")
                .to_string();
            let mode = arguments
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("agent");
            let tab_id = arguments.get("tab_id").and_then(|v| v.as_i64());
            let provider_override = arguments
                .get("provider")
                .and_then(|v| v.as_str())
                .map(String::from);
            // If provider specified but model not, look up from config.toml
            let model_override = arguments
                .get("model")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| {
                    let provider = provider_override.as_ref()?;
                    let agent_config = executor.agent_config()?;
                    agent_config
                        .llm
                        .model_for_provider(provider)
                        .map(String::from)
                });

            // In ACP bridge mode (ClaudeCode/GeminiCli), subagents cannot use the
            // ACP provider directly. Require explicit provider/model specification.
            // If provider given but no model found in config, also ask user.
            if provider_override.is_none()
                || (provider_override.is_some() && model_override.is_none())
            {
                let is_acp_mode = services.llm_config.as_ref().map_or(false, |c| {
                    matches!(
                        c.provider,
                        nevoflux_llm::ProviderType::ClaudeCode
                            | nevoflux_llm::ProviderType::GeminiCli
                            | nevoflux_llm::ProviderType::OpenClaw
                            | nevoflux_llm::ProviderType::Antigravity
                    )
                });
                if is_acp_mode {
                    return Err(
                        "subagent_spawn requires 'provider' and 'model' parameters. \
                         IMPORTANT: Do NOT choose a provider/model yourself. \
                         You MUST ask the user which provider and model to use for this subagent. \
                         Tell the user that subagent needs a direct API provider (not the current ACP provider), \
                         and ask them to specify both provider and model."
                            .to_string(),
                    );
                }
            }

            let agent_mode = match mode {
                "chat" => nevoflux_builtin_wasm::AgentMode::Chat,
                "browser" => nevoflux_builtin_wasm::AgentMode::Browser,
                _ => nevoflux_builtin_wasm::AgentMode::Agent,
            };

            let handle = executor
                .spawn(
                    task,
                    agent_mode,
                    None,
                    tab_id,
                    None,
                    provider_override,
                    model_override,
                )
                .map_err(|e| format!("subagent spawn failed: {e}"))?;

            let id = handle.id;
            Ok(serde_json::json!({"id": id, "status": "spawned"}).to_string())
        }
        "subagent_status" => {
            let id = arguments["id"].as_u64().unwrap_or(0);
            match executor.status(id) {
                Some(status) => Ok(serde_json::json!({
                    "id": id,
                    "status": status.as_str(),
                })
                .to_string()),
                None => Err(format!("subagent {id} not found")),
            }
        }
        "subagent_wait" => {
            let id = arguments["id"].as_u64().unwrap_or(0);
            executor
                .wait(id)
                .await
                .map_err(|e| format!("wait failed: {e}"))
        }
        "subagent_wait_all" => {
            let ids: Vec<u64> = arguments["ids"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                .unwrap_or_default();
            let results = executor.wait_all(&ids).await;
            Ok(serde_json::to_string(&results).unwrap_or_default())
        }
        "subagent_kill" => {
            let id = arguments["id"].as_u64().unwrap_or(0);
            match executor.get(id) {
                Some(handle) => {
                    let killed = handle.kill();
                    Ok(serde_json::json!({"id": id, "killed": killed}).to_string())
                }
                None => Err(format!("subagent {id} not found")),
            }
        }
        "subagent_list" => {
            // List running subagent count — detailed listing not available via public API
            let count = executor.running_count();
            Ok(serde_json::json!({"running_count": count}).to_string())
        }
        _ => Err(format!("unknown subagent tool: {name}")),
    }
}

// ============================================================================
// canvas_attach_asset payload resolution
// ============================================================================

/// Resolved asset payload ready for `CanvasVideoService::attach_asset`.
#[derive(Debug)]
pub struct ResolvedAsset {
    pub name: String,
    pub mime_type: String,
    pub payload_b64: String,
    pub size_bytes: u64,
}

/// Public alias of `resolve_attach_asset_payload` for the direct-API
/// path in `agent_host.rs`. The MCP/ACP path calls the private name
/// directly.
pub async fn resolve_attach_asset_payload_pub(
    req: &nevoflux_protocol::canvas_video::AttachAssetRequest,
) -> Result<ResolvedAsset, String> {
    resolve_attach_asset_payload(req).await
}

/// Convert a `canvas_attach_asset` request's source variant (one of
/// `data_b64` / `url` / `local_path` / `from_tab`) into a normalized
/// `(name, mime_type, base64-payload, size_bytes)` tuple.
async fn resolve_attach_asset_payload(
    req: &nevoflux_protocol::canvas_video::AttachAssetRequest,
) -> Result<ResolvedAsset, String> {
    let chosen: i32 = [
        req.data_b64.is_some(),
        req.url.is_some(),
        req.local_path.is_some(),
        req.from_tab.is_some(),
    ]
    .iter()
    .filter(|x| **x)
    .count() as i32;
    if chosen == 0 {
        return Err("must provide one of data_b64 / url / local_path / from_tab".into());
    }
    if chosen > 1 {
        return Err(
            "only one of data_b64 / url / local_path / from_tab may be set (mutually exclusive)"
                .into(),
        );
    }

    if let Some(b64) = req.data_b64.as_deref() {
        let bytes =
            decode_base64_strict(b64).map_err(|e| format!("data_b64 not valid base64: {e}"))?;
        let mime = req
            .mime_type
            .clone()
            .or_else(|| req.name.as_deref().map(infer_mime_from_name))
            .unwrap_or_else(|| "application/octet-stream".into());
        let name = pick_name(req.name.as_deref(), &mime, "asset");
        return Ok(ResolvedAsset {
            name,
            mime_type: mime,
            payload_b64: b64.to_string(),
            size_bytes: bytes.len() as u64,
        });
    }

    if let Some(url) = req.url.as_deref() {
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err("url must be http:// or https:// (file:/data: rejected)".into());
        }
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| format!("reqwest client init: {e}"))?;
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("fetch failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("fetch returned {}: {url}", resp.status()));
        }
        let mime = req
            .mime_type
            .clone()
            .or_else(|| {
                resp.headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.split(';').next().unwrap_or(s).trim().to_string())
            })
            .or_else(|| Some(infer_mime_from_url(url)))
            .unwrap_or_else(|| "application/octet-stream".into());
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("read body: {e}"))?
            .to_vec();
        let size = bytes.len() as u64;
        let payload_b64 = encode_base64(&bytes);
        let name_from_url = url_basename(url);
        let name = pick_name(
            req.name.as_deref().or(name_from_url.as_deref()),
            &mime,
            "asset",
        );
        return Ok(ResolvedAsset {
            name,
            mime_type: mime,
            payload_b64,
            size_bytes: size,
        });
    }

    if let Some(path_str) = req.local_path.as_deref() {
        if path_str.trim().is_empty() {
            return Err("local_path is empty".into());
        }
        let path = std::path::Path::new(path_str);
        // Reject path traversal / non-absolute paths to keep the agent
        // honest about what it's attaching. The promotion path passes
        // absolute paths from the sidebar's local file picker, so this
        // only blocks accidental relative-path leakage.
        if !path.is_absolute() {
            return Err(format!(
                "local_path must be absolute (got {path_str:?}); the agent receives absolute paths from the user's local_files context"
            ));
        }
        let bytes = std::fs::read(path)
            .map_err(|e| format!("failed to read local_path {path_str:?}: {e}"))?;
        if bytes.is_empty() {
            return Err(format!("local_path {path_str:?} is empty"));
        }
        // Same MIME inference policy as the URL path: caller-provided
        // overrides everything; otherwise infer from path extension.
        // Magic-byte sniffing happens later in asset_inline at render
        // time, so a misnamed extension still renders correctly.
        let mime = req
            .mime_type
            .clone()
            .unwrap_or_else(|| infer_mime_from_name(path_str));
        let size = bytes.len() as u64;
        let payload_b64 = encode_base64(&bytes);
        let name_from_path = path.file_name().map(|n| n.to_string_lossy().to_string());
        let name = pick_name(
            req.name.as_deref().or(name_from_path.as_deref()),
            &mime,
            "asset",
        );
        return Ok(ResolvedAsset {
            name,
            mime_type: mime,
            payload_b64,
            size_bytes: size,
        });
    }

    if req.from_tab.is_some() {
        return Err(
            "from_tab is not yet wired (browser-tool screenshot bridge required); use data_b64 with the screenshot bytes instead"
                .into(),
        );
    }

    Err("no source supplied".into())
}

fn pick_name(provided: Option<&str>, mime: &str, fallback_stem: &str) -> String {
    if let Some(n) = provided {
        if !n.trim().is_empty() {
            return n.to_string();
        }
    }
    let ext = match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/svg+xml" => "svg",
        "image/avif" => "avif",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "audio/mpeg" => "mp3",
        "audio/wav" => "wav",
        "audio/ogg" => "ogg",
        "font/woff2" => "woff2",
        "font/woff" => "woff",
        "font/ttf" => "ttf",
        _ => "bin",
    };
    format!("{fallback_stem}.{ext}")
}

fn url_basename(url: &str) -> Option<String> {
    let stripped = url
        .split('?')
        .next()
        .unwrap_or(url)
        .split('#')
        .next()
        .unwrap_or(url);
    stripped
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty() && s.contains('.'))
        .map(|s| s.to_string())
}

fn infer_mime_from_name(name: &str) -> String {
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "avif" => "image/avif",
        "mp4" | "m4v" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        _ => "application/octet-stream",
    }
    .to_string()
}

fn infer_mime_from_url(url: &str) -> String {
    url_basename(url)
        .map(|n| infer_mime_from_name(&n))
        .unwrap_or_else(|| "application/octet-stream".into())
}

/// Validate base64 input and return decoded bytes (we don't actually
/// keep the bytes; we just need a length + format check). Standard
/// alphabet only; rejects whitespace and url-safe variants.
fn decode_base64_strict(s: &str) -> Result<Vec<u8>, String> {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    STANDARD.decode(s.as_bytes()).map_err(|e| e.to_string())
}

fn encode_base64(bytes: &[u8]) -> String {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    STANDARD.encode(bytes)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_path_reads_disk_bytes_into_resolved_asset() {
        // Real PNG header bytes — 12 bytes is enough for the magic
        // sniffer; resolver doesn't decode here, just stores.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hero.png");
        let bytes: Vec<u8> = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        ];
        std::fs::write(&path, &bytes).unwrap();

        let req = nevoflux_protocol::canvas_video::AttachAssetRequest {
            composition_id: "comp-x".into(),
            name: None,
            mime_type: None,
            data_b64: None,
            url: None,
            local_path: Some(path.to_string_lossy().to_string()),
            from_tab: None,
            role: None,
        };

        let resolved = resolve_attach_asset_payload(&req)
            .await
            .expect("resolve ok");
        assert_eq!(resolved.size_bytes, bytes.len() as u64);
        assert_eq!(resolved.mime_type, "image/png");
        assert_eq!(resolved.name, "hero.png");
        // Round-trip the base64.
        use base64::{engine::general_purpose::STANDARD, Engine};
        let decoded = STANDARD.decode(&resolved.payload_b64).unwrap();
        assert_eq!(decoded, bytes);
        let _ = &dir;
    }

    #[tokio::test]
    async fn local_path_rejects_relative() {
        let req = nevoflux_protocol::canvas_video::AttachAssetRequest {
            composition_id: "comp-x".into(),
            name: None,
            mime_type: None,
            data_b64: None,
            url: None,
            local_path: Some("relative/path.png".into()),
            from_tab: None,
            role: None,
        };
        let err = resolve_attach_asset_payload(&req).await.unwrap_err();
        assert!(err.contains("absolute"), "got: {err}");
    }

    #[tokio::test]
    async fn local_path_rejects_missing_file() {
        let req = nevoflux_protocol::canvas_video::AttachAssetRequest {
            composition_id: "comp-x".into(),
            name: None,
            mime_type: None,
            data_b64: None,
            url: None,
            local_path: Some("/nonexistent/path/foo.png".into()),
            from_tab: None,
            role: None,
        };
        let err = resolve_attach_asset_payload(&req).await.unwrap_err();
        assert!(err.contains("failed to read"), "got: {err}");
    }

    #[tokio::test]
    async fn rejects_multiple_sources() {
        let req = nevoflux_protocol::canvas_video::AttachAssetRequest {
            composition_id: "comp-x".into(),
            name: None,
            mime_type: None,
            data_b64: Some("AAAA".into()),
            url: None,
            local_path: Some("/tmp/x.png".into()),
            from_tab: None,
            role: None,
        };
        let err = resolve_attach_asset_payload(&req).await.unwrap_err();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    #[tokio::test]
    async fn rejects_no_source() {
        let req = nevoflux_protocol::canvas_video::AttachAssetRequest {
            composition_id: "comp-x".into(),
            name: None,
            mime_type: None,
            data_b64: None,
            url: None,
            local_path: None,
            from_tab: None,
            role: None,
        };
        let err = resolve_attach_asset_payload(&req).await.unwrap_err();
        assert!(err.contains("must provide one of"), "got: {err}");
    }

    /// `orchestrate` must be dispatchable on the ACP path, not just advertised.
    ///
    /// The built-in agent intercepts it inside its own loop, so for a long time
    /// an ACP agent could find it via `tool_search` and then get
    /// "unknown tool: orchestrate" on calling it — the search result made it
    /// look available, which is worse than absent.
    #[tokio::test(flavor = "multi_thread")]
    async fn orchestrate_runs_on_the_acp_path() {
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = Arc::new(McpToolBridge::new());

        let out = execute_mcp_tool(
            "orchestrate",
            &serde_json::json!({"code": "print(6 * 7)"}),
            &services,
            &bridge,
        )
        .await
        .expect("orchestrate must execute, not report an unknown tool");
        assert!(out.contains("42"), "expected the code's output, got: {out}");

        // An empty body is a caller error with a name, not a silent success.
        let err = execute_mcp_tool(
            "orchestrate",
            &serde_json::json!({"code": "   "}),
            &services,
            &bridge,
        )
        .await
        .unwrap_err();
        assert!(
            err.contains("code"),
            "the error should name the argument: {err}"
        );
    }

    #[test]
    /// Every browser-family tool the agent is offered must resolve to an
    /// action. The dispatcher matches on the *suffix* after stripping
    /// `browser_`/`canvas_`, so a name can look absent to a plain search of
    /// this file and still work — and one can look present and not. Both
    /// mistakes were made while auditing this list, so it is asserted from the
    /// agent-facing names rather than the suffixes.
    #[test]
    fn advertised_browser_names_all_resolve() {
        for name in [
            "browser_screenshot",
            "browser_get_markdown",
            "browser_get_content",
            "browser_get_tabs",
            "browser_wait_for",
            "browser_scroll",
            "canvas_extract_visual_identity",
            // Advertised as `web_fetch`; the dispatch was written against
            // `fetch_page`, so only the alias makes the advertised name work.
            "web_fetch",
            "fetch_page",
            "web_search",
        ] {
            assert!(
                tool_name_to_browser_action(name).is_some(),
                "{name} is advertised to the agent but resolves to no action"
            );
        }
    }

    #[test]
    fn a_screenshot_result_carries_an_id_not_a_transcript() {
        // The extension answers under `data`, and the vision lift in wasm::llm
        // looks for `screenshot` — so the bytes used to reach the model as
        // text and come back re-typed with a chunk missing. Re-keying is what
        // makes the lift fire; the id is what the model writes instead.
        let raw = serde_json::json!({ "success": true, "data": "aGVsbG8=" });
        let out = screenshot_result(&Some(raw), "no-portal-attached");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["screenshot"], "aGVsbG8=");
        assert!(
            v.get("data").is_none(),
            "the old key must not survive: {out}"
        );
    }

    #[test]
    fn a_screenshot_payload_is_found_under_any_key_the_extension_has_used() {
        for key in ["screenshot", "data", "data_url", "image", "base64"] {
            let v = serde_json::json!({ key: "QUJD" });
            assert_eq!(
                screenshot_payload(&Some(v)).as_deref(),
                Some("QUJD"),
                "{key}"
            );
        }
        assert_eq!(
            screenshot_payload(&Some(serde_json::json!("QUJD"))).as_deref(),
            Some("QUJD")
        );
        assert!(screenshot_payload(&Some(serde_json::json!({ "data": "" }))).is_none());
        assert!(screenshot_payload(&None).is_none());
    }

    #[test]
    fn a_response_with_no_picture_in_it_is_passed_through() {
        let raw = serde_json::json!({ "success": true, "note": "nothing here" });
        let out = screenshot_result(&Some(raw.clone()), "s");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&out).unwrap(),
            raw
        );
    }

    #[test]
    fn nothing_is_stored_when_no_portal_is_watching() {
        // Nothing to serve it to, so no id and no disk spent — and the turn
        // carries on with the picture, just without a reference.
        assert!(store_screenshot("aGVsbG8=", "nobody-here").is_none());
    }

    #[test]
    fn test_tool_name_to_browser_action() {
        assert_eq!(
            tool_name_to_browser_action("browser_navigate"),
            Some(BrowserToolAction::Navigate)
        );
        assert_eq!(
            tool_name_to_browser_action("browser_snapshot"),
            Some(BrowserToolAction::Snapshot)
        );
        assert_eq!(
            tool_name_to_browser_action("web_search"),
            Some(BrowserToolAction::WebSearch)
        );
        assert_eq!(tool_name_to_browser_action("unknown_tool"), None);
    }

    #[test]
    fn test_tool_name_to_browser_action_bare_names() {
        assert_eq!(
            tool_name_to_browser_action("navigate"),
            Some(BrowserToolAction::Navigate)
        );
        assert_eq!(
            tool_name_to_browser_action("screenshot"),
            Some(BrowserToolAction::Screenshot)
        );
        assert_eq!(
            tool_name_to_browser_action("get_tabs"),
            Some(BrowserToolAction::ListTabs)
        );
        assert_eq!(
            tool_name_to_browser_action("list_tabs"),
            Some(BrowserToolAction::ListTabs)
        );
    }

    #[test]
    fn test_tool_name_to_browser_action_all_variants() {
        let cases = [
            ("browser_go_back", BrowserToolAction::GoBack),
            ("browser_go_forward", BrowserToolAction::GoForward),
            ("browser_click", BrowserToolAction::Click),
            ("browser_click_by_id", BrowserToolAction::ClickById),
            ("browser_type", BrowserToolAction::Type),
            ("browser_type_by_id", BrowserToolAction::TypeById),
            ("browser_fill", BrowserToolAction::Fill),
            ("browser_fill_by_id", BrowserToolAction::FillById),
            ("browser_get_content", BrowserToolAction::GetContent),
            ("browser_get_markdown", BrowserToolAction::GetMarkdown),
            ("browser_eval_js", BrowserToolAction::EvalJs),
            ("browser_wait_for", BrowserToolAction::WaitFor),
            ("browser_wait_for_stable", BrowserToolAction::WaitForStable),
            ("browser_scroll", BrowserToolAction::Scroll),
            ("browser_get_element", BrowserToolAction::GetElement),
            ("browser_get_elements", BrowserToolAction::GetElements),
            ("browser_query_all", BrowserToolAction::QueryAll),
            ("browser_get_tabs", BrowserToolAction::ListTabs),
            ("browser_query_tabs", BrowserToolAction::QueryTabs),
            ("browser_activate_tab", BrowserToolAction::ActivateTab),
            ("browser_key_press", BrowserToolAction::KeyPress),
            ("browser_read_artifact", BrowserToolAction::ReadArtifact),
            ("browser_edit_artifact", BrowserToolAction::EditArtifact),
            ("browser_ask_user", BrowserToolAction::AskUser),
            ("fetch_page", BrowserToolAction::WebFetch),
            ("browser_upload_file", BrowserToolAction::UploadFile),
        ];
        for (name, expected) in cases {
            assert_eq!(
                tool_name_to_browser_action(name),
                Some(expected),
                "failed for {name}"
            );
        }
    }

    #[test]
    fn test_parse_mouse_button() {
        assert_eq!(
            parse_mouse_button(&serde_json::json!({})),
            MouseButton::Left
        );
        assert_eq!(
            parse_mouse_button(&serde_json::json!({"button": "right"})),
            MouseButton::Right
        );
        assert_eq!(
            parse_mouse_button(&serde_json::json!({"button": "middle"})),
            MouseButton::Middle
        );
    }

    #[test]
    fn test_parse_click_type() {
        assert_eq!(parse_click_type(&serde_json::json!({})), ClickType::Single);
        assert_eq!(
            parse_click_type(&serde_json::json!({"click_type": "double"})),
            ClickType::Double
        );
        assert_eq!(
            parse_click_type(&serde_json::json!({"click_type": "triple"})),
            ClickType::Triple
        );
    }

    #[test]
    fn test_parse_key_string() {
        assert!(matches!(
            parse_key_string("enter"),
            Ok(KeyOrChar::Key(Key::Enter))
        ));
        assert!(matches!(
            parse_key_string("escape"),
            Ok(KeyOrChar::Key(Key::Escape))
        ));
        assert!(matches!(parse_key_string("a"), Ok(KeyOrChar::Char('a'))));
        assert!(parse_key_string("unknown_key").is_err());
    }

    fn make_test_bridge() -> Arc<McpToolBridge> {
        Arc::new(McpToolBridge::new())
    }

    // ------------------------------------------------------------------
    // record_acp_tool_result — ACP-bridge tool-result recording (fixes the
    // gap where /loop verify and /goal programmatic checks always failed
    // for claude-code / kimi-agent / etc. sessions, since their tool calls
    // never wrote a `tool_result` row to `messages`).
    // ------------------------------------------------------------------

    /// Seed a `sessions` row so `messages.session_id`'s FK constraint is
    /// satisfied — `messages` REFERENCES `sessions(id)` (see
    /// `crates/storage/src/migrations/001_initial.sql`), so a bare insert
    /// against a session-less in-memory DB fails FK enforcement (silently,
    /// since `record_acp_tool_result` is best-effort and only warns).
    fn seed_session(db: &nevoflux_storage::Database, session_id: &str) {
        nevoflux_storage::repositories::SessionRepository::new(db)
            .create(nevoflux_storage::models::CreateSessionParams::new().with_id(session_id))
            .unwrap();
    }

    #[test]
    fn record_acp_tool_result_writes_tool_result_row() {
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        seed_session(&db, "sess-1");
        let services = HostServices::new(db.clone()).with_session_id("sess-1".to_string());

        record_acp_tool_result(&services, "bash", "exit 0: OK");

        let rows = nevoflux_storage::repositories::MessageRepository::new(&db)
            .list_recent_tool_results("sess-1", 10)
            .unwrap();
        assert_eq!(rows, vec![("bash".to_string(), "exit 0: OK".to_string())]);
    }

    #[test]
    fn record_acp_tool_result_skips_think_tool() {
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        seed_session(&db, "sess-1");
        let services = HostServices::new(db.clone()).with_session_id("sess-1".to_string());

        record_acp_tool_result(&services, "think", "some internal reasoning");

        let rows = nevoflux_storage::repositories::MessageRepository::new(&db)
            .list_recent_tool_results("sess-1", 10)
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn record_acp_tool_result_skips_empty_content() {
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        seed_session(&db, "sess-1");
        let services = HostServices::new(db.clone()).with_session_id("sess-1".to_string());

        record_acp_tool_result(&services, "bash", "   ");

        let rows = nevoflux_storage::repositories::MessageRepository::new(&db)
            .list_recent_tool_results("sess-1", 10)
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn record_acp_tool_result_skips_empty_session_id() {
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        // session_id left at HostServices::new's default: "".
        let services = HostServices::new(db.clone());

        record_acp_tool_result(&services, "bash", "exit 0: OK");

        // Nothing should have been written under the empty session_id either.
        let rows = nevoflux_storage::repositories::MessageRepository::new(&db)
            .list_recent_tool_results("", 10)
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn execute_mcp_tool_wrapper_records_result_for_acp_bridge() {
        // End-to-end: the public `execute_mcp_tool` entry point (the only
        // dispatch surface ACP-bridge providers use) must record a
        // `tool_result` row on success, without any caller opting in.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        seed_session(&db, "sess-wrap");
        let services = HostServices::new(db.clone()).with_session_id("sess-wrap".to_string());
        let bridge = make_test_bridge();

        let result = rt.block_on(execute_mcp_tool(
            "create_plan",
            &serde_json::json!({}),
            &services,
            &bridge,
        ));
        assert_eq!(result, Ok("Plan submitted for review.".to_string()));

        let rows = nevoflux_storage::repositories::MessageRepository::new(&db)
            .list_recent_tool_results("sess-wrap", 10)
            .unwrap();
        assert_eq!(
            rows,
            vec![(
                "create_plan".to_string(),
                "Plan submitted for review.".to_string()
            )]
        );
    }

    #[test]
    fn execute_mcp_tool_wrapper_skips_think_tool() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        seed_session(&db, "sess-think");
        let services = HostServices::new(db.clone()).with_session_id("sess-think".to_string());
        let bridge = make_test_bridge();

        let _ = rt.block_on(execute_mcp_tool(
            "think",
            &serde_json::json!({}),
            &services,
            &bridge,
        ));

        let rows = nevoflux_storage::repositories::MessageRepository::new(&db)
            .list_recent_tool_results("sess-think", 10)
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn execute_mcp_tool_wrapper_does_not_record_on_error() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        seed_session(&db, "sess-err");
        let services = HostServices::new(db.clone()).with_session_id("sess-err".to_string());
        let bridge = make_test_bridge();

        let result = rt.block_on(execute_mcp_tool(
            "nonexistent_tool",
            &serde_json::json!({}),
            &services,
            &bridge,
        ));
        assert!(result.is_err());

        let rows = nevoflux_storage::repositories::MessageRepository::new(&db)
            .list_recent_tool_results("sess-err", 10)
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_dispatch_think_tool() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = make_test_bridge();
        let result = rt.block_on(execute_mcp_tool(
            "think",
            &serde_json::json!({}),
            &services,
            &bridge,
        ));
        assert_eq!(result, Ok("Thought recorded.".to_string()));
    }

    #[test]
    fn test_dispatch_create_plan() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = make_test_bridge();
        let result = rt.block_on(execute_mcp_tool(
            "create_plan",
            &serde_json::json!({}),
            &services,
            &bridge,
        ));
        assert_eq!(result, Ok("Plan submitted for review.".to_string()));
    }

    #[test]
    fn test_dispatch_unknown_tool() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = make_test_bridge();
        let result = rt.block_on(execute_mcp_tool(
            "nonexistent_tool",
            &serde_json::json!({}),
            &services,
            &bridge,
        ));
        assert_eq!(result, Err("unknown tool: nonexistent_tool".to_string()));
    }

    #[test]
    fn test_tool_call_dynamic_requires_tool_name() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = make_test_bridge();
        let result = rt.block_on(execute_mcp_tool(
            "tool_call_dynamic",
            &serde_json::json!({ "arguments": "{}" }),
            &services,
            &bridge,
        ));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing required 'tool_name'"));
    }

    #[test]
    fn test_tool_call_dynamic_self_call_rejected() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = make_test_bridge();
        let result = rt.block_on(execute_mcp_tool(
            "tool_call_dynamic",
            &serde_json::json!({ "tool_name": "tool_call_dynamic", "arguments": "{}" }),
            &services,
            &bridge,
        ));
        assert_eq!(
            result,
            Err("tool_call_dynamic cannot call itself".to_string())
        );
    }

    #[test]
    fn test_tool_call_dynamic_unwraps_and_dispatches_inner() {
        // Proves the ACP-bridge fix: tool_call_dynamic re-dispatches the INNER
        // tool through this executor — the failure surfaces the inner name, not
        // "unknown tool: tool_call_dynamic".
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = make_test_bridge();
        let result = rt.block_on(execute_mcp_tool(
            "tool_call_dynamic",
            &serde_json::json!({ "tool_name": "nonexistent_tool", "arguments": "{}" }),
            &services,
            &bridge,
        ));
        assert_eq!(result, Err("unknown tool: nonexistent_tool".to_string()));
    }

    #[test]
    fn test_tool_call_dynamic_parses_json_string_args_and_runs_inner() {
        // `arguments` is a JSON *string* per the tool schema; it must be parsed
        // and the inner tool ("think") executed successfully.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = make_test_bridge();
        let result = rt.block_on(execute_mcp_tool(
            "tool_call_dynamic",
            &serde_json::json!({ "tool_name": "think", "arguments": "{}" }),
            &services,
            &bridge,
        ));
        assert_eq!(result, Ok("Thought recorded.".to_string()));
    }

    #[test]
    fn test_memory_roundtrip() {
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);

        // Create (now writes to knowledge table as hot entry)
        let result = execute_memory_create(
            &serde_json::json!({"content": "test memory content for roundtrip"}),
            &services,
        );
        assert!(result.is_ok(), "create failed: {:?}", result);
        let created: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        let id = created["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("K-"), "Expected knowledge ID, got: {}", id);

        // View (should find the hot entry)
        let result = execute_memory_view(&serde_json::json!({"limit": 10}), &services);
        assert!(result.is_ok());
        let viewed: Vec<serde_json::Value> = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(
            !viewed.is_empty(),
            "memory_view should return the created entry"
        );
    }

    #[test]
    fn test_memory_search_and_update_on_chunks() {
        // memory_search/update/delete still operate on memory_chunks table
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db.clone());

        // Create a chunk directly in memory_chunks (not via memory_create which now uses knowledge)
        let chunk = nevoflux_storage::MemoryChunk::new("searchable chunk content");
        let chunk_id = chunk.id.clone();
        nevoflux_storage::MemoryRepository::new(&db)
            .create(&chunk)
            .unwrap();

        // Search
        let result = execute_memory_search(
            &serde_json::json!({"query": "searchable", "limit": 5}),
            &services,
        );
        assert!(result.is_ok());
        let found: Vec<serde_json::Value> = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(!found.is_empty());

        // Update
        let result = execute_memory_update(
            &serde_json::json!({"id": chunk_id, "content": "updated content"}),
            &services,
        );
        assert!(result.is_ok());

        // Delete
        let result = execute_memory_delete(&serde_json::json!({"id": chunk_id}), &services);
        assert!(result.is_ok());
    }

    #[test]
    fn test_knowledge_teach() {
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);

        let result = execute_knowledge_teach(
            &serde_json::json!({
                "category": "user_preference",
                "summary": "User prefers dark mode",
                "details": "The user always wants dark theme in all applications"
            }),
            &services,
        );
        assert!(result.is_ok(), "knowledge teach failed: {:?}", result);
        let parsed: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed["status"], "taught");
        assert!(parsed["id"].as_str().is_some());
    }

    #[test]
    fn test_computer_tool_no_controller() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let result = rt.block_on(execute_computer_tool(
            "computer_screenshot",
            &serde_json::json!({}),
            &services,
        ));
        assert_eq!(result, Err("computer controller not available".to_string()));
    }

    #[test]
    fn test_browser_tool_no_context() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let db = std::sync::Arc::new(nevoflux_storage::Database::open_in_memory().unwrap());
        let services = HostServices::new(db);
        let bridge = make_test_bridge();
        let result = rt.block_on(execute_mcp_tool(
            "browser_navigate",
            &serde_json::json!({"url": "https://example.com"}),
            &services,
            &bridge,
        ));
        assert_eq!(result, Err("browser not available".to_string()));
    }

    #[test]
    fn browser_fill_with_long_chinese_text_does_not_panic() {
        // This is the exact tweet that caused the production panic
        let value = "Anthropic 封禁 OpenClaw 的 API key，连 e2e 测试都不放过。AI 公司一边鼓励开发者构建 agent 生态，一边收紧 API 权限——这种矛盾迟早会反噬。开源和本地化才是出路。";
        let args = serde_json::json!({ "element_id": "e0", "value": value });
        let _desc = describe_tool_action("browser_fill_by_id", &args.to_string());
    }

    #[test]
    fn browser_type_with_long_chinese_text_does_not_panic() {
        let text = "你好世界，这是一段测试用的长中文文本，用来验证字符串截断不会崩溃。".repeat(3);
        let args = serde_json::json!({ "element_id": "e0", "text": text });
        let _desc = describe_tool_action("browser_type_by_id", &args.to_string());
    }

    #[test]
    fn browser_fill_with_emoji_does_not_panic() {
        let value =
            "Hello 👋🌍 from NevoFlux 🚀! This has emojis 🎉🎊 that are 4-byte UTF-8 characters.";
        let args = serde_json::json!({ "element_id": "e0", "value": value });
        let _desc = describe_tool_action("browser_fill_by_id", &args.to_string());
    }

    #[test]
    fn run_command_with_chinese_does_not_panic() {
        let cmd =
            "echo '这是一个很长的中文命令，长度超过八十字节限制，用来触发 truncation 代码路径'";
        let args = serde_json::json!({ "command": cmd });
        let _desc = describe_tool_action("run_command", &args.to_string());
    }

    #[test]
    fn subagent_spawn_with_chinese_task_does_not_panic() {
        let task =
            "请帮我分析这个超长的中文任务描述，看看截断逻辑是否会因为多字节字符而崩溃。".repeat(2);
        let args = serde_json::json!({ "task": task });
        let _desc = describe_tool_action("subagent_spawn", &args.to_string());
    }

    #[test]
    fn computer_type_text_with_chinese_does_not_panic() {
        let text = "键盘输入测试：这段文本超过 50 字节，应该被安全截断。";
        let args = serde_json::json!({ "text": text });
        let _desc = describe_tool_action("computer_type_text", &args.to_string());
    }
}
