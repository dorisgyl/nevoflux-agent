//! "Agent execution" permission tiers and the tool → risk-bucket classifier.
//!
//! The browser exposes a single "Agent execution" setting
//! (`config:settings → general.agentExecution`) with four tiers that cumulatively
//! auto-approve more tool calls. This module is the single source of truth for
//! what each tier auto-approves, consumed by both permission gates:
//!   - native/WASM path: `daemon::agent_host::check_tool_permission`
//!   - ACP path:         `llm::providers::acp::mcp_bridge::request_permission`
//!
//! Design (buckets, least → most privileged; each tier adds the next bucket):
//!   R  — read-only: browser observe/navigate/wait, web read (web_fetch/search)
//!   B1 — browser interaction: click / type / fill / scroll / key-press
//!   L0 — local file read: read_file / list_files / glob / grep
//!   X  — everything else: file write, exec, computer, memory writes, MCP,
//!        plugins, browser eval-js, and anything unrecognized (safe default)

use serde::{Deserialize, Serialize};

/// The four "Agent execution" tiers, ascending privilege.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionTier {
    /// Confirm every change; only reads/observation/navigation/fetch auto-run.
    ReadOnly,
    /// + browser interactions auto-run.
    BrowserAuto,
    /// + local file reads auto-run.
    BrowserAutoLocalRead,
    /// Everything auto-runs (no confirmations).
    FullAuto,
}

impl Default for ExecutionTier {
    fn default() -> Self {
        ExecutionTier::ReadOnly
    }
}

impl ExecutionTier {
    /// Parse the stored setting string. Legacy dead-stub values ('confirm',
    /// 'auto') and anything unrecognized fall back to the safest tier — matching
    /// the browser-side normalizer (agent-execution-tiers.mjs). Critically, the
    /// old 'auto' does NOT map to full-auto (it never took effect; preserving it
    /// would silently grant full permissions).
    pub fn from_setting(value: &str) -> Self {
        match value {
            "read-only" => ExecutionTier::ReadOnly,
            "browser-auto" => ExecutionTier::BrowserAuto,
            "browser-auto-local-read" => ExecutionTier::BrowserAutoLocalRead,
            "full-auto" => ExecutionTier::FullAuto,
            _ => ExecutionTier::ReadOnly,
        }
    }

    /// The canonical setting string — the inverse of [`Self::from_setting`].
    /// Used to report the tier to remote gateways without re-deriving the
    /// mapping (and drifting from it) at the call site.
    pub fn as_setting(self) -> &'static str {
        match self {
            ExecutionTier::ReadOnly => "read-only",
            ExecutionTier::BrowserAuto => "browser-auto",
            ExecutionTier::BrowserAutoLocalRead => "browser-auto-local-read",
            ExecutionTier::FullAuto => "full-auto",
        }
    }
}

/// Risk bucket a tool call falls into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskBucket {
    /// Read-only: browser observe/navigate, web read.
    R,
    /// Browser interaction that changes page/site state.
    B1,
    /// Local file read.
    L0,
    /// Everything with external/local side effects (safe default for unknowns).
    X,
}

/// R bucket — auto-approved even in the most restrictive `read-only` tier.
/// Includes browser observe/navigate/wait tools (as dispatched by
/// `execute_browser_action`, i.e. `browser_<snake_case action>`) plus the
/// standalone web-read and agent-internal tools.
const R_TOOLS: &[&str] = &[
    // Browser observe
    "browser_get_content",
    "browser_get_markdown",
    "browser_screenshot",
    "browser_snapshot",
    "browser_get_element",
    "browser_get_elements",
    "browser_query_all",
    "browser_get_tabs",
    "browser_query_tabs",
    "browser_list_tabs",
    "browser_read_artifact",
    // Browser navigate (design: navigation is read-only)
    "browser_navigate",
    "browser_go_back",
    "browser_go_forward",
    // Browser wait / ask (no side effects)
    "browser_wait_for",
    "browser_wait_for_stable",
    "browser_ask_user",
    // Browser-dispatched web read + standalone web read
    "browser_web_fetch",
    "browser_web_search",
    "web_fetch",
    "web_search",
    "fetch_page",
    // Memory / knowledge read
    "memory_search",
    "memory_view",
    // Agent-internal (no side effects)
    "tool_search",
    "skill_load",
    "think",
    "create_plan",
    // Telling the user something. It changes nothing, touches no page, file or
    // network, and is addressed *to* the person who would be asked — gating it
    // means asking permission to be told something.
    //
    // Unclassified it fell to bucket X, which nothing below full-auto
    // approves. On the ACP path that meant every notify_user waited on a
    // decision that never came: no dialog was raised, nothing was logged, the
    // model retried once and gave up. Scheduled reminders fired and then died
    // there in silence — on the sidebar as well as the phone.
    "notify_user",
];

/// B1 bucket — browser interactions that change page/site state.
const B1_TOOLS: &[&str] = &[
    "browser_click",
    "browser_type",
    "browser_fill",
    "browser_click_by_id",
    "browser_type_by_id",
    "browser_fill_by_id",
    "browser_scroll",
    "browser_key_press",
];

/// L0 bucket — local filesystem reads.
const L0_TOOLS: &[&str] = &["read_file", "list_files", "glob", "grep"];

/// Classify a tool call into its risk bucket. An `mcp__server__tool` wrapper
/// prefix is stripped first (mirroring `is_read_only_tool`) so ACP-wrapped
/// read tools resolve; anything not explicitly R/B1/L0 is treated as X (the
/// safe default — includes write_file, run_command, computer_*, memory writes,
/// subagent_spawn, browser_eval_js, and every unknown/MCP tool).
pub fn classify_tool(name: &str) -> RiskBucket {
    let bare = name.rsplit("__").next().unwrap_or(name);
    if R_TOOLS.contains(&bare) {
        RiskBucket::R
    } else if B1_TOOLS.contains(&bare) {
        RiskBucket::B1
    } else if L0_TOOLS.contains(&bare) {
        RiskBucket::L0
    } else {
        RiskBucket::X
    }
}

/// True if `tier` auto-approves `tool_name` (skip the confirmation dialog).
/// Tiers are cumulative: each approves its own bucket plus all lower ones.
pub fn tier_auto_approves(tool_name: &str, tier: ExecutionTier) -> bool {
    let bucket = classify_tool(tool_name);
    match tier {
        ExecutionTier::ReadOnly => bucket == RiskBucket::R,
        ExecutionTier::BrowserAuto => matches!(bucket, RiskBucket::R | RiskBucket::B1),
        ExecutionTier::BrowserAutoLocalRead => {
            matches!(bucket, RiskBucket::R | RiskBucket::B1 | RiskBucket::L0)
        }
        ExecutionTier::FullAuto => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notify_user_needs_no_permission_at_any_tier() {
        // It tells the person something. Asking them to approve being told is
        // the wrong shape, and while it was unclassified the ACP gate waited
        // on a decision nobody was there to make.
        for tier in [
            ExecutionTier::ReadOnly,
            ExecutionTier::BrowserAuto,
            ExecutionTier::BrowserAutoLocalRead,
            ExecutionTier::FullAuto,
        ] {
            assert!(tier_auto_approves("notify_user", tier), "{tier:?}");
        }
        // Including the ACP-wrapped name, which arrives prefixed.
        assert!(tier_auto_approves(
            "mcp__nevoflux__notify_user",
            ExecutionTier::ReadOnly
        ));
    }

    #[test]
    fn from_setting_parses_all_tiers() {
        assert_eq!(
            ExecutionTier::from_setting("read-only"),
            ExecutionTier::ReadOnly
        );
        assert_eq!(
            ExecutionTier::from_setting("browser-auto"),
            ExecutionTier::BrowserAuto
        );
        assert_eq!(
            ExecutionTier::from_setting("browser-auto-local-read"),
            ExecutionTier::BrowserAutoLocalRead
        );
        assert_eq!(
            ExecutionTier::from_setting("full-auto"),
            ExecutionTier::FullAuto
        );
    }

    #[test]
    fn as_setting_round_trips_every_tier() {
        for t in [
            ExecutionTier::ReadOnly,
            ExecutionTier::BrowserAuto,
            ExecutionTier::BrowserAutoLocalRead,
            ExecutionTier::FullAuto,
        ] {
            assert_eq!(ExecutionTier::from_setting(t.as_setting()), t);
        }
    }

    #[test]
    fn from_setting_safety_legacy_and_unknown_fall_back_to_read_only() {
        // Old dead-stub 'auto' must NOT become full-auto.
        assert_eq!(ExecutionTier::from_setting("auto"), ExecutionTier::ReadOnly);
        assert_eq!(
            ExecutionTier::from_setting("confirm"),
            ExecutionTier::ReadOnly
        );
        assert_eq!(ExecutionTier::from_setting(""), ExecutionTier::ReadOnly);
        assert_eq!(ExecutionTier::from_setting("xxx"), ExecutionTier::ReadOnly);
        assert_eq!(ExecutionTier::default(), ExecutionTier::ReadOnly);
    }

    #[test]
    fn classify_browser_reads_and_nav_as_r() {
        for t in [
            "browser_get_content",
            "browser_screenshot",
            "browser_snapshot",
            "browser_navigate",
            "browser_go_back",
            "browser_go_forward",
            "web_fetch",
            "web_search",
        ] {
            assert_eq!(classify_tool(t), RiskBucket::R, "{t} should be R");
        }
    }

    #[test]
    fn classify_browser_interactions_as_b1() {
        for t in [
            "browser_click",
            "browser_type",
            "browser_fill",
            "browser_click_by_id",
            "browser_scroll",
            "browser_key_press",
        ] {
            assert_eq!(classify_tool(t), RiskBucket::B1, "{t} should be B1");
        }
    }

    #[test]
    fn classify_local_reads_as_l0() {
        for t in ["read_file", "list_files", "glob", "grep"] {
            assert_eq!(classify_tool(t), RiskBucket::L0, "{t} should be L0");
        }
    }

    #[test]
    fn classify_side_effects_and_unknowns_as_x() {
        for t in [
            "write_file",
            "edit_file",
            "run_command",
            "computer_click",
            "memory_create",
            "subagent_spawn",
            "browser_eval_js",
            "some_unknown_tool",
        ] {
            assert_eq!(classify_tool(t), RiskBucket::X, "{t} should be X");
        }
    }

    #[test]
    fn classify_strips_mcp_prefix() {
        assert_eq!(
            classify_tool("mcp__nevoflux-tools__web_fetch"),
            RiskBucket::R
        );
        assert_eq!(classify_tool("mcp__srv__read_file"), RiskBucket::L0);
    }

    #[test]
    fn read_only_tier_auto_approves_only_r() {
        assert!(tier_auto_approves(
            "browser_navigate",
            ExecutionTier::ReadOnly
        ));
        assert!(tier_auto_approves("web_fetch", ExecutionTier::ReadOnly));
        assert!(!tier_auto_approves(
            "browser_click",
            ExecutionTier::ReadOnly
        ));
        assert!(!tier_auto_approves("read_file", ExecutionTier::ReadOnly));
        assert!(!tier_auto_approves("write_file", ExecutionTier::ReadOnly));
    }

    #[test]
    fn browser_auto_tier_adds_b1() {
        assert!(tier_auto_approves(
            "browser_click",
            ExecutionTier::BrowserAuto
        ));
        assert!(tier_auto_approves(
            "browser_navigate",
            ExecutionTier::BrowserAuto
        ));
        assert!(!tier_auto_approves("read_file", ExecutionTier::BrowserAuto));
        assert!(!tier_auto_approves(
            "write_file",
            ExecutionTier::BrowserAuto
        ));
    }

    #[test]
    fn browser_auto_local_read_tier_adds_l0() {
        assert!(tier_auto_approves(
            "read_file",
            ExecutionTier::BrowserAutoLocalRead
        ));
        assert!(tier_auto_approves(
            "browser_click",
            ExecutionTier::BrowserAutoLocalRead
        ));
        assert!(!tier_auto_approves(
            "write_file",
            ExecutionTier::BrowserAutoLocalRead
        ));
        assert!(!tier_auto_approves(
            "run_command",
            ExecutionTier::BrowserAutoLocalRead
        ));
    }

    #[test]
    fn full_auto_tier_approves_everything() {
        for t in [
            "write_file",
            "run_command",
            "read_file",
            "browser_click",
            "some_mcp__x__y",
        ] {
            assert!(
                tier_auto_approves(t, ExecutionTier::FullAuto),
                "{t} should auto in full-auto"
            );
        }
    }
}
