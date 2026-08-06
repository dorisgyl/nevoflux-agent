//! Iteration-specific tool guards.
//!
//! Historical note: this module used to maintain a parallel "tool class"
//! taxonomy (Read / Write / DomClick / Nav / etc.) for /loop iterations.
//! That taxonomy used fictitious tool names that didn't match the real
//! `builtin-wasm::Agent::get_chat_tools()` catalog, so iterations ended
//! up with empty allowlists (no `browser_query` etc.).
//!
//! Migration 018 replaced the class system with the existing
//! `AgentMode { Chat, Browser, Agent }` enum from `builtin-wasm`, and
//! the iteration's tool catalog now comes directly from
//! `Agent::get_tools_for_mode(mode)`. The only iteration-specific filter
//! left is [`is_forbidden_in_iteration`].

/// Single source of truth for the iteration-forbidden tool names.
/// `loop_create` would let an iteration spawn nested loops; `ask_user` blocks
/// on a sidebar that may be closed. `goal_set` would let an unattended
/// iteration hijack the interactive session's active goal (goals are
/// session-scoped, single-active), and `schedule_create` would let an
/// iteration self-replicate scheduled jobs — both are catalog-filtered out
/// of direct-API unattended runs here (see `agent_exec::filter_allowlist`).
/// `loop_evolve` and `loop_proposal_respond` are the self-improvement
/// (evolve) feature's propose/accept operations; letting an unattended
/// iteration call both in the same turn would let it approve its own
/// rewrite with zero human review, defeating the whole point of the
/// evolve accept/reject gate.
const ITERATION_FORBIDDEN: &[&str] = &[
    "loop_create",
    "ask_user",
    "goal_set",
    "schedule_create",
    "loop_evolve",
    "loop_proposal_respond",
];

/// Tools that are forbidden inside loop iterations regardless of mode.
pub fn is_forbidden_in_iteration(tool_name: &str) -> bool {
    ITERATION_FORBIDDEN.contains(&tool_name)
}

/// The iteration-forbidden tool names as an owned list.
///
/// Behaviourally equivalent to filtering a catalog with
/// [`is_forbidden_in_iteration`], but materialised so it can be handed to the
/// shared `agent_exec` kernel as `AgentExecRequest::forbidden_tools`.
pub fn iteration_forbidden_tools() -> Vec<String> {
    ITERATION_FORBIDDEN.iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_set() {
        assert!(is_forbidden_in_iteration("loop_create"));
        assert!(is_forbidden_in_iteration("ask_user"));
        // Unattended iterations must not hijack the session goal or
        // self-replicate schedules.
        assert!(is_forbidden_in_iteration("goal_set"));
        assert!(is_forbidden_in_iteration("schedule_create"));
        // Evolve propose/accept must never run unattended.
        assert!(is_forbidden_in_iteration("loop_evolve"));
        assert!(is_forbidden_in_iteration("loop_proposal_respond"));
        assert!(!is_forbidden_in_iteration("read"));
        assert!(!is_forbidden_in_iteration("loop_scratchpad_set"));
        assert!(!is_forbidden_in_iteration("browser_get_content"));
        // Read-only goal/schedule tools stay available inside iterations.
        assert!(!is_forbidden_in_iteration("goal_status"));
        assert!(!is_forbidden_in_iteration("schedule_list"));
    }

    /// `orchestrate` is how a loop iteration reaches Code Mode, and Code Mode
    /// is where the Monty interpreter — and every constraint documented in
    /// `agent::code_mode::monty_capabilities` — actually applies.
    ///
    /// Forbidding it here would not merely remove a tool: it would silently
    /// sever loops from Code Mode entirely, and no test elsewhere covers that,
    /// because `code_mode` has no notion of running inside an iteration.
    #[test]
    fn orchestrate_stays_available_to_iterations() {
        assert!(!is_forbidden_in_iteration("orchestrate"));
        assert!(!iteration_forbidden_tools().contains(&"orchestrate".to_string()));
    }

    #[test]
    fn forbidden_list_matches_predicate() {
        let list = iteration_forbidden_tools();
        // Every listed name is forbidden by the predicate, and all entries
        // are present — the kernel filter and the predicate stay in lockstep.
        assert_eq!(
            list,
            vec![
                "loop_create".to_string(),
                "ask_user".to_string(),
                "goal_set".to_string(),
                "schedule_create".to_string(),
                "loop_evolve".to_string(),
                "loop_proposal_respond".to_string(),
            ]
        );
        for name in &list {
            assert!(is_forbidden_in_iteration(name));
        }
    }
}
