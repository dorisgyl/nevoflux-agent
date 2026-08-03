//! Goals subsystem — session-scoped success conditions with an LLM evaluator.
//!
//! A *goal* is a natural-language success condition attached to a chat session
//! (e.g. "the report has been posted to #general"). After every chat turn the
//! [`manager::GoalManager::after_turn`] hook asks a zero-tool LLM *evaluator*
//! whether the condition is met yet, given the tail of the conversation. When
//! it is not met and the turn budget is not exhausted, `after_turn` hands the
//! caller a synthetic `<GOAL-CONTINUATION>` directive to drive another turn;
//! when it is met (or the budget runs out, or the evaluator is broken) it
//! returns `None` and the loop stops.
//!
//! Task 2.1 landed the storage layer (`GoalRepository`, `GoalRecord`). Task 2.2
//! added:
//! - [`evaluator`]: `resolve_evaluator` (pick a direct-API provider/model/key,
//!   rejecting ACP providers), the verbatim evaluator system prompt, the pure
//!   `parse_verdict` / `clip_transcript` cores, and the async `evaluate` call.
//! - [`events`]: the `system:goal:*` EventBus surface (mirrors
//!   `schedules::events`).
//! - [`manager`]: `GoalManager` (`set` / `status` / `clear` / `after_turn`) and
//!   the pure `apply_verdict` decision core.
//!
//! Task 2.3 adds [`tools`] (the `goal_*` LLM-callable tool dispatcher).

pub mod check;
pub mod evaluator;
pub mod events;
pub mod manager;
pub mod tools;

pub use evaluator::{evaluate, evaluate_with_choice, resolve_evaluator, EvaluatorChoice, Verdict};
pub use events::GoalEvents;
pub use manager::GoalManager;
pub use tools::execute_goal_tool;

/// Process-global handle to the daemon's `GoalManager`, set once at daemon
/// startup (see `server.rs` right after the manager is constructed).
///
/// Mirrors [`crate::loops::CURRENT_LOOP_MANAGER`]. Used by
/// `agent_exec::run_agent_once` to back-fill `HostServices.goal_manager` into
/// the per-run services clone: the automation/schedule-runner services
/// snapshots are captured BEFORE `with_goal_manager` runs (chicken-and-egg at
/// boot), so an unattended run's read-only `goal_status` tool would otherwise
/// fail with a misleading "daemon was started without a GoalManager" error.
pub static CURRENT_GOAL_MANAGER: std::sync::OnceLock<std::sync::Arc<GoalManager>> =
    std::sync::OnceLock::new();
