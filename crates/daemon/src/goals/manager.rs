//! GoalManager — the session-scoped goal engine facade.
//!
//! `GoalManager` owns the goal lifecycle (`set` / `status` / `clear`) and the
//! post-turn hook (`after_turn`). Unlike `ScheduleManager` it runs no
//! background task: it is a thin, cheap facade over [`GoalRepository`] plus the
//! evaluator, holding an `Option<Arc<EventBus>>` (via [`GoalEvents`]) so tests
//! without a bus still work.
//!
//! ## The post-turn decision core ([`apply_verdict`])
//!
//! The branchy part of `after_turn` — Achieved vs Expired vs Continue — is a
//! pure function of `(turns_used_after_increment, max_turns, verdict_met)` and
//! is unit-tested exhaustively (including the `turns_used == max_turns`
//! boundary). `after_turn` composes: repo bookkeeping + the evaluator call +
//! `apply_verdict` + event emission.
//!
//! ## Fail-safe evaluator contract
//!
//! `after_turn` runs after EVERY chat turn, so its fast path (no active goal ⇒
//! `None`) must be cheap. When an evaluator is broken — provider resolution
//! fails (a key vanished, or the stored provider is somehow ACP), or the LLM
//! call returns a transport `Err` — `after_turn` records a turn with an
//! `"evaluator error: ..."` reason, emits `evaluated {met:false}` +
//! `state_changed`, and returns `None`. It NEVER emits a continuation directive
//! on a broken evaluator, so a persistently-failing evaluator can never spin
//! the agent in an infinite turn loop.

use crate::config::AgentConfig;
use crate::event_bus::EventBus;
use crate::goals::evaluator::{
    clip_transcript, evaluate_with_choice, resolve_evaluator_for_goal, Verdict,
    TRANSCRIPT_MAX_BYTES, TRANSCRIPT_MAX_MESSAGES,
};
use crate::goals::events::GoalEvents;
use nevoflux_storage::models::current_timestamp;
use nevoflux_storage::models::goal::{GoalRecord, GoalStatus};
use nevoflux_storage::repositories::{GoalRepository, MessageRepository};
use nevoflux_storage::Database;
use serde_json::{json, Value};
use std::sync::Arc;

/// Maximum length of a goal condition (characters).
const CONDITION_MAX_CHARS: usize = 4000;
/// Default turn budget when the caller doesn't specify one (or specifies ≤0).
const DEFAULT_MAX_TURNS: i64 = 20;
/// How many recent tool results feed the programmatic check and the
/// continuation progress anchor (spec §4.1-4.3).
const TOOL_RESULTS_WINDOW: u32 = 6;
/// How many recent actions to list in the continuation progress anchor.
const PROGRESS_ANCHOR_K: usize = 5;
/// Per-message clamp (bytes) for a tool result folded into the evaluator
/// transcript, so one giant output can't evict all other context.
const TOOL_RESULT_MAX_BYTES: usize = 2048;

/// The pure post-turn decision, computed from the turn count *after* the
/// increment and whether the evaluator judged the condition met.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterTurnAction {
    /// Condition met — stamp `Achieved`.
    Achieved,
    /// Condition unmet and the turn budget is exhausted — stamp `Expired`.
    Expired,
    /// Condition unmet with budget remaining — hand back a continuation.
    Continue,
}

/// Pure decision core. `met` always wins (even at/over budget); otherwise the
/// budget boundary (`turns_used >= max_turns`) expires the goal.
pub fn apply_verdict(
    turns_used_after_increment: i64,
    max_turns: i64,
    verdict_met: bool,
) -> AfterTurnAction {
    if verdict_met {
        AfterTurnAction::Achieved
    } else if turns_used_after_increment >= max_turns {
        AfterTurnAction::Expired
    } else {
        AfterTurnAction::Continue
    }
}

/// Render a compact "Recent actions" anchor from the newest-first tool results,
/// one line per action (up to `k`). Each summary is single-lined and clipped so
/// the anchor stays small. Injected into the continuation so a resumed turn
/// knows what already happened and does not repeat completed steps (spec §4.2).
pub fn derive_progress_anchor(recent: &[(String, String)], k: usize) -> String {
    if recent.is_empty() {
        return "(no tool actions last turn)".to_string();
    }
    let mut out = String::new();
    for (name, summary) in recent.iter().take(k) {
        let one_line = summary.replace('\n', " ");
        let clipped: String = one_line.chars().take(120).collect();
        out.push_str("- ");
        out.push_str(name);
        out.push_str(" → ");
        out.push_str(&clipped);
        out.push('\n');
    }
    out.trim_end().to_string()
}

pub struct GoalManager {
    db: Database,
    events: GoalEvents,
    config: Arc<AgentConfig>,
}

impl GoalManager {
    pub fn new(db: Database, bus: Option<Arc<EventBus>>, config: Arc<AgentConfig>) -> Arc<Self> {
        Arc::new(Self {
            db,
            events: GoalEvents::new(bus),
            config,
        })
    }

    /// Set (replace) the active goal for a session. Validates the condition,
    /// resolves the evaluator eagerly (fail fast — an ACP active provider or a
    /// missing key errors here, not later), stores the resolved provider/model
    /// on the record, and emits `state_changed` (sticky). `GoalRepository`'s
    /// `create` clears any prior active goal in the same transaction.
    pub async fn set(
        &self,
        session_id: &str,
        condition: &str,
        provider: Option<String>,
        model: Option<String>,
        max_turns: Option<i64>,
    ) -> Result<GoalRecord, String> {
        self.set_checked(session_id, condition, provider, model, max_turns, None)
            .await
    }

    /// Like [`set`], plus an optional programmatic `check` (spec §4.3 route A).
    /// When a `check` is present the goal can complete with NO evaluator model,
    /// so evaluator resolution is best-effort rather than fail-fast.
    #[allow(clippy::too_many_arguments)]
    pub async fn set_checked(
        &self,
        session_id: &str,
        condition: &str,
        provider: Option<String>,
        model: Option<String>,
        max_turns: Option<i64>,
        check: Option<crate::goals::check::GoalCheck>,
    ) -> Result<GoalRecord, String> {
        let trimmed = condition.trim();
        if trimmed.is_empty() {
            return Err("goal condition must not be empty".to_string());
        }
        let char_count = trimmed.chars().count();
        if char_count > CONDITION_MAX_CHARS {
            return Err(format!(
                "goal condition too long: {char_count} characters (max {CONDITION_MAX_CHARS})"
            ));
        }

        // Resolve the evaluator eagerly so a bad provider/key surfaces at set
        // time. A check-only goal (`check` present) tolerates an unresolvable
        // evaluator — the programmatic check is its completion criterion, so it
        // needs no model at all (spec §4.3 route A: works with no API key).
        let choice =
            match resolve_evaluator_for_goal(&self.config, provider.as_deref(), model.as_deref()) {
                Ok(c) => Some(c),
                Err(e) => {
                    if check.is_some() {
                        None
                    } else {
                        return Err(e);
                    }
                }
            };
        let (evaluator_provider, evaluator_model) = match &choice {
            Some(c) => (Some(c.provider.clone()), Some(c.model.clone())),
            None => (provider.clone(), model.clone()),
        };

        let now = current_timestamp();
        let max_turns = max_turns.filter(|n| *n > 0).unwrap_or(DEFAULT_MAX_TURNS);
        // Fresh uuid (simple form). Goals are per-session with one active at a
        // time, so a full uuid is ample and avoids coupling to loop id gen.
        let id = uuid::Uuid::new_v4().simple().to_string();
        let rec = GoalRecord {
            id,
            session_id: session_id.to_string(),
            condition: trimmed.to_string(),
            evaluator_provider,
            evaluator_model,
            max_turns,
            turns_used: 0,
            status: GoalStatus::Active,
            last_reason: None,
            created_at: now,
            updated_at: now,
            achieved_at: None,
            check_json: check.as_ref().and_then(|c| serde_json::to_string(c).ok()),
        };
        GoalRepository::new(&self.db)
            .create(&rec)
            .map_err(|e| e.to_string())?;

        self.events
            .state_changed(
                session_id,
                &rec.id,
                rec.status.as_str(),
                &rec.condition,
                rec.turns_used,
                rec.max_turns,
                rec.last_reason.as_deref(),
            )
            .await;
        Ok(rec)
    }

    /// Report the goal status for a session as JSON. Prefers the active goal,
    /// else the most recent (already-resolved) one; `{"status":"none"}` when
    /// the session never had a goal.
    pub async fn status(&self, session_id: &str) -> Result<Value, String> {
        let repo = GoalRepository::new(&self.db);
        let rec = match repo.get_active(session_id).map_err(|e| e.to_string())? {
            Some(r) => Some(r),
            None => repo.latest(session_id).map_err(|e| e.to_string())?,
        };
        let Some(r) = rec else {
            return Ok(json!({ "status": "none" }));
        };
        let mut obj = json!({
            "condition": r.condition,
            "status": r.status.as_str(),
            "turns_used": r.turns_used,
            "max_turns": r.max_turns,
            "last_reason": r.last_reason,
            "evaluator": {
                "provider": r.evaluator_provider,
                "model": r.evaluator_model,
            },
        });
        if let Some(at) = r.achieved_at {
            obj["achieved_at"] = json!(at);
        }
        Ok(obj)
    }

    /// Clear the active goal for a session (→ `Cleared`). Returns `true` if a
    /// goal was cleared, `false` if there was no active goal.
    pub async fn clear(&self, session_id: &str) -> Result<bool, String> {
        let repo = GoalRepository::new(&self.db);
        let Some(r) = repo.get_active(session_id).map_err(|e| e.to_string())? else {
            return Ok(false);
        };
        let now = current_timestamp();
        repo.set_status(&r.id, GoalStatus::Cleared, now)
            .map_err(|e| e.to_string())?;
        self.events
            .state_changed(
                session_id,
                &r.id,
                GoalStatus::Cleared.as_str(),
                &r.condition,
                r.turns_used,
                r.max_turns,
                r.last_reason.as_deref(),
            )
            .await;
        Ok(true)
    }

    /// The post-turn hook, run after every chat turn. See the module docs for
    /// the full contract. Returns `Some(directive)` only when the caller should
    /// run another turn with that synthetic user message; `None` when the goal
    /// is done, absent, or the evaluator is broken (fail-safe).
    pub async fn after_turn(&self, session_id: &str) -> Option<String> {
        let repo = GoalRepository::new(&self.db);

        // (1) Cheap fast path: no active goal ⇒ nothing to do.
        let rec = match repo.get_active(session_id) {
            Ok(Some(r)) => r,
            Ok(None) => return None,
            Err(e) => {
                tracing::warn!(session_id, error = %e, "goal after_turn: get_active failed");
                return None;
            }
        };

        // (2) Recent tool results — shared by the programmatic check and the
        // continuation progress anchor.
        let tool_results = MessageRepository::new(&self.db)
            .list_recent_tool_results(session_id, TOOL_RESULTS_WINDOW)
            .unwrap_or_default();
        let check = rec
            .check_json
            .as_deref()
            .and_then(|s| serde_json::from_str::<crate::goals::check::GoalCheck>(s).ok());

        // (2b) Programmatic check short-circuit (spec §4.3 route A): a machine
        // check that holds against recent tool results achieves the goal with
        // NO model call at all — works with no API key / ACP-only.
        if let Some(c) = &check {
            if crate::goals::check::eval_check(c, &tool_results) {
                let now = current_timestamp();
                let turns_used = repo
                    .increment_turns(&rec.id, "programmatic check matched", now)
                    .unwrap_or(rec.turns_used + 1);
                let _ = repo.set_status(&rec.id, GoalStatus::Achieved, now);
                self.emit_both(
                    session_id,
                    &rec,
                    GoalStatus::Achieved.as_str(),
                    true,
                    "programmatic check matched",
                    turns_used,
                )
                .await;
                return None;
            }
        }

        // (3-4) Model verdict. A check-only goal whose model is unusable simply
        // continues (met=false) instead of fail-safe stopping — the check is
        // its completion criterion, so keep working until it matches or the
        // budget is exhausted.
        let verdict = match resolve_evaluator_for_goal(
            &self.config,
            rec.evaluator_provider.as_deref(),
            rec.evaluator_model.as_deref(),
        ) {
            Ok(choice) => {
                let transcript = self.load_transcript(session_id);
                // ACP evaluator (route B) goes through the one-shot ACP adapter;
                // direct-API evaluators through the standard path.
                let result = evaluate_with_choice(&choice, &rec.condition, &transcript).await;
                match result {
                    Ok(v) => v,
                    Err(e) => {
                        // A model error with a check present is not fatal — the
                        // check is the real completion criterion, so keep working
                        // rather than fail-safe stopping.
                        if check.is_some() {
                            Verdict {
                                met: false,
                                reason: "awaiting programmatic check".to_string(),
                                tokens_used: 0,
                            }
                        } else {
                            return self.record_evaluator_error(&repo, &rec, &e).await;
                        }
                    }
                }
            }
            Err(e) => {
                if check.is_some() {
                    Verdict {
                        met: false,
                        reason: "awaiting programmatic check".to_string(),
                        tokens_used: 0,
                    }
                } else {
                    return self.record_evaluator_error(&repo, &rec, &e).await;
                }
            }
        };

        // (5) Count the turn with the verdict's reason.
        let now = current_timestamp();
        let turns_used = match repo.increment_turns(&rec.id, &verdict.reason, now) {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(goal_id = %rec.id, error = %e, "goal after_turn: increment_turns failed");
                return None;
            }
        };

        // (6) Pure decision + emit on every transition.
        match apply_verdict(turns_used, rec.max_turns, verdict.met) {
            AfterTurnAction::Achieved => {
                let _ = repo.set_status(&rec.id, GoalStatus::Achieved, now);
                self.emit_both(
                    session_id,
                    &rec,
                    GoalStatus::Achieved.as_str(),
                    true,
                    &verdict.reason,
                    turns_used,
                )
                .await;
                None
            }
            AfterTurnAction::Expired => {
                let _ = repo.set_status(&rec.id, GoalStatus::Expired, now);
                self.emit_both(
                    session_id,
                    &rec,
                    GoalStatus::Expired.as_str(),
                    false,
                    &verdict.reason,
                    turns_used,
                )
                .await;
                None
            }
            AfterTurnAction::Continue => {
                self.emit_both(
                    session_id,
                    &rec,
                    GoalStatus::Active.as_str(),
                    false,
                    &verdict.reason,
                    turns_used,
                )
                .await;
                let anchor = derive_progress_anchor(&tool_results, PROGRESS_ANCHOR_K);
                Some(format!(
                    "<GOAL-CONTINUATION>\nGoal not yet met: {condition}\nEvaluator: {reason}\nRecent actions (do NOT repeat completed steps):\n{anchor}\nContinue working toward the goal. Turn {used}/{max}.\n</GOAL-CONTINUATION>",
                    condition = rec.condition,
                    reason = verdict.reason,
                    anchor = anchor,
                    used = turns_used,
                    max = rec.max_turns,
                ))
            }
        }
    }

    /// Load the transcript tail for the evaluator: the last
    /// [`TRANSCRIPT_MAX_MESSAGES`] messages (fetched efficiently), then clipped
    /// to [`TRANSCRIPT_MAX_BYTES`] (oldest dropped first).
    fn load_transcript(&self, session_id: &str) -> Vec<(String, String)> {
        use nevoflux_storage::models::ContentType;
        let msgs = MessageRepository::new(&self.db)
            .list_recent(session_id, TRANSCRIPT_MAX_MESSAGES as u32)
            .unwrap_or_default();
        let pairs: Vec<(String, String)> = msgs
            .into_iter()
            .map(|m| {
                // Label tool results distinctly so the evaluator reads them as
                // observed output, not the model's own claim; clamp huge ones.
                if m.content_type == ContentType::ToolResult {
                    let content = if m.content.len() > TOOL_RESULT_MAX_BYTES {
                        let mut c: String = m.content.chars().take(TOOL_RESULT_MAX_BYTES).collect();
                        c.push_str("…[truncated]");
                        c
                    } else {
                        m.content
                    };
                    ("tool".to_string(), content)
                } else {
                    (m.role.as_str().to_string(), m.content)
                }
            })
            .collect();
        clip_transcript(pairs, TRANSCRIPT_MAX_MESSAGES, TRANSCRIPT_MAX_BYTES)
    }

    /// Fail-safe evaluator-error path: record a turn with an
    /// `"evaluator error: ..."` reason, emit `evaluated {met:false}` +
    /// `state_changed` (goal stays in its current status), and return `None`.
    /// Never yields a continuation, so a broken evaluator cannot spin turns.
    async fn record_evaluator_error(
        &self,
        repo: &GoalRepository<'_>,
        rec: &GoalRecord,
        err: &str,
    ) -> Option<String> {
        let reason = format!("evaluator error: {err}");
        tracing::warn!(goal_id = %rec.id, error = %err, "goal after_turn: evaluator error (fail-safe)");
        let now = current_timestamp();
        let turns_used = repo
            .increment_turns(&rec.id, &reason, now)
            .unwrap_or(rec.turns_used + 1);
        self.events
            .evaluated(&rec.session_id, &rec.id, false, &reason, turns_used)
            .await;
        self.events
            .state_changed(
                &rec.session_id,
                &rec.id,
                rec.status.as_str(),
                &rec.condition,
                turns_used,
                rec.max_turns,
                Some(&reason),
            )
            .await;
        None
    }

    /// Emit both the sticky `state_changed` snapshot and the ephemeral
    /// `evaluated` signal for a single verdict.
    async fn emit_both(
        &self,
        session_id: &str,
        rec: &GoalRecord,
        status: &str,
        met: bool,
        reason: &str,
        turns_used: i64,
    ) {
        self.events
            .state_changed(
                session_id,
                &rec.id,
                status,
                &rec.condition,
                turns_used,
                rec.max_turns,
                Some(reason),
            )
            .await;
        self.events
            .evaluated(session_id, &rec.id, met, reason, turns_used)
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_bus::types::{BackpressurePolicy, SubscriberIdentity, TopicPattern};
    use nevoflux_storage::repositories::SessionRepository;
    use nevoflux_storage::CreateSessionParams;

    // ---- apply_verdict (pure) ---------------------------------------------

    #[test]
    fn apply_verdict_met_is_achieved_regardless_of_turns() {
        assert_eq!(apply_verdict(1, 20, true), AfterTurnAction::Achieved);
        // met wins even at/over the budget.
        assert_eq!(apply_verdict(20, 20, true), AfterTurnAction::Achieved);
        assert_eq!(apply_verdict(25, 20, true), AfterTurnAction::Achieved);
    }

    #[test]
    fn apply_verdict_unmet_under_budget_continues() {
        assert_eq!(apply_verdict(1, 20, false), AfterTurnAction::Continue);
        assert_eq!(apply_verdict(19, 20, false), AfterTurnAction::Continue);
    }

    // ---- derive_progress_anchor (pure) ------------------------------------

    #[test]
    fn progress_anchor_lists_recent_actions_newest_first() {
        let recent = vec![
            (
                "create_artifact".to_string(),
                "Artifact created and sent to canvas: art-123".to_string(),
            ),
            ("skill_load".to_string(), "ok".to_string()),
        ];
        let s = derive_progress_anchor(&recent, 5);
        assert!(s.contains("create_artifact"));
        assert!(s.contains("art-123"));
        assert!(s.contains("skill_load"));
        // newest first
        assert!(s.find("create_artifact").unwrap() < s.find("skill_load").unwrap());
    }

    #[test]
    fn progress_anchor_caps_at_k() {
        let recent: Vec<(String, String)> = (0..10)
            .map(|i| (format!("tool{i}"), "r".to_string()))
            .collect();
        let s = derive_progress_anchor(&recent, 3);
        assert_eq!(s.matches("- ").count(), 3);
    }

    #[test]
    fn progress_anchor_empty_is_explicit() {
        assert_eq!(
            derive_progress_anchor(&[], 5),
            "(no tool actions last turn)"
        );
    }

    #[test]
    fn apply_verdict_unmet_at_budget_boundary_expires() {
        // turns_used == max_turns is the boundary: expired.
        assert_eq!(apply_verdict(20, 20, false), AfterTurnAction::Expired);
    }

    #[test]
    fn apply_verdict_unmet_over_budget_expires() {
        assert_eq!(apply_verdict(21, 20, false), AfterTurnAction::Expired);
    }

    // ---- test scaffolding --------------------------------------------------

    fn seed_session(db: &Database, id: &str) {
        SessionRepository::new(db)
            .create(CreateSessionParams::new().with_id(id))
            .unwrap();
    }

    fn config_anthropic() -> Arc<AgentConfig> {
        let mut cfg = AgentConfig::default();
        cfg.llm.provider = Some("anthropic".to_string());
        cfg.llm.anthropic.api_key = Some("sk-ant-test".to_string());
        cfg.llm.anthropic.model = Some("claude-haiku-4-5".to_string());
        Arc::new(cfg)
    }

    fn config_no_provider() -> Arc<AgentConfig> {
        Arc::new(AgentConfig::default())
    }

    // ---- set ---------------------------------------------------------------

    /// A check-only goal can be set with NO usable evaluator provider — the
    /// programmatic check is its completion criterion (spec §4.3 route A).
    #[tokio::test]
    async fn set_checked_check_only_succeeds_without_evaluator() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_no_provider());

        // No provider → a plain goal cannot be set.
        assert!(mgr.set("sess-1", "done", None, None, None).await.is_err());

        // But a check-only goal can (no model needed).
        let check = crate::goals::check::GoalCheck {
            tool: None,
            matches: "PASS".into(),
            negate: false,
        };
        let rec = mgr
            .set_checked("sess-1", "tests pass", None, None, None, Some(check))
            .await
            .expect("check-only goal sets without an evaluator");
        assert!(rec.check_json.is_some());
    }

    #[tokio::test]
    async fn set_persists_and_stores_resolved_evaluator() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());

        let rec = mgr
            .set("sess-1", "  the PR is merged  ", None, None, Some(15))
            .await
            .expect("set succeeds");
        assert_eq!(rec.condition, "the PR is merged", "condition is trimmed");
        assert_eq!(rec.evaluator_provider.as_deref(), Some("anthropic"));
        assert_eq!(rec.evaluator_model.as_deref(), Some("claude-haiku-4-5"));
        assert_eq!(rec.max_turns, 15);
        assert_eq!(rec.status, GoalStatus::Active);

        let active = GoalRepository::new(&db)
            .get_active("sess-1")
            .unwrap()
            .expect("active goal");
        assert_eq!(active.id, rec.id);
    }

    #[tokio::test]
    async fn set_defaults_max_turns_when_absent_or_nonpositive() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());

        let rec = mgr.set("sess-1", "done", None, None, None).await.unwrap();
        assert_eq!(rec.max_turns, DEFAULT_MAX_TURNS);

        let rec2 = mgr
            .set("sess-1", "done again", None, None, Some(0))
            .await
            .unwrap();
        assert_eq!(rec2.max_turns, DEFAULT_MAX_TURNS, "0 falls back to default");
    }

    #[tokio::test]
    async fn set_rejects_empty_and_whitespace_condition() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());

        assert!(mgr
            .set("sess-1", "", None, None, None)
            .await
            .unwrap_err()
            .contains("empty"));
        assert!(mgr
            .set("sess-1", "   \n\t ", None, None, None)
            .await
            .unwrap_err()
            .contains("empty"));
    }

    #[tokio::test]
    async fn set_rejects_over_length_condition() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());

        let too_long = "x".repeat(CONDITION_MAX_CHARS + 1);
        let err = mgr
            .set("sess-1", &too_long, None, None, None)
            .await
            .unwrap_err();
        assert!(err.contains("too long"), "unexpected: {err}");

        // Exactly at the limit is allowed.
        let at_limit = "y".repeat(CONDITION_MAX_CHARS);
        assert!(mgr.set("sess-1", &at_limit, None, None, None).await.is_ok());
    }

    /// With ACP-evaluator support (spec §4.3 route B) an ACP active provider is
    /// now ACCEPTED as the evaluator (judged one-shot over a connected ACP
    /// session), no longer rejected at set time.
    #[tokio::test]
    async fn set_accepts_acp_active_provider_as_evaluator() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mut cfg = AgentConfig::default();
        cfg.llm.provider = Some("claude-code".to_string());
        let mgr = GoalManager::new(db.clone(), None, Arc::new(cfg));

        let rec = mgr
            .set("sess-1", "done", None, None, None)
            .await
            .expect("ACP provider is now an accepted evaluator");
        assert_eq!(rec.evaluator_provider.as_deref(), Some("claude-code"));
        assert!(GoalRepository::new(&db)
            .get_active("sess-1")
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn set_replaces_prior_active_goal() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());

        let first = mgr
            .set("sess-1", "first goal", None, None, None)
            .await
            .unwrap();
        let second = mgr
            .set("sess-1", "second goal", None, None, None)
            .await
            .unwrap();

        let repo = GoalRepository::new(&db);
        assert_eq!(
            repo.get(&first.id).unwrap().unwrap().status,
            GoalStatus::Cleared,
            "prior goal cleared"
        );
        let active = repo.get_active("sess-1").unwrap().unwrap();
        assert_eq!(active.id, second.id);
    }

    // ---- status ------------------------------------------------------------

    #[tokio::test]
    async fn status_none_when_no_goal() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());
        let s = mgr.status("sess-1").await.unwrap();
        assert_eq!(s, json!({ "status": "none" }));
    }

    #[tokio::test]
    async fn status_active_shape() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());
        mgr.set("sess-1", "the report is posted", None, None, Some(12))
            .await
            .unwrap();

        let s = mgr.status("sess-1").await.unwrap();
        assert_eq!(s["condition"], json!("the report is posted"));
        assert_eq!(s["status"], json!("active"));
        assert_eq!(s["turns_used"], json!(0));
        assert_eq!(s["max_turns"], json!(12));
        assert_eq!(s["last_reason"], json!(null));
        assert_eq!(s["evaluator"]["provider"], json!("anthropic"));
        assert_eq!(s["evaluator"]["model"], json!("claude-haiku-4-5"));
        assert!(
            s.get("achieved_at").is_none(),
            "no achieved_at while active"
        );
    }

    #[tokio::test]
    async fn status_reports_resolved_goal_via_latest() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());
        let rec = mgr.set("sess-1", "done", None, None, None).await.unwrap();

        // Resolve it (achieved) so there is no active goal, only a latest one.
        GoalRepository::new(&db)
            .set_status(&rec.id, GoalStatus::Achieved, current_timestamp())
            .unwrap();

        let s = mgr.status("sess-1").await.unwrap();
        assert_eq!(s["status"], json!("achieved"));
        assert!(
            s.get("achieved_at").is_some(),
            "achieved_at present when set"
        );
    }

    // ---- clear -------------------------------------------------------------

    #[tokio::test]
    async fn clear_active_goal_returns_true() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());
        mgr.set("sess-1", "done", None, None, None).await.unwrap();

        assert!(mgr.clear("sess-1").await.unwrap());
        assert!(GoalRepository::new(&db)
            .get_active("sess-1")
            .unwrap()
            .is_none());
        // Idempotent-ish: a second clear finds nothing active.
        assert!(!mgr.clear("sess-1").await.unwrap());
    }

    #[tokio::test]
    async fn clear_no_goal_returns_false() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());
        assert!(!mgr.clear("sess-1").await.unwrap());
    }

    // ---- after_turn --------------------------------------------------------

    #[tokio::test]
    async fn after_turn_fast_path_no_goal_returns_none() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());
        assert!(mgr.after_turn("sess-1").await.is_none());
    }

    fn seed_tool_result(db: &Database, session_id: &str, tool: &str, content: &str) {
        use nevoflux_storage::models::{ContentType, CreateMessageParams, MessageRole};
        let mut meta = std::collections::HashMap::new();
        meta.insert("tool_name".to_string(), serde_json::json!(tool));
        MessageRepository::new(db)
            .create(
                CreateMessageParams::new(session_id, MessageRole::Assistant, content)
                    .with_content_type(ContentType::ToolResult)
                    .with_metadata(meta),
            )
            .unwrap();
    }

    fn seed_goal_with_check(
        db: &Database,
        id: &str,
        session_id: &str,
        check: crate::goals::check::GoalCheck,
    ) {
        let now = current_timestamp();
        let rec = GoalRecord {
            id: id.to_string(),
            session_id: session_id.to_string(),
            condition: "condition text".to_string(),
            // ACP provider: resolve_evaluator would ERROR if the model path ran.
            evaluator_provider: Some("claude-code".to_string()),
            evaluator_model: Some("sonnet".to_string()),
            max_turns: 20,
            turns_used: 0,
            status: GoalStatus::Active,
            last_reason: None,
            created_at: now,
            updated_at: now,
            achieved_at: None,
            check_json: Some(serde_json::to_string(&check).unwrap()),
        };
        GoalRepository::new(db).create(&rec).unwrap();
    }

    /// A matching programmatic check achieves the goal WITHOUT ever calling the
    /// (deliberately broken/ACP) evaluator model. Spec §4.3 route A.
    #[tokio::test]
    async fn after_turn_check_hit_achieves_without_model() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());
        seed_goal_with_check(
            &db,
            "goal-check",
            "sess-1",
            crate::goals::check::GoalCheck {
                tool: Some("canvas_eval".into()),
                matches: "15".into(),
                negate: false,
            },
        );
        seed_tool_result(&db, "sess-1", "canvas_eval", "display=15");

        let out = mgr.after_turn("sess-1").await;
        assert!(out.is_none(), "check matched → achieved, no continuation");
        let status = mgr.status("sess-1").await.unwrap();
        assert_eq!(status["status"], "achieved");
    }

    /// A check-only goal whose check is unmet and whose model is unusable must
    /// CONTINUE (keep working) rather than fail-safe stop, and the continuation
    /// carries the progress anchor. Spec §4.2/§4.3.
    #[tokio::test]
    async fn after_turn_check_only_unmet_continues_with_anchor() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let mgr = GoalManager::new(db.clone(), None, config_anthropic());
        seed_goal_with_check(
            &db,
            "goal-check",
            "sess-1",
            crate::goals::check::GoalCheck {
                tool: Some("canvas_eval".into()),
                matches: "15".into(),
                negate: false,
            },
        );
        // Non-matching result → check unmet, but it should appear in the anchor.
        seed_tool_result(&db, "sess-1", "canvas_eval", "display=20");

        let out = mgr.after_turn("sess-1").await;
        let cont = out.expect("check-only unmet → continuation, not fail-safe stop");
        assert!(cont.contains("GOAL-CONTINUATION"));
        assert!(cont.contains("Recent actions"));
        assert!(cont.contains("canvas_eval"));
    }

    /// Fail-safe: a goal whose stored evaluator is an ACP provider (seeded
    /// directly to bypass `set`'s eager rejection) must NOT loop. `after_turn`
    /// records an `"evaluator error"` turn, emits both events, and returns
    /// `None` — never a continuation.
    #[tokio::test]
    async fn after_turn_broken_evaluator_fails_safe_without_looping() {
        let db = Database::open_in_memory().unwrap();
        seed_session(&db, "sess-1");
        let bus = Arc::new(EventBus::new());
        let mut handle = bus
            .subscribe(
                TopicPattern::double_wildcard("system:goal"),
                SubscriberIdentity::Internal,
                BackpressurePolicy::DropNewest,
                16,
            )
            .unwrap();
        let mgr = GoalManager::new(db.clone(), Some(bus.clone()), config_anthropic());

        // Seed an active goal with an ACP evaluator provider directly.
        let now = current_timestamp();
        let rec = GoalRecord {
            id: "goal-broken".to_string(),
            session_id: "sess-1".to_string(),
            condition: "impossible to evaluate".to_string(),
            evaluator_provider: Some("claude-code".to_string()),
            evaluator_model: Some("sonnet".to_string()),
            max_turns: 20,
            turns_used: 0,
            status: GoalStatus::Active,
            last_reason: None,
            created_at: now,
            updated_at: now,
            achieved_at: None,
            check_json: None,
        };
        GoalRepository::new(&db).create(&rec).unwrap();

        let out = mgr.after_turn("sess-1").await;
        assert!(
            out.is_none(),
            "broken evaluator must not yield a continuation"
        );

        // A turn was counted with an evaluator-error reason.
        let after = GoalRepository::new(&db)
            .get("goal-broken")
            .unwrap()
            .unwrap();
        assert_eq!(after.turns_used, 1);
        assert!(after.status == GoalStatus::Active, "status unchanged");
        assert!(after
            .last_reason
            .as_deref()
            .unwrap()
            .contains("evaluator error"));

        // Both events were emitted (evaluated + state_changed).
        let mut topics = Vec::new();
        while let Ok(ev) = handle.rx.try_recv() {
            topics.push(ev.topic.clone());
        }
        assert!(topics.iter().any(|t| t == "system:goal:evaluated"));
        assert!(topics.iter().any(|t| t == "system:goal:state_changed"));

        // Running it again still fails safe (never loops) and counts another turn.
        assert!(mgr.after_turn("sess-1").await.is_none());
        assert_eq!(
            GoalRepository::new(&db)
                .get("goal-broken")
                .unwrap()
                .unwrap()
                .turns_used,
            2
        );
    }
}
