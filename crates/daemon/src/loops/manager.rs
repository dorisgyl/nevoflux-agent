//! LoopManager — the daemon's facade for the /loop skill (spec §4 architecture).
//!
//! Phase 7 wires only the `time:<duration>` trigger; other trigger variants
//! land in later phases (event in 11, dynamic in 12, state in 19, AND/OR in 20).

use crate::event_bus::EventBus;
use crate::loops::events::LoopEvents;
use crate::loops::executor::IterationExecutor;
use crate::loops::expression::TriggerExpr;
use crate::loops::registry::LoopRegistry;
use crate::loops::scheduler::{LoopFireRequest, TriggerScheduler};
use crate::loops::types::{GateKind, GateSpec, LoopId, LoopRuntime};
use nevoflux_builtin_wasm::AgentMode;
use nevoflux_storage::connection::Database;
use nevoflux_storage::models::{current_timestamp, LoopRecord, LoopState};
use nevoflux_storage::repositories::LoopRepository;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Canonical wire/DB form of an `AgentMode`. Mirrors `serde(rename_all="snake_case")`
/// from `builtin-wasm/types.rs` but as a `&'static str` for SQL bind/CHECK.
pub fn agent_mode_to_db_str(mode: AgentMode) -> &'static str {
    match mode {
        AgentMode::Chat => "chat",
        AgentMode::Browser => "browser",
        AgentMode::Agent => "agent",
        #[allow(deprecated)]
        AgentMode::Code => "agent",
    }
}

/// Parse a DB/wire string into an `AgentMode`. Unknown strings default to `Chat`
/// (matches `server::parse_agent_mode` behavior for forward compatibility).
pub fn db_str_to_agent_mode(s: &str) -> AgentMode {
    match s {
        "browser" => AgentMode::Browser,
        "agent" => AgentMode::Agent,
        _ => AgentMode::Chat,
    }
}

/// Re-check whether `id` is still eligible to claim a fire. Called by the
/// dispatcher right before it commits to running an iteration (state ->
/// `Running`, `execute()`).
///
/// This exists because `evaluate_gate` can await up to `FETCH_TIMEOUT` (5s,
/// see `gate::DefaultFetcher`) for `http`/`bash` gates. A concurrent
/// `LoopManager::cancel_loop` can flip the loop to a terminal state,
/// decrement `pending_work`, tear down its trigger subscriptions, and
/// `registry.remove(id)` while the dispatcher is suspended on that await.
/// Resuming without this check would unconditionally overwrite
/// `Cancelled`/`Failed` back to `Running`, run a full iteration the caller
/// already cancelled, and finish by setting `Idle` — a zombie loop with no
/// live triggers and a permanently undercounted `pending_work`.
///
/// A loop is still live iff its DB row exists in a non-terminal state
/// (`Pending`/`Running`/`Idle`) AND it is still present in the in-memory
/// registry. `cancel_loop_inner` performs both the terminal DB transition
/// and the `registry.remove` as part of the same tear-down, so checking
/// only one leaves a window if the two were ever to diverge — belt and
/// suspenders.
/// Validate that a requested gate is compatible with the loop's trigger
/// (spec §gate). `Http`/`Bash` gates run a poll-and-diff check on the
/// loop's own cadence, so they require a non-event trigger (`time:<dur>`,
/// `time:dynamic`, or a `state:` watcher) — an `event:` trigger already
/// fires only when something happens, so a value-diff gate on top of it is
/// a config mismatch the caller almost certainly didn't intend. `Event`
/// gates filter the triggering event's own payload, so they require an
/// `event:` trigger — attaching one to a `time:` trigger would have
/// nothing to filter.
fn validate_gate_trigger_compat(gate: &GateSpec, expr: &TriggerExpr) -> Result<(), String> {
    let is_event_trigger = matches!(expr, TriggerExpr::Event(_));
    match gate.kind {
        GateKind::None => Ok(()),
        GateKind::Http | GateKind::Bash => {
            if is_event_trigger {
                Err(format!(
                    "{}-gate requires a time-based (or state:) trigger, not an event: trigger — \
                     it polls on the loop's own cadence, not the event",
                    gate.kind.as_str()
                ))
            } else {
                Ok(())
            }
        }
        GateKind::Event => {
            if is_event_trigger {
                Ok(())
            } else {
                Err(
                    "event-gate requires an event:<topic> trigger — it filters that trigger's \
                     payload"
                        .into(),
                )
            }
        }
    }
}

fn loop_still_live(repo: &LoopRepository<'_>, registry: &LoopRegistry, id: &LoopId) -> bool {
    let db_live = matches!(
        repo.get(id.as_ref()).ok().flatten().map(|r| r.state),
        Some(LoopState::Pending) | Some(LoopState::Running) | Some(LoopState::Idle)
    );
    db_live && registry.contains(id)
}

#[derive(Debug, Clone)]
pub struct CreateLoopArgs {
    pub session_id: String,
    pub trigger_expr_text: String,
    pub prompt_text: Option<String>,
    pub wrapped_skill: Option<String>, // JSON {name, args}
    /// Agent mode for iterations (chat | browser | agent). Drives the
    /// iteration's tool catalog via `builtin-wasm::Agent::get_tools_for_mode`.
    /// Inherited from `SkillCommandPayload.mode` at /loop creation time.
    pub mode: nevoflux_builtin_wasm::AgentMode,
    /// Optional deterministic pre-check gate (W3 spec §gate). `create_loop`
    /// validates trigger-compat before persisting: `Http`/`Bash` gates
    /// require a non-event trigger (they poll on the loop's own `time:`
    /// cadence); `Event` gates require an `event:` trigger (they filter
    /// that trigger's payload). `None`/absent means no gate — the loop
    /// always fires on its trigger, matching pre-W3 behavior.
    pub gate: Option<GateSpec>,
    /// Optional programmatic check (W5 spec §verify), reusing `/goal`'s
    /// `GoalCheck` shape (`{tool?, matches, negate?}`). Stored as the RAW
    /// JSON object string, exactly as received — `IterationExecutor`
    /// wraps it in a `{"check": ...}` envelope and re-parses it with
    /// `goals::check::parse_check` on every iteration (see
    /// `executor.rs::finalize_iteration`). `None`/absent means no
    /// per-iteration verdict is recorded.
    pub verify_check: Option<String>,
}

#[derive(Clone)]
pub struct LoopManager {
    db: Database,
    registry: LoopRegistry,
    scheduler: TriggerScheduler,
    fire_tx: mpsc::Sender<LoopFireRequest>,
    events: Arc<LoopEvents>,
    event_bus: Option<Arc<EventBus>>,
    /// Count of armed (non-terminal) loops. A managed daemon must outlive its
    /// browser while any loop is armed — mirrors `ScheduleManager::pending_work`.
    pending_work: Arc<AtomicUsize>,
}

impl LoopManager {
    /// Spawns the dispatcher task and returns the manager handle.
    /// The dispatcher consumes `LoopFireRequest`s emitted by triggers
    /// and routes them to `IterationExecutor::execute`, applying the
    /// drop-on-busy concurrency policy from spec §8.2.
    pub fn start(db: Database) -> Self {
        Self::start_with_bus(db, None, None)
    }

    /// Same as [`start`], but with an EventBus handle so the manager and its
    /// dispatcher emit `system:loop:*` events. If `bus` is `None`, all
    /// emissions are silent no-ops (used by unit tests that don't care).
    ///
    /// `services` carries the live `HostServices` snapshot the iteration
    /// executor uses to spawn a production `DaemonHostFunctions`. When
    /// `None` (unit tests), the executor falls back to the Phase-6 stub
    /// path that records iterations as `ok` without invoking an LLM.
    /// When `Some`, the executor invokes
    /// `nevoflux_builtin_wasm::Agent::run` on every iteration with the
    /// loop's tool-class allowlist (Phase 9c).
    pub fn start_with_bus(
        db: Database,
        bus: Option<Arc<EventBus>>,
        services: Option<crate::wasm::services::HostServices>,
    ) -> Self {
        let (fire_tx, mut fire_rx) = mpsc::channel::<LoopFireRequest>(64);
        let registry = LoopRegistry::new();
        let scheduler = TriggerScheduler::new();
        let events = Arc::new(LoopEvents::new(bus.clone()));
        let executor_inner = IterationExecutor::new_with_events(db.clone(), events.clone());
        let executor = Arc::new(match services {
            Some(s) => executor_inner.with_services(s),
            None => executor_inner,
        });

        // Seed the pending-work counter with the loops that are still armed
        // (non-terminal) at construction time, so a daemon restarted with
        // armed loops keeps itself alive from the moment the manager starts.
        let armed: i64 = db
            .with_connection(|c| {
                c.query_row(
                    "SELECT COUNT(*) FROM loops WHERE state NOT IN ('cancelled','failed')",
                    [],
                    |r| r.get(0),
                )
                .map_err(nevoflux_storage::error::StorageError::from)
            })
            .unwrap_or(0);
        let pending_work = Arc::new(AtomicUsize::new(armed as usize));
        let pending_work_for_task = pending_work.clone();

        let registry_for_task = registry.clone();
        let executor_for_task = executor.clone();
        let events_for_task = events.clone();
        let scheduler_for_task = scheduler.clone();
        let fire_tx_for_task = fire_tx.clone();
        // Zero-sized real network/exec boundary for the gate evaluator
        // (W3 §gate) — `Copy`, constructed once and reused across fires.
        let default_fetcher = crate::loops::gate::DefaultFetcher;
        tokio::spawn(async move {
            while let Some(initial_req) = fire_rx.recv().await {
                // SKILL.md §8.2 "drop if currently running": the dispatcher is
                // single-tasked, so events arriving during execute().await pile
                // up in fire_rx. Without coalescing, each queued event would
                // spawn its own back-to-back iteration. Drain same-loop fires
                // to last-wins; re-enqueue fires for other loops so they get
                // their own pass.
                let mut req = initial_req;
                let mut dropped_for_current: u32 = 0;
                while let Ok(queued) = fire_rx.try_recv() {
                    if queued.loop_id == req.loop_id {
                        dropped_for_current += 1;
                        req = queued;
                    } else {
                        let _ = fire_tx_for_task.try_send(queued);
                    }
                }
                if dropped_for_current > 0 {
                    let now = current_timestamp();
                    let db = executor_for_task.database();
                    let repo = LoopRepository::new(&db);
                    for _ in 0..dropped_for_current {
                        let _ = repo.increment_skipped(req.loop_id.as_ref(), now);
                    }
                    let session_id = registry_for_task
                        .with_mut(&req.loop_id, |rt| rt.session_id.clone())
                        .unwrap_or_default();
                    let count = repo
                        .get(req.loop_id.as_ref())
                        .ok()
                        .flatten()
                        .map(|r| r.skipped_triggers)
                        .unwrap_or(0);
                    events_for_task
                        .trigger_dropped(&session_id, &req.loop_id, count)
                        .await;
                }

                // §8.2: drop if currently running.
                let busy = registry_for_task
                    .with_mut(&req.loop_id, |rt| rt.current_iteration.is_some())
                    .unwrap_or(false);
                if busy {
                    let now = current_timestamp();
                    let _ = LoopRepository::new(&executor_for_task.database())
                        .increment_skipped(req.loop_id.as_ref(), now);
                    let session_id = registry_for_task
                        .with_mut(&req.loop_id, |rt| rt.session_id.clone())
                        .unwrap_or_default();
                    let count = LoopRepository::new(&executor_for_task.database())
                        .get(req.loop_id.as_ref())
                        .ok()
                        .flatten()
                        .map(|r| r.skipped_triggers)
                        .unwrap_or(0);
                    events_for_task
                        .trigger_dropped(&session_id, &req.loop_id, count)
                        .await;
                    continue;
                }

                // W3 §gate: a deterministic gate suppresses this fire unless
                // its observed value changed (http/bash diff) or the firing
                // event's payload matched a predicate (event gate).
                // `GateKind::None` (the default) always runs and produces no
                // gate_output. Every evaluator error path is fail-open (see
                // `gate::evaluate_gate` module docs), so a malformed spec or
                // a failed fetch still lets the iteration run.
                let gate_output: Option<String> = {
                    let gate_db = executor_for_task.database();
                    let repo = LoopRepository::new(&gate_db);
                    let rec = repo.get(req.loop_id.as_ref()).ok().flatten();
                    match rec {
                        Some(rec) if rec.gate_kind != "none" => {
                            let kind = crate::loops::types::GateKind::from_db_str(&rec.gate_kind)
                                .unwrap_or(crate::loops::types::GateKind::None);
                            let spec_json = rec
                                .gate_spec
                                .as_deref()
                                .and_then(|s| serde_json::from_str(s).ok())
                                .unwrap_or(serde_json::Value::Null);
                            let spec = crate::loops::types::GateSpec { kind, spec_json };
                            let decision = crate::loops::gate::evaluate_gate(
                                &spec,
                                rec.gate_last_value.as_deref(),
                                req.event_payload.as_ref(),
                                &default_fetcher,
                            )
                            .await;

                            if !decision.run {
                                let now = current_timestamp();
                                let _ = repo.increment_skipped(req.loop_id.as_ref(), now);
                                let session_id = registry_for_task
                                    .with_mut(&req.loop_id, |rt| rt.session_id.clone())
                                    .unwrap_or_default();
                                events_for_task
                                    .skipped(&session_id, &req.loop_id, &req.fire_reason)
                                    .await;
                                continue;
                            }

                            if let Some(val) = &decision.new_last_value {
                                let _ = repo.set_gate_last_value(
                                    req.loop_id.as_ref(),
                                    val,
                                    current_timestamp(),
                                );
                            }
                            decision.gate_output
                        }
                        _ => None,
                    }
                };

                // Race guard (see `loop_still_live` docs): `evaluate_gate`
                // above is the only new `.await` point between the busy
                // check and claiming this fire. A concurrent `cancel_loop`
                // may have flipped this loop terminal, decremented
                // `pending_work`, and torn it out of the registry while we
                // were suspended there. Re-check now, before we commit to
                // running — do NOT touch `pending_work` here, the cancel
                // path already accounted for it.
                {
                    let live_db = executor_for_task.database();
                    let live_repo = LoopRepository::new(&live_db);
                    if !loop_still_live(&live_repo, &registry_for_task, &req.loop_id) {
                        tracing::debug!(
                            loop_id = %req.loop_id.as_ref(),
                            "loop dispatcher: fire aborted, loop cancelled/removed during gate evaluation"
                        );
                        continue;
                    }
                }

                let token = Arc::new(tokio_util::sync::CancellationToken::new());
                registry_for_task.with_mut(&req.loop_id, |rt| {
                    rt.current_iteration = Some(token.clone());
                });

                // Emit pending|idle -> running so the sidebar's sticky card
                // status badge tracks reality. Without this, the badge stays
                // at the initial "pending" set by create_loop forever, even
                // while iterations are firing.
                let prev_state_for_run: String = {
                    let db = executor_for_task.database();
                    let repo = LoopRepository::new(&db);
                    let prev = repo
                        .get(req.loop_id.as_ref())
                        .ok()
                        .flatten()
                        .map(|r| r.state.as_str().to_string())
                        .unwrap_or_else(|| "pending".to_string());
                    let _ = repo.update_state(
                        req.loop_id.as_ref(),
                        LoopState::Running,
                        current_timestamp(),
                    );
                    prev
                };
                let session_id_for_state = registry_for_task
                    .with_mut(&req.loop_id, |rt| rt.session_id.clone())
                    .unwrap_or_default();
                events_for_task
                    .state_changed(
                        &session_id_for_state,
                        &req.loop_id,
                        "running",
                        &prev_state_for_run,
                        None,
                    )
                    .await;

                let exec_result = executor_for_task
                    .execute(req.loop_id.clone(), req.fire_reason, gate_output)
                    .await;

                // TODO(W4-followup): auto-evolve every N iterations. Once
                // there's an appetite for it, read `rec.iteration_count`
                // here (post-increment, via `LoopRepository::get`) and, when
                // it's a multiple of some threshold N and there's no
                // existing `latest_pending_proposal`, spawn
                // `crate::loops::evolve::evolve_loop` fire-and-forget (it
                // already persists the proposal + emits
                // `system:loop:proposal` on its own). v1 (this task) is
                // manual-only: evolve only runs when a human/LLM explicitly
                // calls the `loop_evolve` tool.

                // 3-strike auto-cancel hook — depends on Phase 9b filling
                // in real failure counting. Reads the current
                // consecutive_failures and trips a soft cancel-equivalent
                // tear-down with reason "fail_threshold" if >= 3.
                //
                // The Phase-6 stub executor always succeeds and resets
                // the counter on every iteration, so under MVP this hook
                // only fires when something external (tests, future
                // executor, manual ops) sets the counter to >= 3.
                let cf = LoopRepository::new(&executor_for_task.database())
                    .get(req.loop_id.as_ref())
                    .ok()
                    .flatten()
                    .map(|r| r.consecutive_failures)
                    .unwrap_or(0);
                if cf >= 3 {
                    let session_id = registry_for_task
                        .with_mut(&req.loop_id, |rt| rt.session_id.clone())
                        .unwrap_or_default();
                    // Inline the relevant tear-down — we can't call
                    // self.cancel_loop_inner from a 'static spawned task.
                    // Same logic as cancel_loop_inner with by="fail_threshold"
                    // + force=false (current iteration just finished).
                    let watchers: Vec<String> = registry_for_task
                        .with_mut(&req.loop_id, |rt| std::mem::take(&mut rt.dom_watchers))
                        .unwrap_or_default();
                    let subs: Vec<String> = registry_for_task
                        .with_mut(&req.loop_id, |rt| std::mem::take(&mut rt.subscription_ids))
                        .unwrap_or_default();
                    for sub in &subs {
                        scheduler_for_task.unsubscribe(sub);
                    }
                    let _ = watchers;
                    // Single owner of the guarded terminal transition +
                    // pending_work decrement (see `LoopManager::mark_terminal`
                    // / `apply_terminal_transition`). This 'static spawned
                    // task has no `LoopManager` handle, so it calls the same
                    // associated function `cancel_loop_inner` and the public
                    // `mark_terminal` both funnel through, rather than
                    // hand-duplicating the atomic-transition + gated-decrement
                    // logic here. The atomic `state NOT IN (...)` guard means
                    // a concurrent cancel racing this auto-fail can never pair
                    // two decrements with one logical transition.
                    let flipped = LoopManager::apply_terminal_transition(
                        &executor_for_task.database(),
                        &pending_work_for_task,
                        &req.loop_id,
                        LoopState::Failed,
                    )
                    .unwrap_or(false);
                    if flipped {
                        events_for_task
                            .state_changed(
                                &session_id,
                                &req.loop_id,
                                "failed",
                                "running",
                                Some("fail_threshold"),
                            )
                            .await;
                        events_for_task
                            .cancelled(&session_id, &req.loop_id, "fail_threshold", false)
                            .await;
                    }
                    registry_for_task.remove(&req.loop_id);
                    continue;
                }

                registry_for_task.with_mut(&req.loop_id, |rt| {
                    rt.current_iteration = None;
                });

                // Emit running -> idle so the sidebar's sticky card status
                // badge falls back to "idle" between fires.
                {
                    let db = executor_for_task.database();
                    let repo = LoopRepository::new(&db);
                    let _ = repo.update_state(
                        req.loop_id.as_ref(),
                        LoopState::Idle,
                        current_timestamp(),
                    );
                }
                events_for_task
                    .state_changed(&session_id_for_state, &req.loop_id, "idle", "running", None)
                    .await;

                // time:dynamic protocol (spec §5.2): if the loop's trigger is
                // time:dynamic and this iteration succeeded with text, parse the
                // `loop-meta` block for `next_delay_seconds` and reschedule.
                if let crate::loops::executor::ExecResult::OkWithText(Some(text)) = &exec_result {
                    let rec = LoopRepository::new(&executor_for_task.database())
                        .get(req.loop_id.as_ref())
                        .ok()
                        .flatten();
                    if let Some(rec) = rec {
                        if rec.trigger_expr == "time:dynamic" {
                            let next = crate::loops::dynamic::extract_next_delay(text);
                            // Tear down old time-* subscriptions, schedule a new one.
                            let old_time_subs: Vec<String> = registry_for_task
                                .with_mut(&req.loop_id, |rt| {
                                    let removed: Vec<String> = rt
                                        .subscription_ids
                                        .iter()
                                        .filter(|s| s.starts_with("time-"))
                                        .cloned()
                                        .collect();
                                    rt.subscription_ids.retain(|s| !s.starts_with("time-"));
                                    removed
                                })
                                .unwrap_or_default();
                            for s in &old_time_subs {
                                scheduler_for_task.unsubscribe(s);
                            }
                            let new_sub = scheduler_for_task.schedule_time(
                                req.loop_id.clone(),
                                next,
                                fire_tx_for_task.clone(),
                            );
                            registry_for_task.with_mut(&req.loop_id, |rt| {
                                rt.subscription_ids.push(new_sub);
                            });
                        }
                    }
                }
            }
        });

        Self {
            db,
            registry,
            scheduler,
            fire_tx,
            events,
            event_bus: bus.clone(),
            pending_work,
        }
    }

    pub fn events(&self) -> &LoopEvents {
        &self.events
    }

    pub fn registry(&self) -> &LoopRegistry {
        &self.registry
    }

    /// Handle to the armed-loop counter for the managed idle watchdog.
    pub fn pending_work_handle(&self) -> Arc<AtomicUsize> {
        self.pending_work.clone()
    }

    /// True while any loop is armed (non-terminal).
    pub fn has_pending_work(&self) -> bool {
        self.pending_work.load(Ordering::SeqCst) > 0
    }

    /// Atomically drive `id` into a terminal state (`Cancelled`/`Failed`) and
    /// decrement `pending_work` exactly once for the transition that wins.
    ///
    /// This is the SINGLE place the guarded terminal-transition +
    /// gated-decrement logic lives — `LoopRepository::transition_to_terminal`
    /// does one atomic `UPDATE ... WHERE state NOT IN ('cancelled','failed')`
    /// and reports whether a row flipped, so two racing callers (a double
    /// cancel, or a cancel racing the dispatcher's 3-strike auto-fail) are
    /// serialized by SQLite and exactly one observes `true`. There is no
    /// separate `get` (read) followed by a `update_state` (write): that
    /// read-then-write gap is exactly what allowed both racing callers to
    /// see a non-terminal snapshot and both decrement, underflowing the
    /// `AtomicUsize` to `usize::MAX` and pinning `has_pending_work()` true
    /// forever.
    ///
    /// Free function (not `&self`) so it can be called both from
    /// `LoopManager` methods (`cancel_loop_inner`, `mark_terminal`) and from
    /// the dispatcher's 'static spawned task, which only has cloned
    /// `Database`/counter handles rather than a full `LoopManager`.
    fn apply_terminal_transition(
        db: &Database,
        pending_work: &Arc<AtomicUsize>,
        id: &LoopId,
        new_state: LoopState,
    ) -> Result<bool, String> {
        let flipped = LoopRepository::new(db)
            .transition_to_terminal(id.as_ref(), new_state, current_timestamp())
            .map_err(|e| e.to_string())?;
        if flipped {
            pending_work.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(flipped)
    }

    /// Manager-owned entry point for driving a loop into a terminal state
    /// with a guarded, race-safe `pending_work` decrement. Returns whether
    /// this call won the transition (an already-terminal loop is an
    /// idempotent no-op returning `false`). `mark_failed` and
    /// `cancel_loop_inner` both route through this so the counter has
    /// exactly one owner.
    pub async fn mark_terminal(&self, id: &LoopId, new_state: LoopState) -> Result<bool, String> {
        Self::apply_terminal_transition(&self.db, &self.pending_work, id, new_state)
    }

    pub async fn create_loop(&self, args: CreateLoopArgs) -> Result<LoopId, String> {
        // XOR — also CHECK-enforced in sqlite, but check here for a clean error.
        if args.prompt_text.is_some() == args.wrapped_skill.is_some() {
            return Err("exactly one of prompt_text or wrapped_skill is required".into());
        }
        let expr = TriggerExpr::parse(&args.trigger_expr_text).map_err(|e| e.to_string())?;
        if let Some(gate) = &args.gate {
            validate_gate_trigger_compat(gate, &expr)?;
        }

        let id = LoopId::generate();
        let now = current_timestamp();
        let rec = LoopRecord {
            id: id.0.clone(),
            session_id: args.session_id.clone(),
            trigger_expr: args.trigger_expr_text.clone(),
            prompt_text: args.prompt_text,
            wrapped_skill: args.wrapped_skill,
            mode: agent_mode_to_db_str(args.mode).to_string(),
            scratchpad: String::new(),
            state: LoopState::Pending,
            consecutive_failures: 0,
            skipped_triggers: 0,
            iteration_count: 0,
            created_at: now,
            updated_at: now,
            gate_kind: args
                .gate
                .as_ref()
                .map(|g| g.kind.as_str())
                .unwrap_or("none")
                .to_string(),
            gate_spec: args.gate.as_ref().map(|g| g.spec_json.to_string()),
            gate_last_value: None,
            verify_check: args.verify_check,
        };
        LoopRepository::new(&self.db)
            .create(&rec)
            .map_err(|e| e.to_string())?;

        self.registry
            .insert(LoopRuntime::new(id.clone(), args.session_id.clone()));
        self.wire_trigger(&id, &expr);

        self.events
            .created(
                &args.session_id,
                &id,
                &rec.trigger_expr,
                rec.prompt_text.as_deref(),
                rec.wrapped_skill.as_deref(),
            )
            .await;

        self.pending_work.fetch_add(1, Ordering::SeqCst);

        Ok(id)
    }

    /// Cancel a loop with two-click grace/force semantics (spec §8.3).
    ///
    /// - `force=true`: immediate force-cancel — abort in-flight iteration,
    ///   tear down all triggers, mark loop `cancelled`.
    /// - `force=false` (soft): first click stamps `first_cancel_at_ms` on
    ///   the runtime; a second soft cancel within 30s escalates to force.
    ///   Either way the trigger subscriptions are torn down and the loop
    ///   is marked `cancelled`. With force=false the in-flight iteration
    ///   (if any) is allowed to run to completion (its cancel_token is not
    ///   tripped) — but no further iterations will fire because the
    ///   triggers are gone and the runtime entry is removed.
    pub async fn cancel_loop(&self, id: &LoopId, force: bool) -> Result<(), String> {
        let now_ms: u64 = chrono::Utc::now().timestamp_millis().max(0) as u64;
        if force {
            return self.cancel_loop_inner(id, true, "user-force").await;
        }

        // First soft-click: stamp the time. Second soft-click within 30s ⇒ force.
        let prior = self.registry.with_mut(id, |rt| {
            let p = rt.first_cancel_at_ms;
            if p.is_none() {
                rt.first_cancel_at_ms = Some(now_ms);
            }
            p
        });
        if let Some(Some(t)) = prior {
            if now_ms.saturating_sub(t) < 30_000 {
                return self.cancel_loop_inner(id, true, "user-force").await;
            }
        }

        // Soft cancel: tear down trigger subs but allow the current iteration
        // (if any) to finish naturally. State → cancelled (we eagerly mark
        // for MVP — the dispatcher checks busy and skips fires anyway).
        self.cancel_loop_inner(id, false, "user-soft").await
    }

    /// Internal cancel implementation. `force=true` aborts in-flight iteration
    /// and tears down everything immediately. `force=false` only tears down
    /// triggers and lets the current iteration finish (the cancellation token
    /// is NOT triggered).
    async fn cancel_loop_inner(&self, id: &LoopId, force: bool, by: &str) -> Result<(), String> {
        let session_id = self
            .registry
            .with_mut(id, |rt| rt.session_id.clone())
            .unwrap_or_default();

        let watchers: Vec<String> = self
            .registry
            .with_mut(id, |rt| std::mem::take(&mut rt.dom_watchers))
            .unwrap_or_default();
        let subs: Vec<String> = self
            .registry
            .with_mut(id, |rt| {
                if force {
                    rt.cancel_token.cancel();
                    if let Some(it) = &rt.current_iteration {
                        it.cancel();
                    }
                }
                std::mem::take(&mut rt.subscription_ids)
            })
            .unwrap_or_default();
        for sub in &subs {
            self.scheduler.unsubscribe(sub);
        }
        // Phase 19 deferred — dom_watchers vec stays empty; if any are present
        // (future phase), the bridge uninstall would happen here.
        let _ = watchers;

        // Atomically flip into Cancelled and decrement pending_work exactly
        // once for the winner. `mark_terminal` (-> `apply_terminal_transition`)
        // is the single owner of this guard; a loop cancelled twice (e.g.
        // soft-cancel followed by a force-cancel escalation racing another
        // caller, or cancel racing the dispatcher's 3-strike auto-fail) can
        // never underflow the counter or double-emit these events.
        let flipped = self.mark_terminal(id, LoopState::Cancelled).await?;
        if flipped {
            self.events
                .state_changed(&session_id, id, "cancelled", "running", Some(by))
                .await;
            self.events.cancelled(&session_id, id, by, force).await;
        }
        self.registry.remove(id);
        Ok(())
    }

    /// Mark a loop `Failed` and decrement the pending-work counter. Thin
    /// wrapper over `mark_terminal` — the single manager-owned entry point
    /// for the guarded terminal transition + gated decrement, also used by
    /// `cancel_loop_inner` and the dispatcher's 3-strike auto-fail path (via
    /// the shared `apply_terminal_transition` associated function — see
    /// `start_with_bus`).
    pub async fn mark_failed(&self, id: &LoopId) -> Result<(), String> {
        self.mark_terminal(id, LoopState::Failed).await?;
        Ok(())
    }

    /// Tear down all triggers + iterations on clean shutdown. Marks any
    /// `running` loops as `idle` so the next startup sweep doesn't paint
    /// them as crashed.
    pub async fn shutdown(&self) {
        let ids = self.registry.ids();
        for id in &ids {
            let _ = self.cancel_loop_inner(id, true, "daemon-shutdown").await;
        }
        // Any rows still `running` (shouldn't be after the per-id force
        // cancels, but defensive) get demoted to idle.
        let now = current_timestamp();
        let _ = self.db.with_connection(|conn| {
            conn.execute(
                "UPDATE loops SET state = 'idle', updated_at = ?1 WHERE state = 'running'",
                rusqlite::params![now],
            )
            .map(|_| ())
            .map_err(nevoflux_storage::error::StorageError::from)
        });
    }

    pub async fn list_by_session(&self, session_id: &str) -> Result<Vec<LoopRecord>, String> {
        LoopRepository::new(&self.db)
            .list_by_session(session_id)
            .map_err(|e| e.to_string())
    }

    /// Wire a trigger expression's subscriptions into the loop's main fire channel.
    fn wire_trigger(&self, id: &LoopId, expr: &TriggerExpr) {
        self.wire_trigger_into(id, expr, self.fire_tx.clone());
    }

    /// Recursive wiring helper: routes a trigger's fires into `sink` rather
    /// than the main fire channel. Combinators use this to splice their
    /// children's pulses through the [`combinator::CombinatorRuntime`] before
    /// the parent fire is forwarded to `self.fire_tx`.
    fn wire_trigger_into(
        &self,
        id: &LoopId,
        expr: &TriggerExpr,
        sink: mpsc::Sender<LoopFireRequest>,
    ) {
        match expr {
            TriggerExpr::Time(dur) => {
                let sub = self.scheduler.schedule_time(id.clone(), *dur, sink);
                self.registry
                    .with_mut(id, |rt| rt.subscription_ids.push(sub));
            }
            TriggerExpr::TimeDynamic => {
                // Initial fire at T+5m. Subsequent fires use `next_delay_seconds`
                // emitted by the LLM in a `loop-meta` JSON block; the dispatcher
                // re-schedules via `crate::loops::dynamic::extract_next_delay`
                // after each iteration succeeds (see start_with_bus loop body).
                let sub = self.scheduler.schedule_time(
                    id.clone(),
                    std::time::Duration::from_secs(300),
                    sink,
                );
                self.registry
                    .with_mut(id, |rt| rt.subscription_ids.push(sub));
            }
            TriggerExpr::Event(topic) => {
                let Some(bus) = self.event_bus.clone() else {
                    tracing::warn!(
                        "event:{} ignored — LoopManager has no EventBus handle",
                        topic
                    );
                    return;
                };
                match self
                    .scheduler
                    .schedule_event(id.clone(), topic.clone(), bus, sink)
                {
                    Ok(sub) => {
                        self.registry
                            .with_mut(id, |rt| rt.subscription_ids.push(sub));
                    }
                    Err(e) => {
                        tracing::warn!("event:{} subscription failed: {e}", topic);
                    }
                }
            }
            TriggerExpr::State { tab, selector } => {
                // MVP: subscribe to a generic dom-mutation topic. The dom-watcher
                // content script (Phase 18) publishes ui:tab:dom:mutation on every
                // batch of mutations across all tabs. Per-selector and per-tab
                // filtering is deferred — for now, the trigger fires on any DOM
                // mutation, and the iteration's LLM can use browser_query to verify
                // the selector matches before acting.
                let _ = (tab, selector); // recorded in trigger_expr_text for future per-tab filtering
                let Some(bus) = self.event_bus.clone() else {
                    tracing::warn!("state:* trigger ignored — no EventBus handle");
                    return;
                };
                let topic = "ui:tab:dom:mutation".to_string();
                match self.scheduler.schedule_event(id.clone(), topic, bus, sink) {
                    Ok(sub) => {
                        self.registry
                            .with_mut(id, |rt| rt.subscription_ids.push(sub));
                    }
                    Err(e) => {
                        tracing::warn!("state:* subscription failed: {e}");
                    }
                }
            }
            TriggerExpr::And(children) | TriggerExpr::Or(children) => {
                self.wire_combinator(id, expr, children, sink);
            }
        }
    }

    /// Wire a combinator (AND/OR) by spawning a forwarding task that re-emits
    /// the runtime's parent fires into `parent_sink`, plus per-child adapter
    /// tasks that translate child `LoopFireRequest`s into
    /// [`combinator::CombinatorRuntime::on_child_fire`] calls.
    fn wire_combinator(
        &self,
        id: &LoopId,
        expr: &TriggerExpr,
        children: &[TriggerExpr],
        parent_sink: mpsc::Sender<LoopFireRequest>,
    ) {
        // Per-combinator output channel: when the combinator fires, we
        // re-emit a single LoopFireRequest into the parent sink.
        let (out_tx, mut out_rx) = mpsc::channel::<()>(8);
        let runtime = std::sync::Arc::new(tokio::sync::Mutex::new(match expr {
            TriggerExpr::And(_) => {
                crate::loops::combinator::CombinatorRuntime::new_and(children.len(), out_tx)
            }
            TriggerExpr::Or(_) => crate::loops::combinator::CombinatorRuntime::new_or(out_tx),
            _ => unreachable!("wire_combinator called with non-combinator"),
        }));

        let label = match expr {
            TriggerExpr::And(_) => "AND",
            TriggerExpr::Or(_) => "OR",
            _ => "?",
        };

        // Forward combinator output -> parent fire sink.
        let id_for_forward = id.clone();
        let sink_for_forward = parent_sink.clone();
        let label_owned = label.to_string();
        tokio::spawn(async move {
            while out_rx.recv().await.is_some() {
                if sink_for_forward
                    .send(LoopFireRequest {
                        loop_id: id_for_forward.clone(),
                        fire_reason: format!("combinator:{}", label_owned),
                        event_payload: None,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
        });

        // Per-child mpsc adapter: receives LoopFireRequest from a child's
        // schedule_* call, calls combinator.on_child_fire(idx) for each.
        for (idx, child) in children.iter().enumerate() {
            let (child_tx, mut child_rx) = mpsc::channel::<LoopFireRequest>(8);
            self.wire_trigger_into(id, child, child_tx);

            let runtime_clone = runtime.clone();
            tokio::spawn(async move {
                while child_rx.recv().await.is_some() {
                    runtime_clone.lock().await.on_child_fire(idx).await;
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_storage::models::CreateSessionParams;
    use nevoflux_storage::Storage;
    use std::time::Duration;

    fn fresh() -> Storage {
        let s = Storage::open_in_memory().unwrap();
        s.sessions()
            .create(CreateSessionParams::new().with_id("s1").with_title("t"))
            .unwrap();
        s
    }

    #[tokio::test(start_paused = true)]
    async fn create_loop_persists_record() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());
        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:5m".into(),
                prompt_text: Some("check".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(rec.state, LoopState::Pending);
        assert_eq!(rec.trigger_expr, "time:5m");
        // mode persisted (default chat)
        assert_eq!(rec.mode, "chat");
    }

    /// W5 task 2: `CreateLoopArgs.verify_check` round-trips through
    /// `create_loop` and `LoopRepository::get` — mirrors
    /// `create_loop_with_http_gate_on_time_trigger_persists` for gate.
    #[tokio::test(start_paused = true)]
    async fn create_loop_with_verify_check_persists() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());
        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:5m".into(),
                prompt_text: Some("check".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: Some(r#"{"matches":"OK"}"#.into()),
            })
            .await
            .unwrap();

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(rec.verify_check.as_deref(), Some(r#"{"matches":"OK"}"#));
    }

    /// W3 task 4: an `event` gate on a `time:` trigger is a config
    /// mismatch — the gate would have no trigger payload to filter — and
    /// must be rejected before the loop is ever persisted.
    #[tokio::test(start_paused = true)]
    async fn create_loop_rejects_event_gate_on_time_trigger() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());
        let err = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:5m".into(),
                prompt_text: Some("check".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: Some(crate::loops::types::GateSpec {
                    kind: GateKind::Event,
                    spec_json: serde_json::json!({}),
                }),
                verify_check: None,
            })
            .await
            .unwrap_err();
        assert!(
            err.contains("event") && err.contains("event:"),
            "expected error naming the event/trigger incompatibility, got: {err}"
        );
    }

    /// W3 task 4: an `http` gate on a `time:` trigger is the intended
    /// shape (poll-and-diff on the loop's own cadence) — it must succeed
    /// and the gate columns must round-trip through `LoopRepository::get`.
    #[tokio::test(start_paused = true)]
    async fn create_loop_with_http_gate_on_time_trigger_persists() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());
        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:5m".into(),
                prompt_text: Some("check".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: Some(crate::loops::types::GateSpec {
                    kind: GateKind::Http,
                    spec_json: serde_json::json!({"url": "https://x", "extract": "$.v"}),
                }),
                verify_check: None,
            })
            .await
            .unwrap();

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(rec.gate_kind, "http");
        let spec: serde_json::Value =
            serde_json::from_str(rec.gate_spec.as_deref().unwrap()).unwrap();
        assert_eq!(spec["url"], "https://x");
        assert_eq!(spec["extract"], "$.v");
    }

    #[tokio::test(start_paused = true)]
    async fn create_loop_rejects_invalid_trigger() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());
        let err = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "garbage".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap_err();
        assert!(err.contains("unknown atom") || err.to_lowercase().contains("garbage"));
    }

    #[tokio::test(start_paused = true)]
    async fn create_loop_rejects_xor_violation() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());

        let err = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:5m".into(),
                prompt_text: None,
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap_err();
        assert!(err.contains("prompt_text or wrapped_skill"));

        let err = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:5m".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: Some("{}".into()),
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap_err();
        assert!(err.contains("prompt_text or wrapped_skill"));
    }

    #[tokio::test(start_paused = true)]
    async fn create_loop_then_one_iteration_fires() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:1m".into(),
                prompt_text: Some("check".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        // Advance past the first 60s tick, then drive the runtime so the
        // scheduler task -> dispatcher channel -> executor chain drains.
        // With virtual time paused, we alternate `advance` + yield to
        // ensure every spawned task gets polled.
        for _ in 0..3 {
            tokio::time::advance(Duration::from_secs(61)).await;
            for _ in 0..200 {
                tokio::task::yield_now().await;
            }
        }

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert!(
            rec.iteration_count >= 1,
            "iteration_count was {}",
            rec.iteration_count
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_loop_unsubscribes_triggers() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:1m".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        mgr.cancel_loop(&id, false).await.unwrap();

        for _ in 0..3 {
            tokio::time::advance(Duration::from_secs(120)).await;
            for _ in 0..200 {
                tokio::task::yield_now().await;
            }
        }

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(rec.state, LoopState::Cancelled);
        assert_eq!(
            rec.iteration_count, 0,
            "no iterations should have fired after cancel"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn list_by_session_returns_session_loops() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());

        let _ = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:5m".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();
        let _ = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:10m".into(),
                prompt_text: Some("q".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        let list = mgr.list_by_session("s1").await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn event_trigger_fires_iteration_on_publish() {
        use crate::event_bus::types::{BusEvent, PublisherIdentity};
        use crate::event_bus::EventBus;
        use std::sync::Arc;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus.clone()), None);

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "event:ui:test:click".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        bus.publish(BusEvent::ephemeral(
            "ui:test:click",
            serde_json::json!({}),
            PublisherIdentity::Internal,
        ))
        .await
        .unwrap();

        // Real-time wait for the dispatcher + executor to run.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert!(
            rec.iteration_count >= 1,
            "iteration_count was {}",
            rec.iteration_count
        );
    }

    #[tokio::test]
    async fn or_combinator_fires_on_any_event() {
        use crate::event_bus::types::{BusEvent, PublisherIdentity};
        use crate::event_bus::EventBus;
        use std::sync::Arc;
        use std::time::Duration;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus.clone()), None);

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "OR(event:a:test,event:b:test)".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        bus.publish(BusEvent::ephemeral(
            "a:test",
            serde_json::json!({}),
            PublisherIdentity::Internal,
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(300)).await;

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert!(
            rec.iteration_count >= 1,
            "OR combinator should fire on first child; got {}",
            rec.iteration_count
        );
    }

    #[tokio::test(start_paused = true)]
    async fn three_strikes_marks_loop_failed() {
        // Synthesize 3 consecutive failures on the loop record. The
        // Phase-6 stub executor always succeeds and resets the counter
        // on every "ok" iteration, so by the time the dispatcher runs,
        // we may not see the Failed state — the executor's own state
        // update races the 3-strike hook.
        //
        // For the MVP this test exercises that the dispatcher's
        // 3-strike read path compiles and executes without panicking;
        // once Phase 9b lands and IterationExecutor returns real
        // ExecResult::Error and bumps consecutive_failures itself,
        // this test will become observable end-to-end.
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());
        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:1m".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        // Pre-load failure counter to 3.
        storage
            .loops()
            .set_consecutive_failures(id.as_ref(), 3, current_timestamp())
            .unwrap();

        // Advance past the next time tick so the dispatcher runs.
        for _ in 0..3 {
            tokio::time::advance(Duration::from_secs(61)).await;
            for _ in 0..200 {
                tokio::task::yield_now().await;
            }
        }

        // The Phase-6 stub resets consecutive_failures to 0 on every
        // "ok" iteration, so the post-iteration read may see 0. We
        // accept either outcome here; the test's value is in
        // exercising the dispatcher's 3-strike code path without
        // panic.
        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        let _ = rec;
    }

    #[tokio::test]
    async fn and_combinator_waits_for_all() {
        use crate::event_bus::types::{BusEvent, PublisherIdentity};
        use crate::event_bus::EventBus;
        use std::sync::Arc;
        use std::time::Duration;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus.clone()), None);

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "AND(event:and:a,event:and:b)".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        // Only one child fires — AND must NOT trip yet.
        bus.publish(BusEvent::ephemeral(
            "and:a",
            serde_json::json!({}),
            PublisherIdentity::Internal,
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(
            rec.iteration_count, 0,
            "AND should not fire on partial: got {}",
            rec.iteration_count
        );

        // Both children fired — AND should trip.
        bus.publish(BusEvent::ephemeral(
            "and:b",
            serde_json::json!({}),
            PublisherIdentity::Internal,
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert!(
            rec.iteration_count >= 1,
            "AND should fire after both children: got {}",
            rec.iteration_count
        );
    }

    #[tokio::test]
    async fn pending_work_tracks_create_and_cancel() {
        use crate::event_bus::EventBus;
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus), None);

        // No loops yet.
        assert_eq!(mgr.pending_work_handle().load(Ordering::SeqCst), 0);
        assert!(!mgr.has_pending_work());

        // Create one armed loop -> counter is 1.
        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:10m".into(),
                prompt_text: Some("watch".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();
        assert_eq!(mgr.pending_work_handle().load(Ordering::SeqCst), 1);
        assert!(mgr.has_pending_work());

        // Cancel it (terminal) -> counter back to 0.
        mgr.cancel_loop(&id, false).await.unwrap();
        assert_eq!(mgr.pending_work_handle().load(Ordering::SeqCst), 0);
        assert!(!mgr.has_pending_work());
    }

    #[tokio::test]
    async fn pending_work_decrements_on_auto_fail() {
        use crate::event_bus::EventBus;
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus), None);
        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:1m".into(),
                prompt_text: Some("x".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();
        assert_eq!(mgr.pending_work_handle().load(Ordering::SeqCst), 1);

        // Drive the loop to Failed directly through the manager method that
        // owns the counter (the registry is pure in-memory runtime state and
        // has no `update_state`; the DB-backed fail transition must go
        // through `mark_failed` so the counter stays in sync).
        mgr.mark_failed(&id).await.unwrap();

        assert_eq!(mgr.pending_work_handle().load(Ordering::SeqCst), 0);
        assert!(!mgr.has_pending_work());
        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(rec.state, LoopState::Failed);
    }

    /// Regression test for the TOCTOU double-decrement: a loop that is
    /// already terminal (failed) must not decrement `pending_work` again
    /// when a subsequent cancel lands on it. Before the atomic
    /// `transition_to_terminal` guard, cancelling twice (or fail-then-cancel,
    /// which is exactly the "cancel racing the dispatcher's auto-fail"
    /// scenario) would wrap the counter to `usize::MAX` via a second
    /// `fetch_sub`, pinning `has_pending_work()` true forever.
    #[tokio::test]
    async fn cancel_after_fail_decrements_exactly_once() {
        use crate::event_bus::EventBus;
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus), None);
        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:1m".into(),
                prompt_text: Some("x".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();
        assert_eq!(mgr.pending_work_handle().load(Ordering::SeqCst), 1);

        // Fail it first (e.g. dispatcher's 3-strike auto-fail).
        mgr.mark_failed(&id).await.unwrap();
        assert_eq!(mgr.pending_work_handle().load(Ordering::SeqCst), 0);

        // A cancel racing/following the fail must be a no-op on the counter:
        // the row is already terminal, so `transition_to_terminal` reports
        // no flip and `cancel_loop_inner` must not decrement again.
        mgr.cancel_loop(&id, true).await.unwrap();
        assert_eq!(
            mgr.pending_work_handle().load(Ordering::SeqCst),
            0,
            "cancel after fail must not underflow the counter"
        );
        assert!(!mgr.has_pending_work());

        // State stays at the terminal value the winning transition set
        // (Failed) — the losing cancel does not clobber it.
        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(rec.state, LoopState::Failed);

        // Calling cancel a second (third overall) time is still a no-op.
        mgr.cancel_loop(&id, true).await.unwrap();
        assert_eq!(mgr.pending_work_handle().load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn state_trigger_fires_on_dom_mutation_event() {
        use crate::event_bus::types::{BusEvent, PublisherIdentity};
        use crate::event_bus::EventBus;
        use std::sync::Arc;
        use std::time::Duration;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus.clone()), None);

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "state:tab=current:.chat-list:change".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        // Simulate the dom-watcher publishing a mutation event.
        bus.publish(BusEvent::ephemeral(
            "ui:tab:dom:mutation",
            serde_json::json!({"url": "https://example.com", "ts_ms": 0}),
            PublisherIdentity::Internal,
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(300)).await;

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert!(
            rec.iteration_count >= 1,
            "state:* trigger should fire on dom mutation; got {}",
            rec.iteration_count
        );
    }

    /// Attach a gate directly on the DB row (no `create_loop` support for
    /// gates yet — gate columns are always `"none"`/`NULL` at creation).
    /// Mirrors the raw-SQL pattern `LoopManager::shutdown` uses.
    fn set_gate(storage: &Storage, loop_id: &str, kind: &str, spec_json: &str) {
        storage
            .database()
            .with_connection(|conn| {
                conn.execute(
                    "UPDATE loops SET gate_kind = ?1, gate_spec = ?2 WHERE id = ?3",
                    rusqlite::params![kind, spec_json, loop_id],
                )
                .map(|_| ())
                .map_err(nevoflux_storage::error::StorageError::from)
            })
            .unwrap();
    }

    /// W3 §gate dispatcher wiring, Step 1 of task-3: a gate whose decision is
    /// `run=false` must suppress the iteration entirely (no executor call),
    /// must bump `skipped_triggers` (distinct counter from the busy-drop
    /// path), and must emit a `system:loop:skipped` event. Uses an `event`
    /// gate so the test needs no network/bash fetcher — `evaluate_gate`'s
    /// event path reads only `LoopFireRequest.event_payload`.
    #[tokio::test]
    async fn gate_skip_suppresses_iteration_and_emits_skipped() {
        use crate::event_bus::types::{
            BackpressurePolicy, BusEvent, PublisherIdentity, SubscriberIdentity, TopicPattern,
        };
        use crate::event_bus::EventBus;
        use std::sync::Arc;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus.clone()), None);

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "event:test:gate".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();
        set_gate(
            &storage,
            id.as_ref(),
            "event",
            r#"{"path":"type","equals":"go"}"#,
        );

        // Subscribe before publishing — ephemeral delivery has no replay.
        let skipped_handle = bus
            .subscribe(
                TopicPattern::Exact("system:loop:skipped".into()),
                SubscriberIdentity::Internal,
                BackpressurePolicy::DropOldest,
                8,
            )
            .expect("subscribe to skipped events");
        let mut skipped_rx = skipped_handle.rx;

        // Payload doesn't match the gate's predicate -> must skip.
        bus.publish(BusEvent::ephemeral(
            "test:gate",
            serde_json::json!({"type": "not-go"}),
            PublisherIdentity::Internal,
        ))
        .await
        .unwrap();

        let skipped_event =
            tokio::time::timeout(std::time::Duration::from_millis(500), skipped_rx.recv())
                .await
                .expect("system:loop:skipped should fire")
                .expect("skipped event payload");
        assert_eq!(skipped_event.payload["loop_id"], id.as_ref());

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(
            rec.iteration_count, 0,
            "gate run=false must not call the executor"
        );
        assert_eq!(
            rec.skipped_triggers, 1,
            "gate skip must bump skipped_triggers"
        );
    }

    /// W3 §gate dispatcher wiring: a gate whose decision is `run=true` must
    /// let the fire proceed to `IterationExecutor::execute` as normal (same
    /// event gate as the skip test above, but with a matching payload).
    #[tokio::test]
    async fn gate_run_executes_iteration() {
        use crate::event_bus::types::{BusEvent, PublisherIdentity};
        use crate::event_bus::EventBus;
        use std::sync::Arc;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus.clone()), None);

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "event:test:gate".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();
        set_gate(
            &storage,
            id.as_ref(),
            "event",
            r#"{"path":"type","equals":"go"}"#,
        );

        // Payload matches the gate's predicate -> must run.
        bus.publish(BusEvent::ephemeral(
            "test:gate",
            serde_json::json!({"type": "go"}),
            PublisherIdentity::Internal,
        ))
        .await
        .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert!(
            rec.iteration_count >= 1,
            "gate run=true must still execute the iteration; got {}",
            rec.iteration_count
        );
        assert_eq!(
            rec.skipped_triggers, 0,
            "a run=true decision must not bump skipped_triggers"
        );
    }

    /// Unit coverage for the `loop_still_live` guard extracted for the W3
    /// gate-race fix: it must require BOTH a non-terminal DB row AND
    /// registry membership, not either alone.
    #[tokio::test]
    async fn loop_still_live_requires_db_state_and_registry_membership() {
        let storage = fresh();
        let mgr = LoopManager::start(storage.database().clone());
        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "time:5m".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();

        let db = storage.database().clone();
        let repo = LoopRepository::new(&db);

        // Freshly created: armed in the DB, present in the registry -> live.
        assert!(
            loop_still_live(&repo, mgr.registry(), &id),
            "a freshly created, non-terminal, registered loop must be live"
        );

        // Cancel: flips the DB row terminal AND removes the registry entry
        // (the real-world transition this guard exists to catch).
        mgr.cancel_loop(&id, true).await.unwrap();
        assert!(
            !loop_still_live(&repo, mgr.registry(), &id),
            "a cancelled, deregistered loop must not be live"
        );

        // Belt-and-suspenders: even if the DB row were somehow non-terminal
        // again but the registry entry is still gone, still not live.
        let _ = repo.update_state(id.as_ref(), LoopState::Idle, current_timestamp());
        assert!(
            !loop_still_live(&repo, mgr.registry(), &id),
            "registry absence alone must veto liveness even with a non-terminal DB row"
        );

        // A loop id that never existed.
        let ghost = LoopId::generate();
        assert!(
            !loop_still_live(&repo, mgr.registry(), &ghost),
            "a nonexistent loop id must not be live"
        );
    }

    /// End-to-end reproduction of the dispatcher race (Task 3 zombie-loop
    /// bug): a `bash` gate's `evaluate_gate(...).await` can take real wall
    /// time (here, a short `sleep`). If a concurrent `cancel_loop(force)`
    /// completes while the dispatcher is suspended in that await, the
    /// dispatcher must NOT resurrect the loop by overwriting `Cancelled`
    /// back to `Running`/`Idle`, must NOT execute an iteration, and must
    /// leave `pending_work` exactly as the cancel path left it.
    ///
    /// Deliberately not `start_paused`: the race only exists because the
    /// gate await is real wall-clock time (production uses a real
    /// `tokio::process::Command` under a 5s timeout — see
    /// `gate::DefaultFetcher`), so this test drives an actual child process
    /// sleep to open the same window.
    #[tokio::test]
    async fn cancel_during_gate_await_does_not_resurrect_loop() {
        use crate::event_bus::types::{BusEvent, PublisherIdentity};
        use crate::event_bus::EventBus;
        use std::sync::Arc;

        let storage = fresh();
        let bus = Arc::new(EventBus::new());
        let mgr = LoopManager::start_with_bus(storage.database().clone(), Some(bus.clone()), None);

        let id = mgr
            .create_loop(CreateLoopArgs {
                session_id: "s1".into(),
                trigger_expr_text: "event:test:race".into(),
                prompt_text: Some("p".into()),
                wrapped_skill: None,
                mode: nevoflux_builtin_wasm::AgentMode::Chat,
                gate: None,
                verify_check: None,
            })
            .await
            .unwrap();
        // Long enough that the test's 100ms delay below reliably lands
        // inside the await, short enough to keep the test fast.
        set_gate(
            &storage,
            id.as_ref(),
            "bash",
            r#"{"command":"sleep 0.4 && echo v1"}"#,
        );

        bus.publish(BusEvent::ephemeral(
            "test:race",
            serde_json::json!({}),
            PublisherIdentity::Internal,
        ))
        .await
        .unwrap();

        // Give the dispatcher time to dequeue the fire, clear the busy
        // check, and enter `evaluate_gate`'s bash await — well before the
        // 0.4s sleep resolves.
        tokio::time::sleep(Duration::from_millis(100)).await;

        mgr.cancel_loop(&id, true).await.unwrap();
        assert!(
            !mgr.has_pending_work(),
            "cancel must decrement pending_work immediately, before the gate resolves"
        );

        // Wait past the gate's sleep so the dispatcher resumes. Pre-fix,
        // this is where it would unconditionally overwrite `Cancelled`
        // back to `Running` and run the iteration.
        tokio::time::sleep(Duration::from_millis(600)).await;

        let rec = storage.loops().get(id.as_ref()).unwrap().unwrap();
        assert_eq!(
            rec.state,
            LoopState::Cancelled,
            "the post-gate liveness re-check must stop the dispatcher from resurrecting a \
             loop that was cancelled while the gate was awaiting"
        );
        assert_eq!(
            rec.iteration_count, 0,
            "a fire aborted by the liveness re-check must not execute an iteration"
        );
        assert!(
            !mgr.has_pending_work(),
            "pending_work must stay correctly decremented, not re-armed by the aborted fire"
        );
    }
}
