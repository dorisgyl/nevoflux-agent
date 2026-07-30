//! In-daemon task queue (P4). Accepts `TaskRequest`s, runs them one at a time
//! (the headless model has exactly one browser), and tracks per-task status.
//!
//! Execution is provided as a `Runner` so the queue is testable without the
//! agent loop or a browser; the automation session runner (P3) plugs in here
//! in production.

use crate::http::types::{TaskRequest, TaskResponse, TaskStatus};
use crate::script_backend::DeltaSink;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Runs one task to a terminal [`TaskResponse`]. Implemented by the automation
/// session runner (P3); mocked in tests.
///
/// `sink` 是可选的增量出口：流式请求由 HTTP 层建通道传入，非流式请求同样
/// 传入（只是在进程内收集），因为结构化结果（`tool_calls` / `usage`）无法从
/// `TaskResponse.output` 这个字符串通道回传。
pub type Runner = Arc<
    dyn Fn(String, TaskRequest, Option<DeltaSink>) -> BoxFuture<'static, TaskResponse>
        + Send
        + Sync,
>;

/// Accepts and tracks tasks.
pub struct TaskQueue {
    statuses: Arc<RwLock<HashMap<String, TaskResponse>>>,
    runner: Runner,
    seq: AtomicU64,
}

fn queued(id: &str) -> TaskResponse {
    TaskResponse {
        id: id.to_string(),
        status: TaskStatus::Queued,
        attempts: 0,
        output: None,
        error: None,
        artifacts: Vec::new(),
    }
}

impl TaskQueue {
    /// Create a queue backed by `runner`.
    pub fn new(runner: Runner) -> Self {
        Self {
            statuses: Arc::new(RwLock::new(HashMap::new())),
            runner,
            seq: AtomicU64::new(0),
        }
    }

    /// Submit a task; returns its id immediately (status `Queued`).
    pub fn submit(&self, req: TaskRequest) -> String {
        self.submit_with(req, None)
    }

    /// 同 [`Self::submit`]，但把增量出口交给 runner。
    pub fn submit_streaming(&self, req: TaskRequest, sink: DeltaSink) -> String {
        self.submit_with(req, Some(sink))
    }

    fn submit_with(&self, req: TaskRequest, sink: Option<DeltaSink>) -> String {
        let id = format!("task-{}", self.seq.fetch_add(1, Ordering::Relaxed));
        self.statuses
            .write()
            .unwrap()
            .insert(id.clone(), queued(&id));

        let runner = self.runner.clone();
        let statuses = self.statuses.clone();
        let run_id = id.clone();
        tokio::spawn(async move {
            if let Some(r) = statuses.write().unwrap().get_mut(&run_id) {
                r.status = TaskStatus::Running;
            }
            let resp = runner(run_id.clone(), req, sink).await;
            statuses.write().unwrap().insert(run_id, resp);
        });
        id
    }

    /// Current status snapshot for `id`.
    pub fn status(&self, id: &str) -> Option<TaskResponse> {
        self.statuses.read().unwrap().get(id).cloned()
    }

    /// Submit `req` and poll until it reaches a terminal status (or `timeout`).
    /// Used by the synchronous front-ends (OpenAI-compatible / MCP). On timeout
    /// returns the last-known snapshot (still `Running`).
    pub async fn submit_and_wait(
        &self,
        req: TaskRequest,
        timeout: std::time::Duration,
    ) -> TaskResponse {
        let id = self.submit(req);
        let start = std::time::Instant::now();
        loop {
            if let Some(r) = self.status(&id) {
                if matches!(r.status, TaskStatus::Succeeded | TaskStatus::Failed) {
                    return r;
                }
                if start.elapsed() >= timeout {
                    return r;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    }

    /// Request cancellation. Marks a `Queued`/`Running` task `Failed` and returns
    /// `true` if the id exists. (Cooperative interrupt of a *running* attempt is
    /// delivered by the session runner, P3 Task 6; this is the queue-level hook.)
    pub fn cancel(&self, id: &str) -> bool {
        let mut map = self.statuses.write().unwrap();
        match map.get_mut(id) {
            Some(r) => {
                if matches!(r.status, TaskStatus::Queued | TaskStatus::Running) {
                    r.status = TaskStatus::Failed;
                    r.error = Some("cancelled".into());
                }
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::types::PolicyRequest;
    use std::time::Duration;

    fn sample_request() -> TaskRequest {
        TaskRequest {
            task: "open example.com".into(),
            mode: "browser".into(),
            profile: None,
            policy: PolicyRequest::default(),
            wall_clock_secs: None,
            token_budget: None,
            idempotent: false,
            no_retry: false,
            end_session: false,
            save_profile: false,
            save_profile_as: None,
            chat_request: None,
        }
    }

    #[tokio::test]
    async fn queue_runs_task_and_tracks_status() {
        let runner: Runner = Arc::new(|id, _req, _sink| {
            Box::pin(async move {
                TaskResponse {
                    id,
                    status: TaskStatus::Succeeded,
                    attempts: 1,
                    output: Some("ok".into()),
                    error: None,
                    artifacts: vec![],
                }
            })
        });
        let q = TaskQueue::new(runner);
        let id = q.submit(sample_request());
        assert!(q.status(&id).is_some());
        for _ in 0..200 {
            if let Some(r) = q.status(&id) {
                if r.status == TaskStatus::Succeeded {
                    assert_eq!(r.output.as_deref(), Some("ok"));
                    assert_eq!(r.attempts, 1);
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("task did not reach Succeeded");
    }

    #[tokio::test]
    async fn submit_streaming_hands_the_sink_to_the_runner() {
        use crate::script_backend::{Delta, DeltaSink, FinishPayload};
        let runner: Runner = Arc::new(|id, _req, sink| {
            Box::pin(async move {
                if let Some(s) = sink {
                    s.text("增量");
                    s.finish(FinishPayload::from_text("完成".into()));
                }
                TaskResponse {
                    id,
                    status: TaskStatus::Succeeded,
                    attempts: 1,
                    output: Some("完成".into()),
                    error: None,
                    artifacts: vec![],
                }
            })
        });
        let q = TaskQueue::new(runner);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _id = q.submit_streaming(sample_request(), DeltaSink::new(tx));
        assert_eq!(rx.recv().await.unwrap(), Delta::Text("增量".into()));
        match rx.recv().await.unwrap() {
            Delta::Finish(p) => assert_eq!(p.content, "完成"),
            other => panic!("expected finish, got {other:?}"),
        }
    }

    #[test]
    fn unknown_task_status_is_none() {
        let runner: Runner =
            Arc::new(|id, _req, _sink| Box::pin(async move { super::queued(&id) }));
        let q = TaskQueue::new(runner);
        assert!(q.status("nope").is_none());
    }

    #[tokio::test]
    async fn cancel_marks_queued_task_failed() {
        // Runner would sleep forever; but the worker is never scheduled because
        // this test never awaits after submit, so the task stays Queued and
        // cancel wins deterministically.
        let runner: Runner = Arc::new(|id, _req, _sink| {
            Box::pin(async move {
                tokio::time::sleep(Duration::from_secs(60)).await;
                super::queued(&id)
            })
        });
        let q = TaskQueue::new(runner);
        let id = q.submit(sample_request());
        assert!(q.cancel(&id));
        let st = q.status(&id).unwrap();
        assert_eq!(st.status, TaskStatus::Failed);
        assert_eq!(st.error.as_deref(), Some("cancelled"));
        assert!(!q.cancel("nope"));
    }
}
