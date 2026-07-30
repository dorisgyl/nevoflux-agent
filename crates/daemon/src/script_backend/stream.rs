//! 增量通道：脚本执行过程中的输出出口。
//!
//! 非流式请求同样使用它——结构化结果（`tool_calls` / `usage` /
//! `finish_reason`）无法从 `TaskResponse.output` 这个字符串通道穿过，
//! 所以两种模式统一经 [`Delta::Finish`] 取结果。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

use crate::script_backend::contract::{OutcomeBody, ScriptOutcome, ScriptToolCall, Usage};

/// 终帧载荷：一次调用的最终结果。
#[derive(Debug, Clone, PartialEq)]
pub struct FinishPayload {
    /// 最终正文（增量拼接或脚本显式返回）。
    pub content: String,
    /// 请求客户端执行的工具调用；空则无。
    pub tool_calls: Vec<ScriptToolCall>,
    /// 终止原因。
    pub finish_reason: String,
    /// 可选用量。
    pub usage: Option<Usage>,
    /// 错误信息（message, type, code），与前三者互斥。
    pub error: Option<(String, String, Option<String>)>,
}

impl FinishPayload {
    /// 由脚本结果构造。
    pub fn from_outcome(outcome: ScriptOutcome) -> Self {
        let finish_reason = outcome.finish_reason.clone();
        let usage = outcome.usage;
        match outcome.body {
            OutcomeBody::Content(text) => Self {
                content: text,
                tool_calls: Vec::new(),
                finish_reason,
                usage,
                error: None,
            },
            OutcomeBody::ToolCalls(calls) => Self {
                content: String::new(),
                tool_calls: calls,
                finish_reason,
                usage,
                error: None,
            },
            OutcomeBody::Error(e) => Self {
                content: String::new(),
                tool_calls: Vec::new(),
                finish_reason,
                usage,
                error: Some((e.message, e.kind, e.code)),
            },
        }
    }

    /// 由一段纯文本构造（老入口、或 agent 循环的输出）。
    pub fn from_text(text: String) -> Self {
        Self {
            content: text,
            tool_calls: Vec::new(),
            finish_reason: "stop".to_string(),
            usage: None,
            error: None,
        }
    }

    /// 由一条错误构造。
    pub fn from_error(message: String, kind: &str, code: &str) -> Self {
        Self {
            content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "error".to_string(),
            usage: None,
            error: Some((message, kind.to_string(), Some(code.to_string()))),
        }
    }
}

/// 一次增量事件。
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    /// 正文增量（`emit_text`）。
    Text(String),
    /// 进度提示（`emit_progress`）——OpenAI 侧降级成 SSE 注释帧，
    /// MCP 第二期映射到 `notifications/progress`。
    Progress(String),
    /// 终帧。每次调用**恰好一个**，且必为最后一条。
    Finish(Box<FinishPayload>),
}

/// 增量出口。可克隆，跨线程发送。
///
/// `finish` 由 `AtomicBool` 守卫：第一次调用发出终帧，后续调用是空操作。
/// 这让“脚本自己发终帧”和“runner 兜底补发”可以共存而不会重复。
#[derive(Debug, Clone)]
pub struct DeltaSink {
    tx: UnboundedSender<Delta>,
    finished: Arc<AtomicBool>,
}

impl DeltaSink {
    /// 绑定一个发送端。
    pub fn new(tx: UnboundedSender<Delta>) -> Self {
        Self {
            tx,
            finished: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 发送正文增量。终帧之后的增量被丢弃。
    pub fn text(&self, chunk: impl Into<String>) {
        if !self.is_finished() {
            let _ = self.tx.send(Delta::Text(chunk.into()));
        }
    }

    /// 发送进度提示。
    pub fn progress(&self, note: impl Into<String>) {
        if !self.is_finished() {
            let _ = self.tx.send(Delta::Progress(note.into()));
        }
    }

    /// 发送终帧。只有第一次调用生效，返回是否真的发出。
    pub fn finish(&self, payload: FinishPayload) -> bool {
        if self
            .finished
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        self.tx.send(Delta::Finish(Box::new(payload))).is_ok()
    }

    /// 终帧是否已发出。
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script_backend::contract::OutcomeError;

    fn sink() -> (DeltaSink, tokio::sync::mpsc::UnboundedReceiver<Delta>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (DeltaSink::new(tx), rx)
    }

    #[tokio::test]
    async fn deltas_arrive_in_order_with_finish_last() {
        let (s, mut rx) = sink();
        s.progress("正在打开");
        s.text("你");
        s.text("好");
        assert!(s.finish(FinishPayload::from_text("你好".into())));

        assert_eq!(rx.recv().await.unwrap(), Delta::Progress("正在打开".into()));
        assert_eq!(rx.recv().await.unwrap(), Delta::Text("你".into()));
        assert_eq!(rx.recv().await.unwrap(), Delta::Text("好".into()));
        match rx.recv().await.unwrap() {
            Delta::Finish(p) => assert_eq!(p.content, "你好"),
            other => panic!("expected finish, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn second_finish_is_a_no_op() {
        let (s, mut rx) = sink();
        assert!(s.finish(FinishPayload::from_text("第一次".into())));
        assert!(!s.finish(FinishPayload::from_text("第二次".into())));
        match rx.recv().await.unwrap() {
            Delta::Finish(p) => assert_eq!(p.content, "第一次"),
            other => panic!("expected finish, got {other:?}"),
        }
        // 只有一条
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn text_after_finish_is_dropped() {
        let (s, mut rx) = sink();
        s.finish(FinishPayload::from_text("done".into()));
        s.text("迟到的增量");
        let _ = rx.recv().await;
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn finish_payload_maps_tool_calls_outcome() {
        let outcome = ScriptOutcome {
            body: OutcomeBody::ToolCalls(vec![ScriptToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "/tmp/a"}),
            }]),
            finish_reason: "tool_calls".into(),
            usage: None,
        };
        let p = FinishPayload::from_outcome(outcome);
        assert!(p.content.is_empty());
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.finish_reason, "tool_calls");
    }

    #[test]
    fn finish_payload_maps_error_outcome() {
        let outcome = ScriptOutcome {
            body: OutcomeBody::Error(OutcomeError {
                message: "boom".into(),
                kind: "server_error".into(),
                code: Some("script_error".into()),
            }),
            finish_reason: "error".into(),
            usage: None,
        };
        let p = FinishPayload::from_outcome(outcome);
        let (msg, kind, code) = p.error.expect("error");
        assert_eq!(msg, "boom");
        assert_eq!(kind, "server_error");
        assert_eq!(code.as_deref(), Some("script_error"));
    }
}
