//! 把入站的语音消息路由到对应的 utterance(P2)。
//!
//! 一个 session 在任一时刻至多有一段 utterance 在转写 —— 人只有一张嘴。注册表
//! 就是这条不变量的执行者。
//!
//! ## 为什么每条消息都要核对 utterance_id
//!
//! 取消之后、或新一段开始之后,**上一段的 chunk 仍可能在路上**。不核对的话它们
//! 会被追加进新缓冲,症状是「转写里混进了上一句的尾巴」—— 一个查起来极慢的 bug,
//! 因为音频拼接不留任何痕迹。`UtteranceBuffer` 已经按 seq 挡了一层,但那只在同一
//! 段之内有效;跨段必须靠 id。

use std::collections::HashMap;
use std::sync::Arc;

use nevoflux_asr::Transcriber;
use tokio::sync::{mpsc, Mutex};

use super::runner::{run_utterance, Command, Emit, UtteranceSpec};
use super::scheduler::AsrScheduler;

struct Active {
    utterance_id: String,
    tx: mpsc::UnboundedSender<Command>,
}

/// 每个 session 的活动 utterance。
pub struct SpeechRegistry {
    active: Mutex<HashMap<String, Active>>,
    scheduler: Arc<AsrScheduler>,
}

/// 路由结果。调用方据此决定要不要记一笔日志。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Routed {
    /// 已投递给正在跑的那一段。
    Delivered,
    /// utterance_id 对不上 —— 属于已结束或已取消的一段,丢弃。
    Stale,
    /// 这个 session 没有活动 utterance。
    NoActive,
}

impl SpeechRegistry {
    pub fn new(scheduler: Arc<AsrScheduler>) -> Self {
        Self {
            active: Mutex::new(HashMap::new()),
            scheduler,
        }
    }

    pub fn scheduler(&self) -> &Arc<AsrScheduler> {
        &self.scheduler
    }

    /// 开始新的一段。同 session 上已有的一段会被取消。
    ///
    /// 覆盖而非拒绝,是因为「上一段还开着」在真实链路里就是会发生的:端点消息
    /// 丢了、页面刷新了、VAD 抖了一下。拒绝新段会让语音从此卡死,而取消旧段
    /// 最多损失一句已经过时的话。
    pub async fn start(
        &self,
        session_id: &str,
        utterance_id: &str,
        sample_rate: u32,
        language: Option<String>,
        transcriber: Arc<dyn Transcriber>,
        out: mpsc::UnboundedSender<Emit>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let mut map = self.active.lock().await;
            if let Some(prev) = map.insert(
                session_id.to_string(),
                Active {
                    utterance_id: utterance_id.to_string(),
                    tx,
                },
            ) {
                let _ = prev.tx.send(Command::Cancel);
                tracing::debug!(
                    target: "speech",
                    session = session_id,
                    replaced = %prev.utterance_id,
                    "a new utterance replaced one still running"
                );
            }
        }

        let scheduler = self.scheduler.clone();
        let session = session_id.to_string();
        let utterance = utterance_id.to_string();
        tokio::spawn(async move {
            run_utterance(
                UtteranceSpec {
                    session_id: session,
                    utterance_id: utterance,
                    sample_rate,
                    language,
                },
                transcriber,
                scheduler,
                rx,
                out,
            )
            .await;
        });
    }

    async fn route(&self, session_id: &str, utterance_id: &str, cmd: Command) -> Routed {
        let map = self.active.lock().await;
        match map.get(session_id) {
            None => Routed::NoActive,
            Some(a) if a.utterance_id != utterance_id => Routed::Stale,
            Some(a) => {
                if a.tx.send(cmd).is_ok() {
                    Routed::Delivered
                } else {
                    // 跑者已经退出但还没被摘掉。对调用方而言与「没有活动段」等价。
                    Routed::NoActive
                }
            }
        }
    }

    pub async fn chunk(
        &self,
        session_id: &str,
        utterance_id: &str,
        seq: u32,
        pcm: String,
    ) -> Routed {
        self.route(session_id, utterance_id, Command::Chunk { seq, pcm })
            .await
    }

    pub async fn end(&self, session_id: &str, utterance_id: &str) -> Routed {
        let routed = self.route(session_id, utterance_id, Command::End).await;
        if routed == Routed::Delivered {
            self.retire(session_id, utterance_id).await;
        }
        routed
    }

    pub async fn cancel(&self, session_id: &str, utterance_id: &str) -> Routed {
        let routed = self.route(session_id, utterance_id, Command::Cancel).await;
        if routed == Routed::Delivered {
            self.retire(session_id, utterance_id).await;
        }
        routed
    }

    /// 从活动表里摘掉,但只在它仍是同一段时 —— 否则会误摘刚开始的新段。
    async fn retire(&self, session_id: &str, utterance_id: &str) {
        let mut map = self.active.lock().await;
        if map.get(session_id).map(|a| a.utterance_id.as_str()) == Some(utterance_id) {
            map.remove(session_id);
        }
    }

    /// 会话结束 / 通道断开:取消这个 session 上的一切。
    pub async fn forget(&self, session_id: &str) {
        if let Some(a) = self.active.lock().await.remove(session_id) {
            let _ = a.tx.send(Command::Cancel);
        }
    }

    #[cfg(test)]
    async fn active_utterance(&self, session_id: &str) -> Option<String> {
        self.active
            .lock()
            .await
            .get(session_id)
            .map(|a| a.utterance_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nevoflux_asr::{AsrError, Segment, Transcript};

    struct Silent;
    impl Transcriber for Silent {
        fn transcribe(&self, _s: &[f32], _l: Option<&str>) -> Result<Transcript, AsrError> {
            Ok(Transcript {
                text: String::new(),
                segments: Vec::<Segment>::new(),
                language: "zh".into(),
                audio_event: Some("Speech".into()),
            })
        }
    }

    fn reg() -> SpeechRegistry {
        SpeechRegistry::new(Arc::new(AsrScheduler::new()))
    }

    async fn start(r: &SpeechRegistry, s: &str, u: &str) -> mpsc::UnboundedReceiver<Emit> {
        let (otx, orx) = mpsc::unbounded_channel();
        r.start(s, u, 16_000, None, Arc::new(Silent), otx).await;
        orx
    }

    #[tokio::test]
    async fn chunks_for_the_wrong_utterance_are_dropped() {
        let r = reg();
        let _rx = start(&r, "s1", "u1").await;
        assert_eq!(
            r.chunk("s1", "u1", 0, "AAA=".into()).await,
            Routed::Delivered
        );
        // 上一段的迟到片 —— 必须丢,否则会污染 u1 的缓冲。
        assert_eq!(r.chunk("s1", "u0", 0, "AAA=".into()).await, Routed::Stale);
    }

    #[tokio::test]
    async fn a_new_utterance_replaces_one_still_running() {
        let r = reg();
        let _rx1 = start(&r, "s1", "u1").await;
        let _rx2 = start(&r, "s1", "u2").await;
        assert_eq!(r.active_utterance("s1").await.as_deref(), Some("u2"));
        // 旧段的片此时已是 stale。
        assert_eq!(r.chunk("s1", "u1", 0, "AAA=".into()).await, Routed::Stale);
    }

    #[tokio::test]
    async fn end_retires_the_utterance() {
        let r = reg();
        let _rx = start(&r, "s1", "u1").await;
        assert_eq!(r.end("s1", "u1").await, Routed::Delivered);
        assert_eq!(r.active_utterance("s1").await, None);
        // 端点之后再来的片没有去处。
        assert_eq!(
            r.chunk("s1", "u1", 1, "AAA=".into()).await,
            Routed::NoActive
        );
    }

    #[tokio::test]
    async fn retiring_does_not_evict_a_newer_utterance() {
        // 端点消息比新一段的开始晚到时,不能把新段摘掉。
        let r = reg();
        let _rx1 = start(&r, "s1", "u1").await;
        let _rx2 = start(&r, "s1", "u2").await;
        r.end("s1", "u1").await; // 已是 stale,不该动 u2
        assert_eq!(r.active_utterance("s1").await.as_deref(), Some("u2"));
    }

    #[tokio::test]
    async fn sessions_do_not_interfere() {
        let r = reg();
        let _a = start(&r, "s1", "u1").await;
        let _b = start(&r, "s2", "u1").await;
        assert_eq!(
            r.chunk("s1", "u1", 0, "AAA=".into()).await,
            Routed::Delivered
        );
        assert_eq!(
            r.chunk("s2", "u1", 0, "AAA=".into()).await,
            Routed::Delivered
        );
        r.forget("s1").await;
        assert_eq!(
            r.chunk("s1", "u1", 1, "AAA=".into()).await,
            Routed::NoActive
        );
        assert_eq!(
            r.chunk("s2", "u1", 1, "AAA=".into()).await,
            Routed::Delivered
        );
    }

    #[tokio::test]
    async fn unknown_session_is_reported_not_panicked() {
        let r = reg();
        assert_eq!(
            r.chunk("nope", "u1", 0, "AAA=".into()).await,
            Routed::NoActive
        );
        assert_eq!(r.end("nope", "u1").await, Routed::NoActive);
        r.forget("nope").await;
    }
}
