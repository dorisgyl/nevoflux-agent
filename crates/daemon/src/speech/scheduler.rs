//! ASR 的优先级调度(P2 / Q40-B)。
//!
//! 全进程只有一个 SenseVoice 实例,而且它内部是 `Mutex<Session>`
//! (`crates/asr/src/sensevoice/mod.rs`)。对话链路的滚动重转写与
//! `tts_transcribe`(离线长音频、字幕、Loops 跑批)**争同一把锁**。
//!
//! 后果很具体:一个正在转 2 小时视频的任务在跑时,partial 的延迟会变成不可
//! 预测 —— 每次都要等对方一整段推理跑完才拿得到锁。而 partial 存在的全部理由
//! 是「说话过程中的活体反馈」,一个抖动到几秒的活体反馈还不如没有。
//!
//! ## 为什么这比听起来便宜
//!
//! 离线路径是**按 VAD span 逐段**调 `engine.transcribe` 的
//! (`crates/asr/src/segmented.rs`),所以**天然的让路点已经存在**。不需要真正的
//! 抢占,只需要让离线侧在每次取锁之前先看一眼有没有对话请求在等。
//!
//! 竞态是可接受的:最坏情况是一个离线 span 抢在一个刚到的对话请求前面跑完 ——
//! 一个 span 的延迟,不是一整个 2 小时任务的延迟。**这个粒度就是设计目标。**

use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{Mutex, MutexGuard, Notify};

/// 谁在要 ASR。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// 对话链路:用户正在等屏幕上的字动。插队。
    Conversation,
    /// 离线批处理:字幕、Loops、长音频。让路。
    Offline,
}

/// 一把带优先级的锁。
///
/// 它保护的不是数据,而是**引擎的使用权** —— 所以 guard 是 `()`,调用方拿到
/// 它之后自己去用全局引擎。把引擎本身塞进来会让这个模块依赖 ASR 的构造方式,
/// 而那正是它不该知道的事。
#[derive(Debug)]
pub struct AsrScheduler {
    lock: Mutex<()>,
    waiting_high: AtomicUsize,
    drained: Notify,
}

impl Default for AsrScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl AsrScheduler {
    pub fn new() -> Self {
        Self {
            lock: Mutex::new(()),
            waiting_high: AtomicUsize::new(0),
            drained: Notify::new(),
        }
    }

    /// 有多少对话请求正在排队。离线侧据此让路;也用于观测。
    pub fn waiting_conversation(&self) -> usize {
        self.waiting_high.load(Ordering::SeqCst)
    }

    /// 取得引擎使用权。guard 释放即让出。
    pub async fn acquire(&self, priority: Priority) -> MutexGuard<'_, ()> {
        match priority {
            Priority::Conversation => {
                // 先登记再排队:登记晚于排队的话,离线侧会看到 0 而径直取锁。
                self.waiting_high.fetch_add(1, Ordering::SeqCst);
                let guard = self.lock.lock().await;
                if self.waiting_high.fetch_sub(1, Ordering::SeqCst) == 1 {
                    // 最后一个对话请求拿到锁了,放离线侧继续。
                    self.drained.notify_waiters();
                }
                guard
            }
            Priority::Offline => loop {
                if self.waiting_high.load(Ordering::SeqCst) == 0 {
                    return self.lock.lock().await;
                }
                // 不忙等:等「对话请求排空」的通知。
                self.drained.notified().await;
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn a_lone_caller_gets_through() {
        let s = AsrScheduler::new();
        let _g = s.acquire(Priority::Conversation).await;
        assert_eq!(s.waiting_conversation(), 0);
    }

    #[tokio::test]
    async fn conversation_goes_before_offline_that_has_not_started() {
        let s = Arc::new(AsrScheduler::new());

        // 离线侧先占着锁 —— 模拟一个 span 正在推理。
        let held = s.acquire(Priority::Offline).await;

        // 对话请求排队。
        let s2 = s.clone();
        let conv = tokio::spawn(async move {
            let _g = s2.acquire(Priority::Conversation).await;
            "conversation"
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(s.waiting_conversation(), 1, "对话请求应在排队");

        // 下一个离线 span 想进来 —— 它必须让路。
        let s3 = s.clone();
        let offline_next = tokio::spawn(async move {
            let _g = s3.acquire(Priority::Offline).await;
            "offline"
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!offline_next.is_finished(), "离线侧不该在对话排队时抢先");

        drop(held);
        assert_eq!(conv.await.unwrap(), "conversation");
        assert_eq!(offline_next.await.unwrap(), "offline");
    }

    #[tokio::test]
    async fn offline_resumes_once_conversation_drains() {
        let s = Arc::new(AsrScheduler::new());
        let held = s.acquire(Priority::Offline).await;

        let s2 = s.clone();
        let conv = tokio::spawn(async move {
            let g = s2.acquire(Priority::Conversation).await;
            tokio::time::sleep(Duration::from_millis(30)).await;
            drop(g);
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        drop(held);
        conv.await.unwrap();

        // 对话排空之后,离线侧不该被永久挂起 —— 这是 Notify 用法最容易写错的地方。
        let got =
            tokio::time::timeout(Duration::from_millis(500), s.acquire(Priority::Offline)).await;
        assert!(got.is_ok(), "离线侧应在对话排空后恢复");
    }
}
