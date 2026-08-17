//! 活动语音轮次的登记(P3)。
//!
//! 打断要能按 session 找到「正在说话的那一轮」的取消开关。和上行的
//! [`super::registry::SpeechRegistry`] 是同一个形状,也出于同一个理由:
//! **每条消息都要核对 id**。
//!
//! 打断信令与新一轮的开始会并发到达 —— 用户喊停的同时 agent 已经开始说下一句。
//! 不核对 turn_id 的话,一条迟到的打断会掐掉刚开始的那一轮,表现为「它刚要说
//! 话就自己停了」。

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tokio::sync::Mutex;

/// 打断的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BargeIn {
    /// 已停止那一轮。
    Stopped,
    /// turn_id 对不上 —— 属于已结束或已被取代的一轮,忽略。
    Stale,
    /// 这个 session 没有在说话。
    NotSpeaking,
}

#[derive(Default)]
pub struct VoiceRegistry {
    active: Mutex<HashMap<String, (String, Arc<AtomicBool>)>>,
}

impl VoiceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记新一轮。同 session 上已有的一轮会被取消 —— 一张嘴不能同时说两段。
    pub async fn begin(&self, session_id: &str, turn_id: &str, canceller: Arc<AtomicBool>) {
        let mut map = self.active.lock().await;
        if let Some((prev_id, prev)) =
            map.insert(session_id.to_string(), (turn_id.to_string(), canceller))
        {
            prev.store(true, std::sync::atomic::Ordering::SeqCst);
            tracing::debug!(
                target: "speech",
                session = session_id,
                replaced = %prev_id,
                "a new voice turn replaced one still speaking"
            );
        }
    }

    /// 打断。
    pub async fn barge_in(&self, session_id: &str, turn_id: &str) -> BargeIn {
        let map = self.active.lock().await;
        match map.get(session_id) {
            None => BargeIn::NotSpeaking,
            Some((id, _)) if id != turn_id => BargeIn::Stale,
            Some((_, flag)) => {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                BargeIn::Stopped
            }
        }
    }

    /// 一轮正常结束。只在仍是同一轮时摘除,否则会误摘刚开始的下一轮。
    pub async fn end(&self, session_id: &str, turn_id: &str) {
        let mut map = self.active.lock().await;
        if map.get(session_id).map(|(id, _)| id.as_str()) == Some(turn_id) {
            map.remove(session_id);
        }
    }

    /// 会话结束 / 通道断开。
    pub async fn forget(&self, session_id: &str) {
        if let Some((_, flag)) = self.active.lock().await.remove(session_id) {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[cfg(test)]
    async fn speaking(&self, session_id: &str) -> Option<String> {
        self.active
            .lock()
            .await
            .get(session_id)
            .map(|(id, _)| id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[tokio::test]
    async fn barge_in_stops_the_matching_turn() {
        let r = VoiceRegistry::new();
        let f = flag();
        r.begin("s1", "t1", f.clone()).await;
        assert_eq!(r.barge_in("s1", "t1").await, BargeIn::Stopped);
        assert!(f.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_late_barge_in_does_not_cut_off_the_next_turn() {
        // 用户喊停的同时 agent 已经开始说下一句。不核对 id 的话,表现是
        // 「它刚要说话就自己停了」。
        let r = VoiceRegistry::new();
        let old = flag();
        let new = flag();
        r.begin("s1", "t1", old.clone()).await;
        r.begin("s1", "t2", new.clone()).await;
        assert!(old.load(Ordering::SeqCst), "被取代的一轮该停");

        assert_eq!(r.barge_in("s1", "t1").await, BargeIn::Stale);
        assert!(!new.load(Ordering::SeqCst), "新一轮不该被迟到的打断掐掉");
    }

    #[tokio::test]
    async fn ending_does_not_evict_a_newer_turn() {
        let r = VoiceRegistry::new();
        r.begin("s1", "t1", flag()).await;
        r.begin("s1", "t2", flag()).await;
        r.end("s1", "t1").await;
        assert_eq!(r.speaking("s1").await.as_deref(), Some("t2"));
    }

    #[tokio::test]
    async fn a_silent_session_reports_not_speaking() {
        let r = VoiceRegistry::new();
        assert_eq!(r.barge_in("s1", "t1").await, BargeIn::NotSpeaking);
    }

    #[tokio::test]
    async fn forget_stops_whatever_was_speaking() {
        let r = VoiceRegistry::new();
        let f = flag();
        r.begin("s1", "t1", f.clone()).await;
        r.forget("s1").await;
        assert!(f.load(Ordering::SeqCst));
        assert_eq!(r.barge_in("s1", "t1").await, BargeIn::NotSpeaking);
    }
}
