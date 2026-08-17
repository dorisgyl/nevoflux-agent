//! 按轮次的中断旗标(ADR-0002 的前置改造)。
//!
//! ## 原来的形状为什么不够
//!
//! 旧实现是 `HashMap<String, Arc<AtomicBool>>` —— 一个 session 一个槽:
//!
//! - 轮次开始时 `insert`,**顶掉**上一轮的旗标 ⇒ 上一轮从此不可停,会一直跑到
//!   24 小时上限
//! - 轮次结束时无条件 `remove` ⇒ 如果此时已有新一轮在跑,**新一轮的旗标被误摘**,
//!   于是它也不可停
//!
//! 今天这不是 bug,因为「一个 session 同时只有一轮」由 UI 维持(生成期间禁用
//! 发送按钮)。**语音把这个执行者拿掉了:VAD 不会禁用嘴。**
//!
//! ## 令牌
//!
//! 每轮拿一个单调递增的令牌。摘除只在令牌匹配时发生,所以迟到的收尾摘不掉
//! 新一轮。开始新一轮时若已有一轮在跑,**先停掉它再登记** —— 覆盖而非拒绝,
//! 因为「上一轮还开着」在真实链路里就是会发生的(收尾消息丢了、页面刷新了),
//! 拒绝会让这个 session 从此卡死,而取消最多损失一轮已经过时的回答。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex;

/// 一轮的句柄。收尾时把它交回去,摘除才认。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnToken(u64);

#[derive(Default)]
pub struct InterruptRegistry {
    active: Mutex<HashMap<String, (TurnToken, Arc<AtomicBool>)>>,
    next: AtomicU64,
}

impl InterruptRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 开始一轮。已有的一轮会被停掉。
    pub async fn begin(&self, session_id: &str) -> (TurnToken, Arc<AtomicBool>) {
        let token = TurnToken(self.next.fetch_add(1, Ordering::SeqCst));
        let flag = Arc::new(AtomicBool::new(false));
        let mut map = self.active.lock().await;
        if let Some((prev_token, prev_flag)) =
            map.insert(session_id.to_string(), (token, flag.clone()))
        {
            prev_flag.store(true, Ordering::SeqCst);
            tracing::debug!(
                target: "turn",
                session = session_id,
                replaced = prev_token.0,
                "a new turn replaced one still running; the old one was stopped"
            );
        }
        (token, flag)
    }

    /// 停止这个 session 上正在跑的那一轮(stop_generation / 清空会话)。
    ///
    /// 返回是否真的停了一轮 —— 调用方据此区分「停了」与「本来就没有在跑」,
    /// 那是两种要说给用户听的不同结果。
    pub async fn interrupt(&self, session_id: &str) -> bool {
        let mut map = self.active.lock().await;
        match map.remove(session_id) {
            Some((_, flag)) => {
                flag.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    /// 一轮正常收尾。**只在令牌匹配时摘除** —— 否则会摘掉刚开始的下一轮。
    pub async fn end(&self, session_id: &str, token: TurnToken) {
        let mut map = self.active.lock().await;
        if map.get(session_id).map(|(t, _)| *t) == Some(token) {
            map.remove(session_id);
        }
    }

    /// 这个 session 上有没有一轮在跑。
    pub async fn is_running(&self, session_id: &str) -> bool {
        self.active.lock().await.contains_key(session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_second_turn_stops_the_first() {
        // 旧实现在这里静默顶掉旗标,于是第一轮永远停不下来。
        let r = InterruptRegistry::new();
        let (_t1, f1) = r.begin("s1").await;
        let (_t2, f2) = r.begin("s1").await;
        assert!(f1.load(Ordering::SeqCst), "被取代的一轮必须被停掉");
        assert!(!f2.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_late_end_does_not_unregister_the_next_turn() {
        // 旧实现在这里无条件 remove,于是新一轮也变成不可停。
        let r = InterruptRegistry::new();
        let (t1, _f1) = r.begin("s1").await;
        let (_t2, f2) = r.begin("s1").await;
        r.end("s1", t1).await; // 迟到的收尾
        assert!(r.is_running("s1").await, "新一轮不该被上一轮的收尾摘掉");
        assert!(r.interrupt("s1").await);
        assert!(f2.load(Ordering::SeqCst), "新一轮仍须可停");
    }

    #[tokio::test]
    async fn interrupt_reports_whether_anything_was_running() {
        let r = InterruptRegistry::new();
        assert!(!r.interrupt("s1").await, "没在跑就该照实说");
        let (_t, f) = r.begin("s1").await;
        assert!(r.interrupt("s1").await);
        assert!(f.load(Ordering::SeqCst));
        assert!(!r.interrupt("s1").await, "停过之后不该再报停了一轮");
    }

    #[tokio::test]
    async fn ending_the_current_turn_clears_it() {
        let r = InterruptRegistry::new();
        let (t, _f) = r.begin("s1").await;
        r.end("s1", t).await;
        assert!(!r.is_running("s1").await);
    }

    #[tokio::test]
    async fn sessions_are_independent() {
        let r = InterruptRegistry::new();
        let (_t1, f1) = r.begin("s1").await;
        let (_t2, f2) = r.begin("s2").await;
        assert!(r.interrupt("s1").await);
        assert!(f1.load(Ordering::SeqCst));
        assert!(!f2.load(Ordering::SeqCst), "另一个 session 不该被牵连");
    }

    #[tokio::test]
    async fn tokens_never_repeat() {
        // 令牌复用会让「迟到的收尾」重新变得危险。
        let r = InterruptRegistry::new();
        let (a, _) = r.begin("s1").await;
        r.end("s1", a).await;
        let (b, _) = r.begin("s1").await;
        assert_ne!(a, b);
    }
}
