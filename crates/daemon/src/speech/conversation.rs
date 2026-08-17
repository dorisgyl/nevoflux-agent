//! 语音对话的进程级接入点(P3)。
//!
//! 聊天流式路径与消息分发要用到同一批状态,而把它们逐个穿进
//! `handle_chat_message_streaming` 的签名意味着改动产品里最承重的一个函数的
//! 参数表。这里用一个进程级单例代替。
//!
//! **这不违反 ADR-0001。** 那条禁止的是「从全局注册表**发现听众**」——
//! 听众仍然是每轮由发起方显式传入 `VoiceTurn` 的。这里存的是「这个 session
//! 开没开语音」,一个真正的 per-session 设置,与 `SpeechRegistry` 同形。

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use tokio::sync::Mutex;

use super::voice_registry::VoiceRegistry;

pub struct Conversation {
    /// 开了语音的 session。
    voice_on: Mutex<HashSet<String>>,
    /// 活动语音轮次,供打断。
    pub turns: VoiceRegistry,
}

impl Conversation {
    fn new() -> Self {
        Self {
            voice_on: Mutex::new(HashSet::new()),
            turns: VoiceRegistry::new(),
        }
    }

    pub async fn set_voice_mode(&self, session_id: &str, on: bool) {
        let mut set = self.voice_on.lock().await;
        if on {
            set.insert(session_id.to_string());
        } else {
            set.remove(session_id);
        }
    }

    pub async fn voice_mode(&self, session_id: &str) -> bool {
        self.voice_on.lock().await.contains(session_id)
    }
}

/// 进程级单例。
pub fn conversation() -> &'static Arc<Conversation> {
    static C: OnceLock<Arc<Conversation>> = OnceLock::new();
    C.get_or_init(|| Arc::new(Conversation::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn voice_mode_is_per_session() {
        let c = Conversation::new();
        assert!(!c.voice_mode("s1").await);
        c.set_voice_mode("s1", true).await;
        assert!(c.voice_mode("s1").await);
        assert!(!c.voice_mode("s2").await, "不该波及别的 session");
        c.set_voice_mode("s1", false).await;
        assert!(!c.voice_mode("s1").await);
    }

    #[tokio::test]
    async fn the_singleton_is_one_instance() {
        assert!(Arc::ptr_eq(conversation(), conversation()));
    }
}
