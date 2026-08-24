//! 语音对话的 daemon 侧。
//!
//! **上行(P2)** —— 浏览器采集 + VAD,这里转写:
//!
//! - [`utterance`] 累积一段话的音频,并给出自适应节拍的依据
//! - [`runner`] 滚动重转写与端点后的权威转写
//! - [`registry`] 按 session 路由,并核对 utterance_id
//! - [`scheduler`] 让对话链路的 ASR 插到离线批处理前面
//!
//! **下行(P3)** —— 这里合成,浏览器播放:
//!
//! - [`speakable`] 把模型的回答变成能念出来的句子(跳过代码块与 markdown 记号)
//! - [`voice_out`] 合成并投递,**听众由调用点传入**(ADR-0001)
//! - [`voice_registry`] 活动语音轮次,供打断按 session 找到取消开关

pub mod conversation;
pub mod registry;
pub mod runner;
pub mod scheduler;
pub mod speakable;
pub mod utterance;
pub mod voice_out;
pub mod voice_registry;

pub use conversation::{conversation, Conversation};
pub use registry::{Routed, SpeechRegistry};
pub use runner::{run_utterance, Command, Emit, UtteranceSpec};
pub use scheduler::{AsrScheduler, Priority};
pub use speakable::{Speakable, NOTHING_TO_SAY_EN, NOTHING_TO_SAY_ZH};
pub use utterance::{Accepted, UtteranceBuffer};
pub use voice_out::{SpeechSynth, VoiceOut, VoiceTurn};
pub use voice_registry::{BargeIn, VoiceRegistry};
