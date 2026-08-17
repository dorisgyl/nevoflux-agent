//! 语音上行链路的线格式(P2)。
//!
//! 浏览器侧做采集与 VAD,daemon 侧做转写。这个模块只描述两者之间传什么。
//!
//! ## 为什么有 `utterance_id`
//!
//! VAD 判定的一段话是一个 utterance。取消(用户点了停、audio-event 闸门判成
//! 非人声、通道重连)之后,**上一段的 chunk 仍可能在路上**。没有 id 的话,那些
//! 迟到的字节会被追加进下一段的缓冲,而症状是「转写里混进了上一句的尾巴」——
//! 一个查起来极慢的 bug。id 让接收端可以直接丢弃。
//!
//! ## 为什么音频是 i16 而不是 f32
//!
//! 采集侧拿到的是 f32,但麦克风的物理分辨率本来就是 16 位,转成 i16 不损失
//! 可听信息,却把上行体积**减半**(16 kHz 单声道 500 ms:f32 32 KB → i16 16 KB,
//! base64 后 43 KB → 21 KB)。VAD 已经在 f32 上跑完了,送 daemon 的只需喂 ASR。

use serde::{Deserialize, Serialize};

/// 对话链路固定的采样率。VAD、ASR、缓冲全部按它对齐。
pub const SPEECH_SAMPLE_RATE: u32 = 16_000;

// ---------------------------------------------------------------- 上行

/// VAD 判定用户开口,一段新的 utterance 开始。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechStart {
    pub session_id: String,
    pub utterance_id: String,
    /// 采集侧的实际采样率。按契约应恒为 [`SPEECH_SAMPLE_RATE`];**照样传**,
    /// 是为了让不匹配变成一个能被发现的错误,而不是一段听起来慢半拍的音频。
    pub sample_rate: u32,
}

/// 一段音频。`seq` 从 0 递增。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechChunk {
    pub session_id: String,
    pub utterance_id: String,
    pub seq: u32,
    /// base64 的小端 i16 PCM。
    pub pcm: String,
}

/// VAD 判定端点,请求权威转写。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechEnd {
    pub session_id: String,
    pub utterance_id: String,
}

/// 丢弃这一段,不要转写也不要入库。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechCancel {
    pub session_id: String,
    pub utterance_id: String,
    pub reason: SpeechCancelReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechCancelReason {
    /// 用户主动停止。
    UserStopped,
    /// 语音会话结束(静默超时、关麦、通道断开)。
    SessionEnded,
    /// 段太短,VAD 判成 blip。
    TooShort,
}

// ---------------------------------------------------------------- 下行

/// 滚动重转写的中间结果。**永不入库** —— 只占屏幕上那个可替换位置。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechPartial {
    pub session_id: String,
    pub utterance_id: String,
    pub text: String,
    /// 已转写的音频时长,毫秒。UI 用它判断 partial 是不是在推进。
    pub buffered_ms: u32,
}

/// 端点之后的权威转写。只有 `accepted` 为真的才成为一轮用户输入。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechFinal {
    pub session_id: String,
    pub utterance_id: String,
    pub text: String,
    /// SenseVoice 检出的语言标签。
    pub language: String,
    /// SenseVoice 的 audio-event 标签(`Speech` / `BGM` / `Applause` / `Laughter`)。
    ///
    /// 它是误触发闸门的信号源。模型一直在算它,只是过去被丢掉了。
    pub audio_event: String,
    /// 是否通过闸门。
    pub accepted: bool,
    /// 上行时观测到的 seq 空洞数。
    ///
    /// 不为零意味着**转写里有一个听不出来的洞** —— 音频被接在一起,缺失的部分
    /// 不留任何痕迹。对一句命令来说一个洞足以反转语义(「不要删除」变成
    /// 「删除」),所以它必须能被上层看到,不能只躺在缓冲里。
    #[serde(default)]
    pub gaps: u32,
}

/// 转写失败。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechError {
    pub session_id: String,
    pub utterance_id: String,
    pub message: String,
}

// ---------------------------------------------------------------- 下行语音(P3)

/// 一句合成好的回答。
///
/// 字节直接放在 frame 里(ADR-0003),不走 asset offer。一句约 3 秒 = base64 后
/// 约 192 KB,远低于 native messaging 的 900 KB 分片阈值。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceAudio {
    pub session_id: String,
    /// 一轮回答一个 id。打断与投递对账都按它。
    pub turn_id: String,
    /// 从 0 递增。播放端据此保序,并知道自己播到了第几句。
    pub seq: u32,
    pub sample_rate: u32,
    /// base64 的 WAV。
    pub wav: String,
}

/// 这一轮的语音推完了。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceDone {
    pub session_id: String,
    pub turn_id: String,
    /// **实际推出去**的句数,不是生成的句数。
    ///
    /// 投递注记要拿它与播放端实际播出的句数对账(ADR-0004):模型以为自己说完了
    /// 整段而用户只听到前三分之一,是 cascaded S2S 最隐蔽的 bug。
    pub spoken: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceFailed {
    pub session_id: String,
    pub turn_id: String,
    pub message: String,
}

/// 打断:停止这一轮的语音。
///
/// 浏览器侧**先本地静音再发这条**(§6.5)。用户感知的停止延迟等于本地静音延迟
/// (≈0),这条只负责停止继续合成与推送、把算力还回去 —— 两件事解耦,IPC 往返
/// 不在关键路径上。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceBargeIn {
    pub session_id: String,
    pub turn_id: String,
    /// 播放端**实际播完**的句数。写投递注记要用。
    pub played: u32,
}

impl SpeechFinal {
    /// 闸门:非 `Speech` 的一律不接受。
    ///
    /// 挡的是背景音乐、掌声、笑声这类最常见的误触发源 —— 而它**挡不住同事说话**,
    /// 那是 `<|Speech|>`。所以真正的兜底不是这道闸门,而是「可见 + 可撤回」。
    pub fn gate(audio_event: &str) -> bool {
        audio_event.eq_ignore_ascii_case("speech")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_accepts_only_speech() {
        assert!(SpeechFinal::gate("Speech"));
        assert!(SpeechFinal::gate("speech"));
        for tag in ["BGM", "Applause", "Laughter", "", "unknown"] {
            assert!(!SpeechFinal::gate(tag), "should reject {tag}");
        }
    }

    #[test]
    fn chunk_round_trips_with_snake_case_fields() {
        let c = SpeechChunk {
            session_id: "s1".into(),
            utterance_id: "u1".into(),
            seq: 3,
            pcm: "AAA=".into(),
        };
        let v = serde_json::to_value(&c).unwrap();
        // 线格式是跨语言契约:字段名改了浏览器侧会静默收不到。
        assert_eq!(v["session_id"], "s1");
        assert_eq!(v["utterance_id"], "u1");
        assert_eq!(v["seq"], 3);
        assert_eq!(v["pcm"], "AAA=");
        assert_eq!(serde_json::from_value::<SpeechChunk>(v).unwrap(), c);
    }

    #[test]
    fn cancel_reason_is_snake_case_on_the_wire() {
        let v = serde_json::to_value(SpeechCancelReason::UserStopped).unwrap();
        assert_eq!(v, "user_stopped");
    }
}
