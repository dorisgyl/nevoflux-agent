//! 一段话的音频缓冲(P2 / Q40-A)。
//!
//! 浏览器侧按 ~500 ms 一片往上送,daemon 侧累积起来反复重转写。这个类型只管
//! 累积与记账;什么时候重转、由谁重转,是编排层的事。
//!
//! ## 节拍是自适应的,不是定时的
//!
//! v1.2 的设计是「每 400 ms 重转一次」。两个问题:chunk 每 500 ms 才到,所以
//! 五次里有一次看到的是完全相同的缓冲,输出逐字相同 —— 白烧一次 CPU,还会让
//! 屏幕上的 partial **周期性地看起来卡住**,而 partial 存在的唯一理由就是活体
//! 反馈。而且成本是二次的:一段 T 秒的话累计约 `T²/24` 秒 CPU,8 秒时占空比
//! 已经 67%,12 秒到 100%。那个「12 秒阈值」是**饱和点,不是够用点**。
//!
//! 所以这里不提供「该不该按时跑」,只提供 [`has_new_audio`](Self::has_new_audio):
//! 编排层跑完一次就问一次,有新音频就立刻接着跑,没有就等下一片。魔数被**删掉**
//! 而不是调对 —— 快机器自动跑得勤,慢机器自动降频,长句子上的退化从悬崖变成斜坡。

use base64::{engine::general_purpose::STANDARD, Engine as _};

use crate::error::DaemonError;

/// 一段 utterance 的累积缓冲。
#[derive(Debug)]
pub struct UtteranceBuffer {
    id: String,
    sample_rate: u32,
    samples: Vec<f32>,
    /// 下一片期望的 seq。
    expected_seq: u32,
    /// 观测到的 seq 空洞数。
    gaps: u32,
    /// 上一次完成的转写覆盖到哪个样本。
    transcribed_upto: usize,
}

/// `push` 的结果。调用方据此决定要不要记一笔。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accepted {
    /// 正常追加。
    Appended,
    /// seq 比期望的小 —— 重复投递或乱序。已忽略,缓冲未变。
    Duplicate,
    /// seq 比期望的大 —— 中间丢了片。已尽力追加,但缓冲有洞。
    Gap { missing: u32 },
}

impl UtteranceBuffer {
    pub fn new(id: impl Into<String>, sample_rate: u32) -> Self {
        Self {
            id: id.into(),
            sample_rate,
            samples: Vec::new(),
            expected_seq: 0,
            gaps: 0,
            transcribed_upto: 0,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// 缓冲里已有多少毫秒音频。
    pub fn buffered_ms(&self) -> u32 {
        if self.sample_rate == 0 {
            return 0;
        }
        ((self.samples.len() as u64 * 1000) / self.sample_rate as u64) as u32
    }

    /// 至今观测到的 seq 空洞数。
    ///
    /// 不为零意味着**转写会有一个听不出来的洞** —— 音频被接在了一起,而缺失的
    /// 部分不会留下任何痕迹。对一句命令来说,一个洞足以反转语义(「不要删除」
    /// 变成「删除」),所以这个数必须能被上层看到,不能只记日志。
    pub fn gaps(&self) -> u32 {
        self.gaps
    }

    /// 自上次转写以来有没有新音频。自适应节拍的全部依据。
    pub fn has_new_audio(&self) -> bool {
        self.samples.len() > self.transcribed_upto
    }

    /// 迄今全部音频 —— 滚动重转写每次都从头跑。
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// 记下「到此为止已被转写过」。
    pub fn mark_transcribed(&mut self) {
        self.transcribed_upto = self.samples.len();
    }

    /// 收下一片 base64 的小端 i16 PCM。
    pub fn push_b64(&mut self, seq: u32, pcm_b64: &str) -> Result<Accepted, DaemonError> {
        let bytes = STANDARD
            .decode(pcm_b64)
            .map_err(|e| DaemonError::InvalidRequest(format!("speech chunk base64: {e}")))?;
        if bytes.len() % 2 != 0 {
            return Err(DaemonError::InvalidRequest(format!(
                "speech chunk is {} bytes, not a whole number of i16 samples",
                bytes.len()
            )));
        }
        Ok(self.push_pcm(seq, &bytes))
    }

    fn push_pcm(&mut self, seq: u32, bytes: &[u8]) -> Accepted {
        // 迟到或重复:直接丢。取消之后仍在路上的片就长这样,追加进去会污染
        // 下一段的缓冲,而症状是「转写里混进了上一句的尾巴」。
        if seq < self.expected_seq {
            return Accepted::Duplicate;
        }
        let outcome = if seq > self.expected_seq {
            let missing = seq - self.expected_seq;
            self.gaps += missing;
            Accepted::Gap { missing }
        } else {
            Accepted::Appended
        };

        self.samples.reserve(bytes.len() / 2);
        for pair in bytes.chunks_exact(2) {
            let v = i16::from_le_bytes([pair[0], pair[1]]);
            // i16::MIN 除以 32768 才落在 [-1, 1];用 32767 会让最负的样本溢出。
            self.samples.push(v as f32 / 32768.0);
        }
        self.expected_seq = seq + 1;
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(samples: &[i16]) -> String {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        STANDARD.encode(bytes)
    }

    #[test]
    fn appends_in_order_and_converts_to_unit_range() {
        let mut b = UtteranceBuffer::new("u1", 16_000);
        assert_eq!(
            b.push_b64(0, &b64(&[0, 16384, -16384])).unwrap(),
            Accepted::Appended
        );
        assert_eq!(b.samples().len(), 3);
        assert!((b.samples()[1] - 0.5).abs() < 1e-6);
        assert!((b.samples()[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn full_scale_negative_stays_in_range() {
        // 用 32767 做除数的话这里会是 -1.000030…,超出 [-1, 1]。
        let mut b = UtteranceBuffer::new("u1", 16_000);
        b.push_b64(0, &b64(&[i16::MIN])).unwrap();
        assert!(b.samples()[0] >= -1.0, "got {}", b.samples()[0]);
    }

    #[test]
    fn stale_chunks_are_dropped_not_appended() {
        let mut b = UtteranceBuffer::new("u1", 16_000);
        b.push_b64(0, &b64(&[1, 2])).unwrap();
        b.push_b64(1, &b64(&[3, 4])).unwrap();
        // 重投 seq 0 —— 取消后仍在路上的那种。
        assert_eq!(b.push_b64(0, &b64(&[9, 9])).unwrap(), Accepted::Duplicate);
        assert_eq!(b.samples().len(), 4, "缓冲不该被迟到的片污染");
    }

    #[test]
    fn a_gap_is_counted_not_silently_swallowed() {
        let mut b = UtteranceBuffer::new("u1", 16_000);
        b.push_b64(0, &b64(&[1])).unwrap();
        assert_eq!(
            b.push_b64(3, &b64(&[2])).unwrap(),
            Accepted::Gap { missing: 2 }
        );
        assert_eq!(b.gaps(), 2);
        // 尽力追加,但上层必须看得到这个洞。
        assert_eq!(b.samples().len(), 2);
    }

    #[test]
    fn has_new_audio_drives_the_adaptive_cadence() {
        let mut b = UtteranceBuffer::new("u1", 16_000);
        assert!(!b.has_new_audio(), "空缓冲没有可转的东西");

        b.push_b64(0, &b64(&[1; 8000])).unwrap();
        assert!(b.has_new_audio());

        b.mark_transcribed();
        assert!(!b.has_new_audio(), "跑完一次之后应等下一片");

        b.push_b64(1, &b64(&[1; 8000])).unwrap();
        assert!(b.has_new_audio(), "新音频到了应立刻接着跑");
    }

    #[test]
    fn buffered_ms_tracks_the_sample_rate() {
        let mut b = UtteranceBuffer::new("u1", 16_000);
        b.push_b64(0, &b64(&[0; 8000])).unwrap();
        assert_eq!(b.buffered_ms(), 500);
    }

    #[test]
    fn odd_byte_count_is_rejected_rather_than_truncated() {
        let mut b = UtteranceBuffer::new("u1", 16_000);
        let odd = STANDARD.encode([1u8, 2, 3]);
        assert!(b.push_b64(0, &odd).is_err());
    }
}
