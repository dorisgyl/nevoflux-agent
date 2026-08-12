//! Length limits.
//!
//! Resampling and channel folding are ffmpeg's job, on the daemon side; audio
//! is already 16 kHz mono by the time it reaches this crate. What is left here
//! is the ceiling, and the ceiling is load-bearing rather than decorative:
//! none of the three dispatch surfaces imposes a tool timeout, and a
//! transcription holds a runtime worker thread for its entire duration. This
//! is the only guard rail there is.

use crate::error::AsrError;
use crate::{Engine, SAMPLE_RATE};

/// How long a single call may block, in minutes.
///
/// The two ceilings below are this number scaled by each engine's realtime
/// factor, so retuning both is a one-number change.
pub const MAX_BLOCKING_MINUTES: u32 = 10;

/// Provisional, pending the Task 10 benchmark.
///
/// The published SenseVoice figures -- 70 ms for 10 s of audio, 169x realtime
/// -- are GPU measurements, and this runs on CPU. They are not evidence for
/// anything here, so this is a deliberately conservative placeholder rather
/// than a derived number.
pub const SENSEVOICE_ASSUMED_RTF: u32 = 24;

/// Provisional, pending the Task 12 benchmark: Whisper assumed to run at one
/// third of realtime. Expressed as a divisor to keep the arithmetic integral.
pub const WHISPER_ASSUMED_RTF_DIVISOR: u32 = 3;

/// The longest audio, in seconds, this engine will accept in one call.
pub fn max_seconds(engine: Engine) -> u32 {
    match engine {
        Engine::Sensevoice => MAX_BLOCKING_MINUTES * 60 * SENSEVOICE_ASSUMED_RTF,
        Engine::Whisper => MAX_BLOCKING_MINUTES * 60 / WHISPER_ASSUMED_RTF_DIVISOR,
    }
}

pub fn duration_ms(samples: &[f32]) -> u32 {
    ((samples.len() as u64 * 1000) / SAMPLE_RATE as u64) as u32
}

/// Reject audio that would hold a worker thread past the ceiling.
pub fn check_length(samples: &[f32], engine: Engine) -> Result<(), AsrError> {
    if samples.is_empty() {
        return Err(AsrError::InvalidAudio("audio is empty".into()));
    }
    let secs = duration_ms(samples) / 1000;
    let cap = max_seconds(engine);
    if secs <= cap {
        return Ok(());
    }
    // Say what to do about it. The caller is usually a model, and "too long"
    // without a next step tends to produce a retry of the same call.
    let remedy = match engine {
        Engine::Whisper => format!(
            "split it into shorter clips, or -- if the audio is in \
             zh/yue/en/ja/ko -- use engine=\"sensevoice\", whose ceiling is {}s",
            max_seconds(Engine::Sensevoice)
        ),
        Engine::Sensevoice => "split it into shorter clips".to_string(),
    };
    Err(AsrError::InvalidAudio(format!(
        "audio is {secs}s, past the {cap}s ceiling for {}; {remedy}",
        engine.as_str()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seconds_of_audio(secs: u32) -> Vec<f32> {
        vec![0.0f32; SAMPLE_RATE as usize * secs as usize]
    }

    #[test]
    fn duration_of_one_second() {
        assert_eq!(duration_ms(&seconds_of_audio(1)), 1000);
    }

    #[test]
    fn duration_rounds_down_on_partial_samples() {
        let s = vec![0.0f32; SAMPLE_RATE as usize + SAMPLE_RATE as usize / 2];
        assert_eq!(duration_ms(&s), 1500);
    }

    #[test]
    fn empty_audio_is_rejected() {
        let err = check_length(&[], Engine::Sensevoice).unwrap_err();
        assert!(matches!(err, AsrError::InvalidAudio(_)));
        assert!(err.to_string().contains("empty"), "{err}");
    }

    #[test]
    fn short_audio_passes_both_engines() {
        let s = seconds_of_audio(5);
        assert!(check_length(&s, Engine::Sensevoice).is_ok());
        assert!(check_length(&s, Engine::Whisper).is_ok());
    }

    #[test]
    fn whisper_ceiling_is_tighter_than_sensevoice() {
        assert!(max_seconds(Engine::Whisper) < max_seconds(Engine::Sensevoice));
    }

    #[test]
    fn audio_at_exactly_the_ceiling_is_accepted() {
        let s = seconds_of_audio(max_seconds(Engine::Whisper));
        assert!(check_length(&s, Engine::Whisper).is_ok());
    }

    #[test]
    fn over_ceiling_names_a_next_step() {
        let s = seconds_of_audio(max_seconds(Engine::Whisper) + 60);
        let err = check_length(&s, Engine::Whisper).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ceiling"), "{msg}");
        assert!(msg.contains("split it"), "{msg}");
        assert!(msg.contains("sensevoice"), "{msg}");
    }

    #[test]
    fn sensevoice_over_ceiling_does_not_suggest_itself() {
        let s = seconds_of_audio(max_seconds(Engine::Sensevoice) + 60);
        let err = check_length(&s, Engine::Sensevoice).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("split it"), "{msg}");
        assert!(!msg.contains("engine=\"sensevoice\""), "circular advice: {msg}");
    }
}
