//! Length limits.
//!
//! Resampling and channel folding are ffmpeg's job, on the daemon side; audio
//! is already 16 kHz mono by the time it reaches this crate. What is left here
//! is the ceiling, and the ceiling is load-bearing rather than decorative:
//! none of the three dispatch surfaces imposes a tool timeout, a transcription
//! holds a runtime worker for its whole duration, and its activations grow
//! with the length of the audio. This is the only guard rail there is.

use crate::error::AsrError;
use crate::{Engine, SAMPLE_RATE};

/// Peak inference memory a single call may claim, in megabytes.
///
/// The ceiling is derived from memory rather than from time, because memory
/// binds first and it is not close. Measured on the int8 export: activations
/// grow about 6 MB per second of audio on top of the ~240 MB model, so five
/// minutes peaks near 2 GB. Meanwhile the realtime factor decays from 14x to
/// 5.9x across the same range, which would put a ten-minute *blocking* budget
/// somewhere past 1500 s of audio -- by which point the process would be
/// asking for roughly 9 GB.
///
/// An earlier version of this file did derive the ceiling from a blocking
/// budget, and landed on 8400 s. That would have been about 50 GB.
pub const MAX_INFERENCE_MEMORY_MB: u32 = 1500;

/// Measured activation growth: about 6 MB per second of audio.
///
/// CPU, int8 export, four intra-op threads; peak RSS minus the loaded model:
/// 30.5 s → 223 MB, 60.9 s → 454 MB, 121.8 s → 738 MB, 304.6 s → 1771 MB.
/// Close enough to linear over that range to divide by.
pub const SENSEVOICE_MB_PER_SECOND: u32 = 6;

/// Measured throughput on short clips: 14x realtime.
///
/// CPU, int8 export, four threads, over the four sherpa test clips
/// (zh/en/ja/yue, 5-7 s each): 13.9, 14.3, 14.2, 14.1. It decays with length
/// -- 12.9x at 30 s, 9.1x at 122 s, 5.9x at 305 s -- so this describes the
/// short-clip path and is not what the ceiling rests on.
///
/// The published 169x is a GPU number and was never evidence for any of this.
/// Re-measure with `cargo run -p nevoflux-asr --example transcribe` when the
/// export or the thread count changes.
pub const SENSEVOICE_SHORT_CLIP_RTF: u32 = 14;

/// Whisper: unmeasured, and deliberately pessimistic until it is. Assumed to
/// run at a third of realtime. Revisit when the engine lands.
pub const WHISPER_ASSUMED_RTF_DIVISOR: u32 = 3;
const WHISPER_MAX_BLOCKING_MINUTES: u32 = 10;

/// How long a segmented call may block, in minutes.
///
/// Segmenting makes memory flat -- every pass is one span, so peak RSS stops
/// tracking the length of the recording -- which puts the binding constraint
/// back on time, where the first version of this file wrongly assumed it
/// already was. Measured on the segmented path: 305 s of audio at 14.6x
/// realtime and 528 MB, against 5.9x and 2011 MB for the same audio in one
/// pass.
const SEGMENTED_MAX_BLOCKING_MINUTES: u32 = 10;

/// The longest audio, in seconds, an engine accepts when it is cut at pauses
/// first.
///
/// Roughly 2.5 hours for SenseVoice. Whisper keeps its single-pass figure
/// until someone measures the segmented path for it.
pub fn max_seconds_segmented(engine: Engine) -> u32 {
    match engine {
        Engine::Sensevoice => SEGMENTED_MAX_BLOCKING_MINUTES * 60 * SENSEVOICE_SHORT_CLIP_RTF,
        Engine::Whisper => max_seconds(engine),
    }
}

/// The longest audio, in seconds, an engine accepts in a single pass.
///
/// Bounded by memory rather than time: activations grow with the length of the
/// audio. [`max_seconds_segmented`] is the limit that applies once the audio
/// is cut at pauses, and it is much higher because each pass is then short.
pub fn max_seconds(engine: Engine) -> u32 {
    match engine {
        Engine::Sensevoice => MAX_INFERENCE_MEMORY_MB / SENSEVOICE_MB_PER_SECOND,
        Engine::Whisper => WHISPER_MAX_BLOCKING_MINUTES * 60 / WHISPER_ASSUMED_RTF_DIVISOR,
    }
}

pub fn duration_ms(samples: &[f32]) -> u32 {
    ((samples.len() as u64 * 1000) / SAMPLE_RATE as u64) as u32
}

/// Reject audio too long for the path that will actually run it.
///
/// `segmented` says whether the caller will cut at pauses first, which is what
/// decides whether memory or time is the binding constraint.
pub fn check_length_for(samples: &[f32], engine: Engine, segmented: bool) -> Result<(), AsrError> {
    if samples.is_empty() {
        return Err(AsrError::InvalidAudio("audio is empty".into()));
    }
    let secs = duration_ms(samples) / 1000;
    let cap = if segmented {
        max_seconds_segmented(engine)
    } else {
        max_seconds(engine)
    };
    if secs <= cap {
        return Ok(());
    }
    Err(AsrError::InvalidAudio(format!(
        "audio is {secs}s, past the {cap}s ceiling for {}; split it into shorter clips",
        engine.as_str()
    )))
}

/// Reject audio that would hold a worker thread past the single-pass ceiling.
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
    fn sensevoice_ceiling_keeps_peak_memory_within_budget() {
        let secs = max_seconds(Engine::Sensevoice);
        assert!(
            secs * SENSEVOICE_MB_PER_SECOND <= MAX_INFERENCE_MEMORY_MB,
            "{secs}s of audio would want {} MB",
            secs * SENSEVOICE_MB_PER_SECOND
        );
        assert!(secs >= 120, "a two-minute clip must still fit; got {secs}s");
    }

    #[test]
    fn segmenting_raises_the_ceiling_by_an_order_of_magnitude() {
        // The whole point: cutting at pauses makes memory flat, so the limit
        // stops being about how much audio fits in RAM at once.
        assert!(
            max_seconds_segmented(Engine::Sensevoice) > 10 * max_seconds(Engine::Sensevoice),
            "{} vs {}",
            max_seconds_segmented(Engine::Sensevoice),
            max_seconds(Engine::Sensevoice)
        );
    }

    #[test]
    fn an_hour_is_accepted_segmented_and_refused_in_one_pass() {
        let hour = vec![0.0f32; SAMPLE_RATE as usize * 3600];
        assert!(check_length_for(&hour, Engine::Sensevoice, true).is_ok());
        assert!(check_length_for(&hour, Engine::Sensevoice, false).is_err());
    }

    #[test]
    fn no_single_pass_ceiling_is_near_the_old_time_derived_one() {
        // 8400 s came from a blocking budget that ignored memory; it would
        // have asked for roughly 50 GB. Guard against that order returning.
        for engine in [Engine::Sensevoice, Engine::Whisper] {
            assert!(
                max_seconds(engine) < 1000,
                "{engine:?}: {}",
                max_seconds(engine)
            );
        }
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
        assert!(
            !msg.contains("engine=\"sensevoice\""),
            "circular advice: {msg}"
        );
    }
}
