//! Whisper against real weights.
//!
//! Skipped unless `just whisper-model <size>` has run. Prefers whichever size
//! is present, smallest first: these check wiring -- language selection, the
//! tag tokens, timestamps -- not transcription quality, and tiny answers that
//! in a second where large-v3-turbo needs half a minute and 4.8 GB.

#![cfg(feature = "whisper")]

use nevoflux_asr::whisper::WhisperEngine;
use nevoflux_asr::Transcriber;
use std::path::{Path, PathBuf};

/// Smallest available model wins: these tests are about plumbing.
fn model_dir() -> Option<PathBuf> {
    let base = nevoflux_asr::default_model_dir()?;
    ["tiny", "base", "small", "medium", "large-v3-turbo"]
        .into_iter()
        .map(|s| base.join(format!("whisper-{s}")))
        .find(|d| d.join("model.safetensors").exists())
}

fn engine() -> Option<WhisperEngine> {
    Some(WhisperEngine::new(&model_dir()?).expect("load Whisper"))
}

fn read_wav(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).expect("fixture");
    let mut pos = 12;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = pos + 8;
        if id == b"data" {
            let end = (body + size).min(b.len());
            return b[body..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
        }
        pos = body + size + (size & 1);
    }
    panic!("no data chunk");
}

fn fixture(name: &str) -> Vec<f32> {
    read_wav(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
}

#[test]
fn a_language_hint_is_honoured() {
    let Some(w) = engine() else { return };
    let t = w.transcribe(&fixture("ja.wav"), Some("ja")).unwrap();
    assert_eq!(t.language, "ja");
    assert!(
        t.text
            .chars()
            .any(|c| ('\u{3040}'..='\u{30ff}').contains(&c)),
        "no kana in a Japanese transcript: {}",
        t.text
    );
}

#[test]
fn a_missing_language_is_detected_not_left_unset() {
    // This is the one that has to hold. A multilingual Whisper decodes
    // SOT -> language -> task; leaving the language token out does not make it
    // guess, it makes it emit a run of one repeated character. Measured before
    // detection was wired: the Japanese clip came back as 普 forty times.
    let Some(w) = engine() else { return };
    for (name, expect) in [("ja.wav", "ja"), ("en.wav", "en"), ("zh.wav", "zh")] {
        let t = w.transcribe(&fixture(name), None).unwrap();
        assert_eq!(t.language, expect, "{name} detected as {}", t.language);

        // Degenerate output check: a repeated-character run has almost no
        // distinct characters relative to its length.
        let chars: Vec<char> = t.text.chars().collect();
        let distinct: std::collections::HashSet<&char> = chars.iter().collect();
        assert!(
            chars.len() < 8 || distinct.len() * 4 > chars.len(),
            "{name} looks degenerate ({} distinct of {}): {}",
            distinct.len(),
            chars.len(),
            t.text
        );
    }
}

#[test]
fn a_region_subtag_is_accepted() {
    let Some(w) = engine() else { return };
    assert_eq!(
        w.transcribe(&fixture("ja.wav"), Some("ja-JP"))
            .unwrap()
            .language,
        "ja"
    );
}

#[test]
fn an_unknown_language_tag_falls_back_to_detection() {
    // Better to transcribe under a detected language than to refuse because
    // the caller named something this tokenizer does not carry.
    let Some(w) = engine() else { return };
    let t = w.transcribe(&fixture("en.wav"), Some("klingon")).unwrap();
    assert_eq!(t.language, "en");
}

#[test]
fn special_tokens_never_reach_the_transcript() {
    let Some(w) = engine() else { return };
    for name in ["en.wav", "ja.wav"] {
        let t = w.transcribe(&fixture(name), None).unwrap();
        assert!(!t.text.contains("<|"), "{name} leaked a tag: {}", t.text);
        assert!(!t.text.contains("|>"), "{name} leaked a tag: {}", t.text);
    }
}

#[test]
fn segments_are_ordered_and_inside_the_recording() {
    let Some(w) = engine() else { return };
    let samples = fixture("en.wav");
    let duration_ms = (samples.len() as f32 / 16.0) as u32;
    let t = w.transcribe(&samples, Some("en")).unwrap();
    assert!(!t.segments.is_empty(), "no segments");
    for s in &t.segments {
        assert!(s.start_ms <= s.end_ms, "inverted: {s:?}");
        // Whisper times against its 30 s window, so a short clip's last
        // segment can run to the window edge rather than the audio edge.
        assert!(
            s.end_ms <= duration_ms.max(30_000) + 100,
            "{s:?} past the window"
        );
    }
    for pair in t.segments.windows(2) {
        assert!(
            pair[0].start_ms <= pair[1].start_ms,
            "out of order: {pair:?}"
        );
    }
}

#[test]
fn the_transcript_is_the_segments_joined() {
    let Some(w) = engine() else { return };
    let t = w.transcribe(&fixture("en.wav"), Some("en")).unwrap();
    let joined: String = t.segments.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(t.text, joined.trim());
}

#[test]
fn transcription_is_reproducible() {
    // The temperature ladder samples when it falls back, so this would drift
    // if the sampler were seeded from the clock.
    let Some(w) = engine() else { return };
    let a = w.transcribe(&fixture("en.wav"), Some("en")).unwrap();
    let b = w.transcribe(&fixture("en.wav"), Some("en")).unwrap();
    assert_eq!(a.text, b.text);
}

#[test]
fn audio_shorter_than_a_window_is_padded_rather_than_refused() {
    // Whisper differs from SenseVoice here, legitimately: it always decodes a
    // 30 s window and pads whatever it is given, so eight samples is a valid
    // -- if empty -- request rather than an error. Asserting the SenseVoice
    // contract here would be asserting a similarity the engines do not have.
    let Some(w) = engine() else { return };
    let t = w
        .transcribe(&[0.0f32; 8], None)
        .expect("padding, not an error");
    assert!(
        t.text.chars().count() < 40,
        "eight samples of silence transcribed to: {}",
        t.text
    );
}
