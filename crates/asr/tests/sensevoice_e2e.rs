//! SenseVoice against real weights and real speech.
//!
//! The unit tests cover the pipeline on synthetic signals, which can show that
//! the arithmetic is self-consistent but never that it agrees with what the
//! model was trained on. A window function or a mel edge that is subtly wrong
//! passes every one of them and only shows up as a worse transcript. These
//! tests are the ones that would catch that.
//!
//! Skipped when `just fetch-asr-models` has not run: 237 MB is not something
//! to make `cargo test` depend on.

#![cfg(all(feature = "sensevoice", feature = "ort-load-dynamic"))]

use nevoflux_asr::sensevoice::SenseVoice;
use nevoflux_asr::Transcriber;
use std::path::{Path, PathBuf};

fn models() -> Option<(PathBuf, PathBuf)> {
    let dir = nevoflux_asr::ort_env::default_model_dir()?;
    let model = dir.join("sensevoice-small.int8.onnx");
    let tokens = dir.join("sensevoice-tokens.txt");
    (model.exists() && tokens.exists()).then_some((model, tokens))
}

/// 16-bit PCM WAV, mono, as the fixtures are.
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
    panic!("no data chunk in {}", path.display());
}

fn fixture(name: &str) -> Vec<f32> {
    read_wav(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
}

fn engine() -> Option<SenseVoice> {
    let (model, tokens) = models()?;
    Some(SenseVoice::new(&model, &tokens, 4).expect("load SenseVoice"))
}

#[test]
fn transcribes_mandarin() {
    let Some(sv) = engine() else { return };
    let t = sv.transcribe(&fixture("zh.wav"), Some("zh")).unwrap();
    // The reference transcript for this clip. Exact-match rather than a CER
    // threshold: the whole point is to catch preprocessing drift, and drift
    // shows up as a handful of wrong characters that a loose threshold hides.
    assert_eq!(t.text, "开饭时间早上9点至下午5点。");
    assert_eq!(t.language, "zh");
}

#[test]
fn inverse_text_normalization_is_on() {
    // "9点" rather than "九点" is ITN doing its job -- which means text_norm
    // was passed as with_itn, not without_itn. Getting that backwards is
    // invisible except here.
    let Some(sv) = engine() else { return };
    let t = sv.transcribe(&fixture("zh.wav"), Some("zh")).unwrap();
    assert!(t.text.contains('9'), "expected digits, got: {}", t.text);
}

#[test]
fn transcribes_english() {
    let Some(sv) = engine() else { return };
    let t = sv.transcribe(&fixture("en.wav"), Some("en")).unwrap();
    assert_eq!(
        t.text,
        "The tribal chieftain called for the boy and presented him with 50 pieces of code."
    );
    assert_eq!(t.language, "en");
}

#[test]
fn english_words_are_spaced_and_chinese_is_not() {
    let Some(sv) = engine() else { return };
    let en = sv.transcribe(&fixture("en.wav"), Some("en")).unwrap();
    let zh = sv.transcribe(&fixture("zh.wav"), Some("zh")).unwrap();
    assert!(en.text.contains(' '), "English lost its word boundaries");
    assert!(
        !zh.text.trim().contains(' '),
        "Chinese gained spaces between characters: {}",
        zh.text
    );
}

#[test]
fn transcribes_cantonese() {
    let Some(sv) = engine() else { return };
    let t = sv.transcribe(&fixture("yue.wav"), Some("yue")).unwrap();
    assert_eq!(t.text, "呢几个字都表达唔到我想讲嘅意思。");
    assert_eq!(t.language, "yue");
}

#[test]
fn detects_the_language_when_none_is_given() {
    // This is what the `auto` route depends on being true.
    let Some(sv) = engine() else { return };
    assert_eq!(
        sv.transcribe(&fixture("zh.wav"), None).unwrap().language,
        "zh"
    );
    assert_eq!(
        sv.transcribe(&fixture("en.wav"), None).unwrap().language,
        "en"
    );
}

#[test]
fn segments_carry_plausible_timestamps() {
    let Some(sv) = engine() else { return };
    let samples = fixture("zh.wav");
    let duration_ms = (samples.len() as f32 / 16.0) as u32;
    let t = sv.transcribe(&samples, Some("zh")).unwrap();

    assert!(!t.segments.is_empty(), "no segments");
    for s in &t.segments {
        assert!(s.start_ms <= s.end_ms, "inverted: {s:?}");
        assert!(
            s.end_ms <= duration_ms,
            "segment ends at {} ms, past the {duration_ms} ms clip",
            s.end_ms
        );
    }
    for w in t.segments.windows(2) {
        assert!(w[0].start_ms <= w[1].start_ms, "segments out of order");
    }
}

#[test]
fn the_four_tag_tokens_never_reach_the_transcript() {
    // Language, emotion, event and ITN tags are model output, not speech.
    let Some(sv) = engine() else { return };
    for name in ["zh.wav", "en.wav", "yue.wav"] {
        let t = sv.transcribe(&fixture(name), None).unwrap();
        assert!(!t.text.contains("<|"), "{name} leaked a tag: {}", t.text);
        assert!(!t.text.contains("|>"), "{name} leaked a tag: {}", t.text);
    }
}

#[test]
fn audio_shorter_than_one_window_is_rejected_not_panicked() {
    let Some(sv) = engine() else { return };
    let err = sv.transcribe(&[0.0f32; 100], None).unwrap_err();
    assert!(err.to_string().contains("window"), "{err}");
}

#[test]
fn silence_does_not_hallucinate_a_transcript() {
    // Digital silence is out of distribution -- no microphone produces exact
    // zeros -- and the model does emit a stray character for it. That is its
    // behaviour, not a fault in this pipeline, so the property worth holding
    // is bounded output rather than empty output: two seconds of nothing must
    // not come back as a sentence.
    let Some(sv) = engine() else { return };
    let t = sv.transcribe(&vec![0.0f32; 16000 * 2], None).unwrap();
    assert!(
        t.text.chars().count() <= 4,
        "silence produced {} characters: {}",
        t.text.chars().count(),
        t.text
    );
}

#[test]
fn timestamps_account_for_the_prepended_tag_frames() {
    // The graph returns input_frames + 4, because it prepends four query
    // positions for the tags. Forgetting that shifts every timestamp 240 ms
    // late, which looks plausible everywhere except at the very end.
    let Some(sv) = engine() else { return };
    let samples = fixture("zh.wav");
    let duration_ms = (samples.len() as f32 / 16.0) as u32;
    let t = sv.transcribe(&samples, Some("zh")).unwrap();
    let last = t.segments.last().expect("a segment");
    assert!(
        last.end_ms <= duration_ms,
        "last segment ends at {} ms but the clip is {duration_ms} ms",
        last.end_ms
    );
}
