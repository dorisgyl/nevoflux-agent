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

// ---------------------------------------------------------------------------
// Segmentation. These need the VAD model as well.
// ---------------------------------------------------------------------------

fn vad() -> Option<nevoflux_asr::vad::Vad> {
    let dir = nevoflux_asr::ort_env::default_model_dir()?;
    let p = dir.join("silero-vad.onnx");
    p.exists()
        .then(|| nevoflux_asr::vad::Vad::new(&p).expect("load VAD"))
}

/// Concatenate clips with `gap_ms` of silence between them.
fn joined(names: &[&str], gap_ms: usize) -> Vec<f32> {
    let gap = vec![0.0f32; 16 * gap_ms];
    let mut out = Vec::new();
    for n in names {
        out.extend(fixture(n));
        out.extend_from_slice(&gap);
    }
    out
}

#[test]
fn vad_finds_one_span_per_utterance() {
    let (Some(v), Some(_)) = (vad(), engine()) else {
        return;
    };
    let audio = joined(&["zh.wav", "en.wav", "yue.wav"], 1000);
    let spans = v
        .detect(&audio, &nevoflux_asr::vad::VadOptions::default())
        .unwrap();
    // Each clip is one or two sentences, so a handful of spans -- but nothing
    // like one per window, and nothing like a single span for all three.
    assert!(
        (3..=8).contains(&spans.len()),
        "{} spans: {spans:?}",
        spans.len()
    );
    for w in spans.windows(2) {
        assert!(w[0].end_ms <= w[1].start_ms, "spans overlap: {w:?}");
    }
}

#[test]
fn segmenting_recovers_languages_a_single_pass_loses() {
    // This is the case that motivates segmentation. SenseVoice classifies
    // language once per call, so all four clips in one pass come back as
    // whichever language wins -- the other three vanish entirely.
    let (Some(v), Some(sv)) = (vad(), engine()) else {
        return;
    };
    let audio = joined(&["zh.wav", "en.wav", "yue.wav", "ja.wav"], 1000);

    let one_pass = sv.transcribe(&audio, None).unwrap();
    let segmented = nevoflux_asr::segmented::transcribe_segmented(
        &v,
        &sv,
        &audio,
        None,
        &nevoflux_asr::vad::VadOptions::default(),
    )
    .unwrap();

    // The Chinese clip's content survives segmenting and does not survive one
    // pass over mixed-language audio.
    // Assert on the stable part of the Chinese clip: the homophone in the
    // first word shifts with span context, the rest does not.
    assert!(
        segmented.text.contains("时间早上9点"),
        "segmented lost the Chinese: {}",
        segmented.text
    );
    assert!(
        segmented.text.contains("tribal chieftain"),
        "segmented lost the English: {}",
        segmented.text
    );
    assert!(
        !one_pass.text.contains("时间早上9点") || !one_pass.text.contains("tribal chieftain"),
        "a single pass unexpectedly kept both languages; if the model changed, \
         this test's premise needs revisiting: {}",
        one_pass.text
    );
}

#[test]
fn padding_keeps_the_onset_of_a_span_intact() {
    // A span starting exactly where the detector heard speech clips the quiet
    // onset, and the encoder then mis-reads the first syllable: at silero's
    // default 30 ms of padding this clip began 菜 rather than 开 -- a
    // different initial consonant, not a near miss.
    //
    // The assertion is on the onset only. Which homophone follows a correct
    // onset (开饭 or 开放, both kāifàn(g)) moves with how much context the
    // span carries, and is not what padding controls.
    let (Some(v), Some(sv)) = (vad(), engine()) else {
        return;
    };
    let audio = joined(&["zh.wav", "en.wav"], 1000);
    let t = nevoflux_asr::segmented::transcribe_segmented(
        &v,
        &sv,
        &audio,
        None,
        &nevoflux_asr::vad::VadOptions::default(),
    )
    .unwrap();
    assert!(
        t.text.starts_with('开'),
        "onset was clipped -- expected the transcript to open on 开: {}",
        t.text
    );
    assert!(t.text.contains("时间早上9点"), "{}", t.text);
}

#[test]
fn segment_timestamps_stay_inside_the_recording() {
    let (Some(v), Some(sv)) = (vad(), engine()) else {
        return;
    };
    let audio = joined(&["zh.wav", "en.wav", "yue.wav"], 800);
    let duration_ms = (audio.len() as f32 / 16.0) as u32;
    let t = nevoflux_asr::segmented::transcribe_segmented(
        &v,
        &sv,
        &audio,
        None,
        &nevoflux_asr::vad::VadOptions::default(),
    )
    .unwrap();
    assert!(!t.segments.is_empty());
    for s in &t.segments {
        assert!(s.start_ms < s.end_ms, "inverted: {s:?}");
        assert!(s.end_ms <= duration_ms, "{s:?} past {duration_ms} ms");
    }
    for w in t.segments.windows(2) {
        assert!(w[0].start_ms <= w[1].start_ms, "out of order: {w:?}");
    }
}

#[test]
fn a_later_utterance_is_timed_from_the_recording_not_its_span() {
    // The rebasing that `stitch` exists for: the third clip starts around
    // 14 s into the concatenation and its segments must say so.
    let (Some(v), Some(sv)) = (vad(), engine()) else {
        return;
    };
    let audio = joined(&["zh.wav", "en.wav", "yue.wav"], 1000);
    let t = nevoflux_asr::segmented::transcribe_segmented(
        &v,
        &sv,
        &audio,
        None,
        &nevoflux_asr::vad::VadOptions::default(),
    )
    .unwrap();
    let cantonese = t
        .segments
        .iter()
        .find(|s| s.text.contains("表达唔到"))
        .expect("the Cantonese clip");
    assert!(
        cantonese.start_ms > 12_000,
        "Cantonese segment starts at {} ms; it is third in the recording",
        cantonese.start_ms
    );
}

#[test]
fn silence_between_utterances_produces_no_segments() {
    let (Some(v), Some(sv)) = (vad(), engine()) else {
        return;
    };
    let audio = joined(&["zh.wav"], 3000);
    let t = nevoflux_asr::segmented::transcribe_segmented(
        &v,
        &sv,
        &audio,
        None,
        &nevoflux_asr::vad::VadOptions::default(),
    )
    .unwrap();
    // The three seconds of trailing silence must not become a segment.
    let speech_end = t.segments.last().map(|s| s.end_ms).unwrap_or(0);
    assert!(
        speech_end < 7000,
        "a segment ran into the trailing silence: {:?}",
        t.segments
    );
}
