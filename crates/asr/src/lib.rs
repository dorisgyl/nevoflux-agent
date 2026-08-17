//! Local speech recognition.
//!
//! Two engines behind one trait: SenseVoice (ONNX via `ort`) is the default --
//! it is the accurate one for Chinese and the fast one for everything -- and
//! Whisper (Candle) covers the languages SenseVoice cannot tell apart.
//!
//! The crate takes PCM and returns text. Base64, artifacts, compositions and
//! MCP stop at the daemon, exactly as they do for `nevoflux-tts`; that is what
//! keeps the inference here testable without any of it.

pub mod audio;
pub mod error;
#[cfg(feature = "sensevoice")]
pub mod ort_env;
pub mod route;
#[cfg(feature = "sensevoice")]
pub mod segmented;
#[cfg(feature = "sensevoice")]
pub mod sensevoice;
pub mod stitch;
#[cfg(feature = "sensevoice")]
pub mod vad;
#[cfg(feature = "whisper")]
pub mod whisper;

pub use error::AsrError;
pub use route::route;

/// Where model files live when config does not say.
///
/// At the crate root rather than beside either engine: it is a cache
/// directory, not an ONNX detail, and gating it behind one engine's feature
/// made the other engine's tests unable to find their own weights.
pub fn default_model_dir() -> Option<std::path::PathBuf> {
    dirs::cache_dir().map(|d| d.join("nevoflux").join("models"))
}

/// Every engine here consumes 16 kHz mono. Resampling happens before the
/// crate boundary, where ffmpeg already is.
pub const SAMPLE_RATE: u32 = 16000;

/// Which backend transcribes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Sensevoice,
    Whisper,
}

impl Engine {
    /// The wire name. Must match what the protocol's `engine` field carries
    /// in both directions -- it is parsed from requests and reported back on
    /// responses.
    pub fn as_str(&self) -> &'static str {
        match self {
            Engine::Sensevoice => "sensevoice",
            Engine::Whisper => "whisper",
        }
    }
}

/// One stretch of speech, placed on the original audio's timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub start_ms: u32,
    pub end_ms: u32,
    pub text: String,
}

/// What an engine returns for one call.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<Segment>,
    /// The language the engine reports having heard.
    pub language: String,
    /// What kind of sound this was, when the engine says: `Speech`, `BGM`,
    /// `Applause`, `Laughter`.
    ///
    /// `None` means the engine does not report it — not that the audio was
    /// silent. Callers gating on this must treat `None` as "cannot judge" and
    /// let the audio through; rejecting on `None` would make voice input stop
    /// working entirely under any engine that lacks the tag, which is a far
    /// worse failure than losing a filter that never could stop a colleague
    /// talking anyway.
    pub audio_event: Option<String>,
}

/// The seam both engines sit behind.
pub trait Transcriber: Send + Sync {
    /// `samples` is 16 kHz mono f32 in [-1.0, 1.0].
    fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<Transcript, AsrError>;
}
