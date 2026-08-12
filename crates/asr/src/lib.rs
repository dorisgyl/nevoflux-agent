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
pub mod route;
pub mod stitch;

pub use error::AsrError;
pub use route::route;

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
}

/// The seam both engines sit behind.
pub trait Transcriber: Send + Sync {
    /// `samples` is 16 kHz mono f32 in [-1.0, 1.0].
    fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<Transcript, AsrError>;
}
