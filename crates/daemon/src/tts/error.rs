//! TTS error type with mapping to dispatch-layer error codes.
//!
//! Daemon dispatch arms convert `TtsError` to `HostError` (direct API
//! path) or to a stringified MCP error (ACP path). The `code()` method
//! returns the canonical numeric code documented in the protocol crate's
//! tool error taxonomy.

use std::fmt;

#[derive(Debug)]
pub enum TtsError {
    /// 4001 — request shape invalid (empty text, oversize, etc.)
    InvalidRequest(String),
    /// 4002 — backend not configured (missing API key etc.)
    ConfigMissing(String),
    /// 4003 — API auth failure (401 from provider).
    AuthFailed(String),
    /// 4004 — provider rate limit or quota exceeded (429).
    RateLimit(String),
    /// 4005 — network / IO failure.
    Network(String),
    /// 4006 — provider returned non-2xx for non-auth reasons.
    BackendError { status: u16, body: String },
    /// 4007 — the engine is not compiled into this build.
    ///
    /// Deliberately not `ConfigMissing`. That one means "the model file is not
    /// on disk", and its remedy is a download or a path in config.toml. This
    /// means "that code is not in the binary", and no amount of editing
    /// config.toml will help. Collapsing the two sends people to the wrong
    /// file and leaves them there.
    EngineUnavailable(String),
    /// 4099 — unexpected daemon-side error (deserialization, etc.).
    Internal(String),
}

impl TtsError {
    pub fn code(&self) -> u32 {
        match self {
            TtsError::InvalidRequest(_) => 4001,
            TtsError::ConfigMissing(_) => 4002,
            TtsError::AuthFailed(_) => 4003,
            TtsError::RateLimit(_) => 4004,
            TtsError::Network(_) => 4005,
            TtsError::BackendError { .. } => 4006,
            TtsError::EngineUnavailable(_) => 4007,
            TtsError::Internal(_) => 4099,
        }
    }
}

impl fmt::Display for TtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TtsError::InvalidRequest(m) => write!(f, "invalid request: {m}"),
            TtsError::ConfigMissing(m) => write!(f, "config missing: {m}"),
            TtsError::AuthFailed(m) => write!(f, "auth failed: {m}"),
            TtsError::RateLimit(m) => write!(f, "rate limit: {m}"),
            TtsError::Network(m) => write!(f, "network error: {m}"),
            TtsError::BackendError { status, body } => {
                write!(f, "backend error ({status}): {body}")
            }
            TtsError::EngineUnavailable(m) => write!(f, "engine unavailable: {m}"),
            TtsError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for TtsError {}

impl From<reqwest::Error> for TtsError {
    fn from(e: reqwest::Error) -> Self {
        // Network-vs-backend distinction made at the call site; this
        // From impl is the catch-all for "couldn't even send the request".
        TtsError::Network(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_unavailable_has_its_own_code() {
        let e = TtsError::EngineUnavailable("whisper".into());
        assert_eq!(e.code(), 4007);
        assert_ne!(e.code(), TtsError::ConfigMissing("x".into()).code());
    }

    #[test]
    fn engine_unavailable_message_points_at_build_not_config() {
        let e = TtsError::EngineUnavailable("whisper is not in this build".into());
        let msg = e.to_string();
        assert!(msg.contains("engine unavailable"), "{msg}");
        assert!(!msg.contains("config missing"), "{msg}");
    }

    #[test]
    fn codes_are_distinct_across_variants() {
        let codes = [
            TtsError::InvalidRequest(String::new()).code(),
            TtsError::ConfigMissing(String::new()).code(),
            TtsError::AuthFailed(String::new()).code(),
            TtsError::RateLimit(String::new()).code(),
            TtsError::Network(String::new()).code(),
            TtsError::BackendError {
                status: 500,
                body: String::new(),
            }
            .code(),
            TtsError::EngineUnavailable(String::new()).code(),
            TtsError::Internal(String::new()).code(),
        ];
        let mut sorted = codes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            codes.len(),
            "duplicate error codes: {codes:?}"
        );
    }
}
