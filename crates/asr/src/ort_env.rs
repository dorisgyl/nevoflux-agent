//! ONNX Runtime bootstrap and session construction.
//!
//! # This file is a third copy
//!
//! The dylib resolution below is duplicated, near-verbatim, in:
//!   * `crates/tts/src/model.rs`   (Kokoro)
//!   * `crates/llm/src/embedding.rs` (fastembed)
//!
//! It is copied rather than shared because collapsing the three means editing
//! two crates that currently work, which is not this change's business. If a
//! fourth appears, stop and extract it -- and note that all three must move
//! together, since they load one library into one process and disagreeing
//! about its version is what the check below exists to prevent.
//!
//! That check errors rather than warns for a specific reason: `ort`'s own
//! error path for a too-old runtime re-enters its `OnceLock` and deadlocks.
//! The symptom is a hang with no message, not a failure anyone can catch.

use crate::error::AsrError;
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// ONNX Runtime version the `ort` crate in this build links against.
///
/// Must move in lockstep with the `ort` pin (ort 2.0.0-rc.12 → ONNX Runtime
/// 1.24.x), and with `EXPECTED_ORT_VERSION` in `nevoflux-llm`'s embedding.rs:
/// both crates load the same library into the same process.
#[cfg(feature = "ort-load-dynamic")]
const EXPECTED_ORT_VERSION: &str = "1.24.2";

#[cfg(feature = "ort-load-dynamic")]
fn onnxruntime_lib_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libonnxruntime.dylib"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "libonnxruntime.so"
    }
}

/// First existing ONNX Runtime library under `base`: `<base>/lib/<name>`
/// (official tarball layout) then `<base>/<name>` (flat).
#[cfg(feature = "ort-load-dynamic")]
fn find_dylib_in(base: &Path) -> Option<PathBuf> {
    let name = onnxruntime_lib_name();
    [base.join("lib").join(name), base.join(name)]
        .into_iter()
        .find(|p| p.exists())
}

/// Decide which ONNX Runtime library to load (pure; no side effects).
///
/// `ORT_DYLIB_PATH` wins so operators can override. Otherwise look next to
/// the executable, then in its parent — test and example binaries run from
/// `target/<profile>/deps/` while `just ort-fetch` installs the runtime into
/// `target/<profile>/lib/`, so without the parent tier every test that builds
/// a session would hang.
#[cfg(feature = "ort-load-dynamic")]
fn resolve_dylib(env_override: Option<PathBuf>, exe_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = env_override {
        return Some(p);
    }
    let exe_dir = exe_dir?;
    find_dylib_in(exe_dir).or_else(|| exe_dir.parent().and_then(find_dylib_in))
}

/// Best-effort version extraction from a library file name, handling
/// `libonnxruntime.so.1.24.2` and `libonnxruntime.1.24.2.dylib`. `None` when
/// the name carries no version, in which case no check is possible.
#[cfg(feature = "ort-load-dynamic")]
fn parse_version(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    for run in name.split(|c: char| !(c.is_ascii_digit() || c == '.')) {
        let trimmed = run.trim_matches('.');
        if trimmed.contains('.') && trimmed.starts_with(|c: char| c.is_ascii_digit()) {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// ONNX Runtime's C API version tracks the minor release, so major and minor
/// must match; the patch may differ.
#[cfg(feature = "ort-load-dynamic")]
fn version_compatible(found: &str, expected: &str) -> bool {
    let major_minor = |v: &str| -> Option<(u32, u32)> {
        let mut it = v.split('.');
        Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
    };
    match (major_minor(found), major_minor(expected)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// Point `ort` at a validated runtime before it initializes.
///
/// Both halves of this matter. Resolving nothing is an error rather than a
/// warning because `ort`'s own fallback search can pick up an incompatible
/// runtime, and resolving a mismatched version is an error because `ort`'s
/// error path for a too-old runtime re-enters its own `OnceLock` and
/// deadlocks silently — a hang with no message, not a failure you can catch.
#[cfg(feature = "ort-load-dynamic")]
fn prepare_dylib() -> Result<(), AsrError> {
    static PREPARED: OnceLock<Result<(), String>> = OnceLock::new();
    PREPARED
        .get_or_init(|| {
            let env_override = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from);
            let already_set = env_override.is_some();
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(Path::to_path_buf));
            let Some(path) = resolve_dylib(env_override, exe_dir.as_deref()) else {
                return Err(format!(
                    "No ONNX Runtime dynamic library found: ORT_DYLIB_PATH is unset and no \
                     `{}` exists in <exe_dir>/lib/, <exe_dir>/, or <exe_dir>/../lib/. Run \
                     `just ort-fetch` to bundle ONNX Runtime {}, or set ORT_DYLIB_PATH.",
                    onnxruntime_lib_name(),
                    EXPECTED_ORT_VERSION
                ));
            };
            if let Some(found) = parse_version(&path) {
                if !version_compatible(&found, EXPECTED_ORT_VERSION) {
                    return Err(format!(
                        "ONNX Runtime version mismatch: {} reports {found}, but this build \
                         requires {EXPECTED_ORT_VERSION} (major.minor must match). Loading \
                         it would make `ort` deadlock on startup; refusing to continue.",
                        path.display()
                    ));
                }
            }
            if !already_set {
                // Safe here: this runs before any session is built, and the
                // OnceLock makes it happen exactly once.
                std::env::set_var("ORT_DYLIB_PATH", &path);
            }
            Ok(())
        })
        .clone()
        .map_err(AsrError::ModelCorrupt)
}

#[cfg(not(feature = "ort-load-dynamic"))]
fn prepare_dylib() -> Result<(), AsrError> {
    // Statically linked: nothing to resolve.
    Ok(())
}

/// Build an inference session.
///
/// `threads` is the intra-op width; see [`default_threads`].
pub fn load_session(path: &Path, threads: usize) -> Result<Session, AsrError> {
    if !path.exists() {
        return Err(AsrError::ModelNotFound(path.display().to_string()));
    }
    prepare_dylib()?;
    // Steps rather than a chain: ort's error type carries the builder as a
    // type parameter, so the intermediate Results do not unify.
    let corrupt = |e: String| AsrError::ModelCorrupt(format!("{}: {e}", path.display()));
    let builder = Session::builder().map_err(|e| corrupt(e.to_string()))?;
    let builder = builder
        .with_intra_threads(threads.max(1))
        .map_err(|e| corrupt(e.to_string()))?;
    let mut builder = builder
        .with_inter_threads(1)
        .map_err(|e| corrupt(e.to_string()))?;
    builder
        .commit_from_file(path)
        .map_err(|e| corrupt(e.to_string()))
}

/// Intra-op width when config does not say: logical cores capped at four.
///
/// Same reasoning as Kokoro's. Past the physical core count throughput falls
/// off, and a daemon serving several sessions wants the rest for concurrency.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 4)
}

/// Where model files live when config does not say.
pub fn default_model_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("nevoflux").join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_reports_not_found() {
        let err = load_session(Path::new("/nonexistent/sensevoice.onnx"), 1).unwrap_err();
        assert!(matches!(err, AsrError::ModelNotFound(_)), "got: {err}");
    }

    #[test]
    fn default_threads_is_within_the_cap() {
        let n = default_threads();
        assert!((1..=4).contains(&n), "{n}");
    }
}
