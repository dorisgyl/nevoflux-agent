//! ONNX session construction and model path resolution.

use crate::error::TtsError;
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
fn prepare_dylib() -> Result<(), TtsError> {
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
        .map_err(TtsError::ModelCorrupt)
}

#[cfg(not(feature = "ort-load-dynamic"))]
fn prepare_dylib() -> Result<(), TtsError> {
    // Statically linked: nothing to resolve.
    Ok(())
}

/// Build an inference session for the Kokoro ONNX graph.
///
/// `threads` is the intra-op width. It is a parameter rather than a constant
/// because the useful value is the machine's *physical* core count and the
/// curve past it slopes back down — measured on a 4-core/8-thread i7-7700K
/// with the fp32 model: 1.45x realtime at one thread, 3.36x at four, then
/// 2.66x at five. Hyperthreads are contention here, not capacity.
pub fn load_session(path: &Path, threads: usize) -> Result<Session, TtsError> {
    if !path.exists() {
        return Err(TtsError::ModelNotFound(path.display().to_string()));
    }
    prepare_dylib()?;
    // Written as steps rather than a chain: ort's error type carries the
    // builder as a type parameter, so the intermediate Results do not unify.
    let corrupt = |e: String| TtsError::ModelCorrupt(format!("{}: {e}", path.display()));
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

/// Intra-op width to use when config does not say.
///
/// Capped at four for two reasons: past the physical core count the measured
/// throughput falls off, and a daemon serving several sessions wants the
/// remaining cores for concurrency rather than for one utterance. Machines
/// report logical cores, so on a hyperthreaded box this lands on roughly the
/// physical count by accident of the cap — which is the number that matters.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 4)
}

/// Where the model files live when config does not say.
///
/// Config wins; this is the fallback that lets a fresh install work after a
/// plain download with no config edit.
pub fn default_model_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("nevoflux").join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_reports_not_found() {
        let err = load_session(Path::new("/nonexistent/kokoro.onnx"), 1).unwrap_err();
        assert!(matches!(err, TtsError::ModelNotFound(_)), "got: {err}");
    }

    #[test]
    fn corrupt_model_reports_corrupt() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b"not an onnx file").unwrap();
        let err = load_session(f.path(), 1).unwrap_err();
        assert!(matches!(err, TtsError::ModelCorrupt(_)), "got: {err}");
    }

    #[test]
    fn default_threads_stays_in_the_useful_range() {
        let t = default_threads();
        assert!((1..=4).contains(&t), "got {t}");
    }

    #[cfg(feature = "ort-load-dynamic")]
    #[test]
    fn env_override_wins_over_bundled() {
        let p = PathBuf::from("/custom/libonnxruntime.so");
        assert_eq!(
            resolve_dylib(Some(p.clone()), Some(Path::new("/exe"))),
            Some(p)
        );
    }

    #[cfg(feature = "ort-load-dynamic")]
    #[test]
    fn version_must_match_major_minor() {
        assert!(version_compatible("1.24.2", "1.24.2"));
        assert!(version_compatible("1.24.0", "1.24.2"), "patch may differ");
        assert!(!version_compatible("1.23.9", "1.24.2"), "minor must match");
        assert!(!version_compatible("2.24.2", "1.24.2"), "major must match");
    }

    #[cfg(feature = "ort-load-dynamic")]
    #[test]
    fn parses_versions_from_both_naming_conventions() {
        assert_eq!(
            parse_version(Path::new("libonnxruntime.so.1.24.2")).as_deref(),
            Some("1.24.2")
        );
        assert_eq!(
            parse_version(Path::new("libonnxruntime.1.24.2.dylib")).as_deref(),
            Some("1.24.2")
        );
        assert_eq!(parse_version(Path::new("libonnxruntime.so")), None);
    }

    /// Confirms the graph signature this crate is written against. Needs the
    /// real 92 MB model, so it is opt-in.
    #[test]
    #[ignore]
    fn real_model_has_expected_signature() {
        let path = default_model_dir().unwrap().join("kokoro-v1.0.int8.onnx");
        let session = load_session(&path, 1).expect("model should load");
        let inputs: Vec<_> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let outputs: Vec<_> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        assert_eq!(inputs, vec!["tokens", "style", "speed"], "input names");
        assert_eq!(outputs, vec!["audio"], "output names");
    }
}
