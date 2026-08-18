//! Getting the speech models onto the user's disk.
//!
//! ## Why the daemon does this at all
//!
//! `just fetch-asr-models` says, in a comment: "The daemon fetches nothing
//! itself. Network behaviour in a native-messaging host expected to start in
//! under a second is a bad trade, and pulling hundreds of megabytes at a moment
//! nobody chose is a worse one."
//!
//! That still holds, and this does not contradict it. What it rules out is
//! *implicit* fetching — at startup, on first use, in the background because
//! something was missing. Every entry point here is a request someone made, and
//! nothing in this module runs unless one arrives. A developer recipe cannot
//! serve users who have no checkout, which is the only reason this exists.
//!
//! ## Why tiers
//!
//! Dictation needs 240 MB. Speech synthesis needs another 120 MB today and
//! around 730 MB once MOSS lands. Making the first depend on the second means
//! nobody talks to the thing until all of it has arrived, over connections that
//! cannot reach the upstream hosts at all.

pub mod catalog;
pub mod fetch;

use std::path::{Path, PathBuf};

pub use catalog::{Asset, Tier, ASSETS};
use tokio_util::sync::CancellationToken;

/// Where the models live. Matches `tts/asr.rs` and `tts/kokoro.rs`, which
/// resolve the same directory independently — this does not move without them.
pub fn models_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("nevoflux").join("models"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AssetState {
    Present,
    /// An interrupted download that a resume can continue.
    Partial {
        have: u64,
    },
    Missing,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AssetStatus {
    pub id: &'static str,
    pub tier: &'static str,
    pub file: &'static str,
    pub bytes: u64,
    #[serde(flatten)]
    pub state: AssetState,
}

/// What is on disk right now.
///
/// Presence is decided by exact size, without hashing. Hashing 240 MB to answer
/// "is voice input available" would cost about a second of disk read every time
/// a panel opens, and the digest was already checked at the one moment it can
/// actually catch something — before the file was renamed into place. A file
/// that changes afterwards is a machine problem, not a download problem.
pub fn state_of(asset: &Asset, dir: &Path) -> AssetState {
    let dest = dir.join(asset.file);
    if let Ok(m) = std::fs::metadata(&dest) {
        if m.len() == asset.bytes {
            return AssetState::Present;
        }
        // Right name, wrong size: not ours. Reported as missing rather than
        // partial, because a resume would append to somebody else's file.
        return AssetState::Missing;
    }
    match std::fs::metadata(fetch::part_path(&dest)) {
        Ok(m) if m.len() > 0 && m.len() < asset.bytes => AssetState::Partial { have: m.len() },
        _ => AssetState::Missing,
    }
}

pub fn status(dir: &Path) -> Vec<AssetStatus> {
    ASSETS
        .iter()
        .map(|a| AssetStatus {
            id: a.id,
            tier: a.tier.id(),
            file: a.file,
            bytes: a.bytes,
            state: state_of(a, dir),
        })
        .collect()
}

/// Whether everything a tier needs is on disk.
pub fn tier_ready(tier: Tier, dir: &Path) -> bool {
    catalog::of_tier(tier).all(|a| state_of(a, dir) == AssetState::Present)
}

/// Bytes still to fetch for a tier, counting a resumable partial as progress.
pub fn tier_remaining(tier: Tier, dir: &Path) -> u64 {
    catalog::of_tier(tier)
        .map(|a| match state_of(a, dir) {
            AssetState::Present => 0,
            AssetState::Partial { have } => a.bytes.saturating_sub(have),
            AssetState::Missing => a.bytes,
        })
        .sum()
}

/// Identifies one download run, so that finishing can only clear the run that
/// is actually finishing.
///
/// The same shape as `InterruptRegistry`'s turn token, for the same reason: a
/// registry keyed only by what is being downloaded lets a slow run's cleanup
/// remove the entry belonging to the run that replaced it, leaving something
/// live that nothing can cancel. There it was one turn per session held by the
/// UI; here it is a user who cancelled a download and immediately started it
/// again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunToken(u64);

/// The downloads currently running, one per tier at most.
#[derive(Default)]
pub struct Downloads {
    active:
        std::sync::Mutex<std::collections::HashMap<&'static str, (RunToken, CancellationToken)>>,
    next: std::sync::atomic::AtomicU64,
}

/// Process-level, because a download outlives the connection that asked for it:
/// closing the sidebar mid-download should not abandon 200 MB.
pub fn downloads() -> &'static Downloads {
    static D: std::sync::OnceLock<Downloads> = std::sync::OnceLock::new();
    D.get_or_init(Downloads::default)
}

impl Downloads {
    /// Claim a tier. `None` means one is already running — the second request
    /// is a duplicate click, not a reason to fetch the same bytes twice.
    pub fn begin(&self, tier: Tier) -> Option<(RunToken, CancellationToken)> {
        let mut map = self.active.lock().unwrap();
        if map.contains_key(tier.id()) {
            return None;
        }
        let token = RunToken(self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
        let cancel = CancellationToken::new();
        map.insert(tier.id(), (token, cancel.clone()));
        Some((token, cancel))
    }

    /// Release a tier, but only if this run still owns it.
    pub fn finish(&self, tier: Tier, token: RunToken) {
        let mut map = self.active.lock().unwrap();
        if map.get(tier.id()).map(|(t, _)| *t) == Some(token) {
            map.remove(tier.id());
        }
    }

    /// Ask a running download to stop. `false` if nothing was running.
    pub fn cancel(&self, tier: Tier) -> bool {
        let map = self.active.lock().unwrap();
        match map.get(tier.id()) {
            Some((_, c)) => {
                c.cancel();
                true
            }
            None => false,
        }
    }

    pub fn is_running(&self, tier: Tier) -> bool {
        self.active.lock().unwrap().contains_key(tier.id())
    }
}

/// One tier's headline, for the panel that offers to download it.
///
/// `remaining` rather than a percentage: a resumed download starts at 40% and a
/// percentage alone cannot say whether that is progress or a stale figure.
pub fn tier_report(tier: Tier, dir: &Path) -> serde_json::Value {
    serde_json::json!({
        "id": tier.id(),
        "ready": tier_ready(tier, dir),
        "downloading": downloads().is_running(tier),
        "bytes": catalog::tier_bytes(tier),
        "remaining": tier_remaining(tier, dir),
    })
}

/// Whether this progress update is worth sending.
///
/// A 229 MB download arrives in tens of thousands of chunks. Forwarding each
/// one would put more traffic on the native-messaging channel than the
/// conversation itself, so updates are rate-limited — but completion always
/// goes through, because the last update is the one that changes the UI from
/// "downloading" to "ready".
pub fn should_emit(done: u64, total: u64, last_emitted: u64, since: std::time::Duration) -> bool {
    done >= total || since >= std::time::Duration::from_millis(250) || done < last_emitted
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("no cache directory on this system")]
    NoModelsDir,
    #[error("cancelled")]
    Cancelled,
    #[error("could not fetch {asset} from any source ({upstream}):\n{}", .attempts.join("\n"))]
    AllSourcesFailed {
        asset: &'static str,
        upstream: &'static str,
        attempts: Vec<String>,
    },
}

/// An HTTP client shaped for large files.
///
/// No total-request timeout on purpose: `Client::timeout` covers the whole
/// exchange, so any value large enough for 240 MB on a slow line is too large
/// to detect a stalled connection, and any value small enough to detect one
/// kills healthy downloads partway. A connect timeout plus a per-read timeout
/// says the thing actually meant — "give up if nothing is arriving" — without
/// putting a clock on the total size.
pub fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .read_timeout(std::time::Duration::from_secs(30))
        .user_agent(concat!("nevoflux-agent/", env!("CARGO_PKG_VERSION")))
        .build()
        .unwrap_or_default()
}

/// Fetch one asset, trying its sources in order.
///
/// Fails fast in the sense that matters: it stops at the end of the source list
/// and reports what each one said, rather than retrying a mirror that is
/// serving 404s until someone notices the panel has been spinning for an hour.
pub async fn download_asset(
    client: &reqwest::Client,
    asset: &'static Asset,
    dir: &Path,
    cancel: &CancellationToken,
    on_progress: &mut (dyn FnMut(&'static Asset, u64, u64) + Send),
) -> Result<(), ModelError> {
    if state_of(asset, dir) == AssetState::Present {
        on_progress(asset, asset.bytes, asset.bytes);
        return Ok(());
    }

    let dest = dir.join(asset.file);
    let mut attempts = Vec::new();

    for source in asset.sources {
        if cancel.is_cancelled() {
            return Err(ModelError::Cancelled);
        }
        let result = fetch::fetch_to(
            client,
            source,
            &dest,
            asset.bytes,
            asset.sha256,
            cancel,
            &mut |done, total| on_progress(asset, done, total),
        )
        .await;

        match result {
            Ok(()) => return Ok(()),
            Err(e) if !e.worth_another_source() => return Err(ModelError::Cancelled),
            Err(e) => {
                tracing::warn!(target: "models", asset = asset.id, error = %e, "source failed");
                attempts.push(format!("  {e}"));
            }
        }
    }

    Err(ModelError::AllSourcesFailed {
        asset: asset.id,
        upstream: asset.upstream,
        attempts,
    })
}

/// Fetch everything a tier needs, in catalog order.
///
/// Stops at the first asset that cannot be had. A tier is all-or-nothing to the
/// user — dictation with no tokens file is not partly working — so continuing
/// past a failure would only produce a longer wait before the same bad news.
pub async fn download_tier(
    client: &reqwest::Client,
    tier: Tier,
    dir: &Path,
    cancel: &CancellationToken,
    on_progress: &mut (dyn FnMut(&'static Asset, u64, u64) + Send),
) -> Result<(), ModelError> {
    for asset in catalog::of_tier(tier) {
        download_asset(client, asset, dir, cancel, on_progress).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(id: &str) -> &'static Asset {
        catalog::by_id(id).expect("known asset")
    }

    #[test]
    fn a_file_of_exactly_the_right_size_counts_as_present() {
        let dir = tempfile::tempdir().unwrap();
        let a = asset("silero-vad");
        std::fs::write(dir.path().join(a.file), vec![0u8; a.bytes as usize]).unwrap();
        assert_eq!(state_of(a, dir.path()), AssetState::Present);
    }

    #[test]
    fn a_wrong_sized_file_under_the_right_name_is_missing_not_partial() {
        // Resuming onto it would append to whatever it actually is.
        let dir = tempfile::tempdir().unwrap();
        let a = asset("silero-vad");
        std::fs::write(dir.path().join(a.file), b"not the model").unwrap();
        assert_eq!(state_of(a, dir.path()), AssetState::Missing);
    }

    #[test]
    fn an_interrupted_download_is_partial_and_says_how_far_it_got() {
        let dir = tempfile::tempdir().unwrap();
        let a = asset("silero-vad");
        std::fs::write(dir.path().join(format!("{}.part", a.file)), vec![0u8; 4096]).unwrap();
        assert_eq!(state_of(a, dir.path()), AssetState::Partial { have: 4096 });
    }

    #[test]
    fn an_empty_partial_is_just_missing() {
        let dir = tempfile::tempdir().unwrap();
        let a = asset("silero-vad");
        std::fs::write(dir.path().join(format!("{}.part", a.file)), b"").unwrap();
        assert_eq!(state_of(a, dir.path()), AssetState::Missing);
    }

    #[test]
    fn nothing_downloaded_means_nothing_is_ready() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!tier_ready(Tier::Transcribe, dir.path()));
        assert_eq!(
            tier_remaining(Tier::Transcribe, dir.path()),
            catalog::tier_bytes(Tier::Transcribe)
        );
    }

    #[test]
    fn a_tier_is_ready_only_when_every_one_of_its_files_is_there() {
        let dir = tempfile::tempdir().unwrap();
        let all: Vec<_> = catalog::of_tier(Tier::Transcribe).collect();
        for (i, a) in all.iter().enumerate() {
            std::fs::write(dir.path().join(a.file), vec![0u8; a.bytes as usize]).unwrap();
            let last = i + 1 == all.len();
            assert_eq!(
                tier_ready(Tier::Transcribe, dir.path()),
                last,
                "ready flipped with {} of {} files present",
                i + 1,
                all.len()
            );
        }
    }

    #[test]
    fn one_tier_being_ready_says_nothing_about_the_other() {
        // The whole point of tiering: dictation works while synthesis is still
        // arriving.
        let dir = tempfile::tempdir().unwrap();
        for a in catalog::of_tier(Tier::Transcribe) {
            std::fs::write(dir.path().join(a.file), vec![0u8; a.bytes as usize]).unwrap();
        }
        assert!(tier_ready(Tier::Transcribe, dir.path()));
        assert!(!tier_ready(Tier::Speak, dir.path()));
    }

    #[test]
    fn a_partial_counts_towards_what_is_left() {
        let dir = tempfile::tempdir().unwrap();
        let a = asset("silero-vad");
        std::fs::write(dir.path().join(format!("{}.part", a.file)), vec![0u8; 1000]).unwrap();
        assert_eq!(
            tier_remaining(Tier::Transcribe, dir.path()),
            catalog::tier_bytes(Tier::Transcribe) - 1000
        );
    }

    #[test]
    fn status_reports_every_asset_and_serialises_its_state() {
        let dir = tempfile::tempdir().unwrap();
        let s = status(dir.path());
        assert_eq!(s.len(), ASSETS.len());
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"state\":\"missing\""), "{json}");
        assert!(json.contains("sensevoice-model"), "{json}");
    }

    #[tokio::test]
    async fn a_present_asset_is_not_downloaded_again() {
        // No client is usable here, which is the assertion: it must not be
        // reached. A download that re-fetches 229 MB it already has is the
        // difference between opening a panel and opening a panel for a minute.
        let dir = tempfile::tempdir().unwrap();
        let a = asset("silero-vad");
        std::fs::write(dir.path().join(a.file), vec![0u8; a.bytes as usize]).unwrap();

        let mut seen = Vec::new();
        let r = download_asset(
            &reqwest::Client::new(),
            a,
            dir.path(),
            &CancellationToken::new(),
            &mut |asset, done, total| seen.push((asset.id, done, total)),
        )
        .await;
        assert!(r.is_ok());
        assert_eq!(seen, vec![(a.id, a.bytes, a.bytes)]);
    }

    #[tokio::test]
    async fn an_already_cancelled_download_does_not_start() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let r = download_asset(
            &reqwest::Client::new(),
            asset("silero-vad"),
            dir.path(),
            &cancel,
            &mut |_, _, _| {},
        )
        .await;
        assert!(matches!(r, Err(ModelError::Cancelled)), "{r:?}");
    }

    #[test]
    fn a_tier_can_only_be_downloading_once() {
        let d = Downloads::default();
        let (token, _c) = d.begin(Tier::Transcribe).expect("first claim");
        assert!(d.begin(Tier::Transcribe).is_none(), "started twice");
        // Tiers are independent: synthesis may download while dictation does.
        assert!(d.begin(Tier::Speak).is_some());
        d.finish(Tier::Transcribe, token);
        assert!(d.begin(Tier::Transcribe).is_some(), "not released");
    }

    #[test]
    fn a_finished_run_cannot_release_the_run_that_replaced_it() {
        // The InterruptRegistry bug in another costume: cancel, restart, then
        // the old run's cleanup arrives and clears the new run's entry, leaving
        // a download nothing can cancel.
        let d = Downloads::default();
        let (first, _c1) = d.begin(Tier::Transcribe).unwrap();
        d.cancel(Tier::Transcribe);
        d.finish(Tier::Transcribe, first);

        let (_second, _c2) = d.begin(Tier::Transcribe).unwrap();
        d.finish(Tier::Transcribe, first); // the straggler
        assert!(
            d.is_running(Tier::Transcribe),
            "the stale cleanup released the live download"
        );
    }

    #[test]
    fn cancelling_reaches_the_token_the_download_is_watching() {
        let d = Downloads::default();
        let (_t, cancel) = d.begin(Tier::Speak).unwrap();
        assert!(!cancel.is_cancelled());
        assert!(d.cancel(Tier::Speak));
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn cancelling_nothing_says_so_rather_than_pretending() {
        assert!(!Downloads::default().cancel(Tier::Speak));
    }

    #[test]
    fn progress_is_rate_limited_but_completion_always_gets_through() {
        use std::time::Duration;
        let quiet = Duration::from_millis(10);
        let elapsed = Duration::from_millis(300);

        assert!(
            !should_emit(1_000, 100_000, 900, quiet),
            "flooded the channel"
        );
        assert!(should_emit(1_000, 100_000, 900, elapsed), "never updated");
        // The update that flips the UI to "ready" must never be the one dropped.
        assert!(should_emit(100_000, 100_000, 900, quiet));
        // A restart from zero is a real change even though it moves backwards.
        assert!(should_emit(0, 100_000, 50_000, quiet));
    }

    #[test]
    fn failure_names_every_source_and_where_to_get_it_by_hand() {
        // What the user sees when nothing works has to be actionable; "download
        // failed" is not.
        let e = ModelError::AllSourcesFailed {
            asset: "sensevoice-model",
            upstream: "csukuangfj/sherpa-onnx-sense-voice",
            attempts: vec![
                "  https://mirror/x refused the request: HTTP 404".into(),
                "  https://upstream/x: connection timed out".into(),
            ],
        };
        let text = e.to_string();
        assert!(text.contains("sensevoice-model"), "{text}");
        assert!(
            text.contains("csukuangfj/sherpa-onnx-sense-voice"),
            "{text}"
        );
        assert!(text.contains("404"), "{text}");
        assert!(text.contains("connection timed out"), "{text}");
    }
}
