//! File watcher for the soul directory that detects external edits.
//!
//! Uses `notify::RecommendedWatcher` to monitor two things directly in the soul
//! directory: the five soul documents (IDENTITY.md, SOUL.md, USER.md, TOOLS.md,
//! AGENTS.md) and the container→soul bindings (space_souls.toml). Only `Modify`
//! and `Create` events on recognised filenames are forwarded through an async
//! channel so that the consumer can validate, reload, and log the change.

use std::path::{Path, PathBuf};

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::agent::space_souls::BINDINGS_FILE;
use crate::error::{DaemonError, Result};

/// Filenames that the watcher considers "soul documents".
const WATCHED_FILES: [&str; 5] = ["IDENTITY.md", "SOUL.md", "USER.md", "TOOLS.md", "AGENTS.md"];

/// What changed in the soul directory.
///
/// The two carry different reload costs and different failure modes, so the
/// consumer decides what to do rather than re-deriving it from a path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoulDirChange {
    /// One of the five global soul documents was written.
    SoulDoc(PathBuf),
    /// `space_souls.toml` was written: the container→soul bindings changed.
    Bindings(PathBuf),
}

impl SoulDirChange {
    /// The file that changed.
    pub fn path(&self) -> &Path {
        match self {
            Self::SoulDoc(p) | Self::Bindings(p) => p,
        }
    }

    /// Classify a written path, or `None` if this watcher does not care about it.
    fn classify(path: &Path, soul_dir: &Path) -> Option<Self> {
        // Only files sitting directly in the soul directory count; role
        // directories under agents/ have their own lifecycle.
        if path.parent() != Some(soul_dir) {
            return None;
        }
        let filename = path.file_name()?.to_str()?;
        if WATCHED_FILES.contains(&filename) {
            Some(Self::SoulDoc(path.to_path_buf()))
        } else if filename == BINDINGS_FILE {
            Some(Self::Bindings(path.to_path_buf()))
        } else {
            None
        }
    }
}

/// Watches the soul directory for external modifications to soul documents.
///
/// Internally holds a [`RecommendedWatcher`] that is kept alive as long as
/// the `SoulWatcher` value exists. Changed file paths are delivered via an
/// async channel and can be consumed with [`next_change`](Self::next_change).
pub struct SoulWatcher {
    /// The directory being watched.
    soul_dir: PathBuf,
    /// Internal watcher handle. Dropping this stops watching.
    _watcher: RecommendedWatcher,
    /// Channel receiver for change events.
    rx: mpsc::Receiver<SoulDirChange>,
}

impl SoulWatcher {
    /// Start watching the soul directory for changes.
    ///
    /// Returns a `SoulWatcher` that yields changed file paths via
    /// [`next_change`](Self::next_change). Only changes to the five
    /// recognised soul documents are reported; all other files in the
    /// directory are silently ignored.
    pub fn start(soul_dir: &Path) -> Result<Self> {
        let soul_dir = soul_dir.to_path_buf();
        let (tx, rx) = mpsc::channel(32);

        let soul_dir_clone = soul_dir.clone();

        let mut watcher =
            notify::recommended_watcher(move |res: std::result::Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    match event.kind {
                        EventKind::Modify(_) | EventKind::Create(_) => {
                            for path in &event.paths {
                                if let Some(change) = SoulDirChange::classify(path, &soul_dir_clone)
                                {
                                    let _ = tx.blocking_send(change);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            })
            .map_err(|e| {
                DaemonError::InternalError(format!("Failed to create file watcher: {}", e))
            })?;

        watcher
            .watch(&soul_dir, RecursiveMode::NonRecursive)
            .map_err(|e| {
                DaemonError::InternalError(format!("Failed to watch soul directory: {}", e))
            })?;

        Ok(Self {
            soul_dir,
            _watcher: watcher,
            rx,
        })
    }

    /// Wait for the next file change event.
    ///
    /// Returns `Some(change)` when a recognised file is modified or created, or
    /// `None` if the watcher channel is closed (which happens when the watcher
    /// is dropped from another context).
    pub async fn next_change(&mut self) -> Option<SoulDirChange> {
        self.rx.recv().await
    }

    /// Returns the soul directory being watched.
    pub fn soul_dir(&self) -> &Path {
        &self.soul_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    #[tokio::test]
    async fn watcher_detects_file_creation() {
        let tmp = TempDir::new().unwrap();
        let soul_dir = tmp.path().to_path_buf();

        // Create initial file
        std::fs::write(soul_dir.join("TOOLS.md"), "# initial").unwrap();

        let mut watcher = SoulWatcher::start(&soul_dir).unwrap();

        // Modify the file after a brief delay so the watcher is ready
        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(soul_dir.join("TOOLS.md"), "# modified").unwrap();

        // Wait for the event with timeout
        let result = tokio::time::timeout(Duration::from_secs(5), watcher.next_change()).await;
        assert!(result.is_ok(), "Should receive change event within timeout");

        let change = result.unwrap().unwrap();
        assert!(matches!(change, SoulDirChange::SoulDoc(_)));
        assert!(change.path().ends_with("TOOLS.md"));
    }

    /// The bindings file lives beside the soul documents but reloads differently,
    /// so the watcher must tell them apart.
    #[tokio::test]
    async fn watcher_reports_bindings_changes_separately() {
        let tmp = TempDir::new().unwrap();
        let soul_dir = tmp.path().to_path_buf();

        let mut watcher = SoulWatcher::start(&soul_dir).unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(
            soul_dir.join(BINDINGS_FILE),
            "[bindings]\n\"firefox-default\" = \"tester\"\n",
        )
        .unwrap();

        let result = tokio::time::timeout(Duration::from_secs(5), watcher.next_change()).await;
        let change = result.expect("timed out").expect("channel closed");

        assert!(matches!(change, SoulDirChange::Bindings(_)));
        assert!(change.path().ends_with(BINDINGS_FILE));
    }

    /// A role's own files live under agents/<slug>/ and have their own lifecycle;
    /// the soul-document watcher must not claim them.
    #[test]
    fn classify_ignores_files_outside_the_soul_dir() {
        let soul_dir = Path::new("/config/nevoflux");

        assert!(SoulDirChange::classify(
            &soul_dir.join("agents").join("alex").join("SOUL.md"),
            soul_dir
        )
        .is_none());
        assert!(SoulDirChange::classify(&soul_dir.join("notes.txt"), soul_dir).is_none());
        assert!(SoulDirChange::classify(&soul_dir.join("SOUL.md"), soul_dir).is_some());
    }

    #[tokio::test]
    async fn watcher_ignores_non_soul_files() {
        let tmp = TempDir::new().unwrap();
        let soul_dir = tmp.path().to_path_buf();

        let mut watcher = SoulWatcher::start(&soul_dir).unwrap();

        // Create a non-soul file
        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(soul_dir.join("random.txt"), "not a soul file").unwrap();

        // Should NOT get an event (use short timeout)
        let result = tokio::time::timeout(Duration::from_millis(500), watcher.next_change()).await;
        assert!(
            result.is_err(),
            "Should NOT receive event for non-soul files"
        );
    }

    #[test]
    fn watcher_soul_dir_accessor() {
        let tmp = TempDir::new().unwrap();
        let soul_dir = tmp.path().to_path_buf();
        let watcher = SoulWatcher::start(&soul_dir).unwrap();
        assert_eq!(watcher.soul_dir(), soul_dir);
    }

    #[tokio::test]
    async fn watcher_detects_user_md_change() {
        let tmp = TempDir::new().unwrap();
        let soul_dir = tmp.path().to_path_buf();

        // Create initial USER.md
        std::fs::write(soul_dir.join("USER.md"), "# user").unwrap();

        let mut watcher = SoulWatcher::start(&soul_dir).unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(soul_dir.join("USER.md"), "# updated user").unwrap();

        let result = tokio::time::timeout(Duration::from_secs(5), watcher.next_change()).await;
        assert!(result.is_ok(), "Should detect USER.md change");
    }
}
