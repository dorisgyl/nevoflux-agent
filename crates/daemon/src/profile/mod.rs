//! Per-task profile management (P5): clone a named base-profile into an
//! ephemeral copy (login state carried), inject the headless automation pref,
//! and clean up after the task.
//!
//! The base profile is a first-class credential resource (a human logs in once
//! into `base-profiles/<name>/`); each task runs on a throwaway clone so tasks
//! sharing a base share only what that base already implies (login + tenant
//! brain), and never pollute the base.

pub mod filter;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic sequence for unique clone directory names (no time/rand — keeps
/// behavior deterministic and replay-safe).
static CLONE_SEQ: AtomicU64 = AtomicU64::new(0);

/// The exact pref line the extension reads pre-connection (see plan P1).
const AUTOMATION_PREF_LINE: &str = "user_pref(\"nevoflux.headless.automation\", true);\n";

/// Error managing profiles.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// Filesystem error while copying/injecting/cleaning.
    #[error("profile io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Clones base-profiles into ephemeral per-task copies.
pub struct ProfileManager {
    /// Directory containing `base-profiles/<name>/`.
    pub base_dir: PathBuf,
    /// Directory under which ephemeral clones are created.
    pub work_dir: PathBuf,
}

impl ProfileManager {
    /// Clone `base_name` from `base_dir` into a fresh dir under `work_dir`.
    /// A missing base (blank profile) yields an empty clone dir (allowed).
    pub fn clone_base(&self, base_name: &str) -> Result<PathBuf, ProfileError> {
        let seq = CLONE_SEQ.fetch_add(1, Ordering::Relaxed);
        let clone = self.work_dir.join(format!("{base_name}-{seq}"));
        std::fs::create_dir_all(&clone)?;
        let base = self.base_dir.join(base_name);
        if base.is_dir() {
            copy_dir_all(&base, &clone)?;
        }
        Ok(clone)
    }

    /// Append the automation pref to `<clone>/user.js` (creating it if absent).
    pub fn inject_automation_pref(&self, clone: &Path) -> Result<(), ProfileError> {
        use std::io::Write;
        let user_js = clone.join("user.js");
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(user_js)?;
        f.write_all(AUTOMATION_PREF_LINE.as_bytes())?;
        Ok(())
    }

    /// Remove an ephemeral clone (best-effort; ignores if already gone).
    pub fn cleanup(&self, clone: &Path) {
        let _ = std::fs::remove_dir_all(clone);
    }

    /// Persist a live clone dir back to `base_dir/<base_name>`, replacing it.
    /// Reverse of `clone_base`. The caller MUST have stopped the browser first
    /// (files flushed) — this is a plain filesystem copy.
    pub fn save_to_base(&self, clone: &Path, base_name: &str) -> Result<(), ProfileError> {
        std::fs::create_dir_all(&self.base_dir)?;
        let seq = CLONE_SEQ.fetch_add(1, Ordering::Relaxed);
        let dest = self.base_dir.join(base_name);
        let tmp = self.base_dir.join(format!("{base_name}.saving-{seq}"));
        // Copy into a temp sibling first so a mid-copy failure can't destroy the
        // existing base; only swap once the copy fully succeeds.
        let _ = std::fs::remove_dir_all(&tmp);
        copy_dir_filtered(clone, &tmp)?;
        if dest.exists() {
            std::fs::remove_dir_all(&dest)?;
        }
        std::fs::rename(&tmp, &dest)?;
        Ok(())
    }
}

/// Recursively copy `src` into `dst` (dirs + files).
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    copy_tree(src, dst, Path::new(""), false)
}

/// Like [`copy_dir_all`], but strips the injected
/// `nevoflux.headless.automation` pref from `user.js` so a saved base stays a
/// clean human-login profile (it is re-injected per clone).
fn copy_dir_filtered(src: &Path, dst: &Path) -> std::io::Result<()> {
    copy_tree(src, dst, Path::new(""), true)
}

/// Copy `src` into `dst`, skipping anything [`filter::should_copy`] rejects.
///
/// `rel` is the path relative to the profile root, which the filter needs in
/// order to tell `storage/default/` (login state) from `storage/temporary/`
/// (evictable). Firefox lock files are excluded by the filter rather than
/// special-cased here.
fn copy_tree(
    src: &Path,
    dst: &Path,
    rel: &Path,
    strip_automation_pref: bool,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let entry_rel = rel.join(&name);
        if !filter::should_copy(&entry_rel) {
            continue;
        }
        let target = dst.join(&name);
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target, &entry_rel, strip_automation_pref)?;
        } else if strip_automation_pref && name.to_string_lossy() == "user.js" {
            let contents = std::fs::read_to_string(entry.path()).unwrap_or_default();
            let filtered: String = contents
                .lines()
                .filter(|l| !l.contains("nevoflux.headless.automation"))
                .map(|l| format!("{l}\n"))
                .collect();
            std::fs::write(&target, filtered)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the filter: a task's clone must not carry the
    /// ~125 MB of regenerable bulk a real profile accumulates.
    #[test]
    fn clone_skips_regenerable_bulk() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("base-profiles/acme");
        std::fs::create_dir_all(base.join("security_state")).unwrap();
        std::fs::create_dir_all(base.join("storage/default/site")).unwrap();
        std::fs::create_dir_all(base.join("storage/temporary/site")).unwrap();
        std::fs::write(base.join("cookies.sqlite"), b"x").unwrap();
        std::fs::write(base.join("places.sqlite"), b"x").unwrap();
        std::fs::write(base.join("security_state/data.bin"), b"x").unwrap();
        std::fs::write(base.join("storage/default/site/idb"), b"x").unwrap();
        std::fs::write(base.join("storage/temporary/site/junk"), b"x").unwrap();

        let pm = ProfileManager {
            base_dir: tmp.path().join("base-profiles"),
            work_dir: tmp.path().join("work"),
        };
        let clone = pm.clone_base("acme").unwrap();

        assert!(clone.join("cookies.sqlite").exists());
        assert!(clone.join("storage/default/site/idb").exists());
        assert!(!clone.join("places.sqlite").exists());
        assert!(!clone.join("security_state").exists());
        assert!(!clone.join("storage/temporary").exists());
    }

    #[test]
    fn clone_copies_base_contents() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("base-profiles/acme");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("cookies.sqlite"), b"x").unwrap();
        let pm = ProfileManager {
            base_dir: tmp.path().join("base-profiles"),
            work_dir: tmp.path().join("work"),
        };
        let clone = pm.clone_base("acme").unwrap();
        assert!(clone.join("cookies.sqlite").exists());
        assert_ne!(clone, base);
    }

    #[test]
    fn user_js_gets_automation_pref() {
        let tmp = tempfile::tempdir().unwrap();
        let clone = tmp.path().join("clone");
        std::fs::create_dir_all(&clone).unwrap();
        let pm = ProfileManager {
            base_dir: tmp.path().into(),
            work_dir: tmp.path().into(),
        };
        pm.inject_automation_pref(&clone).unwrap();
        let s = std::fs::read_to_string(clone.join("user.js")).unwrap();
        assert!(s.contains(r#"user_pref("nevoflux.headless.automation", true);"#));
    }

    #[test]
    fn blank_base_yields_empty_clone() {
        let tmp = tempfile::tempdir().unwrap();
        let pm = ProfileManager {
            base_dir: tmp.path().join("base-profiles"),
            work_dir: tmp.path().join("work"),
        };
        let clone = pm.clone_base("does-not-exist").unwrap();
        assert!(clone.is_dir());
    }

    #[test]
    fn save_to_base_replaces_and_filters() {
        let tmp = tempfile::tempdir().unwrap();
        let pm = ProfileManager {
            base_dir: tmp.path().join("base"),
            work_dir: tmp.path().join("work"),
        };
        let base = pm.base_dir.join("acme");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("old.txt"), b"old").unwrap(); // must be gone after replace
        let clone = tmp.path().join("clone");
        std::fs::create_dir_all(clone.join("storage")).unwrap();
        std::fs::write(clone.join("cookies.sqlite"), b"c").unwrap();
        std::fs::write(clone.join("storage/s"), b"s").unwrap();
        std::fs::write(
            clone.join("user.js"),
            "user_pref(\"a.b\", 1);\nuser_pref(\"nevoflux.headless.automation\", true);\n",
        )
        .unwrap();
        std::fs::write(clone.join("lock"), b"x").unwrap();

        pm.save_to_base(&clone, "acme").unwrap();

        assert!(base.join("cookies.sqlite").exists()); // new content persisted
        assert!(base.join("storage/s").exists()); // subdirs copied
        assert!(!base.join("old.txt").exists()); // base fully REPLACED
        assert!(!base.join("lock").exists()); // lock file skipped
        let uj = std::fs::read_to_string(base.join("user.js")).unwrap();
        assert!(uj.contains("a.b")); // other prefs kept
        assert!(!uj.contains("nevoflux.headless.automation")); // injected pref stripped
    }

    #[test]
    fn save_to_base_as_new_name_leaves_original() {
        let tmp = tempfile::tempdir().unwrap();
        let pm = ProfileManager {
            base_dir: tmp.path().join("base"),
            work_dir: tmp.path().join("work"),
        };
        let orig = pm.base_dir.join("acme");
        std::fs::create_dir_all(&orig).unwrap();
        std::fs::write(orig.join("keep.txt"), b"k").unwrap();
        let clone = tmp.path().join("clone");
        std::fs::create_dir_all(&clone).unwrap();
        std::fs::write(clone.join("new.txt"), b"n").unwrap();

        pm.save_to_base(&clone, "acme-loggedin").unwrap();

        assert!(pm.base_dir.join("acme-loggedin/new.txt").exists());
        assert!(orig.join("keep.txt").exists()); // original untouched
    }
}
