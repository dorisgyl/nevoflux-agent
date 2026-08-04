//! What of a Firefox profile is worth carrying between machines and clones.
//!
//! A real profile measures ~135 MB, of which ~125 MB regenerates itself:
//! CRLite revocation data, DRM plugins, favicons, browsing history, caches.
//! Copying that per task, and shipping it over HTTP, is pure waste.
//!
//! The rule is "regenerable or telemetry → drop; anything that could hold
//! login state → keep". When in doubt, keep: a profile that is a few MB too
//! big is a nuisance, one missing `key4.db` is a silent logout that surfaces
//! days later as "it worked yesterday".

use std::path::Path;

/// Path prefixes and file names that never need to travel.
///
/// Matched against the profile-relative path, so `storage/` can be split:
/// `storage/default/` is login state (IndexedDB, localStorage), while
/// `storage/temporary/` and `storage/to-be-removed/` are evictable by design.
const EXCLUDED: &[&str] = &[
    // Regenerable
    "security_state/",
    "startupCache/",
    "shader-cache/",
    "thumbnails/",
    // Not present inside the profile on Linux (Firefox puts it under
    // ~/.cache), but it is inside the profile on macOS and Windows.
    "cache2/",
    // Evictable / not login state
    "storage/temporary/",
    "storage/to-be-removed/",
    "sessionstore-backups/",
    "sessionstore.jsonlz4",
    "weave/",
    // Telemetry and crash data
    "datareporting/",
    "saved-telemetry-pings/",
    "minidumps/",
    "crashes/",
    // Runtime locks: a stale lock poisons the base.
    "lock",
    ".parentlock",
];

/// Databases dropped along with all their `-wal` / `-shm` sidecars.
///
/// `places.sqlite` holds browsing history *and* bookmarks in one database, so
/// excluding it loses bookmarks too. That is accepted deliberately: a headless
/// base profile is a login-state template for automation, not someone's daily
/// browser.
const EXCLUDED_DB_STEMS: &[&str] = &["favicons.sqlite", "places.sqlite"];

/// Top-level directory prefixes matched with a version-suffix wildcard, e.g.
/// `gmp-widevinecdm/` and `gmp-gmpopenh264/` — plugins Firefox re-downloads.
const EXCLUDED_PREFIXES: &[&str] = &["gmp-"];

/// Whether a profile entry is worth carrying between machines and clones.
///
/// `relative` is the path inside the profile, e.g. `storage/default/x/y`.
/// Kept by default: an unrecognised entry is more likely to be new login state
/// than new bulk.
pub fn should_copy(relative: &Path) -> bool {
    let path = relative.to_string_lossy().replace('\\', "/");
    if path.is_empty() {
        return true;
    }

    if EXCLUDED.iter().any(|e| match e.strip_suffix('/') {
        Some(dir) => path == dir || path.starts_with(&format!("{dir}/")),
        None => path == *e,
    }) {
        return false;
    }

    let first = path.split('/').next().unwrap_or("");
    if EXCLUDED_PREFIXES.iter().any(|p| first.starts_with(p)) {
        return false;
    }

    // A `-wal` / `-shm` sidecar shares its database's fate. Keeping a cookies
    // wal is essential — unflushed cookies live there, and dropping it is a
    // silent logout — while keeping a places wal would defeat excluding places.
    if EXCLUDED_DB_STEMS.contains(&sidecar_base(first)) {
        return false;
    }

    !extra_exclusions()
        .iter()
        .any(|e| path == *e || path.starts_with(&format!("{e}/")) || matches_glob(e, &path))
}

/// The database name a `-wal` / `-shm` sidecar belongs to, or the name itself.
fn sidecar_base(name: &str) -> &str {
    for suffix in ["-wal", "-shm"] {
        if let Some(base) = name.strip_suffix(suffix) {
            return base;
        }
    }
    name
}

/// Operator-supplied additions from `NEVOFLUX_PROFILE_EXCLUDE` (colon-separated).
///
/// Additions only. There is deliberately no syntax for un-excluding a default:
/// a typo that dropped `key4.db` would log the profile out silently, and the
/// defaults are where this module's correctness lives.
fn extra_exclusions() -> Vec<String> {
    std::env::var("NEVOFLUX_PROFILE_EXCLUDE")
        .ok()
        .map(|raw| {
            raw.split(':')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Minimal trailing-`*` glob; enough for `storage/temporary/*` style rules
/// without pulling in a glob crate for four characters of syntax.
fn matches_glob(pattern: &str, path: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => path.starts_with(prefix),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep(p: &str) -> bool {
        should_copy(Path::new(p))
    }

    /// Anything that could hold login state must survive the trip.
    #[test]
    fn login_state_is_kept() {
        for p in [
            "cookies.sqlite",
            "cookies.sqlite-wal",
            "cookies.sqlite-shm",
            "key4.db",
            "logins.json",
            "cert9.db",
            "permissions.sqlite",
            "prefs.js",
            "user.js",
            "containers.json",
            "extension-preferences.json",
            "extensions/uBlock0@raymondhill.net.xpi",
            "storage/default/https+++mail.example.com/idb/x.sqlite",
        ] {
            assert!(keep(p), "{p} must be kept");
        }
    }

    /// Regenerable bulk and telemetry go.
    #[test]
    fn regenerable_and_telemetry_are_dropped() {
        for p in [
            "security_state/data.safe.bin",
            "gmp-widevinecdm/4.10/libwidevinecdm.so",
            "gmp-gmpopenh264/1.8/libgmpopenh264.so",
            "favicons.sqlite",
            "favicons.sqlite-wal",
            "places.sqlite",
            "places.sqlite-wal",
            "places.sqlite-shm",
            "startupCache/scriptCache.bin",
            "shader-cache/x",
            "thumbnails/y.png",
            "cache2/entries/z",
            "storage/temporary/https+++a.example/x",
            "storage/to-be-removed/x",
            "sessionstore-backups/recovery.jsonlz4",
            "sessionstore.jsonlz4",
            "weave/failed/x.json",
            "datareporting/glean/db/data.safe.bin",
            "saved-telemetry-pings/abc",
            "minidumps/x.dmp",
            "crashes/events/x",
            "lock",
            ".parentlock",
        ] {
            assert!(!keep(p), "{p} must be dropped");
        }
    }

    /// `storage/` is the one place needing a sub-path decision rather than a
    /// top-level one: `default/` is login state, its siblings are not.
    #[test]
    fn storage_is_split_by_subdirectory() {
        assert!(keep("storage/default/https+++x/ls/data.sqlite"));
        assert!(!keep("storage/temporary/https+++x/y"));
        assert!(!keep("storage/to-be-removed/x"));
    }

    /// A wal/shm sidecar follows its database either way.
    #[test]
    fn sidecars_follow_their_database() {
        assert!(keep("cookies.sqlite-wal"));
        assert!(!keep("places.sqlite-wal"));
        assert!(!keep("favicons.sqlite-shm"));
    }

    #[test]
    #[serial_test::serial]
    fn env_can_add_exclusions_but_not_remove_them() {
        let prev = std::env::var("NEVOFLUX_PROFILE_EXCLUDE").ok();

        std::env::set_var("NEVOFLUX_PROFILE_EXCLUDE", "extensions");
        assert!(!keep("extensions/x.xpi"), "env must be able to add");

        // Naming a default exclusion cannot un-exclude it, and there is no
        // syntax that would. A typo here must never be able to drop key4.db.
        std::env::set_var("NEVOFLUX_PROFILE_EXCLUDE", "!places.sqlite");
        assert!(!keep("places.sqlite"), "defaults must not be removable");
        assert!(keep("key4.db"), "unrelated keeps must be unaffected");

        match prev {
            Some(v) => std::env::set_var("NEVOFLUX_PROFILE_EXCLUDE", v),
            None => std::env::remove_var("NEVOFLUX_PROFILE_EXCLUDE"),
        }
    }
}
