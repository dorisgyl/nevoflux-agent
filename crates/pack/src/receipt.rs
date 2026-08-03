//! The install record. Written at install, read+reversed at uninstall.

use std::path::PathBuf;

use semver::Version;
use serde::{Deserialize, Serialize};

pub const RECEIPT_VERSION: &str = "1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PathsSource {
    Daemon,
    Derived,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileReceipt {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Receipt {
    pub receipt_version: String,
    pub protocol: String,
    pub pack: String,
    pub namespace: String,
    pub version: Version,
    pub installed_at: String,
    pub nevoflux_version: Version,
    pub paths_source: PathsSource,
    #[serde(default)]
    pub files: Vec<FileReceipt>,
    #[serde(default)]
    pub artifacts: Vec<String>,
    #[serde(default)]
    pub seeded_pages: Vec<String>,
    #[serde(default)]
    pub imported_sources: Vec<String>,
    /// Provenance: where this pack came from (e.g. "github:u/r@ref"). None for
    /// local-directory installs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// sha256 (hex) of the downloaded source tarball, when installed from a
    /// remote source. None for local installs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tarball_sha256: Option<String>,
}

impl Receipt {
    /// Compute the sha256 of file bytes as lowercase hex (used for FileReceipt).
    pub fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let r = Receipt {
            receipt_version: RECEIPT_VERSION.into(),
            protocol: "pack-protocol/0.1".into(),
            pack: "demo".into(),
            namespace: "demo".into(),
            version: Version::new(0, 1, 0),
            installed_at: "2026-06-09T00:00:00Z".into(),
            nevoflux_version: Version::new(0, 3, 0),
            paths_source: PathsSource::Daemon,
            files: vec![FileReceipt {
                path: "/cfg/skills/a.md".into(),
                sha256: "ab".into(),
            }],
            artifacts: vec!["demo-dashboard".into()],
            seeded_pages: vec!["demo/cv".into()],
            imported_sources: vec!["demo".into()],
            source: None,
            tarball_sha256: None,
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: Receipt = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn sha256_is_stable_lowercase_hex() {
        assert_eq!(
            Receipt::sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn receipt_round_trips_with_source_provenance() {
        let mut r = Receipt {
            receipt_version: RECEIPT_VERSION.into(),
            protocol: "pack-protocol/0.1".into(),
            pack: "demo".into(),
            namespace: "demo".into(),
            version: Version::new(0, 1, 0),
            installed_at: "t".into(),
            nevoflux_version: Version::new(0, 3, 0),
            paths_source: PathsSource::Daemon,
            files: vec![],
            artifacts: vec![],
            seeded_pages: vec![],
            imported_sources: vec![],
            source: Some("github:u/r/sub@v1".into()),
            tarball_sha256: Some("ab".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: Receipt = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);

        // Backward compat: a receipt JSON missing the new fields still parses.
        r.source = None;
        r.tarball_sha256 = None;
        let legacy = serde_json::json!({
            "receipt_version": "1", "protocol": "pack-protocol/0.1", "pack": "demo",
            "namespace": "demo", "version": "0.1.0", "installed_at": "t",
            "nevoflux_version": "0.3.0", "paths_source": "daemon"
        })
        .to_string();
        let parsed: Receipt = serde_json::from_str(&legacy).unwrap();
        assert_eq!(parsed.source, None);
        assert_eq!(parsed.tarball_sha256, None);
    }
}
