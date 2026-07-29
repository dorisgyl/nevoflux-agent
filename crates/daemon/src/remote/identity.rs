//! The persisted identity of a headless remote-control head.
//!
//! Channel id and pairing code are generated once and kept, so a phone pairs
//! with the container a single time and keeps working across restarts. The
//! pairing code is the only secret protecting the channel's contents (design
//! §3), which is why it is born here rather than read from a config file or an
//! environment variable — neither of which the daemon can keep out of reach of
//! the agent's own `bash`, which runs in this very process.

use std::path::Path;

/// What a headless head needs in order to be the same head after a restart.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ControlIdentity {
    /// The relay channel. Not a secret, but unguessable.
    pub channel_id: String,
    /// The E2E secret the phone types in. Never logged.
    pub pairing_code: String,
    /// The one long-lived conversation this head holds.
    pub session_id: String,
}

/// Why an identity could not be established.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    /// The file could not be read or written.
    #[error("identity file io error at {path}: {source}")]
    Io {
        /// Path being read or written.
        path: String,
        /// Underlying error.
        source: std::io::Error,
    },
    /// The file exists but is not a usable identity.
    #[error(
        "identity file at {0} is unreadable; refusing to generate a new one — an already-paired \
         device would silently stop connecting. Inspect it, or remove it deliberately to re-pair."
    )]
    Corrupt(String),
}

/// Load the identity at `path`, or generate and persist one if absent.
///
/// Returns `(identity, was_generated)`. A file that exists but does not parse
/// is [`IdentityError::Corrupt`] and is left untouched: regenerating would
/// invalidate an already-paired device, and the only symptom the operator
/// would see is "it stopped connecting".
pub fn load_or_generate(path: &Path) -> Result<(ControlIdentity, bool), IdentityError> {
    let io = |source: std::io::Error| IdentityError::Io {
        path: path.display().to_string(),
        source,
    };

    match std::fs::read(path) {
        Ok(bytes) => {
            let id: ControlIdentity = serde_json::from_slice(&bytes)
                .map_err(|_| IdentityError::Corrupt(path.display().to_string()))?;
            if id.channel_id.is_empty() || id.pairing_code.is_empty() || id.session_id.is_empty() {
                return Err(IdentityError::Corrupt(path.display().to_string()));
            }
            Ok((id, false))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let id = ControlIdentity {
                channel_id: uuid::Uuid::new_v4().to_string(),
                pairing_code: crate::share::password::generate_password(),
                session_id: format!("remote-control-{}", uuid::Uuid::new_v4()),
            };
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(io)?;
            }
            let body = serde_json::to_vec_pretty(&id).expect("ControlIdentity always serializes");
            std::fs::write(path, &body).map_err(io)?;
            restrict(path);
            Ok((id, true))
        }
        Err(e) => Err(io(e)),
    }
}

/// Best-effort 0600. The container's data volume is the real boundary; this
/// only narrows the blast radius if that volume is ever shared.
#[cfg(unix)]
fn restrict(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict(_path: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("nf-identity-{name}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d.join("remote-control.json")
    }

    #[test]
    fn generates_on_first_run() {
        let p = tmp("gen");
        let (id, fresh) = load_or_generate(&p).unwrap();
        assert!(fresh);
        assert!(p.exists());
        // A uuid the portal's parser will accept, and the sidebar's code shape.
        assert_eq!(id.channel_id.len(), 36);
        assert_eq!(id.channel_id, id.channel_id.to_lowercase());
        assert_eq!(id.pairing_code.len(), 16);
        assert!(!id.session_id.is_empty());
    }

    #[test]
    fn reuses_what_is_already_there() {
        // The whole point of persisting: a restart must not invalidate a phone
        // that has already paired.
        let p = tmp("reuse");
        let (first, fresh) = load_or_generate(&p).unwrap();
        assert!(fresh);
        let (second, fresh2) = load_or_generate(&p).unwrap();
        assert!(!fresh2);
        assert_eq!(first.channel_id, second.channel_id);
        assert_eq!(first.pairing_code, second.pairing_code);
        assert_eq!(first.session_id, second.session_id);
    }

    #[test]
    fn a_corrupt_file_is_an_error_not_a_new_identity() {
        // Regenerating would strand the paired phone: it would keep presenting
        // a code for a channel that no longer exists, and the only symptom
        // would be "it will not connect".
        let p = tmp("corrupt");
        std::fs::write(&p, b"{not json").unwrap();
        let err = load_or_generate(&p).unwrap_err();
        assert!(matches!(err, IdentityError::Corrupt(_)));
        // and the bad file is left for the operator to look at
        assert_eq!(std::fs::read(&p).unwrap(), b"{not json");
    }

    #[test]
    fn an_incomplete_file_is_corrupt_too() {
        let p = tmp("partial");
        std::fs::write(&p, br#"{"channel_id":"x","pairing_code":"","session_id":"s"}"#).unwrap();
        assert!(matches!(
            load_or_generate(&p).unwrap_err(),
            IdentityError::Corrupt(_)
        ));
    }

    #[test]
    fn a_missing_parent_directory_is_created() {
        // The data volume is mounted empty on first run; the service must not
        // require anyone to pre-create the path.
        let deep = std::env::temp_dir()
            .join(format!("nf-identity-deep-{}", uuid::Uuid::new_v4()))
            .join("data")
            .join("remote-control.json");
        let (_, fresh) = load_or_generate(&deep).unwrap();
        assert!(fresh);
        assert!(deep.exists());
    }
}
