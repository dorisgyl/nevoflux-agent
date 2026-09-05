//! What survives a restart when a phone has been paired (design §12.4).
//!
//! Before this, the desktop minted a channel id and a pairing code per
//! `/remote-control` invocation and kept neither: restarting the daemon meant
//! it would never dial that relay channel again, so the phone sat on a channel
//! with nobody on the other end, forever, with nothing to say why. On a laptop
//! that happens daily — a lid, an update, a crash — and it takes the push path
//! down with it, because a subscription travels up the control channel and
//! there is no control channel to travel up.
//!
//! So a pairing is a persistent object, and it is the anchor: channels are
//! rebuilt from it at startup, not minted per command.
//!
//! **The derived keys are stored, not the code.** Three reasons, in order of
//! how much they matter:
//!
//! 1. Deriving costs two 64 MiB Argon2id passes, and paying that on every
//!    daemon start puts a multi-hundred-millisecond memory spike in front of
//!    the first dial for no gain — the answer is the same every time.
//! 2. "Show me the code again" becomes impossible by construction. Wanting a
//!    second device is a reason to pair again, not a reason to keep a
//!    long-lived short passphrase somewhere it can be photographed.
//! 3. What sits on disk stops being something a person can read aloud or type
//!    into any browser, and becomes 32 bytes. The authority is identical; the
//!    ways it can leak are not.
//!
//! What is on disk is readable by the agent's own `bash`, which runs in this
//! process — the same exposure `account-token` beside it already has. That is
//! the price of a pairing that survives a restart, and it is deliberate rather
//! than overlooked.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One paired device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pairing {
    /// The always-on channel carrying the session list.
    pub control_channel_id: String,
    /// The channel a conversation is projected onto when one is attached.
    pub data_channel_id: String,
    /// Derived from `(code, control_channel_id)`. Hex, 32 bytes.
    pub control_key: String,
    /// Derived from `(code, data_channel_id)`. Hex, 32 bytes.
    pub data_key: String,
    /// Unix seconds, for the device list.
    pub created_at: i64,
    /// What the person called this device, once they have said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Where to wake this device, once it has said.
    ///
    /// Lives with the pairing because it is meaningless without one and must
    /// die with it: an endpoint is a bearer capability to make somebody's phone
    /// buzz, and leaving one behind after a device is revoked would be leaving
    /// exactly that behind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub push: Option<super::web_push::Subscription>,
}

impl Pairing {
    /// The control channel's key, or `None` if what is stored is not 32 bytes.
    pub fn control_key(&self) -> Option<[u8; 32]> {
        unhex(&self.control_key)
    }

    /// The data channel's key.
    pub fn data_key(&self) -> Option<[u8; 32]> {
        unhex(&self.data_key)
    }
}

/// Why the store could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum PairingError {
    /// The file could not be read or written.
    #[error("pairing store io error at {path}: {source}")]
    Io {
        /// Path being read or written.
        path: String,
        /// Underlying error.
        source: std::io::Error,
    },
    /// The file exists but is not a usable store.
    #[error(
        "pairing store at {0} is unreadable; refusing to replace it — every paired device would \
         silently stop connecting. Inspect it, or remove it deliberately to re-pair."
    )]
    Corrupt(String),
}

/// The paired devices on disk.
///
/// A list, not a record. One machine can reasonably be paired with a phone and
/// a tablet, and someone replacing a phone has both for a while; discovering
/// that later would mean migrating the file rather than appending to it.
pub struct PairingStore {
    path: PathBuf,
}

impl PairingStore {
    /// A store backed by `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Where the daemon keeps it: beside `account-token`.
    ///
    /// Deliberately the same directory as the account token. Their blast radius
    /// and their backup story are identical, and splitting them across two
    /// places is how one of them ends up outside somebody's backup.
    pub fn default_path() -> PathBuf {
        crate::paths::resolve_from_daemon()
            .data_dir
            .join("pairings.json")
    }

    /// Every pairing, oldest first.
    ///
    /// A file that exists but does not parse is [`PairingError::Corrupt`] and is
    /// left untouched. Replacing it would invalidate every paired device, and
    /// the only symptom anyone would see is "it stopped connecting" — the exact
    /// silent failure this whole path is built to avoid.
    pub fn load(&self) -> Result<Vec<Pairing>, PairingError> {
        let raw = match std::fs::read_to_string(&self.path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(PairingError::Io {
                    path: self.path.display().to_string(),
                    source,
                })
            }
        };
        if raw.trim().is_empty() {
            return Ok(Vec::new());
        }
        serde_json::from_str(&raw)
            .map_err(|_| PairingError::Corrupt(self.path.display().to_string()))
    }

    /// Add one, keeping what is already there.
    pub fn add(&self, pairing: Pairing) -> Result<(), PairingError> {
        let mut all = self.load()?;
        all.retain(|p| p.control_channel_id != pairing.control_channel_id);
        all.push(pairing);
        self.write(&all)
    }

    /// Record where to wake this device, or forget it (`None`).
    ///
    /// Reports whether the pairing existed. A subscription for a pairing that
    /// is gone is not an error worth failing on — it is a device that was
    /// revoked between subscribing and saying so — but it must not be stored.
    pub fn set_push(
        &self,
        control_channel_id: &str,
        sub: Option<super::web_push::Subscription>,
    ) -> Result<bool, PairingError> {
        let mut all = self.load()?;
        let Some(entry) = all
            .iter_mut()
            .find(|p| p.control_channel_id == control_channel_id)
        else {
            return Ok(false);
        };
        entry.push = sub;
        self.write(&all)?;
        Ok(true)
    }

    /// Forget one by its control channel. Reports whether there was one.
    pub fn remove(&self, control_channel_id: &str) -> Result<bool, PairingError> {
        let mut all = self.load()?;
        let before = all.len();
        all.retain(|p| p.control_channel_id != control_channel_id);
        if all.len() == before {
            return Ok(false);
        }
        self.write(&all)?;
        Ok(true)
    }

    fn write(&self, all: &[Pairing]) -> Result<(), PairingError> {
        let io = |source: std::io::Error| PairingError::Io {
            path: self.path.display().to_string(),
            source,
        };
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        let json = serde_json::to_string_pretty(all).expect("pairings serialize");
        // Written beside the target and renamed, so a process that dies
        // mid-write leaves the old file rather than half of a new one — a
        // truncated store reads as Corrupt and locks the user out of their own
        // pairings until they go and look at it.
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, json).map_err(io)?;
        std::fs::rename(&tmp, &self.path).map_err(io)?;
        restrict(&self.path);
        Ok(())
    }
}

/// Narrow the file to its owner where the platform has a notion of that.
///
/// Best-effort: a failure here is not worth refusing to pair over, and on
/// Windows the equivalent is inherited from the data directory.
fn restrict(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

/// Mint a new pairing: two channels, one code, both keys derived.
///
/// Returns the pairing and the code to put in front of the person — the only
/// time the code exists anywhere, by design.
///
/// The two Argon2id passes are the expensive part (64 MiB each), so this is
/// meant to be called from a blocking context; see [`mint_blocking`].
pub fn mint(code: &str) -> Option<Pairing> {
    let control_channel_id = uuid::Uuid::new_v4().to_string();
    let data_channel_id = uuid::Uuid::new_v4().to_string();
    // Different channel ids mean different keys from the same code — the salt
    // is `{code}|{channel_id}`, so the control channel's key cannot open the
    // data channel's frames or the other way round.
    let control_key = super::crypto::derive_channel_key(code, &control_channel_id).ok()?;
    let data_key = super::crypto::derive_channel_key(code, &data_channel_id).ok()?;
    Some(Pairing {
        control_channel_id,
        data_channel_id,
        control_key: hex(&control_key),
        data_key: hex(&data_key),
        created_at: now(),
        label: None,
        push: None,
    })
}

/// [`mint`] off the async runtime.
///
/// Two 64 MiB Argon2id derivations are hundreds of milliseconds of solid CPU;
/// running them on a worker thread would stall every other task on it.
pub async fn mint_blocking(code: String) -> Option<Pairing> {
    tokio::task::spawn_blocking(move || mint(&code)).await.ok()?
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or_default()
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (PairingStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = PairingStore::new(dir.path().join("pairings.json"));
        (store, dir)
    }

    fn pairing(id: &str) -> Pairing {
        Pairing {
            control_channel_id: id.into(),
            data_channel_id: format!("{id}-data"),
            control_key: "aa".repeat(32),
            data_key: "bb".repeat(32),
            created_at: 1,
            label: None,
            push: None,
        }
    }

    #[test]
    fn a_store_that_does_not_exist_yet_is_empty_not_an_error() {
        let (store, _dir) = store();
        assert_eq!(store.load().unwrap(), Vec::new());
    }

    #[test]
    fn what_was_added_comes_back() {
        let (store, _dir) = store();
        store.add(pairing("c1")).unwrap();
        assert_eq!(store.load().unwrap(), vec![pairing("c1")]);
    }

    #[test]
    fn a_second_device_does_not_replace_the_first() {
        // One machine, a phone and a tablet. Discovering this later would mean
        // migrating the file rather than appending to it.
        let (store, _dir) = store();
        store.add(pairing("c1")).unwrap();
        store.add(pairing("c2")).unwrap();
        let all = store.load().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].control_channel_id, "c1");
    }

    #[test]
    fn re_adding_the_same_channel_replaces_it() {
        let (store, _dir) = store();
        store.add(pairing("c1")).unwrap();
        let mut updated = pairing("c1");
        updated.label = Some("iPhone".into());
        store.add(updated.clone()).unwrap();
        assert_eq!(store.load().unwrap(), vec![updated]);
    }

    #[test]
    fn revoking_removes_exactly_one() {
        let (store, _dir) = store();
        store.add(pairing("c1")).unwrap();
        store.add(pairing("c2")).unwrap();
        assert!(store.remove("c1").unwrap());
        assert!(!store.remove("c1").unwrap(), "a second removal is a no-op");
        let left = store.load().unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].control_channel_id, "c2");
    }

    #[test]
    fn a_corrupt_store_is_reported_never_replaced() {
        // Replacing it would silently unpair every device, and the only symptom
        // anyone would see is that it stopped working.
        let (store, dir) = store();
        std::fs::write(dir.path().join("pairings.json"), "{ not json").unwrap();
        assert!(matches!(store.load(), Err(PairingError::Corrupt(_))));
        // And the bad file is still there to be looked at.
        assert!(dir.path().join("pairings.json").exists());
    }

    #[test]
    fn an_empty_file_is_an_empty_store_not_a_corrupt_one() {
        let (store, dir) = store();
        std::fs::write(dir.path().join("pairings.json"), "  \n").unwrap();
        assert_eq!(store.load().unwrap(), Vec::new());
    }

    #[test]
    fn a_subscription_is_kept_with_its_pairing_and_dies_with_it() {
        // An endpoint is a bearer capability to make somebody's phone buzz.
        // Revoking the device has to take it with them.
        let (store, _dir) = store();
        store.add(pairing("c1")).unwrap();
        let sub = super::super::web_push::Subscription {
            endpoint: "https://push.example/abc".into(),
            p256dh: "k".into(),
            auth: "a".into(),
        };
        assert!(store.set_push("c1", Some(sub.clone())).unwrap());
        assert_eq!(store.load().unwrap()[0].push, Some(sub));

        assert!(store.remove("c1").unwrap());
        assert!(store.load().unwrap().is_empty());
    }

    #[test]
    fn subscribing_for_a_pairing_that_is_gone_stores_nothing() {
        let (store, _dir) = store();
        let sub = super::super::web_push::Subscription {
            endpoint: "https://push.example/abc".into(),
            p256dh: "k".into(),
            auth: "a".into(),
        };
        assert!(!store.set_push("nope", Some(sub)).unwrap());
        assert!(store.load().unwrap().is_empty());
    }

    #[test]
    fn keys_round_trip_through_hex() {
        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        assert_eq!(unhex(&hex(&key)), Some(key));
    }

    #[test]
    fn a_key_that_is_not_thirty_two_bytes_is_no_key() {
        let p = Pairing {
            control_key: "aa".into(),
            ..pairing("c1")
        };
        assert_eq!(p.control_key(), None);
        assert!(p.data_key().is_some());
    }

    #[test]
    fn minting_gives_two_channels_with_two_different_keys() {
        // The salt is `{code}|{channel_id}`, so one code over two channels is
        // two keys: the control channel cannot open the data channel's frames.
        let p = mint("A-BCDE-FGHJ-KMNP").expect("mint");
        assert_ne!(p.control_channel_id, p.data_channel_id);
        assert_ne!(p.control_key, p.data_key);
        assert!(p.control_key().is_some());
        assert!(p.data_key().is_some());
    }

    #[test]
    fn the_stored_keys_are_what_the_code_derives() {
        // The reason storing keys rather than the code is safe: they are the
        // same answer, and the daemon no longer has to recompute it at boot.
        let code = "X-7Q2K-9ABC-DEF3";
        let p = mint(code).expect("mint");
        let expected = super::super::crypto::derive_channel_key(code, &p.control_channel_id)
            .expect("derive");
        assert_eq!(p.control_key(), Some(expected));
    }
}
