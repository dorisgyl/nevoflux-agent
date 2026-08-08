//! Downstream media: bytes the head holds, served to the portal on demand.
//!
//! The counterpart to [`super::upload`], and the answer to a defect it makes
//! plain by contrast. A screenshot used to reach the phone by being written
//! into the assistant's own prose as a `data:` URL — twenty thousand
//! characters the model had to produce token by token. What arrived had the
//! right head and the right tail and a couple of thousand characters missing
//! from the middle, so the browser held `naturalWidth` at 0 and drew nothing.
//!
//! Bytes are not text and must not travel as text. They are kept here under an
//! id; the model is told the id and nothing else; the portal asks for the
//! ranges it wants. Nothing in that path can drop a character, because nothing
//! in that path is transcribing anything.
//!
//! Ids are minted here rather than accepted from anywhere, so no caller —
//! model or remote peer — is in a position to name a file this store did not
//! create.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

/// Per-asset ceiling. Generous next to a screenshot because this path also
/// carries recordings, and refusing at the door beats truncating downstream.
pub const MAX_ASSET_BYTES: u64 = 64 * 1024 * 1024;

/// The most one `asset_data` frame may carry, in source bytes. Base64 inflates
/// by a third — 342 KB — which stays under the relay's ~900 KB frame ceiling
/// with room for the envelope. Matches the phone's upload chunking.
pub const CHUNK_BYTES: usize = 256 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("no asset {0}")]
    Unknown(String),
    #[error("asset is {size} bytes, over the {max} byte limit")]
    TooLarge { size: u64, max: u64 },
    #[error("session asset quota exhausted ({used}/{quota} bytes)")]
    QuotaExceeded { used: u64, quota: u64 },
    #[error("offset {offset} is past the end of a {size} byte asset")]
    OutOfRange { offset: u64, size: u64 },
    #[error("could not store the asset: {0}")]
    Io(String),
}

/// What the portal is told about an asset before it asks for any of it.
///
/// Tens of bytes. It rides the ordinary frame stream next to the text, so it
/// costs the turn nothing and blocks no token.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AssetOffer {
    pub id: String,
    pub name: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub size: u64,
    /// Whether an arbitrary offset can be read. Always true here — everything
    /// this store holds is a finished file. A head that later streams while
    /// still producing would say false, and the portal would stop offering to
    /// seek rather than seek into bytes that do not exist yet.
    pub seekable: bool,
}

struct Asset {
    path: PathBuf,
    name: String,
    mime_type: String,
    size: u64,
}

/// One remote session's media, on disk under a per-channel directory.
pub struct AssetStore {
    root: PathBuf,
    quota_bytes: u64,
    used: u64,
    assets: HashMap<String, Asset>,
    /// Offers stored but not yet announced to the portal.
    ///
    /// The head is told the id and given the markdown to show it with, and it
    /// does not always write it — it describes the page in prose instead, and
    /// the picture never appears. Whether a picture is *shown* must not depend
    /// on a model choosing to quote something. So every offer is announced;
    /// the reference, when the head does write one, only decides where in the
    /// text it is drawn.
    pending: Vec<AssetOffer>,
}

impl AssetStore {
    pub fn new(root: PathBuf, quota_bytes: u64) -> Self {
        Self {
            root,
            quota_bytes,
            used: 0,
            assets: HashMap::new(),
            pending: Vec::new(),
        }
    }

    /// This session's directory: `<cache>/nevoflux/remote-assets/<channel>`.
    pub fn root_for(channel: &str) -> PathBuf {
        let base = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
        let safe: String = channel
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        base.join("nevoflux")
            .join("remote-assets")
            .join(if safe.is_empty() {
                "default".to_string()
            } else {
                safe
            })
    }

    /// Take bytes into the store and return what the portal should be told.
    ///
    /// The id is minted here. `name` is a display string that never touches
    /// the path — the file is named after the id — so a caller cannot steer
    /// where this writes.
    pub fn put(
        &mut self,
        bytes: &[u8],
        name: &str,
        mime_type: &str,
    ) -> Result<AssetOffer, AssetError> {
        let size = bytes.len() as u64;
        if size > MAX_ASSET_BYTES {
            return Err(AssetError::TooLarge {
                size,
                max: MAX_ASSET_BYTES,
            });
        }
        if self.used + size > self.quota_bytes {
            return Err(AssetError::QuotaExceeded {
                used: self.used,
                quota: self.quota_bytes,
            });
        }

        let id = uuid::Uuid::new_v4().to_string();
        std::fs::create_dir_all(&self.root).map_err(|e| AssetError::Io(e.to_string()))?;
        let path = self.root.join(&id);
        let mut file = std::fs::File::create(&path).map_err(|e| AssetError::Io(e.to_string()))?;
        file.write_all(bytes)
            .map_err(|e| AssetError::Io(e.to_string()))?;

        let offer = AssetOffer {
            id: id.clone(),
            name: super::translate::sanitize_display_name(name),
            mime_type: mime_type.to_string(),
            size,
            seekable: true,
        };
        self.used += size;
        self.pending.push(offer.clone());
        self.assets.insert(
            id,
            Asset {
                path,
                name: offer.name.clone(),
                mime_type: offer.mime_type.clone(),
                size,
            },
        );
        Ok(offer)
    }

    /// What the portal should be told about an id it mentioned.
    pub fn offer(&self, id: &str) -> Option<AssetOffer> {
        self.assets.get(id).map(|a| AssetOffer {
            id: id.to_string(),
            name: a.name.clone(),
            mime_type: a.mime_type.clone(),
            size: a.size,
            seekable: true,
        })
    }

    /// Offers not yet announced. Draining is what makes an announcement
    /// happen once.
    pub fn take_pending(&mut self) -> Vec<AssetOffer> {
        std::mem::take(&mut self.pending)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.assets.contains_key(id)
    }

    /// Read one range. The caller is a remote peer, so both ends are clamped
    /// rather than trusted: an offset past the end is an error, and a length
    /// reaching past it is shortened to what exists.
    pub fn read(&self, id: &str, offset: u64, length: usize) -> Result<Vec<u8>, AssetError> {
        let asset = self
            .assets
            .get(id)
            .ok_or_else(|| AssetError::Unknown(id.to_string()))?;
        if offset > asset.size {
            return Err(AssetError::OutOfRange {
                offset,
                size: asset.size,
            });
        }
        let want = length.min(CHUNK_BYTES).min((asset.size - offset) as usize);
        let mut file =
            std::fs::File::open(&asset.path).map_err(|e| AssetError::Io(e.to_string()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| AssetError::Io(e.to_string()))?;
        let mut buf = vec![0u8; want];
        file.read_exact(&mut buf)
            .map_err(|e| AssetError::Io(e.to_string()))?;
        Ok(buf)
    }

    /// Is this range the last of the asset?
    pub fn is_eof(&self, id: &str, offset: u64, read: usize) -> bool {
        self.assets
            .get(id)
            .is_some_and(|a| offset + read as u64 >= a.size)
    }

    /// End of session: take everything this channel put on disk with it.
    pub fn cleanup(&mut self) {
        for asset in self.assets.values() {
            let _ = std::fs::remove_file(&asset.path);
        }
        self.assets.clear();
        self.used = 0;
        let _ = std::fs::remove_dir(&self.root);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (AssetStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (
            AssetStore::new(dir.path().to_path_buf(), 10 * 1024 * 1024),
            dir,
        )
    }

    #[test]
    fn put_then_read_returns_the_same_bytes() {
        let (mut s, _d) = store();
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let offer = s.put(&bytes, "shot.png", "image/png").unwrap();
        assert_eq!(offer.size, 1000);
        assert_eq!(s.read(&offer.id, 0, 1000).unwrap(), bytes);
    }

    #[test]
    fn a_range_in_the_middle_comes_back_intact() {
        // The whole point: the portal asks for pieces and they reassemble.
        let (mut s, _d) = store();
        let bytes: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let offer = s.put(&bytes, "x", "image/png").unwrap();

        let mut got = Vec::new();
        let mut at = 0u64;
        while at < offer.size {
            let piece = s.read(&offer.id, at, 700).unwrap();
            assert!(!piece.is_empty());
            at += piece.len() as u64;
            got.extend_from_slice(&piece);
        }
        assert_eq!(got, bytes, "reassembled bytes must equal what went in");
    }

    #[test]
    fn a_length_past_the_end_is_shortened_not_refused() {
        let (mut s, _d) = store();
        let offer = s.put(&[7u8; 100], "x", "image/png").unwrap();
        assert_eq!(s.read(&offer.id, 90, 999).unwrap().len(), 10);
        assert!(s.is_eof(&offer.id, 90, 10));
    }

    #[test]
    fn an_offset_past_the_end_is_an_error() {
        let (mut s, _d) = store();
        let offer = s.put(&[7u8; 100], "x", "image/png").unwrap();
        assert!(matches!(
            s.read(&offer.id, 101, 10),
            Err(AssetError::OutOfRange { .. })
        ));
    }

    #[test]
    fn an_id_nobody_minted_reads_nothing() {
        // Ids come from here alone; anything else is a peer guessing.
        let (s, _d) = store();
        assert!(matches!(
            s.read("../../etc/passwd", 0, 10),
            Err(AssetError::Unknown(_))
        ));
        assert!(!s.contains("../../etc/passwd"));
    }

    #[test]
    fn the_display_name_never_reaches_the_path() {
        let (mut s, _d) = store();
        let offer = s.put(&[1u8; 10], "../../evil.png", "image/png").unwrap();
        assert!(!offer.name.contains(".."), "name: {}", offer.name);
        // The file is named after the id, so the display string cannot steer it.
        assert!(s.read(&offer.id, 0, 10).is_ok());
    }

    #[test]
    fn quota_and_ceiling_refuse_rather_than_truncate() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = AssetStore::new(dir.path().to_path_buf(), 1000);
        s.put(&[0u8; 600], "a", "image/png").unwrap();
        assert!(matches!(
            s.put(&[0u8; 600], "b", "image/png"),
            Err(AssetError::QuotaExceeded { .. })
        ));
    }

    #[test]
    fn an_offer_is_pending_until_it_is_announced() {
        // Announcing must not depend on a model quoting a reference.
        let (mut s, _d) = store();
        s.put(&[1u8; 10], "a.png", "image/png").unwrap();
        s.put(&[2u8; 10], "b.png", "image/png").unwrap();
        assert_eq!(s.take_pending().len(), 2);
        assert!(s.take_pending().is_empty(), "draining announces once");
    }

    #[test]
    fn cleanup_takes_the_files_with_it() {
        let (mut s, _d) = store();
        let offer = s.put(&[1u8; 10], "x", "image/png").unwrap();
        s.cleanup();
        assert!(matches!(
            s.read(&offer.id, 0, 10),
            Err(AssetError::Unknown(_))
        ));
    }
}

/// Every live session's media, so the capture point and the gateway can reach
/// the same bytes.
///
/// A screenshot is taken deep in the agent host, which knows a session id and
/// nothing about portals; the gateway that will serve it knows a channel and
/// nothing about tools. Rather than thread one through to the other, both ask
/// here for the store belonging to a session.
///
/// A `std::sync::Mutex` on purpose. The gateway holds it across a read, but a
/// read is at most a chunk out of a local file — microseconds — and the
/// alternative is making the capture point async for no gain.
type Registry = Mutex<HashMap<String, Arc<Mutex<AssetStore>>>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The store for a session, created on first use.
pub fn store_for(session_id: &str, root: std::path::PathBuf, quota: u64) -> Arc<Mutex<AssetStore>> {
    let mut reg = registry().lock().expect("asset registry");
    reg.entry(session_id.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(AssetStore::new(root, quota))))
        .clone()
}

/// Take media for a session and return the id the model should refer to.
///
/// `None` when no portal is attached to this session — nothing is watching, so
/// there is nothing to serve and no reason to spend the disk.
/// Which sessions currently have a store. For diagnosing a miss: a screenshot
/// that finds nothing here is either on a session no portal is attached to, or
/// on a different one than the gateway registered.
pub fn registered_sessions() -> Vec<String> {
    registry()
        .lock()
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default()
}

pub fn put_for_session(
    session_id: &str,
    bytes: &[u8],
    name: &str,
    mime_type: &str,
) -> Option<AssetOffer> {
    let store = registry().lock().ok()?.get(session_id)?.clone();
    let mut store = store.lock().ok()?;
    match store.put(bytes, name, mime_type) {
        Ok(offer) => Some(offer),
        Err(e) => {
            tracing::warn!(target: "remote", error = %e, "asset refused");
            None
        }
    }
}

/// Offers this session has stored but not yet announced.
pub fn take_pending_for_session(session_id: &str) -> Vec<AssetOffer> {
    let Some(store) = registry()
        .lock()
        .ok()
        .and_then(|r| r.get(session_id).cloned())
    else {
        return Vec::new();
    };
    let mut store = match store.lock() {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    store.take_pending()
}

/// Drop a session's store when its gateway goes away.
pub fn forget_session(session_id: &str) {
    if let Some(store) = registry()
        .lock()
        .ok()
        .and_then(|mut r| r.remove(session_id))
    {
        if let Ok(mut s) = store.lock() {
            s.cleanup();
        }
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn a_session_with_no_portal_stores_nothing() {
        // Nothing is watching, so there is nothing to serve.
        assert!(put_for_session("nobody-is-here", &[1, 2, 3], "x.png", "image/png").is_none());
    }

    #[test]
    fn the_capture_point_and_the_gateway_see_the_same_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_for("sess-1", dir.path().to_path_buf(), 1024 * 1024);
        let offer = put_for_session("sess-1", &[9u8; 64], "shot.png", "image/png").unwrap();
        // The gateway reads out of its own handle to the same store.
        assert_eq!(
            store.lock().unwrap().read(&offer.id, 0, 64).unwrap(),
            vec![9u8; 64]
        );
        forget_session("sess-1");
        assert!(put_for_session("sess-1", &[1], "x", "image/png").is_none());
    }
}
