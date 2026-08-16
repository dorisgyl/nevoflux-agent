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

/// Ceiling for a file this store adopts rather than copies.
///
/// Higher than [`MAX_ASSET_BYTES`] because it is guarding something else. That
/// one bounds what the daemon writes: a caller hands over bytes it is holding,
/// they are copied to disk, and both costs are real. An adopted file is
/// already on disk and stays where it is — nothing is read until the portal
/// asks for a range, and then only that range. What is left to bound is how
/// much a viewer could be made to pull over the relay, and a feature-length
/// film fits under this.
pub const MAX_ADOPTED_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The most one `asset_data` frame may carry, in source bytes. Base64 inflates
/// by a third — 342 KB — which stays under the relay's ~900 KB frame ceiling
/// with room for the envelope. Matches the phone's upload chunking.
pub const CHUNK_BYTES: usize = 256 * 1024;

/// The downlink frame carrying one range of an asset.
///
/// Shared by the pull path and the resume path rather than written out at each,
/// so the two cannot drift. The portal matches a reply by id and offset; a
/// resent frame differing in shape would be an answer it could not use.
pub fn data_frame(id: &str, offset: u64, bytes: &[u8], eof: bool) -> serde_json::Value {
    use base64::Engine;
    serde_json::json!({
        "kind": "asset_data",
        "id": id,
        "offset": offset,
        "data": base64::engine::general_purpose::STANDARD.encode(bytes),
        "eof": eof,
    })
}

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
    /// Which sequence this belongs to, when it is one part of several. Equal
    /// to the first part's id, so the body's single `nevo-asset:` reference
    /// still names something real and the head never has to order anything.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// Position within the group, from zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq: Option<u32>,
    /// How many parts the group will have, when that is known up front. A
    /// producer that cannot know yet leaves it out; it drives a progress label
    /// and nothing else, so absence costs only the label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub of: Option<u32>,
}

struct Asset {
    path: PathBuf,
    name: String,
    mime_type: String,
    size: u64,
    /// Kept so `offer` can rebuild what `put_grouped` said. Without these a
    /// re-announcement quietly demotes a part back to a standalone file, and
    /// the player loses the rest of the sequence mid-playback.
    group: Option<String>,
    seq: Option<u32>,
    of: Option<u32>,
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
        self.store(bytes, name, mime_type, None, None, None)
    }

    /// Serve a file that already exists, without copying it.
    ///
    /// For media the machine is holding anyway — a recording, a download, a
    /// film someone asked to watch. Copying it would double a gigabyte on disk
    /// and read the whole thing into memory to do it, to produce a second copy
    /// of a file that is not going anywhere. `read` already seeks into a path;
    /// this simply records the path it was given.
    ///
    /// The file is not owned, so it is not counted against the quota — the
    /// quota bounds what this store spends, and it spends nothing here. It is
    /// also not guarded: something that deletes or truncates it mid-playback
    /// turns the next range into a read error, which is the honest outcome. A
    /// copy would only have moved that failure earlier.
    pub fn adopt(
        &mut self,
        path: &std::path::Path,
        name: &str,
        mime_type: &str,
    ) -> Result<AssetOffer, AssetError> {
        let meta = std::fs::metadata(path).map_err(|e| AssetError::Io(e.to_string()))?;
        if !meta.is_file() {
            return Err(AssetError::Io(format!(
                "{} is not a file",
                path.display()
            )));
        }
        let size = meta.len();
        if size > MAX_ADOPTED_BYTES {
            return Err(AssetError::TooLarge {
                size,
                max: MAX_ADOPTED_BYTES,
            });
        }

        let id = uuid::Uuid::new_v4().to_string();
        let offer = AssetOffer {
            id: id.clone(),
            name: super::translate::sanitize_display_name(name),
            mime_type: mime_type.to_string(),
            size,
            seekable: true,
            group: None,
            seq: None,
            of: None,
        };
        self.pending.push(offer.clone());
        self.assets.insert(
            id,
            Asset {
                path: path.to_path_buf(),
                name: offer.name.clone(),
                mime_type: offer.mime_type.clone(),
                size,
                group: None,
                seq: None,
                of: None,
            },
        );
        Ok(offer)
    }

    /// Store one part of a sequence.
    ///
    /// Pass `group: None` for the first part: the id is minted inside, so only
    /// this can name the group after it.
    pub fn put_grouped(
        &mut self,
        bytes: &[u8],
        name: &str,
        mime_type: &str,
        group: Option<String>,
        seq: u32,
        of: u32,
    ) -> Result<AssetOffer, AssetError> {
        self.store(bytes, name, mime_type, group, Some(seq), Some(of))
    }

    fn store(
        &mut self,
        bytes: &[u8],
        name: &str,
        mime_type: &str,
        group: Option<String>,
        seq: Option<u32>,
        of: Option<u32>,
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
            // A first part names the group after itself.
            group: group.or_else(|| seq.map(|_| id.clone())),
            seq,
            of,
        };
        self.used += size;
        // An ungrouped asset waits for the next outgoing text to be announced.
        // A grouped one was already announced by whoever pushed it, so queuing
        // it here would send it twice — and the second announcement rebuilds
        // from disk, where the sequence fields would be missing.
        if seq.is_none() {
            self.pending.push(offer.clone());
        }
        self.assets.insert(
            id,
            Asset {
                path,
                name: offer.name.clone(),
                mime_type: offer.mime_type.clone(),
                size,
                group: offer.group.clone(),
                seq,
                of,
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
            group: a.group.clone(),
            seq: a.seq,
            of: a.of,
        })
    }

    /// Offers not yet announced. Draining is what makes an announcement
    /// happen once.
    pub fn take_pending(&mut self) -> Vec<AssetOffer> {
        std::mem::take(&mut self.pending)
    }

    /// What is stored but not yet announced, without taking it.
    ///
    /// [`take_pending`](Self::take_pending) is the announcement path and has to
    /// empty the list. This is for asking what the turn has produced while it
    /// is still being written — which is what a body reference naming nothing
    /// could have meant.
    pub fn pending_ids(&self) -> Vec<String> {
        self.pending.iter().map(|o| o.id.clone()).collect()
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
    fn an_ungrouped_put_carries_no_sequence_fields() {
        let (mut s, _d) = store();
        let offer = s.put(b"hello", "a.wav", "audio/wav").unwrap();
        assert!(offer.group.is_none());
        assert!(offer.seq.is_none());
        assert!(offer.of.is_none());
    }

    #[test]
    fn the_first_part_becomes_the_group_id() {
        let (mut s, _d) = store();
        // The caller cannot know the id before the put, so passing None for the
        // first part is what lets the group name itself.
        let first = s
            .put_grouped(b"one", "0.wav", "audio/wav", None, 0, 3)
            .unwrap();
        assert_eq!(first.group.as_deref(), Some(first.id.as_str()));
        assert_eq!(first.seq, Some(0));
        assert_eq!(first.of, Some(3));

        let second = s
            .put_grouped(b"two", "1.wav", "audio/wav", first.group.clone(), 1, 3)
            .unwrap();
        assert_eq!(second.group, first.group);
        assert_eq!(second.seq, Some(1));
        assert_ne!(second.id, first.id, "each part is its own asset");
    }

    #[test]
    fn offer_rebuilds_the_sequence_fields() {
        let (mut s, _d) = store();
        let first = s
            .put_grouped(b"one", "0.wav", "audio/wav", None, 0, 2)
            .unwrap();
        // This is the path a re-announcement takes — from the body naming the
        // id, or from the pending queue. Rebuilding without the sequence would
        // demote the part to a standalone file and lose the rest of the group.
        let again = s.offer(&first.id).expect("offer should rebuild");
        assert_eq!(again.group, first.group);
        assert_eq!(again.seq, Some(0));
        assert_eq!(again.of, Some(2));
    }

    #[test]
    fn a_grouped_part_does_not_queue_for_the_next_text() {
        let (mut s, _d) = store();
        s.put_grouped(b"one", "0.wav", "audio/wav", None, 0, 1)
            .unwrap();
        assert!(
            s.take_pending().is_empty(),
            "whoever pushed it already announced it"
        );
    }

    #[test]
    fn an_ungrouped_put_still_queues() {
        let (mut s, _d) = store();
        s.put(b"x", "a.png", "image/png").unwrap();
        assert_eq!(s.take_pending().len(), 1);
    }

    #[test]
    fn an_adopted_file_is_read_where_it_lies() {
        let (mut s, d) = store();
        let source = d.path().join("outside.mp4");
        let bytes: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&source, &bytes).unwrap();

        let offer = s.adopt(&source, "outside.mp4", "video/mp4").unwrap();
        assert_eq!(offer.size, 5000);
        assert_eq!(s.read(&offer.id, 1000, 500).unwrap(), bytes[1000..1500]);
        // Nothing was copied: the original is still the only file with these
        // bytes, and the store's own directory did not grow.
        assert!(source.exists());
        assert_eq!(s.used, 0, "an adopted file spends none of the quota");
    }

    #[test]
    fn an_adopted_file_that_goes_away_fails_the_read() {
        // Nobody guards it, so this is the honest outcome rather than a
        // silent zero-length answer.
        let (mut s, d) = store();
        let source = d.path().join("gone.mp4");
        std::fs::write(&source, b"here for now").unwrap();
        let offer = s.adopt(&source, "gone.mp4", "video/mp4").unwrap();
        std::fs::remove_file(&source).unwrap();
        assert!(s.read(&offer.id, 0, 4).is_err());
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

/// Serve a file already on disk to a session. `None` when no portal is
/// attached — same contract as `put_for_session`.
pub fn adopt_for_session(
    session_id: &str,
    path: &std::path::Path,
    name: &str,
    mime_type: &str,
) -> Result<AssetOffer, AssetError> {
    let store = registry()
        .lock()
        .ok()
        .and_then(|r| r.get(session_id).cloned())
        .ok_or_else(|| AssetError::Io("no portal is attached to this session".into()))?;
    let mut store = store.lock().map_err(|_| AssetError::Io("store".into()))?;
    store.adopt(path, name, mime_type)
}

/// Store one part of a sequence for a session. `None` when no portal is
/// attached — same contract as `put_for_session`.
pub fn put_grouped_for_session(
    session_id: &str,
    bytes: &[u8],
    name: &str,
    mime_type: &str,
    group: Option<String>,
    seq: u32,
    of: u32,
) -> Option<AssetOffer> {
    let store = registry().lock().ok()?.get(session_id)?.clone();
    let mut store = store.lock().ok()?;
    match store.put_grouped(bytes, name, mime_type, group, seq, of) {
        Ok(offer) => Some(offer),
        Err(e) => {
            tracing::warn!(target: "remote", error = %e, "asset part refused");
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
