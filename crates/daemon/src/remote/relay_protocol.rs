//! Portal relay wire protocol + Y2 send-side sequencing (design §Q12).
//!
//! Matches the portal `RelayChatTransport` (`nevoflux-portal`
//! `src/lib/chat/{wire,sequence}.ts`): one WS message carries one
//! [`WireMessage`] (JSON; in the E2E mode the whole message is AES-256-GCM
//! sealed via [`super::crypto`]). The daemon assigns a monotonic `seq` (from 0)
//! to each downlink data frame and retains sent frames so it can honor the
//! portal's `resume{from}` requests; a gap larger than the buffer escalates to
//! `resync` (portal then resets its sequencer to 0).
//!
//! The inner business `frame` is left opaque (`serde_json::Value`) at this
//! layer — the typed `InboundFrame`/`OutboundFrame` schema and their
//! translation to/from `DaemonEnvelope` land with the M2 tap wiring.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The wire envelope, discriminated by `k` (matches portal `wire.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "k", rename_all = "lowercase")]
pub enum WireMessage {
    /// A business frame. `seq` is present only on downlink (daemon→portal) data
    /// frames; portal→daemon uplink frames omit it.
    Frame {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seq: Option<u64>,
        frame: Value,
    },
    /// Uplink-only (portal→daemon): resend from `from` (the portal's expected
    /// next seq).
    Resume { from: u64 },
    /// Downlink-only (daemon→portal): abandon incremental catch-up; the portal
    /// full-reloads the transcript and resets its sequencer to 0.
    Resync,
}

/// Upper bound on retained entries. A ceiling on the deque itself, not the
/// memory policy — [`SEND_BUFFER_BYTES`] is what actually decides how far back
/// resume reaches. A `resume{from}` older than what the buffer still holds
/// cannot be honored incrementally and must escalate to a `Resync`.
const SEND_BUFFER_CAP: usize = 4096;

/// How many bytes of sent frames to retain.
///
/// The buffer used to be bounded by frame count alone, which reads as a memory
/// bound only if every frame is about the same size. Media broke that
/// assumption twice over: an `asset_data` frame carries a base64 chunk of ~342
/// KB, so 512 of them retained ~175 MB — and because eviction counted frames,
/// a few hundred of them also pushed every chat frame out of the window, so
/// reconnecting during playback could not resume incrementally and fell back to
/// a full transcript reload. Media no longer costs its bytes here (see
/// [`Retained::Media`]), and what remains is bounded by an honest byte budget.
const SEND_BUFFER_BYTES: usize = 1024 * 1024;

/// The frame kind whose bytes are re-readable, so the buffer need not hold them.
const MEDIA_FRAME_KIND: &str = "asset_data";

/// Roughly what a [`Retained::Media`] stub costs: an id, three numbers, and the
/// deque slot. Not measured — it only has to be non-zero so a flood of media
/// still eventually ages out of the window.
const MEDIA_STUB_COST: usize = 96;

/// What the buffer keeps so a frame can be sent again.
#[derive(Debug, Clone)]
enum Retained {
    /// The frame itself. Chat deltas, asset offers, errors — all small.
    Frame(Value),
    /// A media range, kept as the values that can rebuild it.
    ///
    /// The bytes are still on disk in the `AssetStore` and are read back on
    /// demand, so retaining the encoded body would be keeping a second, far
    /// more expensive copy of something already durable.
    Media {
        id: String,
        offset: u64,
        len: usize,
        /// Which encoding this went out as. A resend has to match: the portal
        /// that asked for bytes is decoding bytes, and one that asked for
        /// base64 would not recognise a binary frame.
        binary: bool,
    },
}

/// One frame to resend, and whether the caller still has to fetch its bytes.
#[derive(Debug, Clone, PartialEq)]
pub enum Resend {
    /// Ready to encode as-is.
    Ready(WireMessage),
    /// A media range whose bytes must be read back before it can go out. The
    /// caller owns the store, so it does the reading.
    Media {
        seq: u64,
        id: String,
        offset: u64,
        len: usize,
        /// Re-encode the way it first went out; see [`Retained::Media`].
        binary: bool,
    },
}

/// Source length of a padded standard-base64 string.
///
/// Exact rather than an estimate: the resent frame has to cover the same range
/// as the original, or the portal — which matches replies by id and offset —
/// would splice a differently-sized chunk into the file it is reassembling.
fn base64_source_len(s: &str) -> usize {
    let pad = s.bytes().rev().take_while(|&b| b == b'=').count();
    (s.len() / 4).saturating_mul(3).saturating_sub(pad)
}

/// Assigns monotonic `seq` to downlink frames and retains them for resume
/// (design Y2). Send-side only; the receive-side gap tracker lives in the
/// portal (`sequence.ts`).
#[derive(Debug, Default)]
pub struct SendSequencer {
    next: u64,
    buffer: std::collections::VecDeque<(u64, Retained, usize)>,
    /// Running sum of the third tuple element, so eviction does not have to
    /// walk the deque on every tag.
    bytes: usize,
}

impl SendSequencer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tag a business `frame` with the next seq, retain it, and return the wire
    /// message to send downlink.
    pub fn tag(&mut self, frame: Value) -> WireMessage {
        let seq = self.next;
        self.next += 1;

        let (retained, cost) = match media_stub(&frame) {
            Some(stub) => (stub, MEDIA_STUB_COST),
            // Serializing to measure costs one small allocation per frame.
            // Affordable precisely because the one frame kind that is *not*
            // small never reaches this arm.
            None => {
                let cost = serde_json::to_string(&frame).map(|s| s.len()).unwrap_or(0);
                (Retained::Frame(frame.clone()), cost)
            }
        };

        self.buffer.push_back((seq, retained, cost));
        self.bytes = self.bytes.saturating_add(cost);
        self.evict();

        WireMessage::Frame {
            seq: Some(seq),
            frame,
        }
    }

    /// Drop the oldest entries until the buffer is back inside its budget.
    ///
    /// Always keeps the newest entry, however large: dropping the frame just
    /// taken would make resume permanently unable to reach the present.
    fn evict(&mut self) {
        while self.buffer.len() > 1
            && (self.bytes > SEND_BUFFER_BYTES || self.buffer.len() > SEND_BUFFER_CAP)
        {
            if let Some((_, _, dropped)) = self.buffer.pop_front() {
                self.bytes = self.bytes.saturating_sub(dropped);
            }
        }
    }

    /// Claim a seq for a media range going out as bytes.
    ///
    /// The binary path has no JSON frame for [`tag`](Self::tag) to inspect, so
    /// it says outright what that one has to infer. Retention is identical —
    /// a range and a format, never the payload.
    pub fn tag_media(&mut self, id: &str, offset: u64, len: usize) -> u64 {
        let seq = self.next;
        self.next += 1;
        self.buffer.push_back((
            seq,
            Retained::Media {
                id: id.to_string(),
                offset,
                len,
                binary: true,
            },
            MEDIA_STUB_COST,
        ));
        self.bytes = self.bytes.saturating_add(MEDIA_STUB_COST);
        self.evict();
        seq
    }

    /// The next seq that will be assigned.
    pub fn next_seq(&self) -> u64 {
        self.next
    }

    /// Bytes currently retained for resume. Diagnostics and tests: the whole
    /// point of the stub is that this stays flat while media streams.
    pub fn retained_bytes(&self) -> usize {
        self.bytes
    }

    /// Resend everything from `from` (inclusive). Returns the resend plan, or
    /// `None` when `from` is older than the buffer still holds — the caller must
    /// then send a [`WireMessage::Resync`] and [`reset`](Self::reset).
    pub fn resend_from(&self, from: u64) -> Option<Vec<Resend>> {
        if from >= self.next {
            return Some(Vec::new()); // already caught up
        }
        match self.buffer.front().map(|(s, _, _)| *s) {
            Some(oldest) if from >= oldest => Some(
                self.buffer
                    .iter()
                    .filter(|(s, _, _)| *s >= from)
                    .map(|(s, r, _)| match r {
                        Retained::Frame(f) => Resend::Ready(WireMessage::Frame {
                            seq: Some(*s),
                            frame: f.clone(),
                        }),
                        Retained::Media {
                            id,
                            offset,
                            len,
                            binary,
                        } => Resend::Media {
                            seq: *s,
                            id: id.clone(),
                            offset: *offset,
                            len: *len,
                            binary: *binary,
                        },
                    })
                    .collect(),
            ),
            _ => None, // buffer doesn't reach back to `from`
        }
    }

    /// Reset the counter and buffer for a fresh transcript (paired with a
    /// `Resync` and the portal's `resetSequence(0)`).
    pub fn reset(&mut self) {
        self.next = 0;
        self.buffer.clear();
        self.bytes = 0;
    }
}

/// Recognize a media data frame and reduce it to what can rebuild it.
///
/// A frame missing any of the three fields is *not* treated as media — it would
/// be unrebuildable, and retaining it whole is the safe reading.
fn media_stub(frame: &Value) -> Option<Retained> {
    if frame.get("kind").and_then(Value::as_str)? != MEDIA_FRAME_KIND {
        return None;
    }
    let id = frame.get("id").and_then(Value::as_str)?.to_string();
    let offset = frame.get("offset").and_then(Value::as_u64)?;
    let len = base64_source_len(frame.get("data").and_then(Value::as_str)?);
    Some(Retained::Media {
        id,
        offset,
        len,
        binary: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn frame(id: &str) -> Value {
        json!({ "kind": "stream_delta", "streamId": id, "delta": "x" })
    }

    #[test]
    fn wire_frame_with_seq_roundtrip() {
        let w = WireMessage::Frame {
            seq: Some(0),
            frame: frame("s1"),
        };
        let s = serde_json::to_string(&w).unwrap();
        assert!(s.contains("\"k\":\"frame\""));
        assert!(s.contains("\"seq\":0"));
        assert_eq!(serde_json::from_str::<WireMessage>(&s).unwrap(), w);
    }

    #[test]
    fn wire_frame_without_seq_omits_field() {
        let w = WireMessage::Frame {
            seq: None,
            frame: json!({ "kind": "user_message", "text": "hi" }),
        };
        let s = serde_json::to_string(&w).unwrap();
        assert!(!s.contains("seq"), "uplink frames omit seq: {s}");
        assert_eq!(serde_json::from_str::<WireMessage>(&s).unwrap(), w);
    }

    #[test]
    fn wire_resume_and_resync_shapes() {
        assert_eq!(
            serde_json::to_string(&WireMessage::Resume { from: 3 }).unwrap(),
            r#"{"k":"resume","from":3}"#
        );
        assert_eq!(
            serde_json::to_string(&WireMessage::Resync).unwrap(),
            r#"{"k":"resync"}"#
        );
        assert_eq!(
            serde_json::from_str::<WireMessage>(r#"{"k":"resync"}"#).unwrap(),
            WireMessage::Resync
        );
    }

    #[test]
    fn sequencer_assigns_monotonic_seq_from_zero() {
        let mut seq = SendSequencer::new();
        for expected in 0..3 {
            match seq.tag(frame("s")) {
                WireMessage::Frame { seq: Some(n), .. } => assert_eq!(n, expected),
                other => panic!("expected Frame with seq, got {other:?}"),
            }
        }
        assert_eq!(seq.next_seq(), 3);
    }

    /// An `asset_data` frame the size the gateway really sends: a 256 KB chunk
    /// base64-encoded to ~342 KB.
    fn media_frame(id: &str, offset: u64, source_bytes: usize) -> Value {
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD.encode(vec![7u8; source_bytes]);
        json!({ "kind": "asset_data", "id": id, "offset": offset, "data": data, "eof": false })
    }

    fn resent_seqs(out: &[Resend]) -> Vec<u64> {
        out.iter()
            .map(|r| match r {
                Resend::Ready(WireMessage::Frame { seq: Some(n), .. }) => *n,
                Resend::Media { seq, .. } => *seq,
                other => panic!("unexpected resend item {other:?}"),
            })
            .collect()
    }

    #[test]
    fn resend_from_returns_buffered_tail() {
        let mut seq = SendSequencer::new();
        for _ in 0..3 {
            seq.tag(frame("s"));
        }
        // resume from 1 → frames 1 and 2.
        assert_eq!(resent_seqs(&seq.resend_from(1).unwrap()), vec![1, 2]);
        // already caught up → empty.
        assert!(seq.resend_from(3).unwrap().is_empty());
    }

    #[test]
    fn resend_from_too_old_returns_none_for_resync() {
        let mut seq = SendSequencer::new();
        // Big enough frames that the byte budget, not the entry cap, is what
        // evicts — the same thing that happens in a long session.
        let big = json!({ "kind": "stream_delta", "delta": "x".repeat(4096) });
        while seq.retained_bytes() + 8192 < SEND_BUFFER_BYTES {
            seq.tag(big.clone());
        }
        for _ in 0..64 {
            seq.tag(big.clone());
        }
        assert!(
            seq.resend_from(0).is_none(),
            "gap older than buffer must force resync"
        );
    }

    #[test]
    fn a_media_frame_is_retained_as_a_stub_not_as_its_bytes() {
        // The defect this guards: one 256 KB chunk used to be retained as ~342 KB
        // of base64, and 512 of them came to ~175 MB of live memory.
        let mut seq = SendSequencer::new();
        seq.tag(media_frame("asset-1", 0, 256 * 1024));
        assert!(
            seq.retained_bytes() < 1024,
            "a media frame must not retain its payload, retained {} bytes",
            seq.retained_bytes()
        );
    }

    #[test]
    fn streaming_media_does_not_grow_the_buffer() {
        let mut seq = SendSequencer::new();
        for i in 0..2_000u64 {
            seq.tag(media_frame("asset-1", i * 256 * 1024, 256 * 1024));
        }
        assert!(
            seq.retained_bytes() <= SEND_BUFFER_BYTES,
            "retained {} bytes",
            seq.retained_bytes()
        );
    }

    #[test]
    fn media_does_not_evict_the_chat_around_it() {
        // The second half of the defect, and the user-visible one. Eviction
        // counted frames, so a few hundred chunks pushed every chat frame out of
        // the window and a reconnect mid-playback could only full-reload the
        // transcript. Playing a file is not a reason to lose the conversation.
        let mut seq = SendSequencer::new();
        seq.tag(frame("chat-before"));
        for i in 0..1_000u64 {
            seq.tag(media_frame("asset-1", i * 256 * 1024, 256 * 1024));
        }
        seq.tag(frame("chat-after"));

        let out = seq
            .resend_from(0)
            .expect("the chat frame at seq 0 must still be reachable");
        assert_eq!(out.len(), 1002);
        assert!(
            matches!(
                &out[0],
                Resend::Ready(WireMessage::Frame { seq: Some(0), .. })
            ),
            "seq 0 should still be the chat frame, got {:?}",
            out[0]
        );
    }

    #[test]
    fn a_media_frame_resends_as_a_request_for_its_bytes() {
        let mut seq = SendSequencer::new();
        seq.tag(media_frame("asset-7", 512, 1000));
        let out = seq.resend_from(0).unwrap();
        assert_eq!(
            out,
            vec![Resend::Media {
                seq: 0,
                id: "asset-7".into(),
                offset: 512,
                len: 1000,
                binary: false,
            }],
            "the range has to come back exactly, or the portal splices a \
             differently-sized chunk into the file it is reassembling"
        );
    }

    #[test]
    fn a_binary_media_range_resends_as_binary() {
        // Format has to survive into the resend. The portal that asked for
        // bytes is decoding bytes; base64 would be a frame it cannot read, on
        // the one seq it is blocked waiting for.
        let mut seq = SendSequencer::new();
        seq.tag_media("asset-9", 8192, 4096);
        assert_eq!(
            seq.resend_from(0).unwrap(),
            vec![Resend::Media {
                seq: 0,
                id: "asset-9".into(),
                offset: 8192,
                len: 4096,
                binary: true,
            }]
        );
    }

    #[test]
    fn a_binary_media_range_is_no_more_expensive_to_retain() {
        let mut seq = SendSequencer::new();
        for i in 0..2_000u64 {
            seq.tag_media("asset-9", i * 256 * 1024, 256 * 1024);
        }
        assert!(
            seq.retained_bytes() <= SEND_BUFFER_BYTES,
            "retained {} bytes",
            seq.retained_bytes()
        );
    }

    #[test]
    fn the_two_media_paths_share_one_seq_space() {
        // They go out the same socket, so the portal orders them together. Two
        // counters would hand it duplicate seqs and stall its tracker.
        let mut seq = SendSequencer::new();
        seq.tag(frame("chat"));
        seq.tag_media("a", 0, 10);
        seq.tag(media_frame("b", 0, 10));
        assert_eq!(resent_seqs(&seq.resend_from(0).unwrap()), vec![0, 1, 2]);
        assert_eq!(seq.next_seq(), 3);
    }

    #[test]
    fn base64_source_len_is_exact_across_padding_cases() {
        use base64::Engine;
        for n in 0..16usize {
            let encoded = base64::engine::general_purpose::STANDARD.encode(vec![1u8; n]);
            assert_eq!(base64_source_len(&encoded), n, "n = {n}");
        }
    }

    #[test]
    fn a_malformed_media_frame_is_retained_whole() {
        // Missing `data` means nothing could rebuild it, so treating it as media
        // would make it permanently unresendable. Keeping it is the safe reading.
        let mut seq = SendSequencer::new();
        seq.tag(json!({ "kind": "asset_data", "id": "x", "offset": 0 }));
        assert!(matches!(
            seq.resend_from(0).unwrap().as_slice(),
            [Resend::Ready(_)]
        ));
    }

    #[test]
    fn the_newest_frame_survives_even_when_it_alone_busts_the_budget() {
        // Otherwise the buffer could evict the frame it just took, and resume
        // would have no way back to the present.
        let mut seq = SendSequencer::new();
        seq.tag(json!({ "kind": "stream_delta", "delta": "x".repeat(SEND_BUFFER_BYTES * 2) }));
        assert_eq!(seq.resend_from(0).map(|v| v.len()), Some(1));
    }

    #[test]
    fn reset_rewinds_to_zero() {
        let mut seq = SendSequencer::new();
        seq.tag(frame("s"));
        seq.tag(frame("s"));
        seq.reset();
        assert_eq!(seq.next_seq(), 0);
        match seq.tag(frame("s")) {
            WireMessage::Frame { seq: Some(0), .. } => {}
            other => panic!("after reset seq restarts at 0, got {other:?}"),
        }
    }
}
