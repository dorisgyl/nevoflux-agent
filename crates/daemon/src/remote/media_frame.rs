//! Media bytes on the wire, without base64.
//!
//! `asset_data` carries a chunk of a file, and it used to carry it as base64
//! inside JSON — a third more bytes than the chunk itself, on a wire that is
//! already binary once the channel is sealed. The encoding bought nothing; it
//! was only there because the frame around it was JSON.
//!
//! This is the same information as a byte layout. It is deliberately *not* a
//! new transport: it rides the identical `Wire` the JSON frames do and is
//! sealed by the identical channel key, so nothing about who can read it
//! changes.
//!
//! # Telling the two apart
//!
//! A decoded wire payload is JSON if it starts with `{` and one of these if it
//! starts with [`MAGIC`]. `N` is not `{`, so the discriminator costs nothing and
//! cannot collide — every JSON frame this protocol has ever sent is an object.
//!
//! # Why nothing has to be negotiated up front
//!
//! The portal asks for a range and says, in that request, whether it can read
//! the answer as bytes (`asset_pull.binary`). A peer that does not know about
//! this never sets the flag and is answered in base64 exactly as before, and a
//! portal that does set it is talking to a head that must have understood the
//! flag to have acted on it. There is no handshake to get out of order and no
//! version to remember — which matters because the two ends are deployed on
//! entirely separate schedules.

/// Marks a wire payload as a binary media frame. Chosen so the first byte
/// cannot be confused with the `{` that opens every JSON frame.
pub const MAGIC: &[u8; 4] = b"NFM1";

/// `magic(4) + seq(8) + id_len(1) + offset(8) + flags(1)`, plus the id itself.
const FIXED_HEADER: usize = 4 + 8 + 1 + 8 + 1;

/// `flags` bit 0: this range ends the asset.
const FLAG_EOF: u8 = 1 << 0;

/// One media range, as it appears on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaFrame {
    pub seq: u64,
    pub id: String,
    pub offset: u64,
    pub eof: bool,
    pub data: Vec<u8>,
}

/// Whether a decoded wire payload is one of these.
pub fn is_media_frame(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == MAGIC
}

/// Serialize.
///
/// The id is length-prefixed with a single byte, which caps it at 255. Ids are
/// minted as UUIDs by the asset store, so 36 is the real number and the cap is
/// only ever reached by something that did not come from there — `encode`
/// refuses rather than truncating, because a truncated id names a different
/// asset and would be answered with the wrong file's bytes.
pub fn encode(frame: &MediaFrame) -> Result<Vec<u8>, String> {
    let id = frame.id.as_bytes();
    let id_len = u8::try_from(id.len())
        .map_err(|_| format!("asset id is {} bytes, over the 255 byte limit", id.len()))?;

    let mut out = Vec::with_capacity(FIXED_HEADER + id.len() + frame.data.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&frame.seq.to_be_bytes());
    out.push(id_len);
    out.extend_from_slice(id);
    out.extend_from_slice(&frame.offset.to_be_bytes());
    out.push(if frame.eof { FLAG_EOF } else { 0 });
    out.extend_from_slice(&frame.data);
    Ok(out)
}

/// Parse.
///
/// Every length is checked against what is actually there. The bytes arrive
/// from a remote peer, so a header claiming an id longer than the buffer is a
/// thing that will be received eventually, and it must be an error rather than
/// a panic in the middle of the read loop.
pub fn decode(bytes: &[u8]) -> Result<MediaFrame, String> {
    if !is_media_frame(bytes) {
        return Err("not a media frame".into());
    }
    if bytes.len() < FIXED_HEADER {
        return Err(format!(
            "media frame is {} bytes, shorter than its {FIXED_HEADER} byte header",
            bytes.len()
        ));
    }

    let seq = u64::from_be_bytes(bytes[4..12].try_into().expect("8 bytes"));
    let id_len = bytes[12] as usize;

    let id_end = 13 + id_len;
    // `id_end + 9` covers the offset and flags that must follow the id.
    if bytes.len() < id_end + 9 {
        return Err(format!(
            "media frame claims a {id_len} byte id but holds only {} bytes",
            bytes.len()
        ));
    }
    let id = std::str::from_utf8(&bytes[13..id_end])
        .map_err(|_| "asset id is not utf-8".to_string())?
        .to_string();

    let offset = u64::from_be_bytes(bytes[id_end..id_end + 8].try_into().expect("8 bytes"));
    let eof = bytes[id_end + 8] & FLAG_EOF != 0;
    let data = bytes[id_end + 9..].to_vec();

    Ok(MediaFrame {
        seq,
        id,
        offset,
        eof,
        data,
    })
}

/// What this saves over the base64 frame carrying the same range.
///
/// Used by the log line at the send site, so the win is observable in the field
/// rather than only asserted in a test.
pub fn overhead_saved(payload_len: usize) -> usize {
    // base64 is 4 bytes per 3, rounded up to a 4-byte group.
    let b64 = payload_len.div_ceil(3) * 4;
    b64.saturating_sub(payload_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(data: Vec<u8>) -> MediaFrame {
        MediaFrame {
            seq: 42,
            id: "3f2504e0-4f89-11d3-9a0c-0305e82c3301".into(),
            offset: 1_048_576,
            eof: true,
            data,
        }
    }

    #[test]
    fn roundtrips_a_realistic_chunk() {
        let f = sample((0..256 * 1024).map(|i| (i % 251) as u8).collect());
        let wire = encode(&f).unwrap();
        assert_eq!(decode(&wire).unwrap(), f);
    }

    #[test]
    fn roundtrips_an_empty_range() {
        // A zero-length read at the exact end of a file is legal and must not
        // be confused with a truncated frame.
        let f = sample(Vec::new());
        assert_eq!(decode(&encode(&f).unwrap()).unwrap(), f);
    }

    #[test]
    fn the_wire_is_only_the_header_larger_than_the_payload() {
        // The entire point: the chunk travels at its own size.
        let payload = 256 * 1024;
        let wire = encode(&sample(vec![0u8; payload])).unwrap();
        let header = wire.len() - payload;
        assert!(header < 80, "header is {header} bytes");
        // What the JSON frame would have cost for the same range.
        assert!(overhead_saved(payload) > 85_000);
    }

    #[test]
    fn eof_survives_the_round_trip_either_way() {
        for eof in [true, false] {
            let mut f = sample(vec![1, 2, 3]);
            f.eof = eof;
            assert_eq!(decode(&encode(&f).unwrap()).unwrap().eof, eof);
        }
    }

    #[test]
    fn json_is_never_mistaken_for_a_media_frame() {
        // The discriminator has to hold for every frame the protocol sends, so
        // check the shape they all share rather than one example.
        assert!(!is_media_frame(br#"{"k":"frame","seq":0,"frame":{}}"#));
        assert!(!is_media_frame(b"{"));
        assert!(!is_media_frame(b""));
        assert!(is_media_frame(MAGIC));
    }

    #[test]
    fn a_header_that_lies_about_its_id_is_an_error_not_a_panic() {
        // These bytes come from a remote peer. Truncation and hostile lengths
        // both arrive eventually; neither may take down the read loop.
        let mut wire = encode(&sample(vec![7; 10])).unwrap();
        wire[12] = 255; // claim a 255-byte id the buffer cannot hold
        assert!(decode(&wire).is_err());
    }

    #[test]
    fn every_truncation_of_a_valid_frame_is_an_error_not_a_panic() {
        let wire = encode(&sample(vec![7; 64])).unwrap();
        // The full frame parses; every prefix of it must fail cleanly.
        assert!(decode(&wire).is_ok());
        for cut in 0..wire.len() {
            let _ = decode(&wire[..cut]);
        }
    }

    #[test]
    fn a_non_utf8_id_is_refused() {
        let mut wire = encode(&sample(vec![1])).unwrap();
        wire[13] = 0xFF;
        assert!(decode(&wire).is_err());
    }

    #[test]
    fn an_over_long_id_is_refused_rather_than_truncated() {
        // A truncated id names a different asset, so this would answer a range
        // with some other file's bytes.
        let mut f = sample(vec![1]);
        f.id = "x".repeat(256);
        assert!(encode(&f).is_err());
    }

    #[test]
    fn a_large_offset_survives() {
        // Adopted files may be gigabytes; the offset must not be truncated to
        // something that reads the wrong part of a film.
        let mut f = sample(vec![9]);
        f.offset = 3_000_000_000;
        assert_eq!(decode(&encode(&f).unwrap()).unwrap().offset, 3_000_000_000);
    }
}

#[cfg(test)]
mod golden {
    use super::*;

    /// A frame whose exact bytes are pinned on both sides of the wire.
    ///
    /// The portal has its own decoder (`src/lib/chat/media.ts`), written from
    /// the same layout but in another language and another repository, released
    /// on another schedule. A drift between them corrupts video and nothing
    /// else — no error, no log line, just a file that will not play. So the
    /// layout is pinned to a literal here and to the identical literal there;
    /// either side changing it alone fails its own suite before it can ship.
    ///
    /// If this test fails, do not update the literal until the portal's
    /// `media frame > matches the daemon's golden vector` has been updated too.
    pub const GOLDEN_HEX: &str = concat!(
        "4e464d31",         // magic "NFM1"
        "000000000000002a", // seq = 42
        "04",               // id_len = 4
        "61622d31",         // id = "ab-1"
        "0000000100000000", // offset = 4294967296 (past 32 bits on purpose)
        "01",               // flags = eof
        "00ff107b"          // payload: a zero, a high byte, and a '{'
    );

    #[test]
    fn the_wire_layout_matches_the_pinned_vector() {
        let frame = MediaFrame {
            seq: 42,
            id: "ab-1".into(),
            offset: 4_294_967_296,
            eof: true,
            data: vec![0x00, 0xff, 0x10, 0x7b],
        };
        let hex: String = encode(&frame)
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, GOLDEN_HEX);
    }

    #[test]
    fn the_pinned_vector_decodes_back() {
        let bytes: Vec<u8> = (0..GOLDEN_HEX.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&GOLDEN_HEX[i..i + 2], 16).unwrap())
            .collect();
        let f = decode(&bytes).unwrap();
        assert_eq!(f.seq, 42);
        assert_eq!(f.id, "ab-1");
        assert_eq!(f.offset, 4_294_967_296);
        assert!(f.eof);
        // The payload deliberately contains a `{`, so a decoder that ever
        // sniffed content instead of the magic would be caught here.
        assert_eq!(f.data, vec![0x00, 0xff, 0x10, 0x7b]);
    }
}
