//! The SentencePiece BPE tokenizer MOSS was trained with.
//!
//! `tokenizer.model` is a SentencePiece `ModelProto`: 16384 pieces, each with a
//! score, plus a trainer spec and a normalizer spec. Read directly rather than
//! through a protobuf crate — the parts that matter are two nested messages of
//! four fields, and a code generator plus a schema is a large amount of
//! machinery for that.
//!
//! ## What the model actually is
//!
//! `model_type: 2` — BPE, not Unigram, which is the more common choice for a
//! `.model` file and would have produced plausible-looking wrong tokens. BPE
//! here means: split into characters, then repeatedly merge whichever adjacent
//! pair forms the *highest-scoring* piece in the vocabulary. There is no
//! separate merges list; the vocabulary and its scores are the merge rules.
//!
//! `byte_fallback` is on, so anything with no piece is emitted as its UTF-8
//! bytes through the `<0xHH>` pieces rather than as `<unk>`. A rare character
//! degrades to something mispronounced instead of something lost.
//!
//! ## What this does not implement
//!
//! The normalizer spec carries a 237 KB precompiled character map — an NFKC
//! variant compiled into a trie. This applies whitespace collapsing and the
//! `▁` escape but not that map. Verified against both worked examples upstream
//! ships in the manifest (Chinese and English, 20 and 69 tokens): they
//! reproduce exactly. Text that NFKC would fold — full-width ASCII, ligatures,
//! some compatibility forms — will tokenize differently from upstream here.
//! For a speech model that means a slightly different pronunciation of unusual
//! input, not a failure, which is why the trie can wait until something needs
//! it.

use std::collections::HashMap;

use crate::error::TtsError;

/// SentencePiece's own piece types.
const TYPE_UNKNOWN: i64 = 2;
const TYPE_CONTROL: i64 = 3;

/// The escape SentencePiece uses for a space (U+2581).
const SPACE: char = '▁';

pub struct SpBpe {
    /// Piece text to (id, score).
    index: HashMap<String, (i32, f32)>,
    /// `<0xHH>` ids, for byte fallback.
    bytes: Vec<i32>,
    unk_id: i32,
}

/// Written by hand: the derived form would print all 16384 pieces, which turns
/// any `unwrap` in a test into a screenful of vocabulary.
impl std::fmt::Debug for SpBpe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpBpe")
            .field("pieces", &self.index.len())
            .field("unk_id", &self.unk_id)
            .finish()
    }
}

fn corrupt(msg: impl std::fmt::Display) -> TtsError {
    TtsError::ModelCorrupt(format!("tokenizer.model: {msg}"))
}

/// Minimal protobuf reader: varints and length-delimited fields, which is all
/// this file uses beyond fixed32 scores.
struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

enum Field<'a> {
    Varint(i64),
    Bytes(&'a [u8]),
    F32(f32),
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Reader<'a> {
        Reader { b, i: 0 }
    }

    fn done(&self) -> bool {
        self.i >= self.b.len()
    }

    fn varint(&mut self) -> Result<u64, TtsError> {
        let mut r = 0u64;
        let mut s = 0u32;
        loop {
            let byte = *self
                .b
                .get(self.i)
                .ok_or_else(|| corrupt("truncated varint"))?;
            self.i += 1;
            r |= ((byte & 0x7f) as u64) << s;
            if byte & 0x80 == 0 {
                return Ok(r);
            }
            s += 7;
            if s > 63 {
                return Err(corrupt("varint too long"));
            }
        }
    }

    /// Next (field number, value). Groups (wire types 3 and 4) are not
    /// produced by any SentencePiece writer and are rejected rather than
    /// skipped, so a malformed file fails loudly.
    fn next(&mut self) -> Result<(u32, Field<'a>), TtsError> {
        let key = self.varint()?;
        let number = (key >> 3) as u32;
        match key & 7 {
            0 => Ok((number, Field::Varint(self.varint()? as i64))),
            1 => {
                let end = self.i + 8;
                if end > self.b.len() {
                    return Err(corrupt("truncated fixed64"));
                }
                self.i = end;
                Ok((number, Field::Varint(0)))
            }
            2 => {
                let len = self.varint()? as usize;
                let end = self
                    .i
                    .checked_add(len)
                    .filter(|e| *e <= self.b.len())
                    .ok_or_else(|| corrupt("length past end of file"))?;
                let out = &self.b[self.i..end];
                self.i = end;
                Ok((number, Field::Bytes(out)))
            }
            5 => {
                let end = self.i + 4;
                if end > self.b.len() {
                    return Err(corrupt("truncated fixed32"));
                }
                let v = f32::from_le_bytes([
                    self.b[self.i],
                    self.b[self.i + 1],
                    self.b[self.i + 2],
                    self.b[self.i + 3],
                ]);
                self.i = end;
                Ok((number, Field::F32(v)))
            }
            other => Err(corrupt(format!("unsupported wire type {other}"))),
        }
    }
}

impl SpBpe {
    /// Parse the model, keeping only pieces that text can legitimately produce.
    ///
    /// `reserved` are ids the caller uses structurally — the turn markers and
    /// audio slots that frame a request. They are excluded from matching, so a
    /// reply containing the literal text `<|im_end|>` is spoken rather than
    /// obeyed. Ordinary text never contains them, so nothing else changes.
    pub fn parse(bytes: &[u8], reserved: &[i32]) -> Result<SpBpe, TtsError> {
        let mut index: HashMap<String, (i32, f32)> = HashMap::new();
        let mut byte_ids = vec![-1i32; 256];
        let mut unk_id = 0i32;
        let mut id = 0i32;

        let mut r = Reader::new(bytes);
        while !r.done() {
            match r.next()? {
                // repeated SentencePiece pieces = 1
                (1, Field::Bytes(piece)) => {
                    let mut text: Option<String> = None;
                    let mut score = 0f32;
                    let mut kind = 1i64;
                    let mut pr = Reader::new(piece);
                    while !pr.done() {
                        match pr.next()? {
                            (1, Field::Bytes(s)) => {
                                text = Some(String::from_utf8_lossy(s).into_owned())
                            }
                            (2, Field::F32(v)) => score = v,
                            (3, Field::Varint(v)) => kind = v,
                            _ => {}
                        }
                    }
                    let text = text.ok_or_else(|| corrupt(format!("piece {id} has no text")))?;
                    if let Some(b) = byte_piece(&text) {
                        byte_ids[b as usize] = id;
                    }
                    // Control and unknown pieces are never produced by encoding
                    // — that is what upstream does, and letting them match here
                    // would let a `<pad>` in the text become a real `<pad>`.
                    let matchable =
                        kind != TYPE_CONTROL && kind != TYPE_UNKNOWN && !reserved.contains(&id);
                    if matchable {
                        index.insert(text, (id, score));
                    }
                    id += 1;
                }
                // trainer_spec = 2, for unk_id
                (2, Field::Bytes(spec)) => {
                    let mut sr = Reader::new(spec);
                    while !sr.done() {
                        if let (40, Field::Varint(v)) = sr.next()? {
                            unk_id = v as i32;
                        }
                    }
                }
                _ => {}
            }
        }

        if index.is_empty() {
            return Err(corrupt("no usable pieces"));
        }
        Ok(SpBpe {
            index,
            bytes: byte_ids,
            unk_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.index.len()
    }

    /// Text to token ids.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let prepared = prepare(text);
        if prepared.is_empty() {
            return Vec::new();
        }

        // Start from characters, then merge the best-scoring adjacent pair
        // until none is left. Quadratic in the symbol count, which is fine
        // because callers hand this one sentence at a time — the whole point of
        // the sentence splitter upstream of it.
        let mut symbols: Vec<String> = prepared.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, f32, String)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let mut merged = String::with_capacity(symbols[i].len() + symbols[i + 1].len());
                merged.push_str(&symbols[i]);
                merged.push_str(&symbols[i + 1]);
                if let Some((_, score)) = self.index.get(&merged) {
                    if best.as_ref().is_none_or(|(_, b, _)| *score > *b) {
                        best = Some((i, *score, merged));
                    }
                }
            }
            let Some((i, _, merged)) = best else { break };
            symbols[i] = merged;
            symbols.remove(i + 1);
        }

        let mut out = Vec::with_capacity(symbols.len());
        for s in symbols {
            match self.index.get(&s) {
                Some((id, _)) => out.push(*id),
                // Byte fallback: a character with no piece becomes its UTF-8
                // bytes, so an unusual name is mispronounced rather than
                // dropped.
                None => {
                    for b in s.bytes() {
                        let id = self.bytes[b as usize];
                        out.push(if id >= 0 { id } else { self.unk_id });
                    }
                }
            }
        }
        out
    }
}

/// `<0xHH>` → the byte it stands for.
fn byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

/// Collapse whitespace, add the leading space SentencePiece expects, and escape
/// spaces as `▁`.
///
/// The leading space is not cosmetic: pieces are learned with it, so "hello"
/// and " hello" tokenize differently and only the second matches what the model
/// saw in training.
fn prepare(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(collapsed.len() + 4);
    out.push(SPACE);
    for c in collapsed.chars() {
        out.push(if c == ' ' { SPACE } else { c });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A miniature model: `a`, `b`, `ab`, `▁`, a byte piece, and a control.
    fn tiny() -> Vec<u8> {
        fn piece(text: &str, score: f32, kind: i64) -> Vec<u8> {
            let mut inner = Vec::new();
            inner.push(0x0a); // field 1, bytes
            inner.push(text.len() as u8);
            inner.extend_from_slice(text.as_bytes());
            inner.push(0x15); // field 2, fixed32
            inner.extend_from_slice(&score.to_le_bytes());
            inner.push(0x18); // field 3, varint
            inner.push(kind as u8);

            let mut out = Vec::new();
            out.push(0x0a); // top-level field 1, bytes
            out.push(inner.len() as u8);
            out.extend_from_slice(&inner);
            out
        }
        let mut m = Vec::new();
        m.extend(piece("<unk>", 0.0, 2)); // id 0
        m.extend(piece("<pad>", 0.0, 3)); // id 1, control
        m.extend(piece("▁", -1.0, 1)); // id 2
        m.extend(piece("a", -3.0, 1)); // id 3
        m.extend(piece("b", -4.0, 1)); // id 4
        m.extend(piece("ab", -2.0, 1)); // id 5
        m.extend(piece("<0xE4>", 0.0, 6)); // id 6
        m.extend(piece("<0xB8>", 0.0, 6)); // id 7
        m.extend(piece("<0x80>", 0.0, 6)); // id 8
        m
    }

    fn model() -> SpBpe {
        SpBpe::parse(&tiny(), &[]).expect("the fixture parses")
    }

    #[test]
    fn pieces_and_their_scores_are_read() {
        let m = model();
        // Nine pieces in, minus `<unk>` (type 2) and `<pad>` (type 3), which
        // encoding must never produce: four text pieces and three byte pieces.
        assert_eq!(m.vocab_size(), 7);
    }

    #[test]
    fn a_control_piece_cannot_be_produced_from_text() {
        // Otherwise the literal text "<pad>" becomes a real pad token, and the
        // prompt structure is whatever the input said it was.
        let ids = model().encode("<pad>");
        assert!(!ids.contains(&1), "the control token was emitted: {ids:?}");
    }

    #[test]
    fn reserved_ids_are_excluded_too() {
        // The turn markers are user-defined pieces, so upstream would match
        // them from text. Here they are the frame around the request, and a
        // reply that mentions one should be spoken rather than obeyed.
        let m = SpBpe::parse(&tiny(), &[5]).unwrap();
        let ids = m.encode("ab");
        assert!(!ids.contains(&5), "a reserved id came back: {ids:?}");
        assert_eq!(ids, vec![2, 3, 4], "should fall back to the parts");
    }

    #[test]
    fn the_highest_scoring_merge_wins() {
        // "ab" (-2) beats keeping "a" (-3) and "b" (-4) apart.
        assert_eq!(model().encode("ab"), vec![2, 5]);
    }

    #[test]
    fn a_leading_space_is_added_and_spaces_become_the_escape() {
        // Pieces are learned with the escape attached; without this, every
        // first word is tokenized as something the model never saw.
        assert_eq!(prepare("a b"), "▁a▁b");
        assert_eq!(prepare("a"), "▁a");
    }

    #[test]
    fn runs_of_whitespace_collapse() {
        assert_eq!(prepare("a  \t\n b"), "▁a▁b");
    }

    #[test]
    fn empty_and_blank_text_produce_nothing() {
        // Not one token for the dummy prefix: a request with no words should
        // synthesize nothing rather than a syllable.
        assert!(model().encode("").is_empty());
        assert!(model().encode("   \n ").is_empty());
    }

    #[test]
    fn an_unknown_character_falls_back_to_its_bytes() {
        // 一 is E4 B8 80 and has no piece here.
        assert_eq!(model().encode("一"), vec![2, 6, 7, 8]);
    }

    #[test]
    fn a_byte_with_no_piece_becomes_unk_rather_than_a_panic() {
        let m = model();
        let ids = m.encode("é"); // C3 A9, neither piece present
        assert_eq!(ids, vec![2, 0, 0]);
    }

    #[test]
    fn a_truncated_model_is_rejected() {
        let mut bytes = tiny();
        bytes.truncate(bytes.len() - 3);
        let err = SpBpe::parse(&bytes, &[]).unwrap_err();
        assert!(err.to_string().contains("tokenizer.model"), "{err}");
    }

    #[test]
    fn a_model_with_nothing_usable_is_rejected() {
        // All-control is not a tokenizer, and failing here beats encoding
        // every sentence to an empty sequence and synthesizing silence.
        let mut m = Vec::new();
        m.extend_from_slice(&tiny()[..0]);
        let err = SpBpe::parse(&m, &[]).unwrap_err();
        assert!(err.to_string().contains("no usable pieces"), "{err}");
    }

    /// The real model, checked against the two worked examples upstream ships.
    ///
    /// This is the only test that can catch a wrong merge rule: everything
    /// above would pass just as happily with, say, Unigram scoring.
    #[test]
    #[ignore = "needs the 460 KB tokenizer.model"]
    fn moss_real_matches_the_upstream_examples() {
        let dir = crate::model::default_model_dir().unwrap();
        let bytes = std::fs::read(dir.join("tokenizer.model")).expect("tokenizer.model");
        let manifest_bytes =
            std::fs::read(dir.join("browser_poc_manifest.json")).expect("the manifest");
        let manifest = crate::moss::Manifest::parse(&manifest_bytes).unwrap();
        let tok = SpBpe::parse(&bytes, &manifest.reserved_token_ids()).unwrap();

        assert!(!manifest.text_samples.is_empty(), "no examples to check");
        for sample in &manifest.text_samples {
            let got = tok.encode(&sample.text);
            assert_eq!(
                got, sample.text_token_ids,
                "\n  text: {}\n  want: {:?}\n  got:  {:?}",
                sample.text, sample.text_token_ids, got
            );
        }
    }
}
