//! The manifest that turns five ONNX graphs into a usable model.
//!
//! `browser_poc_manifest.json` ships beside the weights and carries three
//! things the graphs cannot: the token ids that frame a request, the built-in
//! voices as pre-computed prompt codes, and the generation limits.
//!
//! The voices matter most. MOSS clones a voice from a reference recording, and
//! producing those codes needs the encoder — another 43 MB, plus an encode pass
//! before every first word. The manifest already contains eighteen of them,
//! computed upstream. So the encoder is not downloaded, nothing is precomputed
//! at install time, and no derived artefact of the weights is ever produced or
//! shipped by us (ADR-0005 stays intact by construction).

use serde::Deserialize;

use crate::error::TtsError;

/// Token ids and widths. Names mirror the manifest's own keys.
#[derive(Debug, Clone, Deserialize)]
pub struct TtsConfig {
    pub n_vq: usize,
    pub audio_pad_token_id: i32,
    pub audio_start_token_id: i32,
    pub audio_end_token_id: i32,
    pub audio_user_slot_token_id: i32,
    pub audio_assistant_slot_token_id: i32,
    pub vocab_size: i32,
}

impl TtsConfig {
    /// One text channel plus the audio codebooks.
    pub fn row_width(&self) -> usize {
        self.n_vq + 1
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptTemplates {
    pub user_prompt_prefix_token_ids: Vec<i32>,
    pub user_prompt_after_reference_token_ids: Vec<i32>,
    pub assistant_prompt_prefix_token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationDefaults {
    /// Frames are 80 ms, so 375 of them is thirty seconds — the ceiling on one
    /// synthesis, and the thing that stops a degenerate loop from generating
    /// until the disk fills.
    pub max_new_frames: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinVoice {
    pub voice: String,
    pub display_name: String,
    pub group: String,
    /// `[frames][n_vq]` of reference audio, already tokenised upstream.
    pub prompt_audio_codes: Vec<Vec<i32>>,
}

/// A worked example: text and the token ids it produces.
///
/// Present so the browser demo can run before its tokenizer loads. Here it
/// serves the same purpose one layer down — it lets the graphs be exercised
/// before a SentencePiece implementation exists, which keeps "the model does
/// not speak" and "the tokenizer is wrong" as two separate questions.
#[derive(Debug, Clone, Deserialize)]
pub struct TextSample {
    pub id: String,
    pub text: String,
    pub text_token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub tts_config: TtsConfig,
    pub prompt_templates: PromptTemplates,
    pub generation_defaults: GenerationDefaults,
    pub builtin_voices: Vec<BuiltinVoice>,
    #[serde(default)]
    pub text_samples: Vec<TextSample>,
}

/// The prefill input: one row per position, plus the mask.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestRows {
    /// `[seq][row_width]`, row 0 being the text channel.
    pub rows: Vec<Vec<i32>>,
}

impl RequestRows {
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Flattened for ONNX, which wants `[batch, seq, row_width]`.
    pub fn flat(&self) -> Vec<i32> {
        self.rows.iter().flatten().copied().collect()
    }

    /// All ones: every position of a freshly built request is real.
    pub fn attention_mask(&self) -> Vec<i32> {
        vec![1; self.rows.len()]
    }
}

impl Manifest {
    pub fn parse(bytes: &[u8]) -> Result<Manifest, TtsError> {
        serde_json::from_slice(bytes)
            .map_err(|e| TtsError::ModelCorrupt(format!("browser_poc_manifest.json: {e}")))
    }

    pub fn voice(&self, id: &str) -> Option<&BuiltinVoice> {
        self.builtin_voices.iter().find(|v| v.voice == id)
    }

    /// A text row: the token in channel 0, audio padding everywhere else.
    fn text_row(&self, token: i32) -> Vec<i32> {
        let mut row = vec![self.tts_config.audio_pad_token_id; self.tts_config.row_width()];
        row[0] = token;
        row
    }

    /// An audio row: a slot marker in channel 0, one codebook per channel after.
    fn audio_row(&self, codes: &[i32], slot: i32) -> Vec<i32> {
        let mut row = vec![self.tts_config.audio_pad_token_id; self.tts_config.row_width()];
        row[0] = slot;
        for (i, c) in codes.iter().take(self.tts_config.n_vq).enumerate() {
            row[i + 1] = *c;
        }
        row
    }

    /// Assemble the prompt: template, reference voice, text, hand over to the
    /// assistant.
    ///
    /// The order is upstream's and is not adjustable — the model was trained on
    /// exactly this arrangement, and a plausible-looking rearrangement produces
    /// confident nonsense rather than an error.
    pub fn build_request_rows(&self, voice: &BuiltinVoice, text_token_ids: &[i32]) -> RequestRows {
        let c = &self.tts_config;
        let t = &self.prompt_templates;

        let mut prefix = t.user_prompt_prefix_token_ids.clone();
        prefix.push(c.audio_start_token_id);

        let mut suffix = vec![c.audio_end_token_id];
        suffix.extend_from_slice(&t.user_prompt_after_reference_token_ids);
        suffix.extend_from_slice(text_token_ids);
        suffix.extend_from_slice(&t.assistant_prompt_prefix_token_ids);
        suffix.push(c.audio_start_token_id);

        let mut rows =
            Vec::with_capacity(prefix.len() + voice.prompt_audio_codes.len() + suffix.len());
        rows.extend(prefix.iter().map(|t| self.text_row(*t)));
        rows.extend(
            voice
                .prompt_audio_codes
                .iter()
                .map(|codes| self.audio_row(codes, c.audio_user_slot_token_id)),
        );
        rows.extend(suffix.iter().map(|t| self.text_row(*t)));
        RequestRows { rows }
    }

    /// Ids that frame a request rather than say anything.
    ///
    /// Handed to the tokenizer so text can never produce one. A reply that
    /// happens to contain `<|im_end|>` should be read aloud, not treated as the
    /// end of the turn.
    pub fn reserved_token_ids(&self) -> Vec<i32> {
        let c = &self.tts_config;
        vec![
            c.audio_pad_token_id,
            c.audio_start_token_id,
            c.audio_end_token_id,
            c.audio_user_slot_token_id,
            c.audio_assistant_slot_token_id,
        ]
    }

    /// The row fed back after a generated frame.
    pub fn generated_row(&self, frame: &[i32]) -> Vec<i32> {
        self.audio_row(frame, self.tts_config.audio_assistant_slot_token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped like the real file, with numbers small enough to read.
    fn fixture() -> Manifest {
        Manifest::parse(
            br#"{
              "tts_config": {
                "n_vq": 3,
                "audio_pad_token_id": 99,
                "audio_start_token_id": 6,
                "audio_end_token_id": 7,
                "audio_user_slot_token_id": 8,
                "audio_assistant_slot_token_id": 9,
                "vocab_size": 16384
              },
              "prompt_templates": {
                "user_prompt_prefix_token_ids": [10, 11],
                "user_prompt_after_reference_token_ids": [12],
                "assistant_prompt_prefix_token_ids": [13]
              },
              "generation_defaults": { "max_new_frames": 375 },
              "builtin_voices": [
                {
                  "voice": "Junhao",
                  "display_name": "CN male",
                  "group": "Chinese Male",
                  "prompt_audio_codes": [[100, 200, 300], [101, 201, 301]]
                }
              ],
              "text_samples": [
                { "id": "zh", "text": "hi", "text_token_ids": [55, 66] }
              ]
            }"#,
        )
        .expect("fixture parses")
    }

    #[test]
    fn a_row_is_one_text_channel_plus_the_codebooks() {
        assert_eq!(fixture().tts_config.row_width(), 4);
    }

    #[test]
    fn a_text_row_pads_every_audio_channel() {
        // Padding rather than zero: zero is a valid codebook entry, so a row of
        // zeros is not "no audio here", it is a specific sound.
        let m = fixture();
        assert_eq!(m.text_row(42), vec![42, 99, 99, 99]);
    }

    #[test]
    fn an_audio_row_carries_the_slot_marker_then_the_codes() {
        let m = fixture();
        assert_eq!(m.audio_row(&[7, 8, 9], 8), vec![8, 7, 8, 9]);
    }

    #[test]
    fn extra_codebooks_are_dropped_rather_than_overflowing_the_row() {
        let m = fixture();
        assert_eq!(m.audio_row(&[1, 2, 3, 4, 5], 8), vec![8, 1, 2, 3]);
    }

    #[test]
    fn the_request_is_template_then_voice_then_text_then_handover() {
        let m = fixture();
        let v = m.voice("Junhao").expect("the fixture voice");
        let r = m.build_request_rows(v, &[55, 66]);

        let text_channel: Vec<i32> = r.rows.iter().map(|row| row[0]).collect();
        assert_eq!(
            text_channel,
            vec![
                10, 11, 6, // prefix, then "audio starts"
                8, 8,  // two rows of reference voice
                7,  // "audio ends"
                12, // after-reference template
                55, 66, // the text to speak
                13, // "assistant speaks now"
                6,  // "audio starts"
            ]
        );
    }

    #[test]
    fn the_reference_voice_rows_carry_its_codes() {
        let m = fixture();
        let v = m.voice("Junhao").unwrap();
        let r = m.build_request_rows(v, &[55]);
        assert_eq!(r.rows[3], vec![8, 100, 200, 300]);
        assert_eq!(r.rows[4], vec![8, 101, 201, 301]);
    }

    #[test]
    fn the_mask_covers_every_row_and_the_flat_form_matches() {
        let m = fixture();
        let v = m.voice("Junhao").unwrap();
        let r = m.build_request_rows(v, &[55, 66]);
        assert_eq!(r.attention_mask().len(), r.len());
        assert!(r.attention_mask().iter().all(|&x| x == 1));
        assert_eq!(r.flat().len(), r.len() * m.tts_config.row_width());
    }

    #[test]
    fn a_generated_frame_goes_back_as_the_assistants_own_slot() {
        // Not the user slot: the model has to be able to tell what it said
        // from what it was shown.
        let m = fixture();
        assert_eq!(m.generated_row(&[4, 5, 6]), vec![9, 4, 5, 6]);
        assert_ne!(
            m.generated_row(&[4, 5, 6])[0],
            m.tts_config.audio_user_slot_token_id
        );
    }

    #[test]
    fn the_structural_ids_are_reported_for_exclusion() {
        let r = fixture().reserved_token_ids();
        for id in [99, 6, 7, 8, 9] {
            assert!(r.contains(&id), "{id} is missing from {r:?}");
        }
    }

    #[test]
    fn an_unknown_voice_is_none_rather_than_a_default() {
        // Silently substituting a voice would ship the wrong speaker with no
        // indication anything was wrong.
        assert!(fixture().voice("nobody").is_none());
    }

    #[test]
    fn a_malformed_manifest_says_so() {
        let err = Manifest::parse(b"{\"tts_config\":{}}").unwrap_err();
        assert!(
            err.to_string().contains("browser_poc_manifest.json"),
            "{err}"
        );
    }
}
