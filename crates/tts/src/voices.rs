//! The voice bank.
//!
//! Despite the `.bin` name the file is a zip of `.npy` arrays, one per
//! voice, each shaped `(510, 1, 256)` and little-endian f32. The first axis
//! is indexed by how many tokens are being spoken — a longer utterance gets
//! a different style vector — which is the detail most easily got wrong.

use crate::error::TtsError;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

const STYLE_DIM: usize = 256;

pub struct VoiceBank {
    /// voice id -> rows, each row STYLE_DIM long.
    voices: HashMap<String, Vec<Vec<f32>>>,
}

impl VoiceBank {
    /// Load a bank from either layout.
    ///
    /// v1.0 ships one zip of `.npy`; v1.1-zh ships a directory of raw `.bin`,
    /// one file per voice, no header. Dispatching on what is actually there
    /// beats making every caller know which release it is holding.
    pub fn load(path: &Path) -> Result<VoiceBank, TtsError> {
        if !path.exists() {
            return Err(TtsError::ModelNotFound(path.display().to_string()));
        }
        if path.is_dir() {
            return Self::load_dir(path);
        }
        Self::load_zip(path)
    }

    /// A directory of raw `<voice>.bin`, each `(510, 1, 256)` little-endian f32.
    ///
    /// No header to check, so the length is the only guard: a file that is not
    /// an exact multiple of the row size is a truncated download, not something
    /// to read as far as it goes. Reading a truncated bank produces a voice that
    /// works for short sentences and fails for long ones -- the worst shape a
    /// bug can have here.
    pub fn load_dir(dir: &Path) -> Result<VoiceBank, TtsError> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| TtsError::ModelNotFound(format!("{}: {e}", dir.display())))?;
        let mut voices = HashMap::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(id) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".bin"))
            else {
                continue;
            };
            let bytes = std::fs::read(&path)
                .map_err(|e| TtsError::ModelCorrupt(format!("{}: {e}", path.display())))?;
            let row_bytes = STYLE_DIM * 4;
            if bytes.is_empty() || bytes.len() % row_bytes != 0 {
                return Err(TtsError::ModelCorrupt(format!(
                    "{}: {} bytes is not a whole number of {STYLE_DIM}-float rows",
                    path.display(),
                    bytes.len()
                )));
            }
            let rows = bytes
                .chunks_exact(row_bytes)
                .map(|row| {
                    row.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                })
                .collect();
            voices.insert(id.to_string(), rows);
        }
        if voices.is_empty() {
            return Err(TtsError::ModelCorrupt(format!(
                "{}: no .bin voices in this directory",
                dir.display()
            )));
        }
        Ok(VoiceBank { voices })
    }

    fn load_zip(path: &Path) -> Result<VoiceBank, TtsError> {
        let file = std::fs::File::open(path)
            .map_err(|e| TtsError::ModelNotFound(format!("{}: {e}", path.display())))?;
        let mut zip = zip::ZipArchive::new(file)
            .map_err(|e| TtsError::ModelCorrupt(format!("voices is not a zip: {e}")))?;

        let mut voices = HashMap::new();
        for i in 0..zip.len() {
            let mut entry = zip
                .by_index(i)
                .map_err(|e| TtsError::ModelCorrupt(format!("zip entry {i}: {e}")))?;
            let name = entry.name().to_string();
            let Some(id) = name.strip_suffix(".npy") else {
                continue;
            };
            let id = id.to_string();
            let mut bytes = Vec::new();
            entry
                .read_to_end(&mut bytes)
                .map_err(|e| TtsError::ModelCorrupt(format!("read {name}: {e}")))?;
            voices.insert(id, parse_npy(&bytes, &name)?);
        }
        if voices.is_empty() {
            return Err(TtsError::ModelCorrupt(
                "voice bank has no .npy entries".into(),
            ));
        }
        Ok(VoiceBank { voices })
    }

    pub fn ids(&self) -> Vec<&str> {
        self.voices.keys().map(|s| s.as_str()).collect()
    }

    /// The style vector for an utterance of `token_count` tokens.
    pub fn style(&self, voice_id: &str, token_count: usize) -> Result<Vec<f32>, TtsError> {
        let rows = self
            .voices
            .get(voice_id)
            .ok_or_else(|| TtsError::UnsupportedVoice(voice_id.to_string()))?;
        rows.get(token_count).cloned().ok_or_else(|| {
            TtsError::TextTooLong(format!(
                "{token_count} tokens exceeds the voice bank's {} rows",
                rows.len()
            ))
        })
    }
}

/// Parse a `(N, 1, 256)` little-endian f32 `.npy` into N rows.
///
/// Only the one dtype and rank Kokoro ships are accepted; anything else is
/// a corrupt bank rather than something to coerce.
fn parse_npy(bytes: &[u8], name: &str) -> Result<Vec<Vec<f32>>, TtsError> {
    let corrupt = |m: String| TtsError::ModelCorrupt(format!("{name}: {m}"));
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(corrupt("not a .npy file".into()));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_start = 10;
    let data_start = header_start + header_len;
    if bytes.len() < data_start {
        return Err(corrupt("truncated header".into()));
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .map_err(|e| corrupt(format!("header is not utf-8: {e}")))?;
    if !header.contains("'<f4'") {
        return Err(corrupt(format!(
            "expected little-endian f32, header: {header}"
        )));
    }
    if header.contains("'fortran_order': True") {
        return Err(corrupt("fortran-ordered arrays are not supported".into()));
    }

    let payload = &bytes[data_start..];
    if !payload.len().is_multiple_of(4) {
        return Err(corrupt("payload is not a whole number of f32".into()));
    }
    let total = payload.len() / 4;
    if !total.is_multiple_of(STYLE_DIM) {
        return Err(corrupt(format!(
            "{total} floats is not a multiple of {STYLE_DIM}"
        )));
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(values.chunks_exact(STYLE_DIM).map(|c| c.to_vec()).collect())
}

#[cfg(test)]
mod tests {
    /// v1.1-zh 的音色是一目录裸 f32,没有头,所以长度是唯一的守卫。
    #[test]
    fn a_directory_of_raw_voices_loads() {
        let dir = tempfile::tempdir().unwrap();
        let rows = 3usize;
        let mut bytes = Vec::new();
        for r in 0..rows {
            for c in 0..super::STYLE_DIM {
                bytes.extend_from_slice(&((r * 1000 + c) as f32).to_le_bytes());
            }
        }
        std::fs::write(dir.path().join("zf_001.bin"), &bytes).unwrap();
        std::fs::write(dir.path().join("readme.txt"), b"ignored").unwrap();

        let bank = VoiceBank::load(dir.path()).unwrap();
        assert_eq!(bank.ids(), vec!["zf_001"]);
        let style = bank.style("zf_001", 2).unwrap();
        assert_eq!(style.len(), super::STYLE_DIM);
        assert_eq!(style[0], 2000.0, "取的应该是第 2 行");
    }

    /// 截断的下载要当场报错。读一半的音色在短句上能用、长句上崩 ——
    /// 这是排查起来最费劲的一种形状。
    #[test]
    fn a_truncated_voice_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("zf_001.bin"), vec![0u8; 999]).unwrap();
        // `unwrap_err` 要求 VoiceBank: Debug,而它没必要为了测试实现 Debug。
        let Err(err) = VoiceBank::load(dir.path()) else {
            panic!("截断的音色文件应当被拒绝");
        };
        assert!(format!("{err}").contains("rows"), "{err}");
    }

    #[test]
    fn an_empty_directory_is_an_error_not_an_empty_bank() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(VoiceBank::load(dir.path()), Err(_)));
    }

    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-voices.bin")
    }

    #[test]
    fn lists_voice_ids() {
        let bank = VoiceBank::load(&fixture()).unwrap();
        let mut ids = bank.ids();
        ids.sort();
        assert_eq!(ids, vec!["af_test", "zf_test"]);
    }

    #[test]
    fn style_is_indexed_by_token_count() {
        let bank = VoiceBank::load(&fixture()).unwrap();
        // The fixture stores its own row index at element 0 of each row.
        let style = bank.style("af_test", 2).unwrap();
        assert_eq!(style.len(), 256);
        assert_eq!(style[0], 2.0, "must select row 2, not row 0");
    }

    #[test]
    fn token_count_past_the_bank_is_rejected() {
        let bank = VoiceBank::load(&fixture()).unwrap();
        let err = bank.style("af_test", 99).unwrap_err();
        assert!(matches!(err, TtsError::TextTooLong(_)), "got: {err}");
    }

    #[test]
    fn unknown_voice_is_rejected() {
        let bank = VoiceBank::load(&fixture()).unwrap();
        let err = bank.style("nope", 1).unwrap_err();
        assert!(matches!(err, TtsError::UnsupportedVoice(_)), "got: {err}");
    }

    /// The real bank should hold 54 voices of 510 rows each.
    #[test]
    #[ignore]
    fn real_voice_bank_shape() {
        let path = crate::model::default_model_dir()
            .unwrap()
            .join("kokoro-voices-v1.0.bin");
        let bank = VoiceBank::load(&path).expect("voice bank should load");
        assert_eq!(bank.ids().len(), 54);
        assert_eq!(bank.style("af_heart", 509).unwrap().len(), 256);
        assert!(
            bank.style("af_heart", 510).is_err(),
            "510 rows means 0..=509"
        );
    }
}
