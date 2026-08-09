//! Minimal RIFF/WAVE writer.
//!
//! Hand-rolled rather than pulled from a crate: this is one 44-byte header
//! and a sample loop, and the artifact path only ever needs mono 16-bit.

/// Encode mono f32 samples as a 16-bit PCM WAV file.
pub fn encode(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    const CHANNELS: u16 = 1;
    const BITS: u16 = 16;
    let byte_rate = sample_rate * CHANNELS as u32 * (BITS / 8) as u32;
    let block_align = CHANNELS * (BITS / 8);
    let data_len = (pcm.len() * 2) as u32;

    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format tag
    out.extend_from_slice(&CHANNELS.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in pcm {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * i16::MAX as f32).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_a_riff_wave() {
        let out = encode(&[0.0, 0.0], 24000);
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WAVE");
        assert_eq!(&out[12..16], b"fmt ");
        assert_eq!(&out[36..40], b"data");
    }

    #[test]
    fn declares_mono_16bit_at_the_given_rate() {
        let out = encode(&[0.0], 24000);
        assert_eq!(u16::from_le_bytes([out[22], out[23]]), 1, "channels");
        assert_eq!(
            u32::from_le_bytes([out[24], out[25], out[26], out[27]]),
            24000,
            "sample rate"
        );
        assert_eq!(
            u16::from_le_bytes([out[34], out[35]]),
            16,
            "bits per sample"
        );
    }

    #[test]
    fn sizes_agree_with_the_payload() {
        let out = encode(&[0.0; 10], 24000);
        let data_len = u32::from_le_bytes([out[40], out[41], out[42], out[43]]);
        assert_eq!(data_len, 20, "10 samples at 2 bytes each");
        let riff_len = u32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        assert_eq!(riff_len as usize, out.len() - 8);
    }

    #[test]
    fn clamps_instead_of_wrapping() {
        // Kokoro occasionally overshoots; wrapping would turn a loud sample
        // into a full-scale click of the opposite sign.
        let out = encode(&[2.0, -2.0], 24000);
        // Scaling by i16::MAX keeps the range symmetric, so full negative
        // scale is -32767 rather than i16::MIN.
        assert_eq!(i16::from_le_bytes([out[44], out[45]]), i16::MAX);
        assert_eq!(i16::from_le_bytes([out[46], out[47]]), -i16::MAX);
    }
}
