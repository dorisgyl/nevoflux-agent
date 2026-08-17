//! Low frame rate stacking, and the model's own CMVN.
//!
//! LFR turns `[T, 80]` fbank into `[ceil(T/n), 80*m]` by giving each output
//! frame a window of `m` input frames centred on `i*n`, clamping at both edges
//! rather than padding with zeros. With m=7, n=6 that is the 560-wide input
//! the encoder expects, at one sixth the frame rate.
//!
//! The clamping is the part worth stating: the first output frames reach back
//! before the start and the last reach past the end, and both repeat the
//! nearest real frame. Zero padding would feed the encoder silence that the
//! recording does not contain, at exactly the two places a transcript is most
//! often wrong.
//!
//! Ported from sherpa-onnx `csrc/lfr.cc`, which is what the exported model was
//! validated against.

/// Stack `input` (row-major `[T, input_dim]`) into `[ceil(T/shift), dim*size]`.
pub fn apply_lfr(input: &[f32], input_dim: usize, size: usize, shift: usize) -> Vec<f32> {
    debug_assert!(input_dim > 0 && size > 0 && shift > 0);
    if input.is_empty() {
        return Vec::new();
    }
    let input_frames = input.len() / input_dim;
    let output_frames = 1 + (input_frames - 1) / shift;
    let left_context = (size - 1) / 2;

    let mut out = Vec::with_capacity(output_frames * input_dim * size);
    for i in 0..output_frames {
        let center = i * shift;
        let left_padding = left_context.saturating_sub(center);
        let first = center.saturating_sub(left_context);
        let max_offset = input_frames - 1 - first;

        for j in 0..size {
            let frame = if j < left_padding {
                0
            } else {
                let offset = j - left_padding;
                if offset > max_offset {
                    input_frames - 1
                } else {
                    first + offset
                }
            };
            out.extend_from_slice(&input[frame * input_dim..(frame + 1) * input_dim]);
        }
    }
    out
}

/// `(x + neg_mean) * inv_stddev`, applied per row.
///
/// Both vectors come from the model's own ONNX metadata rather than a side
/// file, so there is exactly one statement of these numbers and it travels
/// with the weights they belong to.
pub fn apply_cmvn(features: &mut [f32], neg_mean: &[f32], inv_stddev: &[f32]) {
    let dim = neg_mean.len();
    debug_assert_eq!(dim, inv_stddev.len());
    for row in features.chunks_exact_mut(dim) {
        for ((v, m), s) in row.iter_mut().zip(neg_mean).zip(inv_stddev) {
            *v = (*v + *m) * *s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[T, 2]` where row t is `[t, -t]`, so a frame is identifiable on sight.
    fn ramp(frames: usize) -> Vec<f32> {
        (0..frames).flat_map(|t| [t as f32, -(t as f32)]).collect()
    }

    fn rows(v: &[f32], dim: usize) -> Vec<Vec<f32>> {
        v.chunks_exact(dim).map(<[f32]>::to_vec).collect()
    }

    #[test]
    fn output_frame_count_is_ceil_of_t_over_shift() {
        for (t, expect) in [(1, 1), (6, 1), (7, 2), (12, 2), (13, 3), (100, 17)] {
            let out = apply_lfr(&ramp(t), 2, 7, 6);
            assert_eq!(out.len() / (2 * 7), expect, "T={t}");
        }
    }

    #[test]
    fn first_frame_is_left_clamped_not_zero_padded() {
        // Centre 0 with 3 frames of left context: the window reaches back
        // before the recording starts, and must repeat frame 0 there.
        let out = apply_lfr(&ramp(10), 2, 7, 6);
        let first: Vec<Vec<f32>> = rows(&out[..2 * 7], 2);
        assert_eq!(first[0], vec![0.0, -0.0]);
        assert_eq!(first[1], vec![0.0, -0.0]);
        assert_eq!(first[2], vec![0.0, -0.0]);
        // Then the real frames 0,1,2,3.
        assert_eq!(first[3], vec![0.0, -0.0]);
        assert_eq!(first[4], vec![1.0, -1.0]);
        assert_eq!(first[5], vec![2.0, -2.0]);
        assert_eq!(first[6], vec![3.0, -3.0]);
    }

    #[test]
    fn last_frame_is_right_clamped_to_the_final_frame() {
        // T=7 gives two output frames; the second is centred at 6, the last
        // real frame, so most of its window runs past the end.
        let out = apply_lfr(&ramp(7), 2, 7, 6);
        let second: Vec<Vec<f32>> = rows(&out[2 * 7..], 2);
        assert_eq!(second[0], vec![3.0, -3.0]);
        assert_eq!(second[1], vec![4.0, -4.0]);
        assert_eq!(second[2], vec![5.0, -5.0]);
        assert_eq!(second[3], vec![6.0, -6.0]);
        // Past the end: repeat the last frame, never zeros.
        assert_eq!(second[4], vec![6.0, -6.0]);
        assert_eq!(second[5], vec![6.0, -6.0]);
        assert_eq!(second[6], vec![6.0, -6.0]);
    }

    #[test]
    fn a_middle_frame_is_centred_on_its_shift_multiple() {
        let out = apply_lfr(&ramp(60), 2, 7, 6);
        // Output frame 5 is centred on input frame 30, window 27..=33.
        let f5: Vec<Vec<f32>> = rows(&out[5 * 2 * 7..6 * 2 * 7], 2);
        for (j, expect) in (27..=33).enumerate() {
            assert_eq!(f5[j], vec![expect as f32, -(expect as f32)], "j={j}");
        }
    }

    #[test]
    fn produces_the_560_wide_rows_the_encoder_wants() {
        let feats = vec![0.5f32; 100 * 80];
        let out = apply_lfr(&feats, 80, 7, 6);
        assert_eq!(out.len() % 560, 0);
        assert_eq!(out.len() / 560, 17); // ceil(100/6)
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(apply_lfr(&[], 80, 7, 6).is_empty());
    }

    #[test]
    fn single_frame_input_still_produces_one_window() {
        let out = apply_lfr(&ramp(1), 2, 7, 6);
        assert_eq!(out.len(), 2 * 7);
        // Every slot clamps to the only frame there is.
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn cmvn_shifts_then_scales() {
        let mut f = vec![1.0, 2.0, 3.0, 4.0];
        apply_cmvn(&mut f, &[-1.0, -2.0], &[2.0, 0.5]);
        // Row 0: (1-1)*2=0, (2-2)*0.5=0. Row 1: (3-1)*2=4, (4-2)*0.5=1.
        assert_eq!(f, vec![0.0, 0.0, 4.0, 1.0]);
    }

    #[test]
    fn cmvn_normalizes_a_constant_input_to_zero() {
        // The property the statistics encode: feed the mean, get zero out.
        let neg_mean: Vec<f32> = (0..560).map(|i| -(i as f32)).collect();
        let inv_stddev = vec![0.25f32; 560];
        let mut f: Vec<f32> = (0..560).map(|i| i as f32).collect();
        apply_cmvn(&mut f, &neg_mean, &inv_stddev);
        assert!(f.iter().all(|v| v.abs() < 1e-6));
    }
}
