//! Kaldi-compatible 80-bin log-mel filterbank.
//!
//! "Kaldi-compatible" is the whole requirement. The model was trained on
//! features from Kaldi's `compute-fbank-feats`, and an encoder is not
//! forgiving about features that are merely close: a window function or a mel
//! edge that disagrees costs accuracy silently, with no error anywhere to
//! trace it back to. So every constant here is Kaldi's, and the ones that
//! differ from Kaldi's own defaults are the ones SenseVoice overrides.
//!
//! Settings, and where each comes from
//! (sherpa-onnx `features.h` defaults, overridden in
//! `offline-recognizer-sense-voice-impl.h::InitFeatConfig`):
//!
//! | setting              | value    | source                       |
//! |----------------------|----------|------------------------------|
//! | sample rate          | 16000    | default                      |
//! | frame length / shift | 25 / 10ms| default                      |
//! | mel bins             | 80       | default                      |
//! | low / high freq      | 20 / 8k  | default / SenseVoice (0 = nyquist) |
//! | window               | hamming  | **SenseVoice** (Kaldi's is povey) |
//! | snip_edges           | true     | **SenseVoice** (sherpa's is false) |
//! | dither               | 0        | default                      |
//! | remove DC offset     | true     | default                      |
//! | preemphasis          | 0.97     | default                      |
//! | round to power of 2  | true     | default → 512-point FFT      |
//! | energy floor         | eps      | Kaldi's `log` floor          |

pub const SAMPLE_RATE: f32 = 16000.0;
pub const FRAME_LENGTH: usize = 400; // 25 ms
pub const FRAME_SHIFT: usize = 160; // 10 ms
pub const FFT_SIZE: usize = 512; // next power of two above FRAME_LENGTH
pub const NUM_BINS: usize = 80;
const LOW_FREQ: f32 = 20.0;
const HIGH_FREQ: f32 = SAMPLE_RATE / 2.0;
const PREEMPH: f32 = 0.97;

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi's Hamming window: `0.54 - 0.46 cos(2*pi*i/(N-1))`.
///
/// Note the `N-1`. Using `N` is the other common convention and produces a
/// window that is subtly wrong at both ends — the kind of difference that
/// never fails a test you thought to write.
fn hamming(n: usize) -> Vec<f32> {
    let a = std::f32::consts::TAU / (n as f32 - 1.0);
    (0..n).map(|i| 0.54 - 0.46 * (a * i as f32).cos()).collect()
}

/// Triangular mel filterbank over the FFT's positive bins, Kaldi's layout.
///
/// Each row is one mel bin, `FFT_SIZE / 2` wide.
fn mel_banks() -> Vec<Vec<f32>> {
    let num_fft_bins = FFT_SIZE / 2;
    let fft_bin_width = SAMPLE_RATE / FFT_SIZE as f32;
    let mel_low = mel_scale(LOW_FREQ);
    let mel_high = mel_scale(HIGH_FREQ);
    let delta = (mel_high - mel_low) / (NUM_BINS + 1) as f32;

    (0..NUM_BINS)
        .map(|bin| {
            let left = mel_low + bin as f32 * delta;
            let center = left + delta;
            let right = center + delta;
            (0..num_fft_bins)
                .map(|i| {
                    let mel = mel_scale(fft_bin_width * i as f32);
                    if mel <= left || mel >= right {
                        0.0
                    } else if mel <= center {
                        (mel - left) / (center - left)
                    } else {
                        (right - mel) / (right - center)
                    }
                })
                .collect()
        })
        .collect()
}

/// In-place iterative radix-2 FFT.
///
/// Hand-written rather than pulled in: 512 points of power-of-two FFT is a
/// well-understood fifty lines, and the alternative is a dependency in a crate
/// that otherwise has almost none. `fft_matches_naive_dft` is what makes that
/// trade safe.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let ang = -std::f32::consts::TAU / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cur_r, mut cur_i) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr0, vi0) = (re[i + k + len / 2], im[i + k + len / 2]);
                let vr = vr0 * cur_r - vi0 * cur_i;
                let vi = vr0 * cur_i + vi0 * cur_r;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let next_r = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = next_r;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Precomputed window and filterbank, so a long utterance builds them once.
pub struct Fbank {
    window: Vec<f32>,
    banks: Vec<Vec<f32>>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Fbank {
    pub fn new() -> Fbank {
        Fbank {
            window: hamming(FRAME_LENGTH),
            banks: mel_banks(),
        }
    }

    /// How many frames `num_samples` yields under `snip_edges = true`:
    /// only whole windows count, and the tail that cannot fill one is dropped.
    pub fn num_frames(num_samples: usize) -> usize {
        if num_samples < FRAME_LENGTH {
            0
        } else {
            1 + (num_samples - FRAME_LENGTH) / FRAME_SHIFT
        }
    }

    /// Row-major `[num_frames, 80]` log-mel energies.
    ///
    /// `samples` must be 16 kHz mono. Scale is the caller's problem: this model
    /// reports `normalize_samples = 0`, meaning it was trained on the int16
    /// range, so the caller multiplies by 32768 before getting here.
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let frames = Self::num_frames(samples.len());
        let mut out = Vec::with_capacity(frames * NUM_BINS);
        let mut re = vec![0.0f32; FFT_SIZE];
        let mut im = vec![0.0f32; FFT_SIZE];
        let mut buf = vec![0.0f32; FRAME_LENGTH];

        for f in 0..frames {
            let start = f * FRAME_SHIFT;
            buf.copy_from_slice(&samples[start..start + FRAME_LENGTH]);

            // Kaldi's order: DC offset, then preemphasis, then window.
            let mean = buf.iter().sum::<f32>() / FRAME_LENGTH as f32;
            for v in buf.iter_mut() {
                *v -= mean;
            }
            // Backwards, so each sample still sees its original predecessor.
            // The first sample uses itself, which is Kaldi's edge convention.
            for i in (1..FRAME_LENGTH).rev() {
                buf[i] -= PREEMPH * buf[i - 1];
            }
            buf[0] -= PREEMPH * buf[0];
            for (v, w) in buf.iter_mut().zip(&self.window) {
                *v *= w;
            }

            re[..FRAME_LENGTH].copy_from_slice(&buf);
            re[FRAME_LENGTH..].fill(0.0);
            im.fill(0.0);
            fft(&mut re, &mut im);

            // Power spectrum over the positive bins.
            let power: Vec<f32> = (0..FFT_SIZE / 2)
                .map(|i| re[i] * re[i] + im[i] * im[i])
                .collect();

            for bank in &self.banks {
                let energy: f32 = bank.iter().zip(&power).map(|(w, p)| w * p).sum();
                out.push(energy.max(f32::EPSILON).ln());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(input: &[f32]) -> Vec<(f32, f32)> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (t, x) in input.iter().enumerate() {
                    let ang = -std::f64::consts::TAU * (k * t) as f64 / n as f64;
                    re += *x as f64 * ang.cos();
                    im += *x as f64 * ang.sin();
                }
                (re as f32, im as f32)
            })
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft() {
        // This is what licenses the hand-written FFT above.
        let n = 64;
        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.37).sin() + 0.5 * (i as f32 * 1.1).cos())
            .collect();
        let expect = naive_dft(&input);
        let mut re = input.clone();
        let mut im = vec![0.0f32; n];
        fft(&mut re, &mut im);
        for k in 0..n {
            assert!(
                (re[k] - expect[k].0).abs() < 1e-3,
                "bin {k} real: {} vs {}",
                re[k],
                expect[k].0
            );
            assert!(
                (im[k] - expect[k].1).abs() < 1e-3,
                "bin {k} imag: {} vs {}",
                im[k],
                expect[k].1
            );
        }
    }

    #[test]
    fn fft_of_a_pure_tone_peaks_at_its_bin() {
        let n = 512;
        let bin = 20;
        let mut re: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * bin as f32 * i as f32 / n as f32).sin())
            .collect();
        let mut im = vec![0.0f32; n];
        fft(&mut re, &mut im);
        let power: Vec<f32> = (0..n / 2).map(|i| re[i] * re[i] + im[i] * im[i]).collect();
        let peak = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, bin);
    }

    #[test]
    fn hamming_uses_the_n_minus_one_convention() {
        let w = hamming(400);
        // Symmetric, and reaching the same endpoint value at both ends --
        // which the N convention does not do.
        assert!((w[0] - 0.08).abs() < 1e-5, "{}", w[0]);
        assert!((w[399] - 0.08).abs() < 1e-5, "{}", w[399]);
        assert!((w[199] - w[200]).abs() < 1e-3);
    }

    #[test]
    fn mel_scale_is_kaldis() {
        assert!((mel_scale(0.0) - 0.0).abs() < 1e-6);
        // 1127 * ln(1 + 1000/700)
        assert!(
            (mel_scale(1000.0) - 999.985_2).abs() < 0.01,
            "{}",
            mel_scale(1000.0)
        );
    }

    #[test]
    fn mel_banks_have_the_expected_shape() {
        let banks = mel_banks();
        assert_eq!(banks.len(), NUM_BINS);
        assert!(banks.iter().all(|b| b.len() == FFT_SIZE / 2));
        // Every bin has some support, and none is all zeros.
        for (i, b) in banks.iter().enumerate() {
            assert!(b.iter().any(|w| *w > 0.0), "bin {i} is empty");
            assert!(
                b.iter().all(|w| *w >= 0.0 && *w <= 1.0),
                "bin {i} out of range"
            );
        }
        // Neighbouring triangles overlap: consecutive bins share support.
        let shared = banks[10]
            .iter()
            .zip(&banks[11])
            .filter(|(a, b)| **a > 0.0 && **b > 0.0)
            .count();
        assert!(shared > 0, "adjacent mel bins should overlap");
    }

    #[test]
    fn frame_count_snips_the_partial_tail() {
        // snip_edges = true: a tail too short for a whole window is dropped.
        assert_eq!(Fbank::num_frames(0), 0);
        assert_eq!(Fbank::num_frames(399), 0);
        assert_eq!(Fbank::num_frames(400), 1);
        assert_eq!(Fbank::num_frames(559), 1);
        assert_eq!(Fbank::num_frames(560), 2);
        // One second of 16 kHz audio.
        assert_eq!(Fbank::num_frames(16000), 98);
    }

    #[test]
    fn compute_produces_one_row_of_80_per_frame() {
        let fb = Fbank::new();
        let samples: Vec<f32> = (0..16000)
            .map(|i| 3000.0 * (std::f32::consts::TAU * 440.0 * i as f32 / SAMPLE_RATE).sin())
            .collect();
        let feats = fb.compute(&samples);
        assert_eq!(feats.len(), 98 * NUM_BINS);
        assert!(feats.iter().all(|v| v.is_finite()), "non-finite energy");
    }

    #[test]
    fn a_tone_lands_in_the_mel_bin_that_covers_it() {
        let fb = Fbank::new();
        let hz = 440.0;
        let samples: Vec<f32> = (0..16000)
            .map(|i| 3000.0 * (std::f32::consts::TAU * hz * i as f32 / SAMPLE_RATE).sin())
            .collect();
        let feats = fb.compute(&samples);
        // Look at a frame in the middle, away from onset effects.
        let row = &feats[50 * NUM_BINS..51 * NUM_BINS];
        let loudest = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // Which bin should hold 440 Hz, by the same mel geometry.
        let mel_low = mel_scale(LOW_FREQ);
        let delta = (mel_scale(HIGH_FREQ) - mel_low) / (NUM_BINS + 1) as f32;
        let expected = ((mel_scale(hz) - mel_low) / delta - 1.0).round() as usize;
        assert!(
            loudest.abs_diff(expected) <= 1,
            "tone at {hz} Hz peaked in bin {loudest}, expected about {expected}"
        );
    }

    #[test]
    fn silence_floors_at_log_epsilon_without_nan() {
        let fb = Fbank::new();
        let feats = fb.compute(&vec![0.0f32; 16000]);
        assert_eq!(feats.len(), 98 * NUM_BINS);
        assert!(
            feats.iter().all(|v| v.is_finite()),
            "silence produced non-finite"
        );
        assert!(feats.iter().all(|v| (*v - f32::EPSILON.ln()).abs() < 1e-3));
    }
}
