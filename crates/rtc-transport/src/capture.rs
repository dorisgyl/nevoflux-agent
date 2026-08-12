//! Screencast: getting the desktop into H.264, on every platform.
//!
//! # Why ffmpeg and not three capture backends
//!
//! `crates/computer` takes screenshots — one full frame, PNG-encoded. At 30 fps
//! that saturates a core before any of it reaches the network, because a
//! screenshot API is built to hand you a bitmap and an encoder is built to be
//! fed a stream. Real screencast needs the platform's *capture* path (DXGI
//! Desktop Duplication, ScreenCaptureKit, X11/PipeWire) wired to a hardware
//! encoder, ideally without the frame ever leaving the GPU.
//!
//! Writing that three times is three platform backends plus three encoder
//! integrations. ffmpeg already has all six, and this repository already
//! depends on it (`crates/daemon/src/canvas_video/ffmpeg.rs`). So capture and
//! encode are one child process per platform, and what is written here is the
//! argument list and the parser for what comes back — both of which are pure,
//! and both of which are therefore testable on a machine that cannot run any of
//! the three capture paths.
//!
//! # What "tested" means here
//!
//! [`ffmpeg_args`] and [`AnnexBReader`] are fully covered. Actually capturing a
//! desktop is not, and cannot be from CI or from the current development box —
//! see the crate README. The split is deliberate: everything that can be got
//! wrong silently is in the tested half.

use std::time::Duration;

/// Which platform's capture path to build arguments for.
///
/// A parameter rather than `cfg!`, so the arguments for all three can be tested
/// from any one of them. The caller passes [`Platform::host`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    MacOs,
    Linux,
}

impl Platform {
    /// The platform this build is for.
    pub const fn host() -> Self {
        #[cfg(target_os = "windows")]
        {
            Platform::Windows
        }
        #[cfg(target_os = "macos")]
        {
            Platform::MacOs
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            Platform::Linux
        }
    }
}

/// Which encoder to ask for.
///
/// Hardware first, with software as the fallback that always exists. A machine
/// with no usable hardware encoder should still be able to share its screen,
/// just at a higher CPU cost — refusing outright would be worse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoder {
    /// NVIDIA. The fast path on the Windows boxes this targets.
    Nvenc,
    /// Apple VideoToolbox.
    VideoToolbox,
    /// Linux VA-API.
    Vaapi,
    /// libx264. Always available; costs a core or two at 1080p30.
    Software,
}

impl Encoder {
    fn name(self) -> &'static str {
        match self {
            Encoder::Nvenc => "h264_nvenc",
            Encoder::VideoToolbox => "h264_videotoolbox",
            Encoder::Vaapi => "h264_vaapi",
            Encoder::Software => "libx264",
        }
    }

    /// The encoder to try first on a platform, before falling back to software.
    pub fn preferred_for(platform: Platform) -> Self {
        match platform {
            Platform::Windows => Encoder::Nvenc,
            Platform::MacOs => Encoder::VideoToolbox,
            Platform::Linux => Encoder::Vaapi,
        }
    }
}

/// How the screencast should look and how hard it may push.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureConfig {
    pub fps: u32,
    /// Target bitrate. Driven by the bandwidth estimate in a running session, so
    /// this is only where it starts.
    pub bitrate_bps: u32,
    /// Seconds between keyframes.
    ///
    /// Short, because a viewer joining or recovering from loss cannot draw
    /// anything until the next one. Long GOPs are for files nobody seeks into.
    pub keyframe_interval: Duration,
    pub encoder: Encoder,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            fps: 30,
            bitrate_bps: 4_000_000,
            keyframe_interval: Duration::from_secs(2),
            encoder: Encoder::preferred_for(Platform::host()),
        }
    }
}

/// Build the ffmpeg command line for capturing this platform's desktop.
///
/// Returns arguments only; the caller supplies the binary, so a build that
/// bundles ffmpeg and one that finds it on PATH share this.
///
/// The output is raw Annex-B H.264 on stdout — no container. A container would
/// mean muxing and demuxing a stream that never touches a disk, and would put a
/// buffer between the encoder and the wire, which is latency this path cannot
/// afford.
pub fn ffmpeg_args(platform: Platform, cfg: &CaptureConfig) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    let s = |v: &str| v.to_string();

    a.push(s("-hide_banner"));
    a.push(s("-loglevel"));
    a.push(s("error"));

    // --- input: the platform's capture device ---------------------------------
    match platform {
        Platform::Windows => {
            // Desktop Duplication. Keeps the frame on the GPU, which is what
            // makes the nvenc path nearly free; `gdigrab` would round-trip it
            // through system memory.
            a.push(s("-f"));
            a.push(s("ddagrab"));
            a.push(s("-framerate"));
            a.push(cfg.fps.to_string());
            a.push(s("-i"));
            a.push(s("desktop"));
        }
        Platform::MacOs => {
            a.push(s("-f"));
            a.push(s("avfoundation"));
            a.push(s("-capture_cursor"));
            a.push(s("1"));
            a.push(s("-framerate"));
            a.push(cfg.fps.to_string());
            a.push(s("-i"));
            // "1:none" is the default screen with no audio. Audio is a separate
            // track and a separate consent decision.
            a.push(s("1:none"));
        }
        Platform::Linux => {
            a.push(s("-f"));
            a.push(s("x11grab"));
            a.push(s("-framerate"));
            a.push(cfg.fps.to_string());
            a.push(s("-i"));
            a.push(s(":0.0"));
        }
    }

    // --- encode ---------------------------------------------------------------
    a.push(s("-c:v"));
    a.push(s(cfg.encoder.name()));

    // Latency, in the three places it hides.
    match cfg.encoder {
        Encoder::Nvenc => {
            a.push(s("-preset"));
            a.push(s("p1")); // fastest
            a.push(s("-tune"));
            a.push(s("ll")); // low latency
            a.push(s("-zerolatency"));
            a.push(s("1"));
            // No lookahead: it buys quality by holding frames, and a held frame
            // is exactly what a screencast cannot pay for.
            a.push(s("-rc-lookahead"));
            a.push(s("0"));
        }
        Encoder::VideoToolbox => {
            a.push(s("-realtime"));
            a.push(s("1"));
        }
        Encoder::Vaapi => {
            a.push(s("-compression_level"));
            a.push(s("1"));
        }
        Encoder::Software => {
            a.push(s("-preset"));
            a.push(s("ultrafast"));
            a.push(s("-tune"));
            a.push(s("zerolatency"));
        }
    }

    // B-frames reorder output, so a decoder has to hold a frame to show the one
    // before it. Fine for a file, wrong for anything live.
    a.push(s("-bf"));
    a.push(s("0"));

    a.push(s("-b:v"));
    a.push(cfg.bitrate_bps.to_string());
    a.push(s("-maxrate"));
    a.push(cfg.bitrate_bps.to_string());
    // One second of buffer. Larger lets the encoder overshoot the estimate for
    // longer than the estimate is valid for.
    a.push(s("-bufsize"));
    a.push(cfg.bitrate_bps.to_string());

    let gop = (cfg.fps as u64 * cfg.keyframe_interval.as_millis() as u64 / 1000).max(1);
    a.push(s("-g"));
    a.push(gop.to_string());

    // WebRTC's baseline-ish profile. High profile is not universally decodable
    // on the phones this targets, and a stream nobody can decode is worse than
    // a slightly larger one.
    a.push(s("-profile:v"));
    a.push(s("constrained_baseline"));
    a.push(s("-pix_fmt"));
    a.push(s("yuv420p"));

    // --- output: raw Annex-B on stdout ---------------------------------------
    a.push(s("-f"));
    a.push(s("h264"));
    a.push(s("-"));
    a
}

/// Splits an Annex-B byte stream into access units.
///
/// ffmpeg writes NAL units separated by three- or four-byte start codes, in a
/// stream that arrives in whatever sized pieces the pipe felt like. This
/// reassembles it, because handing str0m half a frame produces a picture that
/// never decodes and no error anywhere.
///
/// An access unit ends where the next one begins, which is only knowable once
/// the next one has started — so [`push`](Self::push) yields the *previous*
/// complete unit, and [`flush`](Self::flush) yields the last.
#[derive(Debug, Default)]
pub struct AnnexBReader {
    /// Bytes seen since the current access unit started, start code included.
    current: Vec<u8>,
    /// Bytes held back because they might be the beginning of a start code.
    tail: Vec<u8>,
    /// Whether the unit being built already holds a coded slice.
    ///
    /// This is what makes the boundary correct. Splitting on *any* picture-ish
    /// NAL puts SPS and PPS in a unit of their own, and the IDR that follows
    /// then arrives without the parameter sets needed to decode it — so a
    /// viewer joining a running screencast sees nothing at all, with no error
    /// to say why.
    has_slice: bool,
}

/// Whether this NAL is a coded slice — the thing an access unit is built around.
///
/// 1 is a non-IDR slice and 5 an IDR slice. Parameter sets (7, 8) and SEI (6)
/// are not pictures; they prefix the one that follows and belong with it.
fn is_slice(nal_type: u8) -> bool {
    matches!(nal_type, 1 | 5)
}

/// An access unit delimiter, which says outright that a new unit begins.
fn is_delimiter(nal_type: u8) -> bool {
    nal_type == 9
}

impl AnnexBReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes from the pipe. Returns whatever access units completed.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        let mut buf = std::mem::take(&mut self.tail);
        buf.extend_from_slice(chunk);

        let mut out = Vec::new();
        let mut i = 0usize;

        while i + 3 <= buf.len() {
            let four = i + 4 <= buf.len() && buf[i..i + 4] == [0, 0, 0, 1];
            let three = buf[i..i + 3] == [0, 0, 1];
            if !(four || three) {
                self.current.push(buf[i]);
                i += 1;
                continue;
            }

            let code_len = if four { 4 } else { 3 };
            // The type byte follows the start code; without it there is no way
            // to know whether this begins a new access unit, so hold and wait.
            let Some(&type_byte) = buf.get(i + code_len) else {
                break;
            };

            let nal_type = type_byte & 0x1f;
            // A new unit begins at a delimiter, or at the first slice after one
            // this unit already has. Anything else — parameter sets, SEI — is a
            // prefix of the picture still being assembled.
            let boundary = !self.current.is_empty()
                && (is_delimiter(nal_type) || (is_slice(nal_type) && self.has_slice));
            if boundary {
                out.push(std::mem::take(&mut self.current));
                self.has_slice = false;
            }
            self.has_slice |= is_slice(nal_type);
            self.current.extend_from_slice(&buf[i..i + code_len]);
            i += code_len;
        }

        // Anything that could still be the front of a start code stays behind.
        self.tail = buf[i..].to_vec();
        out
    }

    /// The stream ended; give up whatever is left.
    pub fn flush(&mut self) -> Option<Vec<u8>> {
        let tail = std::mem::take(&mut self.tail);
        self.current.extend_from_slice(&tail);
        self.has_slice = false;
        (!self.current.is_empty()).then(|| std::mem::take(&mut self.current))
    }
}

/// Whether an access unit contains an IDR slice.
///
/// A viewer joining, or recovering from loss, can draw nothing until one
/// arrives — so the sender needs to know which frames those are in order to
/// answer a keyframe request rather than waiting out the GOP.
pub fn is_keyframe(access_unit: &[u8]) -> bool {
    let mut i = 0usize;
    while i + 3 <= access_unit.len() {
        let four = i + 4 <= access_unit.len() && access_unit[i..i + 4] == [0, 0, 0, 1];
        let three = access_unit[i..i + 3] == [0, 0, 1];
        if four || three {
            let code_len = if four { 4 } else { 3 };
            if let Some(&b) = access_unit.get(i + code_len) {
                // 5 = IDR slice, 7 = SPS (which only precedes one).
                if matches!(b & 0x1f, 5 | 7) {
                    return true;
                }
            }
            i += code_len;
        } else {
            i += 1;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_for(p: Platform) -> Vec<String> {
        ffmpeg_args(
            p,
            &CaptureConfig {
                encoder: Encoder::preferred_for(p),
                ..Default::default()
            },
        )
    }

    #[test]
    fn every_platform_captures_and_encodes_to_raw_h264_on_stdout() {
        // All three built here, on whichever one this is: the arguments are the
        // part that can be wrong silently, and they must not go untested on two
        // platforms out of three just because the machine is the third.
        for p in [Platform::Windows, Platform::MacOs, Platform::Linux] {
            let a = args_for(p);
            let joined = a.join(" ");
            assert!(
                joined.contains("-f h264"),
                "{p:?} is not raw h264: {joined}"
            );
            assert_eq!(a.last().unwrap(), "-", "{p:?} does not write to stdout");
            assert!(joined.contains("-i "), "{p:?} has no input: {joined}");
        }
    }

    #[test]
    fn each_platform_uses_its_own_capture_device() {
        assert!(args_for(Platform::Windows).contains(&"ddagrab".to_string()));
        assert!(args_for(Platform::MacOs).contains(&"avfoundation".to_string()));
        assert!(args_for(Platform::Linux).contains(&"x11grab".to_string()));
    }

    #[test]
    fn each_platform_reaches_for_its_own_hardware_encoder() {
        assert!(args_for(Platform::Windows).contains(&"h264_nvenc".to_string()));
        assert!(args_for(Platform::MacOs).contains(&"h264_videotoolbox".to_string()));
        assert!(args_for(Platform::Linux).contains(&"h264_vaapi".to_string()));
    }

    #[test]
    fn software_encoding_is_available_on_every_platform() {
        // A machine with no usable hardware encoder must still be able to share
        // its screen, or the feature simply does not exist for that user.
        for p in [Platform::Windows, Platform::MacOs, Platform::Linux] {
            let cfg = CaptureConfig {
                encoder: Encoder::Software,
                ..Default::default()
            };
            let a = ffmpeg_args(p, &cfg);
            assert!(a.contains(&"libx264".to_string()));
            assert!(
                a.contains(&"zerolatency".to_string()),
                "{p:?} not low latency"
            );
        }
    }

    #[test]
    fn nothing_is_allowed_to_hold_a_frame_back() {
        // The three places latency hides in an encoder. Each of these being
        // wrong costs hundreds of milliseconds and none of them shows up as an
        // error — the picture is simply always behind.
        for p in [Platform::Windows, Platform::MacOs, Platform::Linux] {
            let a = args_for(p).join(" ");
            assert!(a.contains("-bf 0"), "{p:?} allows B-frames: {a}");
        }
        let win = args_for(Platform::Windows).join(" ");
        assert!(win.contains("-rc-lookahead 0"), "nvenc lookahead: {win}");
        assert!(win.contains("-tune ll"), "nvenc not low latency: {win}");
    }

    #[test]
    fn the_gop_follows_the_keyframe_interval_and_the_frame_rate() {
        let cfg = CaptureConfig {
            fps: 30,
            keyframe_interval: Duration::from_secs(2),
            ..Default::default()
        };
        let a = ffmpeg_args(Platform::Linux, &cfg);
        let g = a.iter().position(|x| x == "-g").expect("-g");
        assert_eq!(a[g + 1], "60");

        // And never zero, however short the interval — ffmpeg reads -g 0 as
        // all-keyframes, which at 30 fps is far more bitrate than asked for.
        let cfg = CaptureConfig {
            fps: 30,
            keyframe_interval: Duration::from_millis(1),
            ..Default::default()
        };
        let a = ffmpeg_args(Platform::Linux, &cfg);
        let g = a.iter().position(|x| x == "-g").unwrap();
        assert_eq!(a[g + 1], "1");
    }

    // --- Annex-B ------------------------------------------------------------

    /// One NAL with a four-byte start code.
    fn nal(nal_type: u8, body: &[u8]) -> Vec<u8> {
        let mut v = vec![0, 0, 0, 1, nal_type & 0x1f];
        v.extend_from_slice(body);
        v
    }

    #[test]
    fn splits_a_stream_into_access_units() {
        let mut r = AnnexBReader::new();
        let mut stream = Vec::new();
        stream.extend(nal(7, b"sps")); // starts AU 1
        stream.extend(nal(8, b"pps")); // same AU
        stream.extend(nal(5, b"idr")); // same AU (already started)
        stream.extend(nal(1, b"p1")); // starts AU 2
        stream.extend(nal(1, b"p2")); // starts AU 3

        let mut units = r.push(&stream);
        units.extend(r.flush());
        assert_eq!(units.len(), 3, "got {} units", units.len());
        assert!(is_keyframe(&units[0]), "the first unit holds the IDR");
        assert!(!is_keyframe(&units[1]));
    }

    #[test]
    fn parameter_sets_stay_with_the_picture_they_describe() {
        // The defect this guards: splitting on any picture-ish NAL put SPS and
        // PPS in a unit of their own, so the IDR after them arrived without the
        // parameter sets needed to decode it. A viewer joining a running
        // screencast then sees nothing, and nothing reports an error.
        let mut r = AnnexBReader::new();
        let mut stream = Vec::new();
        stream.extend(nal(7, b"sps"));
        stream.extend(nal(8, b"pps"));
        stream.extend(nal(5, b"idr"));
        stream.extend(nal(1, b"p"));

        let mut units = r.push(&stream);
        units.extend(r.flush());

        assert_eq!(units.len(), 2);
        let first = &units[0];
        assert!(is_keyframe(first));
        assert!(
            first.windows(3).any(|w| w == b"sps") && first.windows(3).any(|w| w == b"pps"),
            "the IDR unit must carry its own parameter sets"
        );
    }

    #[test]
    fn an_access_unit_delimiter_always_starts_a_new_unit() {
        let mut r = AnnexBReader::new();
        let mut stream = Vec::new();
        stream.extend(nal(9, b"")); // AUD
        stream.extend(nal(1, b"a"));
        stream.extend(nal(9, b"")); // AUD
        stream.extend(nal(1, b"b"));
        let mut units = r.push(&stream);
        units.extend(r.flush());
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn a_unit_split_across_pipe_reads_is_reassembled() {
        // The pipe hands over whatever size it likes, including a start code cut
        // in half. Handing str0m half a frame produces a picture that never
        // decodes and no error anywhere.
        let mut whole = Vec::new();
        whole.extend(nal(7, b"sps"));
        whole.extend(nal(5, &[9u8; 200]));
        whole.extend(nal(1, &[3u8; 200]));

        for split in 1..whole.len() {
            let mut r = AnnexBReader::new();
            let mut units = r.push(&whole[..split]);
            units.extend(r.push(&whole[split..]));
            units.extend(r.flush());

            let rejoined: Vec<u8> = units.concat();
            assert_eq!(
                rejoined, whole,
                "bytes lost or duplicated when split at {split}"
            );
        }
    }

    #[test]
    fn byte_at_a_time_delivery_still_reassembles() {
        let mut whole = Vec::new();
        whole.extend(nal(7, b"s"));
        whole.extend(nal(5, b"idr"));
        whole.extend(nal(1, b"p"));

        let mut r = AnnexBReader::new();
        let mut units = Vec::new();
        for b in &whole {
            units.extend(r.push(&[*b]));
        }
        units.extend(r.flush());
        assert_eq!(units.concat(), whole);
    }

    #[test]
    fn three_byte_start_codes_are_understood_too() {
        // ffmpeg emits both lengths in the same stream, sometimes within one
        // access unit. Two slices, so there is a boundary to find at all.
        let mut stream = vec![0, 0, 1, 1, b'a']; // 3-byte code, slice
        stream.extend_from_slice(&[0, 0, 0, 1, 1, b'b']); // 4-byte code, slice
        let mut r = AnnexBReader::new();
        let mut units = r.push(&stream);
        units.extend(r.flush());
        assert_eq!(units.len(), 2, "both code lengths must be found");
        assert_eq!(units.concat(), stream, "no byte may be lost or duplicated");
    }

    #[test]
    fn a_parameter_set_before_a_slice_is_one_unit_not_two() {
        // The corrected boundary rule, stated as its own case: parameter sets
        // are a prefix of the picture that follows, never a picture themselves.
        let mut stream = Vec::new();
        stream.extend(nal(7, b"sps"));
        stream.extend(nal(1, b"slice"));
        let mut r = AnnexBReader::new();
        let mut units = r.push(&stream);
        units.extend(r.flush());
        assert_eq!(units.len(), 1);
    }

    #[test]
    fn an_empty_stream_yields_nothing_rather_than_an_empty_frame() {
        let mut r = AnnexBReader::new();
        assert!(r.push(&[]).is_empty());
        assert_eq!(r.flush(), None);
    }

    #[test]
    fn a_keyframe_is_recognised_by_its_idr_not_its_position() {
        assert!(is_keyframe(&nal(5, b"idr")));
        assert!(is_keyframe(&nal(7, b"sps")));
        assert!(!is_keyframe(&nal(1, b"p")));
        assert!(!is_keyframe(b""));
        assert!(!is_keyframe(b"no start code here"));
    }
}

/// The 90 kHz clock every H.264 RTP stream is timed against.
pub const VIDEO_CLOCK_HZ: u64 = 90_000;

/// The RTP timestamp for a frame this far into the capture.
///
/// Derived from elapsed time rather than counted in frames: a capture that
/// drops one — and every screen capture drops some, because nothing changed or
/// the encoder was behind — would otherwise slew permanently ahead of the clock
/// and the far end would play the whole session progressively out of time.
pub fn rtp_time(elapsed: Duration) -> u64 {
    (elapsed.as_nanos() * VIDEO_CLOCK_HZ as u128 / 1_000_000_000u128) as u64
}

#[cfg(feature = "tokio-driver")]
pub use spawn::{run_capture, CaptureHandle, Frame};

#[cfg(feature = "tokio-driver")]
mod spawn {
    use super::*;
    use std::process::Stdio;
    use tokio::io::AsyncReadExt;
    use tokio::sync::mpsc;

    /// One encoded frame, ready for the track.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Frame {
        pub data: Vec<u8>,
        /// Since capture began, for [`rtp_time`].
        pub elapsed: Duration,
        pub keyframe: bool,
    }

    /// A running capture. Dropping it stops ffmpeg.
    pub struct CaptureHandle {
        child: tokio::process::Child,
    }

    impl CaptureHandle {
        /// Stop capturing.
        ///
        /// Killed rather than asked politely: ffmpeg reading a capture device
        /// does not always notice a closed stdout, and a screencast that keeps
        /// running after the session ended is both a privacy problem and a core
        /// burning for nobody.
        pub async fn stop(mut self) {
            let _ = self.child.kill().await;
        }
    }

    /// Start capturing the desktop, delivering access units as they are encoded.
    ///
    /// `ffmpeg` is the binary to run — passed in rather than found here, so a
    /// build that bundles it and one that relies on PATH share this.
    pub async fn run_capture(
        ffmpeg: &std::path::Path,
        cfg: CaptureConfig,
        frames: mpsc::Sender<Frame>,
    ) -> std::io::Result<CaptureHandle> {
        let args = ffmpeg_args(Platform::host(), &cfg);
        tracing::info!(target: "rtc", "starting capture: {} {}", ffmpeg.display(), args.join(" "));

        let mut child = tokio::process::Command::new(ffmpeg)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()?;

        let mut stdout = child.stdout.take().expect("stdout was piped");
        // Drained on its own task. ffmpeg writes progress and warnings here, and
        // a full stderr pipe blocks the encoder — which looks exactly like a
        // capture that mysteriously stops after a few seconds.
        if let Some(mut stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut buf = String::new();
                let _ = tokio::io::AsyncReadExt::read_to_string(&mut stderr, &mut buf).await;
                if !buf.trim().is_empty() {
                    tracing::warn!(target: "rtc", "ffmpeg: {}", buf.trim());
                }
            });
        }

        tokio::spawn(async move {
            let started = std::time::Instant::now();
            let mut reader = AnnexBReader::new();
            let mut buf = vec![0u8; 64 * 1024];

            loop {
                let n = match stdout.read(&mut buf).await {
                    Ok(0) => break, // ffmpeg exited
                    Ok(n) => n,
                    Err(e) => {
                        tracing::warn!(target: "rtc", "capture read failed: {e}");
                        break;
                    }
                };
                for unit in reader.push(&buf[..n]) {
                    let frame = Frame {
                        keyframe: is_keyframe(&unit),
                        data: unit,
                        elapsed: started.elapsed(),
                    };
                    if frames.send(frame).await.is_err() {
                        return; // nobody is watching any more
                    }
                }
            }
            if let Some(unit) = reader.flush() {
                let _ = frames
                    .send(Frame {
                        keyframe: is_keyframe(&unit),
                        data: unit,
                        elapsed: started.elapsed(),
                    })
                    .await;
            }
        });

        Ok(CaptureHandle { child })
    }
}

#[cfg(test)]
mod clock_tests {
    use super::*;

    #[test]
    fn rtp_time_runs_at_ninety_kilohertz() {
        assert_eq!(rtp_time(Duration::ZERO), 0);
        assert_eq!(rtp_time(Duration::from_secs(1)), 90_000);
        assert_eq!(rtp_time(Duration::from_millis(33)), 2_970);
    }

    #[test]
    fn a_dropped_frame_does_not_slew_the_clock() {
        // The reason this is derived from elapsed time and not a frame counter.
        // Every screen capture drops frames — nothing changed, or the encoder
        // was behind — and counting them would put the whole session
        // progressively out of time with no single moment where it broke.
        let at_one_second = rtp_time(Duration::from_secs(1));
        // Ten frames of a 30 fps capture never arrived; the eleventh is still
        // stamped for where it actually is.
        assert_eq!(at_one_second, 90_000);
        assert_eq!(rtp_time(Duration::from_secs(2)) - at_one_second, 90_000);
    }

    #[test]
    fn the_clock_survives_a_long_session() {
        // u64 at 90 kHz overflows after six million years; u32 would wrap in
        // thirteen hours, which a screencast can reach.
        let day = rtp_time(Duration::from_secs(60 * 60 * 24));
        assert_eq!(day, 90_000 * 60 * 60 * 24);
    }
}
