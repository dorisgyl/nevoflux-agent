//! What the speech stack needs on disk, where to get it, and how to tell it
//! apart from something else that arrived under the same name.
//!
//! Everything here is pinned data rather than discovery. A model repository can
//! move a file, re-export it, or replace a Mandarin-first checkpoint with a
//! Cantonese fine-tune under an unchanged filename — that last one has already
//! happened once in this project's history (see `just fetch-asr-models`). A
//! size and a digest turn it from a silent behaviour change into a refused
//! download.
//!
//! ## How these numbers were obtained
//!
//! Not by hashing whatever happened to be on the build machine. Each entry was
//! checked against what the source serves now:
//!
//! - HuggingFace keeps large files in LFS and reports the content digest in the
//!   `x-linked-etag` header, so a `HEAD` verifies a 229 MB file without
//!   transferring it. `sensevoice-small.int8.onnx` matched exactly.
//! - Small files are plain git objects, whose ETag is a git blob SHA-1 rather
//!   than a content SHA-256. Those were verified by downloading them again and
//!   comparing (`tokens.txt`, 309 KB; `silero_vad.onnx`, 2.3 MB).
//! - GitHub release assets expose no digest header at all, so Kokoro's files
//!   were matched on exact byte count against the release and their digests
//!   pinned from the verified local copies.

/// What a tier buys the user.
///
/// The split is the point of tiered downloading: 240 MB gets dictation working,
/// and the far larger speech synthesis can arrive afterwards without blocking
/// it. A single "download everything" step makes the first useful moment wait
/// on the least urgent bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Speech in, text out. Useful on its own.
    Transcribe,
    /// Text out, speech back.
    Speak,
}

impl Tier {
    pub fn id(self) -> &'static str {
        match self {
            Tier::Transcribe => "transcribe",
            Tier::Speak => "speak",
        }
    }

    pub fn parse(s: &str) -> Option<Tier> {
        match s {
            "transcribe" => Some(Tier::Transcribe),
            "speak" => Some(Tier::Speak),
            _ => None,
        }
    }
}

/// One file that has to exist under `~/.cache/nevoflux/models`.
#[derive(Debug, Clone)]
pub struct Asset {
    /// Stable identifier for the wire and for logs. Never the filename: that is
    /// a local convention which a future export may well change.
    pub id: &'static str,
    pub tier: Tier,
    /// Local filename. Model resolution in `tts/asr.rs` and `tts/kokoro.rs`
    /// looks for exactly these — rename here, rename there.
    pub file: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    /// Tried in order. Mirrors first: the users this phase exists for cannot
    /// reach the upstream hosts at all, so upstream-first would open every
    /// download with a timeout.
    pub sources: &'static [&'static str],
    /// Whether NevoFlux may ever host these bytes itself.
    ///
    /// `false` is not a preference. MOSS's weights are published under terms
    /// that may not permit redistribution, and the entire mitigation is that we
    /// never touch the bytes: pointing at someone else's URL is not
    /// redistribution, putting them on our own CDN is. A test enforces this, so
    /// that a later "downloads keep failing, let's just mirror it" cannot
    /// quietly void it.
    pub self_hosting_allowed: bool,
    /// Named when a download fails, so the file can be fetched by hand.
    pub upstream: &'static str,
}

const SENSEVOICE_REPO: &str = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";
const KOKORO_RELEASE: &str = "thewh1teagle/kokoro-onnx @ model-files-v1.0";

pub const ASSETS: &[Asset] = &[
    Asset {
        id: "sensevoice-model",
        tier: Tier::Transcribe,
        file: "sensevoice-small.int8.onnx",
        bytes: 239_233_841,
        sha256: "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
        sources: &[
            "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx",
            "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx",
        ],
        self_hosting_allowed: true,
        upstream: SENSEVOICE_REPO,
    },
    Asset {
        id: "sensevoice-tokens",
        tier: Tier::Transcribe,
        file: "sensevoice-tokens.txt",
        bytes: 315_894,
        sha256: "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
        sources: &[
            "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt",
            "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt",
        ],
        self_hosting_allowed: true,
        upstream: SENSEVOICE_REPO,
    },
    Asset {
        // v6.2.1 from the upstream release rather than a HuggingFace mirror:
        // onnx-community's copy is v5, a year and a major version behind, and
        // takes the same inputs — a substitution nothing would report.
        id: "silero-vad",
        tier: Tier::Transcribe,
        file: "silero-vad.onnx",
        bytes: 2_327_524,
        sha256: "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        sources: &[
            "https://ghfast.top/https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx",
            "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx",
        ],
        self_hosting_allowed: true,
        upstream: "snakers4/silero-vad @ v6.2.1",
    },
    Asset {
        // int8 rather than fp32: a third of the bytes, and `kokoro.rs` takes
        // whichever is present. 233 MB more download is not worth it for an
        // engine that only runs when the primary one is unavailable.
        id: "kokoro-model",
        tier: Tier::Speak,
        file: "kokoro-v1.0.int8.onnx",
        bytes: 92_361_271,
        sha256: "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
        sources: &[
            "https://ghfast.top/https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
        ],
        self_hosting_allowed: true,
        upstream: KOKORO_RELEASE,
    },
    Asset {
        id: "kokoro-voices",
        tier: Tier::Speak,
        file: "kokoro-voices-v1.0.bin",
        bytes: 28_214_398,
        sha256: "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
        sources: &[
            "https://ghfast.top/https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        ],
        self_hosting_allowed: true,
        upstream: KOKORO_RELEASE,
    },
];

/// Hosts NevoFlux controls. Anything here counts as us serving the bytes.
///
/// Used only to enforce `self_hosting_allowed` — a denylist for
/// non-redistributable assets, not an allowlist for anything.
pub const OUR_HOSTS: &[&str] = &["nevoflux.com", "nevoflux.app", "cdn.nevoflux.com"];

pub fn by_id(id: &str) -> Option<&'static Asset> {
    ASSETS.iter().find(|a| a.id == id)
}

pub fn of_tier(tier: Tier) -> impl Iterator<Item = &'static Asset> {
    ASSETS.iter().filter(move |a| a.tier == tier)
}

/// What a tier costs, for saying so before it starts.
pub fn tier_bytes(tier: Tier) -> u64 {
    of_tier(tier).map(|a| a.bytes).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_of(url: &str) -> &str {
        // Deliberately crude: a proxied URL embeds a second scheme
        // (`https://proxy/https://real/...`), and it is the outer host — the
        // one actually contacted — that this wants.
        url.split("://")
            .nth(1)
            .unwrap_or(url)
            .split('/')
            .next()
            .unwrap_or("")
    }

    #[test]
    fn ids_are_unique() {
        let mut ids: Vec<_> = ASSETS.iter().map(|a| a.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "two assets share an id");
    }

    #[test]
    fn filenames_are_unique() {
        // They all land in one directory, so a collision has one asset
        // overwrite another and then pass its own digest check.
        let mut files: Vec<_> = ASSETS.iter().map(|a| a.file).collect();
        files.sort_unstable();
        let before = files.len();
        files.dedup();
        assert_eq!(before, files.len(), "two assets share a filename");
    }

    #[test]
    fn every_asset_is_pinned() {
        for a in ASSETS {
            assert!(a.bytes > 0, "{} has no size", a.id);
            assert_eq!(a.sha256.len(), 64, "{} digest is not a sha256", a.id);
            assert!(
                a.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{} digest is not hex",
                a.id
            );
            assert!(!a.sources.is_empty(), "{} has nowhere to come from", a.id);
        }
    }

    #[test]
    fn every_asset_has_a_fallback_source() {
        // One source is one outage away from the feature not existing, which is
        // the risk this whole phase exists to reduce.
        for a in ASSETS {
            assert!(a.sources.len() >= 2, "{} has a single source", a.id);
        }
    }

    #[test]
    fn sources_are_https() {
        // These are weights that get loaded and executed. Plain HTTP would
        // leave the digest as the only thing between us and a hostile network.
        for a in ASSETS {
            for s in a.sources {
                assert!(s.starts_with("https://"), "{}: {s}", a.id);
            }
        }
    }

    #[test]
    fn non_redistributable_assets_never_point_at_us() {
        // ADR-0005 as a test rather than a promise. When MOSS lands, this is
        // what stops "downloads keep failing, let's mirror it ourselves" from
        // voiding the one mitigation the licensing position rests on.
        for a in ASSETS.iter().filter(|a| !a.self_hosting_allowed) {
            for s in a.sources {
                for ours in OUR_HOSTS {
                    assert!(
                        !s.contains(ours),
                        "{} may not be served from {ours}: {s}",
                        a.id
                    );
                }
            }
        }
    }

    #[test]
    fn the_first_tier_stays_small_enough_for_the_split_to_mean_something() {
        // If synthesis ever migrates into tier one, tiering has stopped meaning
        // anything and the first useful moment moves by hundreds of megabytes.
        // 400 MB is the ceiling that keeps the split honest, not a measurement.
        let t1 = tier_bytes(Tier::Transcribe);
        assert!(t1 > 0);
        assert!(t1 < 400 * 1024 * 1024, "tier one is {t1} bytes");
    }

    #[test]
    fn transcription_needs_a_model_its_tokens_and_a_vad() {
        // Any one of the three missing leaves dictation non-functional, which
        // is what puts them in the same tier.
        let ids: Vec<_> = of_tier(Tier::Transcribe).map(|a| a.id).collect();
        for needed in ["sensevoice-model", "sensevoice-tokens", "silero-vad"] {
            assert!(ids.contains(&needed), "tier one is missing {needed}");
        }
    }

    #[test]
    fn mirrors_come_before_upstream() {
        for a in ASSETS {
            let first = host_of(a.sources[0]);
            let last = host_of(a.sources[a.sources.len() - 1]);
            assert_ne!(first, last, "{}: both ends are {first}", a.id);
            assert!(
                matches!(
                    last,
                    "huggingface.co" | "raw.githubusercontent.com" | "github.com"
                ),
                "{}: the last resort should be upstream, got {last}",
                a.id
            );
        }
    }

    #[test]
    fn lookup_by_id_and_by_tier_agree_with_the_table() {
        assert!(by_id("sensevoice-model").is_some());
        assert!(by_id("no-such-asset").is_none());
        assert_eq!(
            of_tier(Tier::Speak).count() + of_tier(Tier::Transcribe).count(),
            ASSETS.len(),
            "an asset belongs to no tier"
        );
    }

    #[test]
    fn tier_ids_round_trip() {
        for t in [Tier::Transcribe, Tier::Speak] {
            assert_eq!(Tier::parse(t.id()), Some(t));
        }
        assert_eq!(Tier::parse("everything"), None);
    }
}
