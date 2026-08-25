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
    /// The same engine, in the release that can read Chinese.
    ///
    /// Its own tier rather than an upgrade folded into `Speak` for the reason
    /// the split exists at all: this is 375 MB against that tier's 120, and
    /// someone who only ever hears English answers should not pay it. It is
    /// self-sufficient — the v1.1-zh bank carries English voices too — so
    /// `Speak` becomes redundant once this arrives, not a prerequisite.
    ///
    /// Not merged into `SpeakMultilingual` either: MOSS is a different engine
    /// with 20 languages and 717 MB, and it is the one that gets ruled out on
    /// slow machines. Two languages that always work is a different offer from
    /// twenty that might not.
    SpeakChinese,
    /// The voice people actually want: multilingual, natural, and 717 MB.
    ///
    /// Its own tier rather than part of `Speak` because the two are
    /// alternatives, not companions — Kokoro is the fallback that runs when
    /// this is missing or too slow, and 120 MB of fallback should not be
    /// withheld until 717 MB of primary has arrived.
    SpeakMultilingual,
}

impl Tier {
    /// Every tier. `id()` below is an exhaustive match, so the compiler stops
    /// a new variant from being added without being named — but nothing forces
    /// it into this list, which is why the test that walks it also checks that
    /// every asset was reached.
    pub const ALL: &'static [Tier] = &[
        Tier::Transcribe,
        Tier::Speak,
        Tier::SpeakChinese,
        Tier::SpeakMultilingual,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Tier::Transcribe => "transcribe",
            Tier::Speak => "speak",
            Tier::SpeakChinese => "speak-chinese",
            Tier::SpeakMultilingual => "speak-multilingual",
        }
    }

    pub fn parse(s: &str) -> Option<Tier> {
        match s {
            "transcribe" => Some(Tier::Transcribe),
            "speak" => Some(Tier::Speak),
            "speak-chinese" => Some(Tier::SpeakChinese),
            "speak-multilingual" => Some(Tier::SpeakMultilingual),
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

const MOSS_TTS: &str = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX";
const MOSS_CODEC: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX";

/// Build one MOSS entry. The two repositories are laid out identically, and
/// writing ten of these by hand invites exactly one typo in exactly one URL.
macro_rules! moss {
    ($id:literal, $repo:expr, $mirror:literal, $file:literal, $bytes:literal, $sha:literal) => {
        Asset {
            id: $id,
            tier: Tier::SpeakMultilingual,
            file: $file,
            bytes: $bytes,
            sha256: $sha,
            sources: &[
                concat!("https://hf-mirror.com/", $mirror, "/resolve/main/", $file),
                concat!("https://huggingface.co/", $mirror, "/resolve/main/", $file),
            ],
            // ADR-0005. The upstream root LICENSE has since been published as
            // Apache-2.0, which is the condition the model card's clause names
            // — but the card still carries the clause, and pointing at someone
            // else's URL costs nothing while hosting the bytes cannot be undone.
            // Flip this only as a decision, never as a convenience.
            self_hosting_allowed: false,
            upstream: $repo,
        }
    };
}

const KOKORO_ZH_REPO: &str = "onnx-community/Kokoro-82M-v1.1-zh-ONNX";

/// Build one v1.1-zh voice entry.
///
/// The 103 of them differ only by name: the same 522,240 bytes (510 style rows
/// of 256 f32 — the shape `voices.rs` rejects anything else against), the same
/// directory, the same URL. Writing that out by hand is 103 chances to mistype
/// one URL and learn about it months later, when one voice alone never arrives.
macro_rules! kokoro_zh_voice {
    ($name:literal, $sha:literal) => {
        Asset {
            id: concat!("kokoro-zh-voice-", $name),
            tier: Tier::SpeakChinese,
            // A subdirectory, because this release ships voices as separate
            // files rather than v1.0's single bundle. `fetch` creates the
            // parent, and `kokoro.rs` resolves the bank by this directory.
            file: concat!("kokoro-voices-v1.1-zh/", $name, ".bin"),
            bytes: 522_240,
            sha256: $sha,
            sources: &[
                concat!(
                    "https://hf-mirror.com/onnx-community/Kokoro-82M-v1.1-zh-ONNX",
                    "/resolve/main/voices/",
                    $name,
                    ".bin"
                ),
                concat!(
                    "https://huggingface.co/onnx-community/Kokoro-82M-v1.1-zh-ONNX",
                    "/resolve/main/voices/",
                    $name,
                    ".bin"
                ),
            ],
            self_hosting_allowed: true,
            upstream: KOKORO_ZH_REPO,
        }
    };
}

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
    // ── Kokoro v1.1-zh — the release that can read Chinese (P1) ─────────
    //
    // fp32 at 324 MB, which is not an oversight. The repository also publishes
    // int8 (121 MB) and fp16 (156 MB), and both were measured on this hardware
    // before this entry was written, same text, same voice, same thread count:
    //
    //     fp32   RTF 0.428x   audio RMS 0.061      <- what is pinned here
    //     fp16   RTF 0.417x   every sample zero (NaN out of the CPU provider)
    //     int8   RTF 3.17x    audio fine
    //
    // int8 is seven times slower than fp32 — dynamic quantisation inserts
    // Quantize/Dequantize nodes the CPU provider will not fuse — and 3.17x is
    // far past the 0.85 budget that decides whether an engine may speak at
    // all. fp16 is the worse trap: right duration, right speed, silence.
    // Copying v1.0's int8 choice would have shipped one of those two.
    Asset {
        id: "kokoro-zh-model",
        tier: Tier::SpeakChinese,
        // `kokoro.rs::resolve_release` prefers this pair over v1.0 whenever
        // both halves are present, so installing this tier is what switches
        // the engine to the one that can say a Chinese word.
        file: "kokoro-v1.1-zh.onnx",
        bytes: 339_369_442,
        sha256: "94b973941b1852754f979be5d5e20be666d5c81d9bb886b88ae1dc85c9b895ca",
        sources: &[
            "https://hf-mirror.com/onnx-community/Kokoro-82M-v1.1-zh-ONNX/resolve/main/onnx/model.onnx",
            "https://huggingface.co/onnx-community/Kokoro-82M-v1.1-zh-ONNX/resolve/main/onnx/model.onnx",
        ],
        self_hosting_allowed: true,
        upstream: KOKORO_ZH_REPO,
    },
    Asset {
        // `Synthesizer::new` looks for this beside the model under the model's
        // own stem. It is release-specific and being wrong is silent: v1.1-zh
        // renumbered the letters, so v1.0's built-in table maps this model's
        // phonemes to the wrong ids rather than to none — noise, not an error.
        id: "kokoro-zh-vocab",
        tier: Tier::SpeakChinese,
        file: "kokoro-v1.1-zh.tokenizer.json",
        bytes: 4_944,
        sha256: "5715a60b09d5e4b9074435d68c6ccd5675b9d48b220e109fdea3cda681e23d15",
        sources: &[
            "https://hf-mirror.com/onnx-community/Kokoro-82M-v1.1-zh-ONNX/resolve/main/tokenizer.json",
            "https://huggingface.co/onnx-community/Kokoro-82M-v1.1-zh-ONNX/resolve/main/tokenizer.json",
        ],
        self_hosting_allowed: true,
        upstream: KOKORO_ZH_REPO,
    },
    // The voice bank: 100 Chinese and 3 English, 522 KB each. All of them,
    // because a partial bank makes the dropdown and `pick_voice`'s automatic
    // choice differ from the release the model was published with.
    kokoro_zh_voice!("af_maple", "bd5b230b916ea98c67a3a7a833a3ce43e535ce56a81503f4166ee9390b9ddeeb"),
    kokoro_zh_voice!("af_sol", "87c1d6d6a3f13f89ed54a6a67cc1ff75b926aebc3a2aac3086e6dd109a8147e6"),
    kokoro_zh_voice!("bf_vale", "2536f7922d31d96e994135ac2bb73f5a3c01326476200513c57d988822c6ca4d"),
    kokoro_zh_voice!("zf_001", "0a89ec12bb93fb9c74077924daf02568baad64e1f869389f5aaee01a386035f8"),
    kokoro_zh_voice!("zf_002", "452f96e1e3c20b14b228b5336a8d7e833b105f837d98ef53b4ddfce18eed39bf"),
    kokoro_zh_voice!("zf_003", "a70654663d013a700f8afe42d57d1ce03ff49af8bdedbbb56c7e3da6e820b788"),
    kokoro_zh_voice!("zf_004", "81fe28be1c496ccffa73078d5c2e6d9cc5e5f91e354d6ae0f494a9c28a42064d"),
    kokoro_zh_voice!("zf_005", "a91d2c2adecbb7c191dac4fa35e213e8d370e0c733a9bf36797f513924dcb84c"),
    kokoro_zh_voice!("zf_006", "52dee405a0e609d6d2869eef05984745db435a73aea49db7418494d5cb53bf9d"),
    kokoro_zh_voice!("zf_007", "23b6c4312208de7f195e4e78eef0ec9e498cd52907b1678ecb7a0a1996a51573"),
    kokoro_zh_voice!("zf_008", "23e20843a71fb8c5f6b984c8d4e74dd590251aeb8f620cb07a57e1485d726b0c"),
    kokoro_zh_voice!("zf_017", "9c6513e77a8efb7172a4e4ebeb48b3b010152d8cd6a0f2fa1adb353248fd91ed"),
    kokoro_zh_voice!("zf_018", "a9ee6566c7500ce90cbd177f1d6f6cae533356e5f0e37edf3adbd0a8f2d44020"),
    kokoro_zh_voice!("zf_019", "ab5906846335ac1227c47fd04be8b7b10a47c3ee77729296cc4c0f3f8fe79073"),
    kokoro_zh_voice!("zf_021", "c16f939c566f786150be3e3cb6a61f7fd8d4214a26a46d3e80e1d61de06d54ec"),
    kokoro_zh_voice!("zf_022", "c1cde40b2fd522355d2ae152d06d1033b04ad3528b7273cf25dd68f28a5563db"),
    kokoro_zh_voice!("zf_023", "d02846dc3b8a89b4f81634b407ce8dd30d80db40fc952d54f05853a8e26a2190"),
    kokoro_zh_voice!("zf_024", "2d0f2df7b25482f6870cfbe8e83c7d1e74ed8ef8ae72df749207ea9cec7c9102"),
    kokoro_zh_voice!("zf_026", "2a2434fed129796276c61947fc65bf7dadd569468ea4edaca81f3d59802bdc32"),
    kokoro_zh_voice!("zf_027", "9301e7a0b80eedff7f314076e07f9eb74b687afb7fcb2c64ca94168469858aae"),
    kokoro_zh_voice!("zf_028", "8d15e55ad315708010c1528c633828e3179d5cbf0bfcd2db0fec27f63001eaec"),
    kokoro_zh_voice!("zf_032", "2ab0b9b1fe2f1e4cfa08b0126581df3316c12aadeadc51136c826255debd2058"),
    kokoro_zh_voice!("zf_036", "b72873066ef9d57d4ea0605d32a82b2655010af51edbaecd6f926250d84efa4e"),
    kokoro_zh_voice!("zf_038", "569bb54ebf05c00c89eaf56218992474a8a0494d8f94a14e933ae6a5ea9f7e31"),
    kokoro_zh_voice!("zf_039", "92ad68f36d16860dcf125fdf87a4572c7ab6d7efbe6a7197d6772242a2cbbb56"),
    kokoro_zh_voice!("zf_040", "bae25bd2c4dd717ce65effa2350b28a55a73f53a0bcc2cca5876a15e984eaa2d"),
    kokoro_zh_voice!("zf_042", "713160091b44d962ab73e3bf66f0c2f0b9677aaf0c2f0e62faa94ed451540d1e"),
    kokoro_zh_voice!("zf_043", "7d257954b651436e354712738b22e8365ed908923a5d0f154690188b2c73ea78"),
    kokoro_zh_voice!("zf_044", "e166c8b2a680d3f51f1765cedc4ed7a29308ea40081b03347d5cd95f52119cc0"),
    kokoro_zh_voice!("zf_046", "73e8881e76670cfe803c8ab0ccc6b5daae22efdc3a67f8e0e80658a4be21d3af"),
    kokoro_zh_voice!("zf_047", "620f92a680c5a28bd8eb601738721047b73008bdf5b880f34075181afbfdcb62"),
    kokoro_zh_voice!("zf_048", "24e0aa66d1cb264e86adac5ea143f7fb0a9672a8b237ba1ad017aacc1fb0290a"),
    kokoro_zh_voice!("zf_049", "4808ba6dfc1c13af9921bb5072faf875b69bc54777281eb549592b1aa7e8a4bc"),
    kokoro_zh_voice!("zf_051", "64df55d707d3c28dec15c5c9051653b6a6cbbbf85c52e56183d3922e53415e76"),
    kokoro_zh_voice!("zf_059", "f12fd6e34f602445173f930c4f41b4d064e1a981b39b664591ad2476fb7425cc"),
    kokoro_zh_voice!("zf_060", "429ad8ffb6fc8b68d3b77f2ccafba5be50e4ec31d25f890d739a16449ca01748"),
    kokoro_zh_voice!("zf_067", "5669caef309a3746a47588d893d58fa61c721a94ca07e431c732ea9a2b4abc6a"),
    kokoro_zh_voice!("zf_070", "f225305611865b09dc8cb36439143c02a404299b00508e00676655bb117342e2"),
    kokoro_zh_voice!("zf_071", "454cd5aaa9458f3b028e973d19902a002eb59fc743eb94fb3c7c9d5d7a190a07"),
    kokoro_zh_voice!("zf_072", "2274b572b44723e80eee422a914c96b37ee42bf9b5ca60e7f656bede7860a1d3"),
    kokoro_zh_voice!("zf_073", "b8ecf6945ae7bb2a0353422b7af32d3b928c5fdccf3ea553bc61227a57bf4a2c"),
    kokoro_zh_voice!("zf_074", "8e6726977c7494dfd567c580963630834c0971f66111e06c9467e6f7d912ea1e"),
    kokoro_zh_voice!("zf_075", "84e70ca9aac5178ceaeb1eae95dd77278d71e60cfacaf9af99852922b4355560"),
    kokoro_zh_voice!("zf_076", "58351a1be5aa04f874bea0e5228fc0e0047dc2ee801e915d4f59513ca60a3194"),
    kokoro_zh_voice!("zf_077", "e2fd07f3b62e204cd6632696e2a975678e7e6aa5a9b2cb4927f94d875c113625"),
    kokoro_zh_voice!("zf_078", "82d591700a9ecb3cb14d3d05ec5aab18bd7a482d677bf6522cf96ed4c064b23b"),
    kokoro_zh_voice!("zf_079", "2df9a9fbbd39a54077a2c5073c04c7e685ce0342b34f22fc85c25a1f7a3954c7"),
    kokoro_zh_voice!("zf_083", "5557cbc381e0f2cd2a29759fdda25eca664492f1cfa3fcc241e7adeaf4164661"),
    kokoro_zh_voice!("zf_084", "2b2d4fdefa9a6c3d9f472d91600308be8b827a929955ec66afe81aca50c99968"),
    kokoro_zh_voice!("zf_085", "2451d63f576c9ffbe91acd11f2720b6e76dc75b293d7350a9620a28c23da9ef8"),
    kokoro_zh_voice!("zf_086", "c228ba018bf56d5de1339bde10dcbea7700797b85351e4d57ee5307ac7d3ee99"),
    kokoro_zh_voice!("zf_087", "e0347f1430b80781d910bf1499cf64d231c34b70d6e46f30f3598254d5114d44"),
    kokoro_zh_voice!("zf_088", "ee1e80cdeaffb55e1f6c41005c4966b8979b16bb808fa451739e06953d343c13"),
    kokoro_zh_voice!("zf_090", "fe90b3e651c3741edf6bc08a543c531c798ec53fab1fd944ab537e9257270c68"),
    kokoro_zh_voice!("zf_092", "ffd200defd2867f79886eb0e45540d53ed0dc00ceb222c5fa692f004958d3e11"),
    kokoro_zh_voice!("zf_093", "d736ac1738c5ba567f8e5e9bfac8d9a774e9e3a3d001b5976a5d005c9179f811"),
    kokoro_zh_voice!("zf_094", "b37b96e3c51d12f152c9476cae42396ac38cdab76f09cc935f2fbcd8fa012f42"),
    kokoro_zh_voice!("zf_099", "9582b2ffb5027c695f6775161a8856108fa68530c6c1e29d2354d455d9eaa937"),
    kokoro_zh_voice!("zm_009", "7b74d6ed22f201e2fa28758e78ce6197082779f2b80e69ea1bf877908609514a"),
    kokoro_zh_voice!("zm_010", "73b088f7e0dc47adca4d6a642ee68843df90ff56ec2800c29d96609989d6de0a"),
    kokoro_zh_voice!("zm_011", "ec4d7d934b9aa47e0e98e6ec802bb7b0a221be6f1b67d923d33b7082d9bbfe9f"),
    kokoro_zh_voice!("zm_012", "50d1986f71ea1a2b3ab1bfc0c95bedc59e8de45e650bc5bde87fd99ceb83cdfb"),
    kokoro_zh_voice!("zm_013", "2caa23e98910fb232a2bf6aab563666588a82bc37c99b67963ab304c5be66dff"),
    kokoro_zh_voice!("zm_014", "565aa48a99d8a196ebdd1c72f4a0e5760fed65aa95992c887385d3aad9e1d2f8"),
    kokoro_zh_voice!("zm_015", "e695c8d72b3eb4a864f4735db7f4f93de100ff62704e619edfddbebfafb07312"),
    kokoro_zh_voice!("zm_016", "4dc408a11f1e8925ff8bca40c885e7596d08be63c2c61571234b96170f8e6d1b"),
    kokoro_zh_voice!("zm_020", "2a25ca83ddfd003c0b82d97899ff38734db8faf48e5e4076d68c69aea17705c8"),
    kokoro_zh_voice!("zm_025", "cbb4cd1df85b5dde4cf742c61d9b93e179574935c1162d06604bc4be6a8e990d"),
    kokoro_zh_voice!("zm_029", "ff78cb9f64d43fe6179c9479257775bea94dc31eade9362b0c630749025d8ecd"),
    kokoro_zh_voice!("zm_030", "32f8fe4d5f626dfb67adb73499690fd4f5952763746efbcadcf2daad4a3560a6"),
    kokoro_zh_voice!("zm_031", "a5974232ed634be2ae40526101ee7c653120fdb354b1584f2bb38ed4fae4a39e"),
    kokoro_zh_voice!("zm_033", "758e64c15efad492beebf575eb96951e0500da980bf5e8a3850d078d0430dda1"),
    kokoro_zh_voice!("zm_034", "b371bef26b75c826de15c836103da3e037a44f605ffe744d68ad485d17e370ab"),
    kokoro_zh_voice!("zm_035", "8b1c603f6e1eae300ac1a6f1199a5bb1ce9814d3e1e61f7536d21b564d680476"),
    kokoro_zh_voice!("zm_037", "535ec47cb6ed66c203f61ace51eddbc7a13ecaca2458c1d8d5a928fb3db4b315"),
    kokoro_zh_voice!("zm_041", "8b7bd5649a00c62e87a6099eb31d683664bfb0fd55bcd5211da62e95b3b78ffc"),
    kokoro_zh_voice!("zm_045", "68d5e7f3415811a076326178c2d9f5d3aa168cf27deb6ae3f0395ab97ad45225"),
    kokoro_zh_voice!("zm_050", "7869f25a5e71ea9b67a1893777e375ac411bdbfb75feff5efe25fad2fc766c8d"),
    kokoro_zh_voice!("zm_052", "e775a71cd74c5462d5a6c3f0ff7681f04419053043ad2dcbabaa13d0c6de0b08"),
    kokoro_zh_voice!("zm_053", "a549dcf0456b2ba1515455e0a30d111b00d24e748880e03da1965cd4ff10baf7"),
    kokoro_zh_voice!("zm_054", "4f7aeb1627fa1d406bcc1a83e4eeb8b014509dca492dec0ae7dd4d0d1cdbe8b3"),
    kokoro_zh_voice!("zm_055", "3cf77a37899053b5a7c224f51a22371dc37d30667d87e1b481e760ac0f8f0da8"),
    kokoro_zh_voice!("zm_056", "d5f48143152f376a940182741589fb1f588387e09f1a9263760981f8702af816"),
    kokoro_zh_voice!("zm_057", "29cb23f82add956f25c7b663db2027d5a50abb7035c7cb15ee608e193a412f51"),
    kokoro_zh_voice!("zm_058", "0cafc2ef83710c4ce55b384107426cfdcb23aabf5699e106ff392f9571b4d4f3"),
    kokoro_zh_voice!("zm_061", "d57141bd480e7a4f1a3b5d3365ce53ac04cfe6dd9d0f2d3a052213f80657046d"),
    kokoro_zh_voice!("zm_062", "171d8811bfeb198714b0d3733ed740eb56be0aa3e6d2bd7b127904116a199f93"),
    kokoro_zh_voice!("zm_063", "6fbe326e5852cd9dbac7eb7e9c9989a604e51074d251a8ff8365e050c83b2a6e"),
    kokoro_zh_voice!("zm_064", "6e712d18758f2d8654c97fb274c23b5148dbafc1027ee6539446d4c8aff308cd"),
    kokoro_zh_voice!("zm_065", "77918a32b75bf7902a6cee310cd22fb16de7ecbb9aa438a05039ebec73f9e0e8"),
    kokoro_zh_voice!("zm_066", "fcdd1d4c8b2808c4456418c76ed2321910b3f1b5c2646365af4ee0e8c6c53c4d"),
    kokoro_zh_voice!("zm_068", "ae265665e29656ea0d151f3315c649d892ecf053391c3c3bec5f10bf18531a58"),
    kokoro_zh_voice!("zm_069", "54ac2dc25ee5d1dfa72f777ff538faf69033705019a9e15f3138deb79a965774"),
    kokoro_zh_voice!("zm_080", "cbbcae6bfd6c2f3875b2163fffc8851aeae1b8bad2f1006a94ea4e6f9b44ae9e"),
    kokoro_zh_voice!("zm_081", "ffc1341f044c95ad8ef598079df9e3a85d81d093268ae8422848a9b81d386fd2"),
    kokoro_zh_voice!("zm_082", "7351d7652d97ec5fcb7632b317d7c4e062fff3cf4ed0246167d3d15e97046c01"),
    kokoro_zh_voice!("zm_089", "f4315ceab376b4082cc9cdd8028d37f9aa7d43ad67ddb85902a247df0913ca89"),
    kokoro_zh_voice!("zm_091", "6ab4a21b702ec57161f48e350de3fa0b0bd1d13a890657398cfa21452d85a9b3"),
    kokoro_zh_voice!("zm_095", "4d4dac7245ea4c4c4740137fca229932055e376aed2b1ad7cd0e3e3859a13b75"),
    kokoro_zh_voice!("zm_096", "e6b76d86ca459d8ce87d2a41c812319b11e9563ecf1ba80a4e86a368d6c02fbc"),
    kokoro_zh_voice!("zm_097", "e60ec05738261c2eb13c43f7429ece6b32e0992223b0a6336a9fcdec90fb8da4"),
    kokoro_zh_voice!("zm_098", "7dd142334e863af31c7e0d0fbe491b8d54083ff5322441d550fcfc5e4111480d"),
    kokoro_zh_voice!("zm_100", "112de52f1aab3b370fd01f4e2e1d8bb37fee2725e77e14b81c802d50687e0e6e"),
    // ── MOSS-TTS-Nano (P1) ───────────────────────────────────────────────
    //
    // Filenames are upstream's and must stay that way: the `.onnx` graphs
    // reference their `.data` blobs **by name** through ONNX external data, so
    // a tidier local name would load a model with no weights in it.
    //
    // The streaming decoder, the encoder (voice cloning), and the per-channel
    // sampling path are all deliberately absent — 43 MB and 42 MB of files this
    // does not call yet. They can be added the day something uses them.
    moss!(
        "moss-manifest",
        MOSS_TTS,
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "browser_poc_manifest.json",
        503_354,
        "097d80e993dc29f0bae427590b4f77084a161cb578b50d82c29f455d5faa9eee"
    ),
    moss!(
        "moss-tts-prefill",
        MOSS_TTS,
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "moss_tts_prefill.onnx",
        283_305,
        "d56126dcd0574c2f15d98fc6b35eda68d0386b5bd9c5e38e28548d6f2ea8f3db"
    ),
    moss!(
        "moss-tts-decode-step",
        MOSS_TTS,
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "moss_tts_decode_step.onnx",
        291_483,
        "698cbc2fc1c2feca16e5895614ed52bbb32ded10f236c076f477b2e69abf32d8"
    ),
    moss!(
        "moss-tts-sampled-frame",
        MOSS_TTS,
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "moss_tts_local_fixed_sampled_frame.onnx",
        471_262,
        "40cdb00efc171c450cf91468e01429caa41b0252222cd308e978f58fe354afa8"
    ),
    moss!(
        "moss-tts-global-weights",
        MOSS_TTS,
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "moss_tts_global_shared.data",
        440_813_568,
        "bce8312c3df6a44545302cae229b61054fe0672e0b252ba59cba47adeed831dc"
    ),
    moss!(
        "moss-tts-local-weights",
        MOSS_TTS,
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "moss_tts_local_shared.data",
        229_678_080,
        "bae7782032c0fb12490ab42afe009f87ae6c75a0f0596fc7b5c08e4d5ee93916"
    ),
    moss!(
        "moss-tts-tokenizer",
        MOSS_TTS,
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "tokenizer.model",
        470_897,
        "c353ee1479b536bf414c1b247f5542b6607fb8ae91320e5af1781fee200fddff"
    ),
    moss!(
        "moss-codec-decode",
        MOSS_CODEC,
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
        "moss_audio_tokenizer_decode_full.onnx",
        681_902,
        "0fbbafe3fd4afa2a019af5c5ced204af6e2d1db044fa40f021525d2aee95b4ac"
    ),
    moss!(
        "moss-codec-weights",
        MOSS_CODEC,
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
        "moss_audio_tokenizer_decode_shared.data",
        44_198_912,
        "e69d52e0f4e84ca27850557ee54face46632d3a5a16c89bd246c7c408466dcad"
    ),
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
        let reached: usize = Tier::ALL.iter().map(|t| of_tier(*t).count()).sum();
        assert_eq!(
            reached,
            ASSETS.len(),
            "an asset belongs to a tier that Tier::ALL does not list"
        );
    }

    #[test]
    fn every_tier_has_something_in_it() {
        // An empty tier reaches the UI as a row offering a 0 MB download.
        for t in Tier::ALL {
            assert!(of_tier(*t).count() > 0, "{} is empty", t.id());
            assert!(tier_bytes(*t) > 0, "{} costs nothing", t.id());
        }
    }

    #[test]
    fn the_multilingual_voice_is_not_bundled_into_the_fallback_tier() {
        // They are alternatives: Kokoro runs when MOSS is missing or too slow.
        // Merging them would withhold 120 MB of working fallback until 717 MB
        // of primary had arrived.
        assert!(tier_bytes(Tier::SpeakMultilingual) > tier_bytes(Tier::Speak));
        for a in of_tier(Tier::Speak) {
            assert!(
                !a.id.starts_with("moss-"),
                "{} is in the fallback tier",
                a.id
            );
        }
    }

    #[test]
    fn moss_is_never_served_from_our_own_hosts() {
        // The general rule is enforced above; this pins the specific case the
        // rule exists for, so that deleting the flag on these entries fails
        // here by name rather than silently passing a vacuous loop.
        let moss: Vec<_> = ASSETS
            .iter()
            .filter(|a| a.id.starts_with("moss-"))
            .collect();
        assert_eq!(moss.len(), 9, "the MOSS asset set changed");
        for a in moss {
            assert!(!a.self_hosting_allowed, "{} may not be self-hosted", a.id);
        }
    }

    #[test]
    fn moss_keeps_upstream_filenames() {
        // The `.onnx` graphs reference their `.data` blobs by name through ONNX
        // external data. A tidier local name loads a model with no weights in
        // it — and reports nothing until inference produces silence.
        for a in ASSETS.iter().filter(|a| a.id.starts_with("moss-")) {
            let remote = a.sources[0].rsplit('/').next().unwrap();
            assert_eq!(a.file, remote, "{} is renamed on the way in", a.id);
        }
    }

    #[test]
    fn tier_ids_round_trip() {
        for t in Tier::ALL {
            assert_eq!(Tier::parse(t.id()), Some(*t));
        }
        assert_eq!(Tier::parse("everything"), None);
    }

    /// The Chinese tier has to arrive as a whole or not at all.
    ///
    /// `resolve_release` only accepts v1.1-zh when the model *and* the voice
    /// directory are both there, and `Synthesizer::new` reads the vocabulary
    /// from beside the model. Any one of the three missing leaves the tier
    /// downloaded and the engine still falling back to English — the exact
    /// shape of failure this tier exists to end.
    #[test]
    fn the_chinese_tier_carries_a_model_its_vocabulary_and_a_bank() {
        let ids: Vec<_> = of_tier(Tier::SpeakChinese).map(|a| a.id).collect();
        assert!(ids.contains(&"kokoro-zh-model"), "no model");
        assert!(ids.contains(&"kokoro-zh-vocab"), "no vocabulary");

        let voices: Vec<_> = of_tier(Tier::SpeakChinese)
            .filter(|a| a.file.starts_with("kokoro-voices-v1.1-zh/"))
            .collect();
        assert!(voices.len() > 50, "only {} voices", voices.len());
        // The point of the tier. An English-only bank here would download
        // 375 MB and still not say a Chinese word.
        assert!(
            voices
                .iter()
                .any(|a| a.file.contains("/zf_") || a.file.contains("/zm_")),
            "the Chinese tier has no Chinese voice"
        );
    }

    /// The vocabulary must land where the synthesizer looks for it.
    ///
    /// `Synthesizer::new` derives it as `model_path.with_extension(...)`, so
    /// the two filenames are coupled through code that never mentions either
    /// of them. Renaming one and not the other loads the built-in v1.0 table
    /// against a v1.1-zh model, which does not fail — it mispronounces.
    #[test]
    fn the_chinese_vocabulary_sits_beside_its_model() {
        let model = by_id("kokoro-zh-model").expect("model entry");
        let vocab = by_id("kokoro-zh-vocab").expect("vocab entry");
        let stem = model.file.strip_suffix(".onnx").expect("model is .onnx");
        assert_eq!(vocab.file, format!("{stem}.tokenizer.json"));
    }

    /// Whatever else changes, the Chinese release stays the one that was
    /// actually measured. int8 (121 MB) runs at 3.17x real time and fp16
    /// (156 MB) emits nothing but zeroes; both are tempting on size alone and
    /// both are unusable. The byte count is the cheapest way to notice that
    /// someone swapped in a smaller export.
    #[test]
    fn the_chinese_model_is_the_variant_that_was_measured() {
        let model = by_id("kokoro-zh-model").expect("model entry");
        assert_eq!(model.bytes, 339_369_442, "not the fp32 export");
        for s in model.sources {
            assert!(s.ends_with("/onnx/model.onnx"), "quantised export: {s}");
        }
    }
}
