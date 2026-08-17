# experiments/

Evaluation harnesses kept for their findings. Each answers one question that a
design decision hung on, and records the answer next to the code that produced
it — so the conclusion can be re-checked rather than taken on trust.

Distinct from `spike/`, which `.gitignore` reserves for throwaway exploration.
Nothing here is in the workspace: these pull dependency trees (`webrtc` alone is
~250 crates) that have no business in the daemon's build graph while the
decision they inform is still open.

| | question | answer |
|---|---|---|
| [`webrtc-rs-twcc/`](webrtc-rs-twcc/) | Can `webrtc-rs` drive an encoder bitrate from TWCC? | No estimator exists, and a library gap silently kills receive-side feedback entirely |
| [`str0m-bwe/`](str0m-bwe/) | Does any Rust WebRTC stack ship a working one? | `str0m` does; measured good ramp, collapse, and recovery |

## Background: the remote-media transport

In remote-control mode every byte — chat, screenshots, TTS, video — travels the
same Cloudflare Durable Object relay, as base64 inside JSON frames. That is
sound for small, ephemeral media and the wrong shape for large files: no
transport-level flow control, no caching (one 158 MB film was observed pulling
425 MB), head-of-line blocking against chat, and 33% base64 overhead on a wire
that is already binary.

The plan these experiments inform:

- **B — improve the relay path.** Partly landed: media frames no longer retain
  their bytes in the resume buffer, and the buffer is bounded by bytes rather
  than frame count (`crates/daemon/src/remote/relay_protocol.rs`). Remaining
  work needs matching portal changes: a dedicated media channel and binary
  frames. B is also WebRTC's fallback for networks that block UDP, so it is
  worth finishing either way.
- **C — WebRTC.** P2P when it connects, real congestion control, and a media
  track for screencast on the same PeerConnection as the DataChannel. These
  experiments are what makes C costable; `str0m` is the base to build it on.
