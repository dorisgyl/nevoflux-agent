# nevoflux-rtc-transport

WebRTC transport for remote sessions. **Incomplete.** Read this before building
on it.

## Why

Everything a remote session sends crosses a Cloudflare Durable Object relay.
That is right for small ephemeral media and wrong for large files: no
transport-level congestion control, no caching (one 158 MB film was measured
pulling 425 MB), and every byte billed and latency-bound through one hop. A peer
connection, when it forms, is a direct path with real congestion control, and it
carries a media track for screencast on the same connection as the data.

Built on `str0m` rather than `webrtc-rs`, for reasons measured rather than
assumed — see [`experiments/`](../../experiments/) at the repository root.
`str0m` ships a working, test-covered GCC estimator and a pacer; `webrtc-rs`
ships neither, and separately never calls `bind_remote_stream` for a
declared-SSRC track, which silently kills all receive-side feedback.

## What is here

| | |
|---|---|
| `signal.rs` | The four signalling frames, and the rule that keeps them safe |
| `connection.rs` | Offer / answer over those frames, with a data channel |
| `driver.rs` | The pump, and a tokio loop that owns a socket (`tokio-driver`) |

The first two are sans-IO and unit-tested, including a negotiation between two
endpoints passing nothing but serialized `SignalFrame`s. `driver.rs` is split
the same way: `pump` is pure and testable by hand, `run` owns the socket.

`tests/loopback.rs` is the one test that cannot be sans-IO — ICE has to probe,
DTLS has to handshake and SCTP has to open, and SDP that looks plausible and
never connects would pass everything else here. Two peers on real UDP sockets
connect, the channel opens on both, and 3001 bytes of non-ASCII cross it intact
in both directions.

### The property everything else rests on

WebRTC's confidentiality rests on the DTLS fingerprints in the offer and answer.
Whoever controls the signalling path can substitute both, terminate DTLS in the
middle, and read everything — and the relay is exactly a party in the middle.

It cannot do that here, because the wire carrying signalling is sealed with a
key derived from a pairing code the relay never sees. That is load-bearing
rather than incidental, so `SignalGuard` **refuses to negotiate at all** on an
unsealed channel rather than quietly setting up a session the relay could sit
inside.

## What is not here

Each is substantial:

- **ICE against a real network.** The loopback test proves the driver; it does
  not prove hole-punching. Trickled candidates are plumbed
  (`add_remote_candidate`) but nothing generates srflx or relay ones yet.
- **Wiring into the daemon.** Routing `rtc_*` frames off the relay wire,
  preferring the channel over the relay once it is up, and falling back when it
  drops. Nothing routes between the two paths yet.
- **The media track.** Screencast over RTP.
- **Screen capture.** Three platform backends. `crates/computer` is
  screenshot-only — `capture_screen()` takes one full frame and PNG-encodes it,
  which at 30 fps would saturate a core. Real capture means DXGI Desktop
  Duplication on Windows, ScreenCaptureKit on macOS, and X11/PipeWire on Linux;
  the cheapest route is one ffmpeg process doing capture and encode together
  (`ddagrab`/`avfoundation`/`x11grab` → `h264_nvenc`/`videotoolbox`/`vaapi`),
  reading Annex-B off its stdout.
- **TURN.** Roughly 10–20% of connections need it — symmetric NAT at both ends,
  CGNAT, corporate firewalls. Not optional in practice.
- **The fallback.** When no connection forms, the session has to stay on the
  relay path. That path exists and works; nothing routes between them yet.
- **The portal side.** `RTCPeerConnection`, SDP exchange through the sealed
  wire, a `ChatTransport` over the data channel, `<video srcObject>`.

The channel payloads, at least, need no new protocol.
`crates/daemon/src/remote/media_frame.rs` already frames a range as bytes and
the portal already decodes it; the data channel moves opaque blobs, so the same
framing crosses either path and neither end has to learn a second one.

## Not in the daemon's build

Deliberately. `str0m` pulls ~100 crates, and this is not usable yet — wiring it
in would cost every build for something no session can reach. It is a workspace
member so it compiles and tests in CI, and nothing depends on it.

## Testing notes

`cargo test -p nevoflux-rtc-transport --features tokio-driver` covers everything
here, including the loopback connection. Without the feature the driver loop and
its integration test are skipped.

The capture path cannot be verified on the current development box: the Tesla T4
is passthrough and headless (MCDM), so there is no desktop for DXGI Desktop
Duplication to copy and the Windows-optimal path is untestable there. It needs a
machine with a real display output.
