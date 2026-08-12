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
| `capture.rs` | ffmpeg capture+encode arguments for all three platforms, and the Annex-B reader |
| `ice.rs` | STUN/TURN configuration, validated when it is read rather than when a call fails |

The first two are sans-IO and unit-tested, including a negotiation between two
endpoints passing nothing but serialized `SignalFrame`s. `driver.rs` is split
the same way: `pump` is pure and testable by hand, `run` owns the socket.

`tests/loopback.rs` is the one test that cannot be sans-IO — ICE has to probe,
DTLS has to handshake and SCTP has to open, and SDP that looks plausible and
never connects would pass everything else here. Two peers on real UDP sockets
connect, the channel opens on both, and 3001 bytes of non-ASCII cross it intact
in both directions. An encoded H.264 access unit crosses the video track the
same way and arrives marked as a keyframe — a track that negotiates and never
delivers would pass every unit test in the crate.

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
- **Proof on a real network.** Everything here is exercised on loopback, where
  there is no NAT to traverse. Hole-punching, and how often it fails, can only
  be measured between two machines on different networks.
- **STUN/TURN gathering.** `ice.rs` validates the configuration and the driver
  accepts trickled candidates, but nothing yet asks a STUN server for a
  reflexive address or allocates a TURN relay — so today only host candidates
  are offered, which works on a LAN and not across the internet.

The channel payloads, at least, need no new protocol.
`crates/daemon/src/remote/media_frame.rs` already frames a range as bytes and
the portal already decodes it; the data channel moves opaque blobs, so the same
framing crosses either path and neither end has to learn a second one.

## Wired, behind a feature

`nevoflux-daemon` depends on this only under its `webrtc` feature, off by
default: `str0m` pulls ~100 crates and the path is not proven on real networks
yet. With the feature on, a head offers a connection as soon as a portal
attaches (`PortalGateway::offer_peer_connection`), answers and candidates route
through `remote::rtc_peer`, and the session's path moves to `Peer` only while
the channel is genuinely open — back to `Relay` the moment it is not.

Both configurations are built and tested. Without the feature every range takes
the relay, exactly as it does today.

## Testing notes

`cargo test -p nevoflux-rtc-transport --features tokio-driver` covers everything
here, including the loopback connection. Without the feature the driver loop and
its integration test are skipped.

### The portal side

Landed and released: `nevoflux-portal` answers an offer, trickles candidates,
takes bytes on the data channel and the screencast on a track, and keeps
signalling away from chat subscribers. It is inert until a head offers.

### Capture, and what is actually untested

`crates/computer` takes *screenshots* — one frame, PNG-encoded — which at 30 fps
saturates a core before anything reaches the network. Screencast needs the
platform capture path wired to a hardware encoder, and writing that three times
is three capture backends plus three encoder integrations. ffmpeg has all six
and this repository already depends on it, so capture and encode are one child
process and what lives here is the argument list and the parser for what comes
back.

Both of those are pure, and both are tested for **all three platforms from
whichever one you are on** — the arguments are the part that fails silently, and
leaving two of three untested because the machine is the third is how a
screencast ends up permanently a second behind with nothing in the logs.

What is genuinely untested is running it: no CI machine and not the current
development box, where the Tesla T4 is passthrough and headless (MCDM) so there
is no desktop for DXGI Desktop Duplication to copy. That needs a machine with a
real display output, per platform.

### TURN

`ice.rs` validates configuration and reports whether a deployment can relay at
all. It picks no provider — which service, at what cost, with credentials issued
how, are deployment questions, and answering them in code answers them for every
deployment. An empty list is legal and means host candidates only.

Worth knowing before shipping: STUN alone works perfectly in testing and fails
for roughly a fifth of real users, because a symmetric NAT makes a reflexive
address useless to the far end. `can_relay` exists to make that checkable.
