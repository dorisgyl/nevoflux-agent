# str0m BWE spike

Answers the question `../webrtc-rs-twcc/` left open: **is there a Rust WebRTC
stack that gives us a usable bandwidth estimate without writing our own
congestion controller?**

Yes. `str0m` 0.22.0 ships one, and it holds up.

## What str0m already has

```
src/bwe/delay/{arrival_group,trendline,control}.rs   full delay-based controller
src/bwe/loss_controller.rs                           loss-based controller
src/bwe/{alr_detector,probe}/                        ALR detection + probing
src/pacer/leaky.rs                                   a real pacer
src/sctp/                                            DataChannel
tests/bwe/                                           21 simulation tests
```

The pacer matters more than it looks: goog_cc-style probing drives ramp-up by
asking a pacer to emit probe clusters at a chosen rate, so a port without one
has no working ramp-up regardless of how faithful the estimator is.

MIT OR Apache-2.0, `rust-version = 1.85` — builds on stable, unlike the
`webrtc-rs` stack.

The app-facing surface is small:

```rust
let rtc = Rtc::builder().enable_bwe(Some(initial_bitrate)).build(start);
rtc.bwe().set_desired_bitrate(desired);
// then read Event::EgressBitrateEstimate(BweKind::Twcc(bitrate))
```

The receiving peer must call `direct_api().enable_twcc_feedback()`.

## Running `nevoflux.rs`

It is written against str0m's own test harness (`tests/bwe/common.rs`) and the
`str0m-netem` emulator, neither of which is public API — so it is applied onto
str0m's source tree rather than shipped as a standalone crate:

```sh
VERSION=0.22.0
cargo fetch                       # anywhere, to unpack the source
SRC=$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" -maxdepth 2 \
        -type d -name "str0m-$VERSION" | head -1)
cp -r "$SRC" /tmp/str0m-spike && chmod -R u+w /tmp/str0m-spike
rm -f /tmp/str0m-spike/.cargo-checksum.json
cp nevoflux.rs /tmp/str0m-spike/tests/bwe/
echo 'mod nevoflux;' >> /tmp/str0m-spike/tests/bwe/main.rs

cd /tmp/str0m-spike
cargo test --release --test bwe --features _internal_test_exports nevoflux -- --nocapture
```

Upstream's own suite is worth running first (`cargo test --test bwe --features
_internal_test_exports`) — 21 tests, all passing here in 42s.

## What `nevoflux.rs` measures

Unlike str0m's tests, which step the send rate through a fixed plan, this one
**closes the loop**: every second it re-derives the encoder's send rate from the
current estimate, which is what an encoder following BWE actually does. A fixed
ladder cannot expose an unstable controller; an adaptive sender can.

Three phases against a simulated home uplink, 8 Mbps desired:

```
  phase                  t(s)   estimate      sending
  20 Mbps uplink          1      7.99 Mbps    0.60 Mbps
                          6     15.08 Mbps    8.00 Mbps
                         25     15.08 Mbps    8.00 Mbps
  2 Mbps (contended)     26     15.08 Mbps    8.00 Mbps
                         31      2.85 Mbps    2.34 Mbps
                         36      2.21 Mbps    2.21 Mbps
                         50      2.94 Mbps    2.94 Mbps
  20 Mbps restored       51     15.65 Mbps    2.94 Mbps
                         56     15.08 Mbps    8.00 Mbps
                         80     15.08 Mbps    8.00 Mbps

  20 Mbps uplink       ends  15.08 Mbps   (low water   7.99)
  2 Mbps (contended)   ends   2.94 Mbps   (low water   2.01)
  20 Mbps restored     ends  15.08 Mbps   (low water  15.08)
```

- **Ramp**: 0.6 → 7.99 Mbps in one second, plateau by six, then flat.
- **Collapse**: ~5 seconds from 15.08 down to the 2 Mbps link.
- **Capacity**: low water 2.01 Mbps on a 2.00 Mbps link.
- **Recovery**: back inside one second, and it never sags again.

The 2.2–2.9 Mbps band during the degraded phase is AIMD working, not drift:
additive increase until the delay signal trips, then a multiplicative cut. The
test asserts it *touches* real capacity rather than asserting a tight ceiling —
a controller that stopped probing would look steadier and be worse, because that
is how a session ends up latched at 200 kbps after one bad minute.

## Consequence

The largest cost in the WebRTC option was owning and tuning a congestion
controller. On str0m that cost is gone, and the remaining work is ordinary
integration.

Still unverified, and only answerable on real networks and devices: BWE quality
outside the emulator, whether the media-track API suits screencast, and browser
interop. str0m is also pure sans-IO — the event loop is the caller's to drive.
