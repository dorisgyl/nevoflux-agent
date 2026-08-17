# webrtc-rs TWCC spike

Answers one question: **does `webrtc-rs` give us a bandwidth estimate we can
drive an encoder with?**

It does not — and the way it fails is worth keeping, because the failure is
silent.

## Running it

```sh
./setup.sh            # vendors a patched rtc-mdns; see below
cargo run --release
```

`setup.sh` is needed because `rtc-mdns` 0.20.2 calls `Ipv4Addr::from_octets`,
still unstable behind `ip_from` ([rust#131360]), so the dependency does not
build on stable. The script vendors a copy with `Ipv4Addr::from` instead. On
Windows run it from Git Bash.

[rust#131360]: https://github.com/rust-lang/rust/issues/131360

## What it found

### 1. There is no bandwidth estimator

`rtc-interceptor` 0.20.2 ships exactly three interceptors — `nack`, `report`,
`twcc`. No GCC, no trendline filter, no rate controller. Its own README marks
"RTCP feedback processing" as intentionally skipped. TWCC gives you feedback
*packets*; turning them into a number to encode at is left to the caller.

### 2. `bind_remote_stream` is never called for a declared-SSRC track

`rtc-0.20.2/src/peer_connection/handler/endpoint.rs`: `find_track_id_by_ssrc`
resolves a remote track and fires `OnTrack` **without calling
`interceptor.bind_remote_stream()`**, and `find_track_id` tries it before the
two paths that do bind (the MID-extension path and `handle_undeclared_ssrc`).

So for any track whose SSRC is declared in the SDP — the ordinary case — every
receive-side interceptor stays unbound. `TwccReceiverInterceptor.streams` is
empty, so each `handle_read` lookup misses, so its recorder is never built, so
it emits no feedback for the life of the connection. The NACK generator is dead
the same way.

The symptom gives nothing away. Media flows perfectly, `on_track` fires, the SDP
negotiates `a=rtcp-fb:102 transport-cc` and the extmap, the sender stamps every
outgoing packet — and not one RTCP packet ever comes back. The wire tap in this
spike is what makes it legible; it prints, side by side:

```
[OFFERER/sender]   bind_local_stream : [(14600958, [".../transport-wide-cc...#1"])]
                   RTP out: 79930 (with header extension: 79930)
[ANSWERER/receiver] bind_remote_stream: []            <-- the bug
                   RTP in : 79810 (with header extension: 79810)
                   RTCP out: 0
```

Upstream's own test suite steps around this: `rtcp_processing_webrtc2webrtc.rs`
gets sender-side RTCP only because the answerer explicitly calls
`write_rtcp(PLI)` on a timer, and its comment says it does so "without relying
on periodic Receiver Report generation."

### 3. The workaround is ~50 lines

`bind_remote_stream` is a public trait method, so the missing call can be made
from outside. `BindShimInterceptor` in `src/main.rs`, registered outermost,
binds each SSRC on first inbound RTP. With it, feedback appears immediately —
228 `TransportLayerCc` reports at ~10 Hz over the run.

### 4. With feedback flowing, two of GCC's three inputs are usable

Ramping the send rate 2 → 100 Mbps over loopback:

```
  target       sent     TWCC   covered       lost      arrival
    Mbps       pkts  reports      pkts          %         Mbps
       2        830       37       830      0.00%          1.7
       5       2079       37      2079      0.00%          4.4
      10       4157       37      4157      0.00%          8.8
      25      10405       38     10405      0.00%         20.8
      50      20828       39     20828      0.01%         40.7
     100      41653       40     41653      0.34%         75.4
```

`covered` equals `sent` at every rung — nothing is unaccounted for. Loss appears
at the knee. Arrival throughput tracks and then saturates.

The third input, the **one-way delay gradient**, this spike does *not* validate.
The `gradient` column its output prints is an artifact of the sender's own
pacing, not a network signal. Computing it for real needs the send timestamp of
each transport-wide sequence number, and `TwccSenderInterceptor` does not expose
that mapping — so a real GCC would need a replacement sender interceptor too.
`twcc::Recorder` is `pub(crate)`, so it cannot be reused either.

## Conclusion

Everything needed is reachable, but nothing above the raw feedback is provided.
Building on `webrtc-rs` means owning a congestion controller.

**See `../str0m-bwe/` — `str0m` already has one, so this is the road not taken.**
The spike is kept because the `bind_remote_stream` gap is worth reporting
upstream and worth recognising if we ever revisit this crate.
