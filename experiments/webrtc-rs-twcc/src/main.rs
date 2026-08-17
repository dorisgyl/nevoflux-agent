//! Spike: can TWCC feedback in webrtc-rs 0.20.2 actually drive an encoder bitrate?
//!
//! Source archaeology already settled half of it: `rtc-interceptor` ships `nack`,
//! `report`, and `twcc` and nothing else — there is no GCC, no trendline filter,
//! no rate controller. So nobody hands you a bandwidth estimate.
//!
//! What is left to prove is whether the *raw material* is reachable: does a
//! sending peer actually receive `TransportLayerCc` feedback about its own
//! stream, and does that feedback carry enough to compute the three signals a
//! GCC needs — loss, arrival throughput, and the one-way delay gradient?
//!
//! Topology: two in-process peers over loopback UDP. The offerer sends a video
//! track at a *known, ramped* bitrate; the answerer receives. The offerer parses
//! inbound TWCC and prints, per rung, what it could estimate from it.
//!
//! The payload is synthetic rather than real H.264 on purpose. TWCC measures the
//! transport, and it cannot tell an encoded frame from a filler byte — but a
//! precisely-paced sender lets us know `t_send` analytically, which is what
//! makes the delay gradient computable without a send-time map. See the
//! `SEND-TIME MAP` note at the bottom of this file: that shortcut is exactly the
//! thing production would not have.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rtc::interceptor::{Interceptor, Packet, Registry, StreamInfo, TaggedPacket, interceptor};
use rtc::media_stream::MediaStreamTrack;
use rtc::peer_connection::configuration::interceptor_registry::{
    configure_nack, configure_rtcp_reports, configure_twcc,
};
use rtc::peer_connection::configuration::media_engine::{MIME_TYPE_H264, MediaEngine};
use rtc::rtcp::transport_feedbacks::transport_layer_cc::{
    PacketStatusChunk, SymbolTypeTcc, TransportLayerCc,
};
use rtc::rtp_transceiver::rtp_sender::{
    RTCRtpCodec, RTCRtpCodecParameters, RTCRtpCodingParameters, RTCRtpEncodingParameters,
    RtpCodecKind,
};
use rtc::sansio;
use rtc::shared::error::Error;

use webrtc::media_stream::track_local::static_rtp::TrackLocalStaticRTP;
use webrtc::media_stream::track_local::{TrackLocal, TrackLocalEvent};
use webrtc::peer_connection::{
    PeerConnection, PeerConnectionBuilder, PeerConnectionEventHandler, RTCIceGatheringState,
    RTCPeerConnectionState,
};
use webrtc::runtime::{Runtime, Sender, channel, default_runtime};

// ---------------------------------------------------------------------------
// Test shape
// ---------------------------------------------------------------------------

const VIDEO_SSRC: u32 = 0x00DE_CAFE;
const PAYLOAD_BYTES: usize = 1100; // ~1200B on the wire with RTP+SRTP overhead
const WIRE_BYTES: usize = PAYLOAD_BYTES + 100;

/// The bitrate ladder, in Mbps. Each rung is held for `RUNG` seconds. Loopback
/// has no link to saturate, so the knee — when it comes — is the receiving
/// peer's socket buffer, which is a real queue and shows up in TWCC the same way
/// a real bottleneck would.
const LADDER_MBPS: &[f64] = &[2.0, 5.0, 10.0, 25.0, 50.0, 100.0];
const RUNG: Duration = Duration::from_secs(4);

// ---------------------------------------------------------------------------
// RTCP forwarder — the default chain consumes RTCP before the app can see it.
// ---------------------------------------------------------------------------

struct RtcpForwarderBuilder<P> {
    _p: std::marker::PhantomData<P>,
}
impl<P> RtcpForwarderBuilder<P> {
    fn new() -> Self {
        Self {
            _p: std::marker::PhantomData,
        }
    }
    fn build(self) -> impl FnOnce(P) -> RtcpForwarderInterceptor<P> {
        move |inner| RtcpForwarderInterceptor {
            next: inner,
            read_queue: VecDeque::new(),
        }
    }
}

#[derive(Interceptor)]
struct RtcpForwarderInterceptor<P> {
    #[next]
    next: P,
    read_queue: VecDeque<TaggedPacket>,
}

#[interceptor]
impl<P: Interceptor> RtcpForwarderInterceptor<P> {
    #[overrides]
    fn handle_read(&mut self, msg: TaggedPacket) -> Result<(), Self::Error> {
        if let Packet::Rtcp(pkts) = &msg.message {
            self.read_queue.push_back(TaggedPacket {
                now: msg.now,
                transport: msg.transport,
                message: Packet::Rtcp(pkts.clone()),
            });
        }
        self.next.handle_read(msg)
    }

    #[overrides]
    fn poll_read(&mut self) -> Option<Self::Rout> {
        if let Some(p) = self.read_queue.pop_front() {
            return Some(p);
        }
        self.next.poll_read()
    }

    #[overrides]
    fn close(&mut self) -> Result<(), Self::Error> {
        self.read_queue.clear();
        self.next.close()
    }
}

// ---------------------------------------------------------------------------
// Wire tap — registered INNERMOST, so on the write path it sees packets after
// TwccSender has had its turn, and on the read path before TwccReceiver does.
// That is the only vantage point from which "was the extension actually
// stamped?" is answerable.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TapCounts {
    bound_local: Vec<(u32, Vec<String>)>,
    bound_remote: Vec<(u32, Vec<String>)>,
    rtp_out: u64,
    rtp_out_with_ext: u64,
    rtp_in: u64,
    rtp_in_with_ext: u64,
    rtcp_out: u64,
}

struct TapBuilder<P> {
    counts: Arc<Mutex<TapCounts>>,
    _p: std::marker::PhantomData<P>,
}
impl<P> TapBuilder<P> {
    fn new(counts: Arc<Mutex<TapCounts>>) -> Self {
        Self {
            counts,
            _p: std::marker::PhantomData,
        }
    }
    fn build(self) -> impl FnOnce(P) -> TapInterceptor<P> {
        let counts = self.counts;
        move |inner| TapInterceptor { next: inner, counts }
    }
}

#[derive(Interceptor)]
struct TapInterceptor<P> {
    #[next]
    next: P,
    counts: Arc<Mutex<TapCounts>>,
}

#[interceptor]
impl<P: Interceptor> TapInterceptor<P> {
    #[overrides]
    fn bind_local_stream(&mut self, info: &StreamInfo) {
        self.counts.lock().unwrap().bound_local.push((
            info.ssrc,
            info.rtp_header_extensions
                .iter()
                .map(|e| format!("{}#{}", e.uri, e.id))
                .collect(),
        ));
        self.next.bind_local_stream(info);
    }

    #[overrides]
    fn bind_remote_stream(&mut self, info: &StreamInfo) {
        self.counts.lock().unwrap().bound_remote.push((
            info.ssrc,
            info.rtp_header_extensions
                .iter()
                .map(|e| format!("{}#{}", e.uri, e.id))
                .collect(),
        ));
        self.next.bind_remote_stream(info);
    }

    #[overrides]
    fn handle_write(&mut self, msg: TaggedPacket) -> Result<(), Self::Error> {
        {
            let mut c = self.counts.lock().unwrap();
            match &msg.message {
                Packet::Rtp(p) => {
                    c.rtp_out += 1;
                    if p.header.extension {
                        c.rtp_out_with_ext += 1;
                    }
                }
                Packet::Rtcp(_) => c.rtcp_out += 1,
                _ => {}
            }
        }
        self.next.handle_write(msg)
    }

    #[overrides]
    fn handle_read(&mut self, msg: TaggedPacket) -> Result<(), Self::Error> {
        {
            let mut c = self.counts.lock().unwrap();
            if let Packet::Rtp(p) = &msg.message {
                c.rtp_in += 1;
                if p.header.extension {
                    c.rtp_in_with_ext += 1;
                }
            }
        }
        self.next.handle_read(msg)
    }
}

// ---------------------------------------------------------------------------
// Bind shim — the workaround for the gap this spike found.
//
// `find_track_id_by_ssrc` resolves a remote track and fires OnTrack without ever
// calling `interceptor.bind_remote_stream()`, and it is tried before the two
// paths that do bind. So on any stream whose SSRC is declared in the SDP — the
// ordinary case — TwccReceiverInterceptor's `streams` map stays empty, every
// `handle_read` lookup misses, its recorder is never constructed, and it emits
// no feedback for the life of the connection.
//
// `bind_remote_stream` is a public trait method, so the missing call can simply
// be made from outside. Registered OUTERMOST, this sees inbound RTP before the
// TWCC receiver does and binds the stream on first sight.
// ---------------------------------------------------------------------------

struct BindShimBuilder<P> {
    ext_id: u16,
    _p: std::marker::PhantomData<P>,
}
impl<P> BindShimBuilder<P> {
    fn new(ext_id: u16) -> Self {
        Self {
            ext_id,
            _p: std::marker::PhantomData,
        }
    }
    fn build(self) -> impl FnOnce(P) -> BindShimInterceptor<P> {
        let ext_id = self.ext_id;
        move |inner| BindShimInterceptor {
            next: inner,
            ext_id,
            bound: std::collections::HashSet::new(),
        }
    }
}

#[derive(Interceptor)]
struct BindShimInterceptor<P> {
    #[next]
    next: P,
    ext_id: u16,
    bound: std::collections::HashSet<u32>,
}

#[interceptor]
impl<P: Interceptor> BindShimInterceptor<P> {
    #[overrides]
    fn handle_read(&mut self, msg: TaggedPacket) -> Result<(), Self::Error> {
        if let Packet::Rtp(p) = &msg.message
            && self.bound.insert(p.header.ssrc)
        {
            let info = StreamInfo {
                ssrc: p.header.ssrc,
                payload_type: p.header.payload_type,
                rtp_header_extensions: vec![rtc::interceptor::RTPHeaderExtension {
                    uri: rtc::sdp::extmap::TRANSPORT_CC_URI.to_owned(),
                    id: self.ext_id,
                }],
                ..Default::default()
            };
            println!(
                "  [bind-shim] binding remote stream ssrc={} pt={}",
                p.header.ssrc, p.header.payload_type
            );
            self.next.bind_remote_stream(&info);
        }
        self.next.handle_read(msg)
    }
}

// ---------------------------------------------------------------------------
// What one TWCC report told us
// ---------------------------------------------------------------------------

#[derive(Default, Debug, Clone)]
struct Feedback {
    /// Packets the report covers.
    covered: u64,
    /// Of those, how many never arrived.
    lost: u64,
    /// Sum of inter-arrival deltas across the received ones, in microseconds.
    /// This is the wall-clock span the receiver saw those bytes over.
    arrival_span_us: i64,
    /// Received packets that carried a usable delta.
    with_delta: u64,
}

/// Expand the run-length / status-vector chunks back into one symbol per packet,
/// then walk `recv_deltas` alongside them.
///
/// The two lists are parallel but not the same length: only packets that arrived
/// *and* carry a delta consume an entry from `recv_deltas`, so they have to be
/// walked with a cursor rather than zipped by index. Getting that wrong silently
/// attributes one packet's arrival time to another and the gradient turns to noise.
fn parse(cc: &TransportLayerCc) -> Feedback {
    let mut symbols: Vec<SymbolTypeTcc> = Vec::new();
    for chunk in &cc.packet_chunks {
        match chunk {
            PacketStatusChunk::RunLengthChunk(rl) => {
                for _ in 0..rl.run_length {
                    symbols.push(rl.packet_status_symbol);
                }
            }
            PacketStatusChunk::StatusVectorChunk(sv) => {
                for s in &sv.symbol_list {
                    symbols.push(*s);
                }
            }
        }
    }
    symbols.truncate(cc.packet_status_count as usize);

    let mut fb = Feedback::default();
    let mut delta_cursor = 0usize;
    for s in &symbols {
        fb.covered += 1;
        match s {
            SymbolTypeTcc::PacketNotReceived => fb.lost += 1,
            SymbolTypeTcc::PacketReceivedWithoutDelta => {}
            _ => {
                if let Some(d) = cc.recv_deltas.get(delta_cursor) {
                    fb.arrival_span_us += d.delta; // already scaled to microseconds
                    fb.with_delta += 1;
                }
                delta_cursor += 1;
            }
        }
    }
    fb
}

// ---------------------------------------------------------------------------
// Peer plumbing
// ---------------------------------------------------------------------------

struct Handler {
    gather_tx: Sender<()>,
    conn_tx: Sender<()>,
    /// Present on the answerer only: drain the remote track. Without a consumer
    /// the receive pipeline may never bind the stream, and an unbound stream is
    /// one the TWCC receiver never records — hence no feedback at all.
    runtime: Option<Arc<dyn Runtime>>,
    remote_rtp: Option<Arc<AtomicU64>>,
}

#[async_trait::async_trait]
impl PeerConnectionEventHandler for Handler {
    async fn on_ice_gathering_state_change(&self, s: RTCIceGatheringState) {
        if s == RTCIceGatheringState::Complete {
            let _ = self.gather_tx.try_send(());
        }
    }
    async fn on_connection_state_change(&self, s: RTCPeerConnectionState) {
        if s == RTCPeerConnectionState::Connected {
            let _ = self.conn_tx.try_send(());
        }
    }
    async fn on_track(&self, track: Arc<dyn webrtc::media_stream::track_remote::TrackRemote>) {
        let (Some(runtime), Some(counter)) = (self.runtime.clone(), self.remote_rtp.clone()) else {
            return;
        };
        println!("answerer: on_track fired (kind {})", track.kind().await);
        runtime.spawn(Box::pin(async move {
            use webrtc::media_stream::track_remote::TrackRemoteEvent;
            while let Some(evt) = track.poll().await {
                match evt {
                    TrackRemoteEvent::OnRtpPacket(_) => {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                    TrackRemoteEvent::OnEnded | TrackRemoteEvent::OnError => break,
                    _ => {}
                }
            }
        }));
    }
}

fn h264_codec() -> RTCRtpCodec {
    RTCRtpCodec {
        mime_type: MIME_TYPE_H264.to_owned(),
        clock_rate: 90000,
        channels: 0,
        sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"
            .to_owned(),
        rtcp_feedback: vec![],
    }
}

async fn build_peer(
    runtime: Arc<dyn Runtime>,
    handler: Arc<dyn PeerConnectionEventHandler>,
    tap: Arc<Mutex<TapCounts>>,
) -> anyhow::Result<Arc<dyn PeerConnection>> {
    let mut me = MediaEngine::default();
    me.register_codec(
        RTCRtpCodecParameters {
            rtp_codec: h264_codec(),
            payload_type: 102,
            ..Default::default()
        },
        RtpCodecKind::Video,
    )?;

    // NOT `register_default_interceptors`: that one calls `configure_twcc_receiver_only`,
    // so this peer would generate feedback for the remote but never stamp transport-wide
    // sequence numbers on its own packets — and a sender with no sequence numbers gets
    // no feedback to estimate from. `configure_twcc` installs both halves.
    // Innermost: on write it runs last (post-stamp), on read it runs first.
    let registry = Registry::new().with(TapBuilder::new(tap).build());
    let registry = configure_nack(registry, &mut me);
    let registry = configure_rtcp_reports(registry);
    let registry = configure_twcc(registry, &mut me)?;
    // Outermost: sees inbound RTP before the TWCC receiver, and inbound RTCP
    // before the default chain consumes it.
    let registry = registry.with(BindShimBuilder::new(1).build());
    let registry = registry.with(RtcpForwarderBuilder::new().build());

    let pc = PeerConnectionBuilder::new()
        .with_media_engine(me)
        .with_interceptor_registry(registry)
        .with_handler(handler)
        .with_runtime(runtime)
        .with_udp_addrs(vec!["127.0.0.1:0".to_owned()])
        .build()
        .await?;
    Ok(Arc::new(pc) as Arc<dyn PeerConnection>)
}

// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let runtime = default_runtime().expect("tokio runtime feature");
    let mut out: Option<anyhow::Result<()>> = None;
    {
        let slot = &mut out;
        let rt = runtime.clone();
        runtime.block_on(Box::pin(async move {
            *slot = Some(run(rt).await);
        }));
    }
    out.unwrap()
}

async fn run(runtime: Arc<dyn Runtime>) -> anyhow::Result<()> {
    println!("webrtc = 0.20.2 | rtc = 0.20.2\n");

    let tap_off = Arc::new(Mutex::new(TapCounts::default()));
    let tap_ans = Arc::new(Mutex::new(TapCounts::default()));

    let (og, mut ogr) = channel::<()>(1);
    let (oc, mut ocr) = channel::<()>(1);
    let offerer = build_peer(
        runtime.clone(),
        Arc::new(Handler {
            gather_tx: og,
            conn_tx: oc,
            runtime: None,
            remote_rtp: None,
        }),
        Arc::clone(&tap_off),
    )
    .await?;

    let track = Arc::new(TrackLocalStaticRTP::new(MediaStreamTrack::new(
        "spike-stream".to_owned(),
        "spike-video".to_owned(),
        "spike".to_owned(),
        RtpCodecKind::Video,
        vec![RTCRtpEncodingParameters {
            rtp_coding_parameters: RTCRtpCodingParameters {
                ssrc: Some(VIDEO_SSRC),
                ..Default::default()
            },
            codec: h264_codec(),
            ..Default::default()
        }],
    )));
    offerer
        .add_track(Arc::clone(&track) as Arc<dyn TrackLocal>)
        .await?;

    let (ag, mut agr) = channel::<()>(1);
    let (ac, mut acr) = channel::<()>(1);
    let remote_rtp = Arc::new(AtomicU64::new(0));
    let answerer = build_peer(
        runtime.clone(),
        Arc::new(Handler {
            gather_tx: ag,
            conn_tx: ac,
            runtime: Some(runtime.clone()),
            remote_rtp: Some(Arc::clone(&remote_rtp)),
        }),
        Arc::clone(&tap_ans),
    )
    .await?;

    // Non-trickle offer/answer.
    let offer = offerer.create_offer(None).await?;
    offerer.set_local_description(offer).await?;
    webrtc::runtime::timeout(&*runtime, Duration::from_secs(5), ogr.recv())
        .await
        .map_err(|_| anyhow::anyhow!("offerer ICE gather timed out"))?;
    let offer_sdp = offerer.local_description().await.unwrap();

    println!("--- offer SDP: feedback + extension negotiation ---");
    for line in offer_sdp.sdp.lines() {
        if line.contains("transport-cc") || line.starts_with("a=extmap") {
            println!("  {line}");
        }
    }
    println!();

    answerer.set_remote_description(offer_sdp).await?;
    let answer = answerer.create_answer(None).await?;
    answerer.set_local_description(answer).await?;
    webrtc::runtime::timeout(&*runtime, Duration::from_secs(5), agr.recv())
        .await
        .map_err(|_| anyhow::anyhow!("answerer ICE gather timed out"))?;
    let answer_sdp = answerer.local_description().await.unwrap();

    // The sender only stamps transport-wide sequence numbers on a stream whose
    // *negotiated* header extensions include the TWCC URI (`stream_supports_twcc`
    // looks it up in StreamInfo). The offer having it is not enough — the answer
    // has to echo it back, or the intersection is empty and nothing is stamped.
    println!("--- answer SDP: feedback + extension negotiation ---");
    for line in answer_sdp.sdp.lines() {
        if line.contains("transport-cc") || line.starts_with("a=extmap") {
            println!("  {line}");
        }
    }
    println!();

    offerer.set_remote_description(answer_sdp).await?;

    webrtc::runtime::timeout(&*runtime, Duration::from_secs(10), ocr.recv())
        .await
        .map_err(|_| anyhow::anyhow!("offerer never connected"))?;
    webrtc::runtime::timeout(&*runtime, Duration::from_secs(10), acr.recv())
        .await
        .map_err(|_| anyhow::anyhow!("answerer never connected"))?;
    println!("both peers connected\n");

    // --- collect TWCC on the sender side ------------------------------------
    let reports = Arc::new(Mutex::new(Vec::<(Instant, Feedback)>::new()));
    let rtcp_total = Arc::new(AtomicU64::new(0));
    let twcc_total = Arc::new(AtomicU64::new(0));
    {
        let (reports, rtcp_total, twcc_total) = (
            Arc::clone(&reports),
            Arc::clone(&rtcp_total),
            Arc::clone(&twcc_total),
        );
        let poll_track = Arc::clone(&track);
        runtime.spawn(Box::pin(async move {
            while let Some(TrackLocalEvent::OnRtcpPacket(pkts)) = poll_track.poll().await {
                for p in pkts {
                    rtcp_total.fetch_add(1, Ordering::Relaxed);
                    if let Some(cc) = p.as_any().downcast_ref::<TransportLayerCc>() {
                        twcc_total.fetch_add(1, Ordering::Relaxed);
                        reports.lock().unwrap().push((Instant::now(), parse(cc)));
                    }
                }
            }
        }));
    }

    // --- ramp the send rate --------------------------------------------------
    println!(
        "{:>8} {:>10} {:>8} {:>9} {:>10} {:>12} {:>11}",
        "target", "sent", "TWCC", "covered", "lost", "arrival", "gradient"
    );
    println!(
        "{:>8} {:>10} {:>8} {:>9} {:>10} {:>12} {:>11}",
        "Mbps", "pkts", "reports", "pkts", "%", "Mbps", "us/pkt"
    );
    println!("{}", "-".repeat(76));

    let mut seq: u16 = 0;
    let mut ts: u32 = 0;

    for &mbps in LADDER_MBPS {
        let pps = (mbps * 1_000_000.0) / (WIRE_BYTES as f64 * 8.0);
        let ideal_gap_us = 1_000_000.0 / pps;
        reports.lock().unwrap().clear();

        let rung_start = Instant::now();
        let mut sent = 0u64;
        while rung_start.elapsed() < RUNG {
            // Rate control against elapsed time rather than a fixed sleep, so
            // timer jitter cannot silently change the bitrate under test.
            let due = (rung_start.elapsed().as_secs_f64() * pps) as u64;
            while sent < due {
                let pkt = rtc::rtp::packet::Packet {
                    header: rtc::rtp::header::Header {
                        version: 2,
                        payload_type: 102,
                        sequence_number: seq,
                        timestamp: ts,
                        ssrc: VIDEO_SSRC,
                        ..Default::default()
                    },
                    payload: bytes::Bytes::from(vec![0xABu8; PAYLOAD_BYTES]),
                };
                if track.write_rtp(pkt).await.is_err() {
                    break;
                }
                seq = seq.wrapping_add(1);
                ts = ts.wrapping_add(3000);
                sent += 1;
            }
            runtime.sleep(Duration::from_micros(500)).await;
        }
        // Let the last feedback come back before reading the rung.
        runtime.sleep(Duration::from_millis(300)).await;

        let snap = reports.lock().unwrap().clone();
        let n_reports = snap.len();
        let covered: u64 = snap.iter().map(|(_, f)| f.covered).sum();
        let lost: u64 = snap.iter().map(|(_, f)| f.lost).sum();
        let span_us: i64 = snap.iter().map(|(_, f)| f.arrival_span_us).sum();
        let with_delta: u64 = snap.iter().map(|(_, f)| f.with_delta).sum();

        let loss_pct = if covered > 0 {
            100.0 * lost as f64 / covered as f64
        } else {
            0.0
        };
        // Bytes the receiver actually took in, over the wall-clock span it took
        // them in over. This is the "acked rate" a loss-based controller uses.
        let arrival_mbps = if span_us > 0 {
            (with_delta as f64 * WIRE_BYTES as f64 * 8.0) / (span_us as f64)
        } else {
            0.0
        };
        // GCC's core signal: mean (inter-arrival - inter-departure). Positive
        // means the receiver is spreading packets out more than we sent them,
        // i.e. a queue is building somewhere.
        let gradient = if with_delta > 0 {
            (span_us as f64 / with_delta as f64) - ideal_gap_us
        } else {
            0.0
        };

        println!(
            "{:>8.0} {:>10} {:>8} {:>9} {:>9.2}% {:>12.1} {:>+11.1}",
            mbps, sent, n_reports, covered, loss_pct, arrival_mbps, gradient
        );
    }

    println!("{}", "-".repeat(76));
    println!(
        "\nRTP packets the ANSWERER actually received: {}",
        remote_rtp.load(Ordering::Relaxed)
    );
    println!(
        "inbound RTCP packets seen by the SENDER: {}  (of which TransportLayerCc: {})",
        rtcp_total.load(Ordering::Relaxed),
        twcc_total.load(Ordering::Relaxed)
    );

    for (who, tap) in [("OFFERER/sender", &tap_off), ("ANSWERER/receiver", &tap_ans)] {
        let c = tap.lock().unwrap();
        println!("
[{who}] wire tap (innermost interceptor)");
        println!("  bind_local_stream : {:?}", c.bound_local);
        println!("  bind_remote_stream: {:?}", c.bound_remote);
        println!(
            "  RTP out: {} (with header extension: {})",
            c.rtp_out, c.rtp_out_with_ext
        );
        println!(
            "  RTP in : {} (with header extension: {})",
            c.rtp_in, c.rtp_in_with_ext
        );
        println!("  RTCP out: {}", c.rtcp_out);
    }

    offerer.close().await.ok();
    answerer.close().await.ok();
    Ok(())
}

// ---------------------------------------------------------------------------
// SEND-TIME MAP — the shortcut this spike takes, and what production cannot.
//
// The gradient above uses an *analytic* inter-departure time, valid only because
// the send loop paces to a known rate. A real encoder does not: frames are bursty,
// a keyframe is ten times a delta frame, and the pacer reshapes both. GCC needs
// the actual send timestamp of each transport-wide sequence number.
//
// Those sequence numbers are assigned inside `TwccSenderInterceptor`, which does
// not expose the mapping. So a production estimator cannot use the stock sender —
// it needs its own interceptor that stamps the extension *and* records
// (transport_seq -> send_instant, size). That is a small interceptor, but it is
// a required one, and it is not in the crate.
// ---------------------------------------------------------------------------
