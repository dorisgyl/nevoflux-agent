//! The NevoFlux scenario: a daemon at home streaming to a phone.
//!
//! The question this answers is the original one — *can TWCC feedback actually
//! drive an encoder bitrate* — so unlike str0m's own tests, the media send rate
//! here is not stepped by a plan. It is re-derived from the estimate on every
//! slice, which is exactly what an encoder following BWE would do. If the loop
//! is unstable, an adaptive sender is the shape that reveals it; a sender on a
//! fixed ladder is not.
//!
//! Three phases, because the interesting behaviour is not the ramp:
//!   1. 20 Mbps uplink   — climb to the desired rate
//!   2. 2 Mbps uplink    — the collapse. Someone else started an upload, or the
//!                         phone dropped to cellular. Has to be fast: this is
//!                         the case that ruins a call.
//!   3. 20 Mbps restored — recover, without having latched low.
//!
//! Prints the trajectory rather than only asserting on it, so the *shape* is
//! visible — a controller that reaches the right number by oscillating through
//! it is not the same as one that converges, and an assertion cannot tell them
//! apart.

use std::time::Duration;

use netem::{Bitrate, DataSize, NetemConfig};
use str0m::RtcError;

use crate::common::{BweTestContext, connect_with_bwe};
use crate::common::{init_crypto_default, init_log};

/// A 1080p screencast worth of video — what the head would like to send.
const DESIRED: Bitrate = Bitrate::mbps(8);
/// Where the estimator starts before it has any feedback to go on.
const INITIAL: Bitrate = Bitrate::kbps(600);

/// Typical home broadband uplink: not the bottleneck we are testing, just the
/// ceiling. 250 KB of buffer is roughly 100ms at rate — enough to queue, which
/// is what makes the delay signal meaningful.
fn good_uplink() -> NetemConfig {
    NetemConfig::new()
        .latency(Duration::from_millis(25))
        .link(Bitrate::mbps(20), DataSize::bytes(250_000))
        .seed(42)
}

/// The uplink under contention. Deliberately far below `DESIRED`, so the
/// sender is definitely over-driving it at the moment conditions change.
fn degraded_uplink() -> NetemConfig {
    NetemConfig::new()
        .latency(Duration::from_millis(40))
        .link(Bitrate::mbps(2), DataSize::bytes(250_000))
        .seed(42)
}

fn mbps(b: Bitrate) -> f64 {
    b.as_u64() as f64 / 1_000_000.0
}

#[test]
fn nevoflux_screencast_uplink_collapse_and_recovery() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();

    let (mut l, mut r) = connect_with_bwe(INITIAL, DESIRED);
    let mut ctx = BweTestContext::new(&mut l, &mut r);

    let phases: &[(&str, NetemConfig, u64)] = &[
        ("20 Mbps uplink", good_uplink(), 25),
        ("2 Mbps (contended)", degraded_uplink(), 25),
        ("20 Mbps restored", good_uplink(), 30),
    ];

    println!();
    println!(
        "desired = {:.1} Mbps, initial estimate = {:.2} Mbps",
        mbps(DESIRED),
        mbps(INITIAL)
    );
    println!();
    println!("  phase                  t(s)   estimate      sending");
    println!("  {}", "-".repeat(52));

    let mut t = 0u64;
    let mut last = INITIAL;
    let mut min_in_phase;
    let mut summary: Vec<(String, f64, f64)> = Vec::new();

    for (name, config, secs) in phases {
        l.set_netem(*config);
        r.set_netem(*config);
        min_in_phase = f64::MAX;
        let mut end = mbps(last);

        for i in 0..*secs {
            // The closed loop: an encoder would target the estimate, capped by
            // what it actually wants to send. Never below a floor — a real
            // encoder cannot emit zero and still be a video.
            let send_at = Bitrate::bps(last.as_u64().min(DESIRED.as_u64()).max(150_000));
            ctx.set_media_send_rate(send_at);

            if let Some(est) = ctx.run_for_duration(&mut l, &mut r, Duration::from_secs(1))? {
                last = est;
            }
            t += 1;
            end = mbps(last);
            min_in_phase = min_in_phase.min(end);

            // One line per second is too much to read; sample the shape.
            if i % 5 == 0 || i == secs - 1 {
                println!(
                    "  {:<20} {:>4}   {:>7.2} Mbps  {:>6.2} Mbps",
                    if i == 0 { *name } else { "" },
                    t,
                    end,
                    mbps(send_at)
                );
            }
        }
        summary.push((name.to_string(), end, min_in_phase));
    }

    println!("  {}", "-".repeat(52));
    println!();

    for (name, end, min) in &summary {
        println!("  {name:<20} ends {end:>6.2} Mbps   (low water {min:>6.2})");
    }
    println!();

    let (_, after_good, _) = &summary[0];
    let (_, after_bad, low_bad) = &summary[1];
    let (_, after_recovery, _) = &summary[2];

    // Phase 1: has to reach something a screencast can actually use. Not the
    // full 8 — the link is 20 but the sender only ever offers what it estimates,
    // so ramp-up is self-limiting and 4 Mbps is already a usable picture.
    assert!(
        *after_good >= 4.0,
        "should climb on a 20 Mbps uplink, got {after_good:.2} Mbps"
    );

    // Phase 2: the one that matters. Staying high here means pushing 8 Mbps into
    // a 2 Mbps pipe — seconds of queueing delay, then mass loss.
    //
    // The bound is deliberately loose about the exact number. AIMD does not
    // settle *on* the capacity, it oscillates across it: additive increase until
    // the delay signal trips, then a multiplicative cut. Measured band on this
    // link is 2.2–2.9 around a 2.0 Mbps pipe. Asserting a tight ceiling would be
    // asserting that a correct controller stops probing, which would be worse —
    // that is how you get a session latched at 200 kbps after one bad minute.
    // What has to be true is that it left 15 Mbps behind and is tracking the
    // link, not that it sits at exactly 2.0.
    assert!(
        *after_bad <= 4.0,
        "must back off toward the 2 Mbps link, got {after_bad:.2} Mbps"
    );
    assert!(
        *low_bad <= 2.6,
        "should touch the real capacity, low water was {low_bad:.2} Mbps"
    );

    // Phase 3: a controller that backs off and never comes back is not usable
    // either — that is the failure mode where one bad minute costs you the
    // rest of the session.
    assert!(
        *after_recovery >= *after_bad * 2.0,
        "must recover after the link comes back: {after_bad:.2} -> {after_recovery:.2} Mbps"
    );

    Ok(())
}
