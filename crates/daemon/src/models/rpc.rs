//! The three commands the browser calls to get weights onto disk.
//!
//! `models.status` / `models.download` / `models.cancel`, reached through the
//! same `system_command` path as `kb.wizard.*` and `brain.*`. That is not
//! incidental: the settings page and the sidebar both already speak it, and a
//! second mechanism doing the same job is how a codebase ends up with three
//! places to register the same thing and two of them subtly stale.
//!
//! `download` returns as soon as it has started. The work then streams frames
//! on the EventBus topic [`PROGRESS_TOPIC`], exactly as the install wizard
//! does — a request/response round trip cannot describe six minutes of
//! downloading.

use std::sync::Arc;

use crate::event_bus::{BusEvent, EventBus, PublisherIdentity};
use crate::kb_wizard::{err_response, ok_response, CURRENT_EVENT_BUS};

use super::{catalog, downloads, models_dir, status, tier_report, ModelError, Tier};

/// Where download progress is published. `system:` is the daemon's own
/// namespace, and the extension surface is allowed to subscribe to it.
///
/// The separator is `:` throughout. A `.` here would pass every type check,
/// fail the permission match at runtime, find no subscriber, and drop every
/// frame with nothing logged.
pub const PROGRESS_TOPIC: &str = "system:models:progress";

fn request_id(params: &serde_json::Value) -> String {
    params
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn tier_of(params: &serde_json::Value) -> Option<Tier> {
    params
        .get("tier")
        .and_then(|v| v.as_str())
        .and_then(Tier::parse)
}

/// What is on disk, and what each tier would still cost.
pub async fn handle_status(params: &serde_json::Value) -> serde_json::Value {
    let id = request_id(params);
    let Some(dir) = models_dir() else {
        return err_response(
            &id,
            "models.status",
            "NO_CACHE_DIR",
            "this system has no cache directory to download into",
        );
    };
    ok_response(
        &id,
        "models.status",
        serde_json::json!({
            "dir": dir.display().to_string(),
            "assets": status(&dir),
            // Every tier, from the one list of them: a hand-written pair here
            // would silently stop reporting the day a third was added, and the
            // panel would offer a download the daemon never mentions.
            "tiers": Tier::ALL.iter().map(|t| tier_report(*t, &dir)).collect::<Vec<_>>(),
        }),
    )
}

/// Start fetching a tier. Returns immediately; watch [`PROGRESS_TOPIC`].
pub async fn handle_download(params: &serde_json::Value) -> serde_json::Value {
    download_with(params, CURRENT_EVENT_BUS.get().cloned()).await
}

/// The bus is a parameter rather than a global lookup so that a test can say
/// which case it is testing.
///
/// Found the hard way: with the lookup inside, whether this refused or started
/// depended on whether some *other* test had initialised the global bus first
/// — and in the full suite it had, so the test that meant to check the refusal
/// quietly kicked off a 240 MB download instead.
async fn download_with(
    params: &serde_json::Value,
    bus: Option<Arc<EventBus>>,
) -> serde_json::Value {
    let id = request_id(params);
    let Some(tier) = tier_of(params) else {
        return err_response(
            &id,
            "models.download",
            "BAD_TIER",
            &format!(
                "expected one of {}",
                Tier::ALL
                    .iter()
                    .map(|t| t.id())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    };
    let Some(dir) = models_dir() else {
        return err_response(
            &id,
            "models.download",
            "NO_CACHE_DIR",
            "this system has no cache directory to download into",
        );
    };
    let Some(bus) = bus else {
        // Refused rather than started silently: a download nobody can see the
        // progress of looks exactly like one that is not running.
        return err_response(
            &id,
            "models.download",
            "NO_EVENT_BUS",
            "EventBus not initialised; cannot stream progress",
        );
    };
    let Some((run, cancel)) = downloads().begin(tier) else {
        // A second click is a duplicate, not a reason to fetch twice.
        return ok_response(
            &id,
            "models.download",
            serde_json::json!({ "started": false, "reason": "already_running" }),
        );
    };

    // Spawned and deliberately not tied to the caller: closing the settings
    // page during a 240 MB download should not abandon it.
    tokio::spawn(async move {
        let total = catalog::tier_bytes(tier);
        let client = super::http_client();

        // The progress callback is synchronous and cannot await, so it hands
        // frames to a forwarding task — the same shape the speech uplink uses.
        // Collecting them and flushing at the end would deliver the entire
        // progress history at the moment progress stopped being interesting.
        let (ptx, mut prx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
        let fbus = bus.clone();
        let forwarder = tokio::spawn(async move {
            while let Some(frame) = prx.recv().await {
                publish(&fbus, frame).await;
            }
        });

        let mut base = 0u64;
        let mut last_at = std::time::Instant::now();
        let mut last_done = 0u64;

        let result = super::download_tier(
            &client,
            tier,
            &dir,
            &cancel,
            &mut |asset, done, asset_total| {
                let tier_done = base + done;
                if done >= asset_total {
                    base += asset_total;
                }
                if super::should_emit(tier_done, total, last_done, last_at.elapsed()) {
                    last_at = std::time::Instant::now();
                    last_done = tier_done;
                    let _ = ptx.send(serde_json::json!({
                        "tier": tier.id(),
                        "status": "running",
                        "asset": asset.id,
                        "asset_done": done,
                        "asset_total": asset_total,
                        "done": tier_done,
                        "total": total,
                    }));
                }
            },
        )
        .await;

        // Dropping the sender ends the forwarder; awaiting it keeps the
        // terminal frame behind the last progress frame instead of racing it.
        drop(ptx);
        let _ = forwarder.await;

        downloads().finish(tier, run);

        let frame = match result {
            Ok(()) => {
                tracing::info!(target: "models", tier = tier.id(), "ready");
                serde_json::json!({
                    "tier": tier.id(),
                    "status": "ok",
                    "done": total,
                    "total": total,
                })
            }
            Err(ModelError::Cancelled) => serde_json::json!({
                // Not a failure: what arrived is on disk and a later run
                // resumes from it.
                "tier": tier.id(),
                "status": "cancelled",
                "remaining": super::tier_remaining(tier, &dir),
                "total": total,
            }),
            Err(e) => {
                tracing::warn!(target: "models", tier = tier.id(), error = %e, "download failed");
                serde_json::json!({
                    "tier": tier.id(),
                    "status": "failed",
                    "message": e.to_string(),
                    "total": total,
                })
            }
        };
        publish(&bus, frame).await;
    });

    ok_response(
        &id,
        "models.download",
        serde_json::json!({ "started": true, "tier": tier.id() }),
    )
}

/// Stop a running download. What has arrived stays on disk.
pub async fn handle_cancel(params: &serde_json::Value) -> serde_json::Value {
    let id = request_id(params);
    let Some(tier) = tier_of(params) else {
        return err_response(
            &id,
            "models.cancel",
            "BAD_TIER",
            &format!(
                "expected one of {}",
                Tier::ALL
                    .iter()
                    .map(|t| t.id())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    };
    let was_running = downloads().cancel(tier);
    ok_response(
        &id,
        "models.cancel",
        serde_json::json!({ "cancelled": was_running }),
    )
}

/// Publish one frame, swallowing bus errors: a flaky EventBus must not take
/// the download down with it.
async fn publish(bus: &Arc<EventBus>, frame: serde_json::Value) {
    let event = BusEvent::ephemeral(PROGRESS_TOPIC, frame, PublisherIdentity::Internal);
    if let Err(e) = bus.publish(event).await {
        tracing::warn!(target: "models", error = %e, topic = PROGRESS_TOPIC, "publish failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(v: &serde_json::Value) -> &serde_json::Value {
        v.get("payload").expect("system_response envelope")
    }

    fn ok(v: &serde_json::Value) -> bool {
        payload(v)
            .get("success")
            .and_then(|s| s.as_bool())
            .unwrap_or(false)
    }

    #[test]
    fn the_progress_topic_is_colon_separated() {
        // A `.` here type-checks, fails the permission match at runtime, finds
        // no subscriber, and drops every frame silently.
        assert!(PROGRESS_TOPIC.starts_with("system:"));
        assert!(!PROGRESS_TOPIC.contains('.'), "{PROGRESS_TOPIC}");
    }

    #[tokio::test]
    async fn status_reports_both_tiers_and_every_asset() {
        let v = handle_status(&serde_json::json!({ "request_id": "r1" })).await;
        assert!(ok(&v), "{v}");
        let data = payload(&v).get("data").unwrap();
        assert_eq!(data["tiers"].as_array().unwrap().len(), Tier::ALL.len());
        assert_eq!(
            data["assets"].as_array().unwrap().len(),
            catalog::ASSETS.len()
        );
        assert_eq!(payload(&v)["request_id"], "r1");
    }

    #[tokio::test]
    async fn an_unknown_tier_is_refused_rather_than_guessed() {
        for bad in [
            serde_json::json!({ "request_id": "r" }),
            serde_json::json!({ "request_id": "r", "tier": "everything" }),
            serde_json::json!({ "request_id": "r", "tier": 3 }),
        ] {
            let v = download_with(&bad, None).await;
            assert!(!ok(&v), "{v}");
            assert_eq!(payload(&v)["error"]["code"], "BAD_TIER");
            let v = handle_cancel(&bad).await;
            assert!(!ok(&v), "{v}");
        }
    }

    #[tokio::test]
    async fn downloading_without_a_bus_is_refused_not_started_blind() {
        // In tests CURRENT_EVENT_BUS is unset, which is exactly the condition
        // being asserted: a download whose progress nobody can see is
        // indistinguishable from one that never started.
        let v = download_with(
            &serde_json::json!({ "request_id": "r", "tier": "transcribe" }),
            None,
        )
        .await;
        assert!(!ok(&v), "{v}");
        assert_eq!(payload(&v)["error"]["code"], "NO_EVENT_BUS");
        assert!(
            !downloads().is_running(Tier::Transcribe),
            "a refused download still claimed the tier"
        );
    }

    #[tokio::test]
    async fn cancelling_nothing_says_so_instead_of_claiming_success() {
        let v = handle_cancel(&serde_json::json!({
            "request_id": "r",
            "tier": "speak",
        }))
        .await;
        assert!(ok(&v), "{v}");
        assert_eq!(payload(&v)["data"]["cancelled"], false);
    }
}
