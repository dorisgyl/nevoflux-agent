/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! User-facing notifications published on the EventBus.
//!
//! The `notify_user` LLM tool routes through [`publish_user_notification`],
//! which emits an ephemeral `ui:notification:agent` event with an `Internal`
//! publisher identity (the wasm surface cannot emit `ui:*` topics directly).
//!
//! Two consumers pick the event up:
//! - the sidebar renders it as a bottom-right toast (`ui:notification:*`);
//! - `background.js` fires an OS notification via `browser.notifications.create`
//!   when `source == "notify_user"` and the browser window is unfocused.

use crate::event_bus::types::{BusEvent, PublisherIdentity};
use crate::event_bus::EventBus;
use serde_json::json;

/// Cap the notification message so a runaway tool call can't flood the bus.
const NOTIFY_MESSAGE_MAX_BYTES: usize = 4096;

fn cap_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Publish a user-facing notification on `ui:notification:agent`.
///
/// `title` defaults to "NevoFlux" on the consumer side when `None`. `source`
/// identifies the origin (currently always `"notify_user"`); `background.js`
/// gates the OS notification on this value. `schedule_id` is attached when the
/// notification originates from a scheduled run, for future correlation.
pub async fn publish_user_notification(
    bus: &EventBus,
    title: Option<&str>,
    message: &str,
    source: &str,
    schedule_id: Option<&str>,
) {
    // `body` is the canonical toast text field consumed by the sidebar's
    // `ui:notification:*` renderer (see chat-sidebar handler). `source` gates
    // the background OS notification.
    let payload = json!({
        "event_id": uuid::Uuid::new_v4().to_string(),
        "title": title,
        "body": cap_bytes(message, NOTIFY_MESSAGE_MAX_BYTES),
        "source": source,
        "schedule_id": schedule_id,
    });
    let _ = bus
        .publish(BusEvent::ephemeral(
            "ui:notification:agent",
            payload,
            PublisherIdentity::Internal,
        ))
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_bus::types::{Delivery, TopicPattern};
    use crate::event_bus::{BackpressurePolicy, SubscriberIdentity};
    use std::sync::Arc;

    #[tokio::test]
    async fn emits_ui_notification_agent_with_source() {
        let bus = Arc::new(EventBus::new());
        let mut handle = bus
            .subscribe(
                TopicPattern::double_wildcard("ui:notification"),
                SubscriberIdentity::Internal,
                BackpressurePolicy::DropNewest,
                8,
            )
            .expect("subscribe should succeed");

        publish_user_notification(&bus, Some("Reminder"), "drink water", "notify_user", None).await;

        let ev = handle.rx.try_recv().expect("expected a buffered event");
        assert_eq!(ev.topic, "ui:notification:agent");
        assert_eq!(ev.delivery, Delivery::Ephemeral);
        assert_eq!(ev.payload["source"], json!("notify_user"));
        assert_eq!(ev.payload["body"], json!("drink water"));
        assert_eq!(ev.payload["title"], json!("Reminder"));
        assert!(ev.payload["event_id"].is_string());
    }

    #[tokio::test]
    async fn caps_long_message() {
        let bus = Arc::new(EventBus::new());
        let mut handle = bus
            .subscribe(
                TopicPattern::double_wildcard("ui:notification"),
                SubscriberIdentity::Internal,
                BackpressurePolicy::DropNewest,
                8,
            )
            .expect("subscribe should succeed");

        publish_user_notification(&bus, None, &"z".repeat(5000), "notify_user", None).await;

        let ev = handle.rx.try_recv().expect("expected a buffered event");
        assert_eq!(ev.payload["body"].as_str().unwrap().len(), 4096);
    }
}
