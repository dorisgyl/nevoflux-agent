//! Getting TURN credentials that are still valid.
//!
//! Most of the ICE configuration is static: a STUN server is a hostname and
//! nothing else, and a self-hosted coturn has a password that lives in
//! `config.toml` until someone changes it. Cloudflare's TURN is not like that.
//! It issues credentials that expire, minted on demand against a key — so the
//! thing an operator can write down is the key, and the credentials have to be
//! fetched.
//!
//! # Why this is worth the round trip
//!
//! STUN alone gets a connection through the common home router. It does not get
//! one through a symmetric NAT, which is what a phone on a mobile network is
//! usually behind, and that is precisely the case this feature exists for: the
//! phone is not on the same WiFi. Without a relayed candidate those sessions
//! have no path at all and fall back to the relay, where a live screencast
//! cannot go.
//!
//! # Failing quietly
//!
//! Every failure here returns nothing rather than an error. A head that cannot
//! mint credentials still offers host and reflexive candidates, and most
//! connections still form. Refusing to negotiate because the relay was
//! unavailable would turn a degraded path into no path.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use crate::config::{CloudflareTurnConfig, IceServerConfig};

/// Where credentials are minted.
const CLOUDFLARE_RTC: &str = "https://rtc.live.cloudflare.com";

/// How long to wait for the mint before giving up and going without a relay.
///
/// Short. This runs while a portal is waiting to be offered a connection, and
/// an offer that arrives late is worse than one that arrives without a relayed
/// candidate — ICE can add that later, but nothing can start until the offer
/// lands.
const MINT_TIMEOUT: Duration = Duration::from_secs(5);

/// Re-mint this long before the credential actually expires.
///
/// An allocation has to be refreshed for as long as the call lasts, and a
/// refresh signed with an expired credential fails. Renewing early costs one
/// request; renewing late costs the connection mid-call.
const RENEW_MARGIN: Duration = Duration::from_secs(300);

/// Minted credentials, and when they stop being usable.
type Cache = Mutex<HashMap<String, (Instant, Vec<IceServerConfig>)>>;

fn cache() -> &'static Cache {
    static C: OnceLock<Cache> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The ICE servers this key currently grants, minting if needed.
///
/// Cached across sessions on the same key: every head on this machine would
/// otherwise mint its own set on every negotiation, and they are
/// interchangeable.
pub async fn cloudflare(cfg: &CloudflareTurnConfig) -> Vec<IceServerConfig> {
    if let Some(hit) = cache().lock().await.get(&cfg.key_id) {
        if hit.0 > Instant::now() {
            return hit.1.clone();
        }
    }

    let minted = match mint(cfg).await {
        Some(v) if !v.is_empty() => v,
        _ => return Vec::new(),
    };
    // Never negative, and never so long that a clock skew hands out something
    // already dead: the margin is subtracted from the TTL the server was asked
    // for, not from one it reported.
    let good_for = Duration::from_secs(cfg.ttl_seconds).saturating_sub(RENEW_MARGIN);
    cache().lock().await.insert(
        cfg.key_id.clone(),
        (Instant::now() + good_for, minted.clone()),
    );
    tracing::info!(
        target: "rtc",
        count = minted.len(),
        "minted TURN credentials"
    );
    minted
}

/// One trip to Cloudflare for a fresh set.
///
/// Two endpoint names, because the service has been documented under both and
/// an account that only knows the older one would otherwise fail with a 404
/// that reads exactly like a bad key. Both answer in the same shape.
async fn mint(cfg: &CloudflareTurnConfig) -> Option<Vec<IceServerConfig>> {
    for path in ["generate-ice-servers", "generate"] {
        let url = format!(
            "{CLOUDFLARE_RTC}/v1/turn/keys/{}/credentials/{path}",
            cfg.key_id
        );
        let resp = match reqwest::Client::new()
            .post(url)
            .bearer_auth(&cfg.api_token)
            .json(&serde_json::json!({ "ttl": cfg.ttl_seconds }))
            .timeout(MINT_TIMEOUT)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(target: "rtc", "TURN credentials: {e}");
                return None;
            }
        };

        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            continue; // the other spelling
        }
        let body = resp.text().await.ok()?;
        if !status.is_success() {
            // Worth a warning rather than a debug line: unlike a connection
            // that does not form, this is a configuration mistake and nothing
            // will fix itself. The body carries the reason and no secret.
            tracing::warn!(
                target: "rtc",
                %status,
                "TURN credentials refused: {}",
                body.chars().take(200).collect::<String>()
            );
            return None;
        }
        return Some(parse(&body));
    }
    tracing::warn!(target: "rtc", "TURN key {} is not known to Cloudflare", cfg.key_id);
    None
}

/// Pull the usable servers out of a mint response.
///
/// Separate from the request so the shape of the answer is tested without a
/// network or an account — which is the only part of this that can be tested
/// here at all.
fn parse(body: &str) -> Vec<IceServerConfig> {
    let v: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(target: "rtc", "TURN credentials: unreadable answer: {e}");
            return Vec::new();
        }
    };
    // The field has been documented both as one object and as a list of them.
    // Accepting either costs three lines and saves a silent breakage.
    let entries = match v.get("iceServers") {
        Some(serde_json::Value::Array(a)) => a.clone(),
        Some(one) => vec![one.clone()],
        None => return Vec::new(),
    };

    let mut out = Vec::new();
    for e in entries {
        let username = e.get("username").and_then(|u| u.as_str());
        let credential = e.get("credential").and_then(|c| c.as_str());
        let urls = match e.get("urls") {
            Some(serde_json::Value::Array(a)) => {
                a.iter().filter_map(|u| u.as_str()).collect::<Vec<_>>()
            }
            Some(serde_json::Value::String(s)) => vec![s.as_str()],
            _ => continue,
        };
        for url in urls {
            if !usable(url) {
                continue;
            }
            let needs_auth = url.starts_with("turn:") || url.starts_with("turns:");
            out.push(IceServerConfig {
                url: url.to_string(),
                username: needs_auth.then(|| username.unwrap_or_default().to_string()),
                credential: needs_auth.then(|| credential.unwrap_or_default().to_string()),
            });
        }
    }
    out
}

/// Whether this end can actually use a URL it was given.
///
/// The mint answers with every way in, including ones this client does not
/// speak. Keeping a TCP or TLS URL would mean advertising a relayed candidate
/// on a transport nothing here can carry — ICE would select it and then fail
/// on it, which is worse than never offering it.
fn usable(url: &str) -> bool {
    let scheme_ok = url.starts_with("stun:") || url.starts_with("turn:");
    // No `transport` parameter means UDP, which is what RFC 7065 says and what
    // every implementation does.
    let udp = match url.split_once("transport=") {
        Some((_, rest)) => rest.split('&').next() == Some("udp"),
        None => true,
    };
    scheme_ok && udp
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exactly what Cloudflare answered, captured from a real mint.
    ///
    /// Two entries rather than the single object the docs show, and the STUN
    /// one carries no credentials — so the parser has to read username and
    /// credential per entry rather than once for the whole answer.
    const REAL_SHAPE: &str = r#"{
      "iceServers": [
        {
          "urls": [
            "stun:stun.cloudflare.com:3478",
            "stun:stun.cloudflare.com:53"
          ]
        },
        {
          "urls": [
            "turn:turn.cloudflare.com:3478?transport=udp",
            "turn:turn.cloudflare.com:3478?transport=tcp",
            "turns:turn.cloudflare.com:5349?transport=tcp",
            "turn:turn.cloudflare.com:53?transport=udp",
            "turn:turn.cloudflare.com:80?transport=tcp",
            "turns:turn.cloudflare.com:443?transport=tcp"
          ],
          "username": "user-abc",
          "credential": "secret-xyz"
        }
      ]
    }"#;

    #[test]
    fn a_mint_answer_yields_servers_this_client_can_use() {
        let got = parse(REAL_SHAPE);
        let urls: Vec<&str> = got.iter().map(|s| s.url.as_str()).collect();
        assert_eq!(
            urls,
            vec![
                "stun:stun.cloudflare.com:3478",
                "stun:stun.cloudflare.com:53",
                "turn:turn.cloudflare.com:3478?transport=udp",
                "turn:turn.cloudflare.com:53?transport=udp",
            ],
            "kept a transport this client cannot speak"
        );
    }

    #[test]
    fn only_the_turn_entries_carry_credentials() {
        // A STUN entry with a username is not wrong so much as misleading: it
        // would make `allocate_relay` treat it as a TURN candidate and spend a
        // round trip finding out otherwise.
        let got = parse(REAL_SHAPE);
        let stun = got.iter().find(|s| s.url.starts_with("stun:")).unwrap();
        assert_eq!(stun.username, None);
        assert_eq!(stun.credential, None);

        let turn = got.iter().find(|s| s.url.starts_with("turn:")).unwrap();
        assert_eq!(turn.username.as_deref(), Some("user-abc"));
        assert_eq!(turn.credential.as_deref(), Some("secret-xyz"));
    }

    #[test]
    fn a_list_of_entries_is_read_as_well_as_a_single_one() {
        let got = parse(
            r#"{"iceServers":[
                 {"urls":"stun:a.example:3478"},
                 {"urls":["turn:b.example:3478"],"username":"u","credential":"p"}
               ]}"#,
        );
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].username.as_deref(), Some("u"));
    }

    #[test]
    fn an_answer_that_makes_no_sense_yields_nothing_rather_than_panicking() {
        // It comes off the network, so all of these will happen eventually. The
        // session still has host and reflexive candidates.
        for body in [
            "",
            "null",
            "{}",
            r#"{"iceServers":null}"#,
            r#"{"iceServers":{"urls":[]}}"#,
            r#"{"iceServers":{"urls":123}}"#,
            r#"{"errors":[{"message":"bad token"}]}"#,
            "<html>502</html>",
        ] {
            assert!(parse(body).is_empty(), "{body}");
        }
    }

    #[test]
    fn transports_this_client_cannot_speak_are_left_out() {
        assert!(usable("stun:s.example:3478"));
        assert!(usable("turn:t.example:3478"));
        assert!(usable("turn:t.example:3478?transport=udp"));
        assert!(!usable("turn:t.example:3478?transport=tcp"));
        assert!(!usable("turns:t.example:5349?transport=tcp"));
        assert!(!usable("turns:t.example:5349"));
    }
}
