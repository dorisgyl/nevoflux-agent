//! Waking a phone that is not connected (design §6).
//!
//! The daemon posts to the push service itself rather than asking the relay to
//! do it. The usual reason to put a server in the middle — the sender might be
//! offline — cannot arise here: a push is triggered by an agent on this machine
//! stopping to ask a question, so at that instant the daemon is by construction
//! running and connected. What routing through the cloud would buy is nothing,
//! and what it would cost is a device-level identifier (the endpoint URL is a
//! bearer capability to wake that phone) stored somewhere it need not be.
//!
//! Two specs, both implemented here because the alternative is a dependency
//! that pulls its own HTTP client:
//!
//! - **RFC 8292 (VAPID)** — an ES256 JWT identifying this daemon, so the push
//!   service will accept the request without an account anywhere.
//! - **RFC 8291 / RFC 8188 (`aes128gcm`)** — the payload is encrypted to the
//!   subscription's own key. The push service forwards ciphertext it cannot
//!   read, which is the same property the relay has.
//!
//! **What rides in the payload is deliberately almost nothing** — a session id
//! and what kind of question it is. The notification text is fixed and lives in
//! the service worker. A lock screen is read by whoever is standing there, and
//! "confirm the ¥5,000 payment to X" and "something needs confirming" are not
//! the same disclosure.

use aes_gcm::aead::{Aead, Payload};
use aes_gcm::{Aes128Gcm, KeyInit};
use base64::Engine;
use p256::ecdsa::{signature::Signer, Signature, SigningKey};
use p256::elliptic_curve::sec1::ToEncodedPoint;
use p256::{PublicKey, SecretKey};
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// base64url, no padding — what every one of these specs uses.
const B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// How long a VAPID assertion is good for.
///
/// Twelve hours against a 24-hour ceiling: comfortably inside what services
/// accept, and long enough that clock skew on either end is never the reason a
/// push fails.
const VAPID_TTL_SECS: u64 = 12 * 60 * 60;

/// How long the push service should hold an undelivered message.
///
/// Four hours. A question that has gone unanswered longer than that is one the
/// person has already missed, and delivering it then is a notification about
/// something they can no longer usefully act on.
const PUSH_TTL_SECS: u32 = 4 * 60 * 60;

/// Record size for the single-record body. Any value larger than the payload
/// works; this is the conventional one.
const RECORD_SIZE: u32 = 4096;

/// One browser's push subscription, exactly as `PushSubscription.toJSON()`
/// gives it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Subscription {
    /// Where to POST. A bearer capability to wake this device — treat as a
    /// secret even though it is not called one.
    pub endpoint: String,
    /// The subscription's P-256 public key, base64url, uncompressed SEC1.
    pub p256dh: String,
    /// The subscription's 16-byte auth secret, base64url.
    pub auth: String,
}

/// This daemon's VAPID identity.
///
/// Persisted beside `account-token` and the pairings: every subscription a
/// device holds is bound to this public key, so replacing it silently
/// invalidates all of them — and the device is told nothing. That is why it is
/// backed up with the rest, and why the service worker reports the key it holds
/// on every open so a mismatch is noticed by somebody who can act on it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VapidKey {
    /// PKCS#8-free raw scalar, base64url.
    secret: String,
    /// Uncompressed SEC1 public key, base64url. This is what the browser calls
    /// `applicationServerKey`.
    pub public: String,
    /// The `sub` claim. Required: Apple rejects an assertion without one with
    /// `BadJwtToken`, and the failure gives no hint which field was missing.
    pub subject: String,
}

impl VapidKey {
    /// Generate a fresh identity.
    pub fn generate(subject: impl Into<String>) -> Self {
        let secret = SecretKey::random(&mut rand::thread_rng());
        let public = secret.public_key().to_encoded_point(false);
        Self {
            secret: B64.encode(secret.to_bytes()),
            public: B64.encode(public.as_bytes()),
            subject: subject.into(),
        }
    }

    fn signing_key(&self) -> Option<SigningKey> {
        let bytes = B64.decode(&self.secret).ok()?;
        SigningKey::from_slice(&bytes).ok()
    }

    /// The `Authorization` header value for a push to `endpoint`.
    ///
    /// The audience is the endpoint's **origin**, not the whole URL: a service
    /// that is handed the full path rejects the assertion, and says only that
    /// the token is bad.
    pub fn authorization(&self, endpoint: &str, now: u64) -> Option<String> {
        let aud = origin_of(endpoint)?;
        let header = B64.encode(br#"{"typ":"JWT","alg":"ES256"}"#);
        let claims = serde_json::json!({
            "aud": aud,
            "exp": now + VAPID_TTL_SECS,
            "sub": self.subject,
        });
        let claims = B64.encode(serde_json::to_vec(&claims).ok()?);
        let signing_input = format!("{header}.{claims}");
        let signature: Signature = self.signing_key()?.sign(signing_input.as_bytes());
        // JWS wants the fixed-width r‖s form, not the DER the signer prints.
        let jwt = format!("{signing_input}.{}", B64.encode(signature.to_bytes()));
        Some(format!("vapid t={jwt}, k={}", self.public))
    }
}

/// Where the VAPID identity lives, and how it is loaded.
///
/// Beside `account-token` and `pairings.json`, because losing any of the three
/// has the same shape of consequence and they belong in one backup.
///
/// **Replacing this key silently invalidates every subscription** bound to it:
/// the push service still accepts the request, the phone simply never hears
/// anything, and neither end is told. That is why it is generated once and kept
/// — and why the device reports the key it holds every time it opens, so a
/// mismatch is noticed by somebody who can do something about it.
pub struct VapidStore {
    path: std::path::PathBuf,
}

impl VapidStore {
    /// A store backed by `path`.
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The daemon's own location for it.
    pub fn default_path() -> std::path::PathBuf {
        crate::paths::resolve_from_daemon()
            .data_dir
            .join("vapid.json")
    }

    /// Load the identity, generating and persisting one the first time.
    ///
    /// A file that exists but does not parse is an error, never silently
    /// replaced — replacing it is exactly the failure described above, and
    /// doing it automatically would turn a bad file into a fleet of dead
    /// subscriptions with no trace of why.
    pub fn load_or_generate(&self, subject: &str) -> std::io::Result<VapidKey> {
        match std::fs::read_to_string(&self.path) {
            Ok(raw) => serde_json::from_str(&raw).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{} is unreadable; refusing to replace it — every push subscription is bound to it: {e}",
                        self.path.display()
                    ),
                )
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let key = VapidKey::generate(subject);
                if let Some(dir) = self.path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                let tmp = self.path.with_extension("json.tmp");
                std::fs::write(&tmp, serde_json::to_string_pretty(&key)?)?;
                std::fs::rename(&tmp, &self.path)?;
                Ok(key)
            }
            Err(e) => Err(e),
        }
    }
}

/// The `sub` claim every assertion carries.
///
/// Required, not optional: Apple answers `BadJwtToken` without one and says
/// nothing about which field was missing.
pub const DEFAULT_SUBJECT: &str = "mailto:push@nevoflux.app";

/// The scheme + host + port of an endpoint, which is what `aud` must be.
fn origin_of(endpoint: &str) -> Option<String> {
    let (scheme, rest) = endpoint.split_once("://")?;
    let host = rest.split('/').next()?;
    if host.is_empty() {
        return None;
    }
    Some(format!("{scheme}://{host}"))
}

/// What a push says. Deliberately not much — see the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushPayload {
    /// Which session needs attention, so the worker can tag the notification
    /// and later withdraw it.
    pub session_id: String,
    /// `"gate"` or `"plan"`, so the worker can pick its fixed wording.
    pub kind: String,
}

/// Encrypt `plaintext` to `sub` per RFC 8291, producing an `aes128gcm` body.
///
/// `salt` and `ephemeral` are arguments rather than generated inside so the
/// construction can be tested against fixed inputs; [`encrypt`] is the calling
/// convention everything else uses.
fn encrypt_with(
    sub: &Subscription,
    plaintext: &[u8],
    salt: [u8; 16],
    ephemeral: SecretKey,
) -> Option<Vec<u8>> {
    let client_public_bytes = B64.decode(&sub.p256dh).ok()?;
    let auth = B64.decode(&sub.auth).ok()?;
    let client_public = PublicKey::from_sec1_bytes(&client_public_bytes).ok()?;

    let server_public = ephemeral.public_key().to_encoded_point(false);
    let server_public_bytes = server_public.as_bytes();

    // ECDH, then the two-stage derivation RFC 8291 §3.4 specifies: the auth
    // secret salts the first extract, and the info string binds the result to
    // both parties' public keys so a shared secret cannot be replayed against a
    // different subscription.
    let shared = p256::ecdh::diffie_hellman(
        ephemeral.to_nonzero_scalar(),
        client_public.as_affine(),
    );
    let mut key_info = Vec::with_capacity(14 + 1 + 65 + 65);
    key_info.extend_from_slice(b"WebPush: info\0");
    key_info.extend_from_slice(&client_public_bytes);
    key_info.extend_from_slice(server_public_bytes);

    let mut ikm = [0u8; 32];
    hkdf::Hkdf::<sha2::Sha256>::new(Some(&auth), shared.raw_secret_bytes())
        .expand(&key_info, &mut ikm)
        .ok()?;

    let prk = hkdf::Hkdf::<sha2::Sha256>::new(Some(&salt), &ikm);
    let mut cek = [0u8; 16];
    prk.expand(b"Content-Encoding: aes128gcm\0", &mut cek).ok()?;
    let mut nonce = [0u8; 12];
    prk.expand(b"Content-Encoding: nonce\0", &mut nonce).ok()?;

    // RFC 8188 records carry a delimiter; 0x02 marks the last one.
    let mut padded = plaintext.to_vec();
    padded.push(0x02);
    let ciphertext = Aes128Gcm::new_from_slice(&cek)
        .ok()?
        .encrypt(
            (&nonce).into(),
            Payload {
                msg: &padded,
                aad: b"",
            },
        )
        .ok()?;

    // The aes128gcm content-coding header: salt, record size, key id length,
    // key id (here the server's public key), then the record.
    let mut body = Vec::with_capacity(21 + server_public_bytes.len() + ciphertext.len());
    body.extend_from_slice(&salt);
    body.extend_from_slice(&RECORD_SIZE.to_be_bytes());
    body.push(server_public_bytes.len() as u8);
    body.extend_from_slice(server_public_bytes);
    body.extend_from_slice(&ciphertext);
    Some(body)
}

/// Encrypt `plaintext` to `sub` with a fresh salt and ephemeral key.
pub fn encrypt(sub: &Subscription, plaintext: &[u8]) -> Option<Vec<u8>> {
    let mut salt = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut salt);
    encrypt_with(
        sub,
        plaintext,
        salt,
        SecretKey::random(&mut rand::thread_rng()),
    )
}

/// What became of one push.
///
/// Not a `Result<()>`: the interesting distinction is not success versus
/// failure but *which* failure, because one of them means the subscription is
/// dead and should be dropped rather than retried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delivery {
    /// The service accepted it.
    Accepted,
    /// 404/410 — the subscription is gone. Stop using it; the device has to
    /// subscribe again.
    Expired,
    /// 403 or 401 — the service would not take this VAPID assertion. Usually
    /// the key was replaced, which silently invalidates every subscription
    /// bound to the old one.
    Rejected(String),
    /// Anything else, including no answer at all: a firewall that blocks FCM or
    /// APNs looks like this, and it is why "push is not reaching you" has to be
    /// a state somebody can see.
    Failed(String),
}

impl Delivery {
    /// Whether this outcome means the subscription should be forgotten.
    pub fn is_gone(&self) -> bool {
        matches!(self, Delivery::Expired)
    }
}

/// Classify a push service's response.
pub fn classify(status: u16, body: &str) -> Delivery {
    match status {
        200..=299 => Delivery::Accepted,
        404 | 410 => Delivery::Expired,
        401 | 403 => Delivery::Rejected(format!("{status}: {}", body.trim())),
        _ => Delivery::Failed(format!("{status}: {}", body.trim())),
    }
}

/// Post one push.
pub async fn send(
    client: &reqwest::Client,
    key: &VapidKey,
    sub: &Subscription,
    payload: &PushPayload,
) -> Delivery {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let Some(auth) = key.authorization(&sub.endpoint, now) else {
        return Delivery::Failed("could not build a VAPID assertion".into());
    };
    let Some(plaintext) = serde_json::to_vec(payload).ok() else {
        return Delivery::Failed("payload would not serialize".into());
    };
    let Some(body) = encrypt(sub, &plaintext) else {
        return Delivery::Failed("subscription keys would not encrypt".into());
    };

    let res = client
        .post(&sub.endpoint)
        .header("Authorization", auth)
        .header("Content-Encoding", "aes128gcm")
        .header("Content-Type", "application/octet-stream")
        .header("TTL", PUSH_TTL_SECS.to_string())
        // A question waiting on a person is worth waking the screen for; the
        // default would let the service batch it until the device next wakes on
        // its own, which can be a long time on a phone in a pocket.
        .header("Urgency", "high")
        .body(body)
        .send()
        .await;

    match res {
        Ok(r) => {
            let status = r.status().as_u16();
            let text = r.text().await.unwrap_or_default();
            classify(status, &text)
        }
        Err(e) => Delivery::Failed(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p256::ecdh::diffie_hellman;

    /// A subscription with a keypair the test also holds, so what the daemon
    /// encrypts can actually be opened again.
    fn subscription() -> (Subscription, SecretKey) {
        let secret = SecretKey::random(&mut rand::thread_rng());
        let public = secret.public_key().to_encoded_point(false);
        let mut auth = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut auth);
        (
            Subscription {
                endpoint: "https://fcm.googleapis.com/fcm/send/abc123".into(),
                p256dh: B64.encode(public.as_bytes()),
                auth: B64.encode(auth),
            },
            secret,
        )
    }

    /// The receiving half of RFC 8291, as a browser would do it.
    fn decrypt(body: &[u8], sub: &Subscription, client_secret: &SecretKey) -> Vec<u8> {
        let salt = &body[0..16];
        let idlen = body[20] as usize;
        let server_public_bytes = &body[21..21 + idlen];
        let ciphertext = &body[21 + idlen..];

        let server_public = PublicKey::from_sec1_bytes(server_public_bytes).unwrap();
        let shared = diffie_hellman(
            client_secret.to_nonzero_scalar(),
            server_public.as_affine(),
        );
        let client_public_bytes = B64.decode(&sub.p256dh).unwrap();
        let auth = B64.decode(&sub.auth).unwrap();

        let mut key_info = Vec::new();
        key_info.extend_from_slice(b"WebPush: info\0");
        key_info.extend_from_slice(&client_public_bytes);
        key_info.extend_from_slice(server_public_bytes);
        let mut ikm = [0u8; 32];
        hkdf::Hkdf::<sha2::Sha256>::new(Some(&auth), shared.raw_secret_bytes())
            .expand(&key_info, &mut ikm)
            .unwrap();

        let prk = hkdf::Hkdf::<sha2::Sha256>::new(Some(salt), &ikm);
        let mut cek = [0u8; 16];
        prk.expand(b"Content-Encoding: aes128gcm\0", &mut cek).unwrap();
        let mut nonce = [0u8; 12];
        prk.expand(b"Content-Encoding: nonce\0", &mut nonce).unwrap();

        let mut out = Aes128Gcm::new_from_slice(&cek)
            .unwrap()
            .decrypt(
                (&nonce).into(),
                Payload {
                    msg: ciphertext,
                    aad: b"",
                },
            )
            .expect("the subscription's own key must open it");
        // Drop the RFC 8188 record delimiter.
        assert_eq!(out.pop(), Some(0x02));
        out
    }

    #[test]
    fn what_is_encrypted_to_a_subscription_opens_with_its_key() {
        let (sub, client_secret) = subscription();
        let body = encrypt(&sub, b"{\"session_id\":\"s1\",\"kind\":\"gate\"}").unwrap();
        assert_eq!(
            decrypt(&body, &sub, &client_secret),
            b"{\"session_id\":\"s1\",\"kind\":\"gate\"}"
        );
    }

    #[test]
    fn the_body_has_the_aes128gcm_header_a_service_expects() {
        let (sub, _) = subscription();
        let body = encrypt(&sub, b"x").unwrap();
        assert_eq!(&body[16..20], &RECORD_SIZE.to_be_bytes());
        // Uncompressed SEC1 P-256: 65 bytes, leading 0x04.
        assert_eq!(body[20], 65);
        assert_eq!(body[21], 0x04);
        assert!(body.len() > 21 + 65);
    }

    #[test]
    fn a_fresh_salt_and_key_every_time() {
        // Reusing either would leak that two pushes carry the same thing, and
        // reusing the nonce with the same key would be much worse than that.
        let (sub, _) = subscription();
        let a = encrypt(&sub, b"same").unwrap();
        let b = encrypt(&sub, b"same").unwrap();
        assert_ne!(a[0..16], b[0..16], "salt must differ");
        assert_ne!(a[21..86], b[21..86], "ephemeral key must differ");
    }

    #[test]
    fn a_subscription_with_unusable_keys_fails_rather_than_panicking() {
        let (mut sub, _) = subscription();
        sub.p256dh = "not base64!!".into();
        assert!(encrypt(&sub, b"x").is_none());
        let (mut sub, _) = subscription();
        sub.p256dh = B64.encode([0u8; 10]);
        assert!(encrypt(&sub, b"x").is_none());
    }

    #[test]
    fn the_assertion_names_the_origin_not_the_whole_url() {
        // A service handed the full path rejects the token and says only that
        // it is bad, which is a long afternoon if you do not know this.
        let key = VapidKey::generate("mailto:push@nevoflux.app");
        let auth = key
            .authorization("https://fcm.googleapis.com/fcm/send/abc123", 1_000)
            .unwrap();
        let jwt = auth
            .strip_prefix("vapid t=")
            .unwrap()
            .split(',')
            .next()
            .unwrap();
        let claims: serde_json::Value =
            serde_json::from_slice(&B64.decode(jwt.split('.').nth(1).unwrap()).unwrap()).unwrap();
        assert_eq!(claims["aud"], "https://fcm.googleapis.com");
        assert_eq!(claims["sub"], "mailto:push@nevoflux.app");
        assert_eq!(claims["exp"], 1_000 + VAPID_TTL_SECS as i64);
    }

    #[test]
    fn the_assertion_carries_the_key_the_browser_subscribed_with() {
        let key = VapidKey::generate("mailto:push@nevoflux.app");
        let auth = key.authorization("https://example.com/p/1", 0).unwrap();
        assert!(auth.contains(&format!("k={}", key.public)));
    }

    #[test]
    fn the_signature_verifies_against_the_advertised_key() {
        use p256::ecdsa::signature::Verifier;
        use p256::ecdsa::VerifyingKey;

        let key = VapidKey::generate("mailto:push@nevoflux.app");
        let auth = key.authorization("https://example.com/p/1", 0).unwrap();
        let jwt = auth
            .strip_prefix("vapid t=")
            .unwrap()
            .split(',')
            .next()
            .unwrap();
        let (signing_input, sig_b64) = jwt.rsplit_once('.').unwrap();
        let sig = Signature::from_slice(&B64.decode(sig_b64).unwrap()).unwrap();
        let public = PublicKey::from_sec1_bytes(&B64.decode(&key.public).unwrap()).unwrap();
        VerifyingKey::from(public)
            .verify(signing_input.as_bytes(), &sig)
            .expect("a service must be able to check this");
    }

    #[test]
    fn an_endpoint_that_is_not_a_url_produces_no_assertion() {
        let key = VapidKey::generate("mailto:x@y.z");
        assert!(key.authorization("not-a-url", 0).is_none());
        assert!(key.authorization("https://", 0).is_none());
    }

    #[test]
    fn a_dead_subscription_is_told_apart_from_a_blocked_network() {
        // The distinction the caller acts on: one means drop the subscription,
        // the other means say so and keep it.
        assert_eq!(classify(201, ""), Delivery::Accepted);
        assert!(classify(410, "").is_gone());
        assert!(classify(404, "").is_gone());
        assert!(!classify(403, "bad vapid").is_gone());
        assert!(matches!(classify(403, "bad vapid"), Delivery::Rejected(_)));
        assert!(matches!(classify(500, "oops"), Delivery::Failed(_)));
        assert!(!classify(500, "oops").is_gone());
    }

    #[test]
    fn an_identity_is_generated_once_and_then_kept() {
        // Every subscription is bound to this key. Generating a second one
        // would invalidate all of them, and nothing on either end would say so.
        let dir = tempfile::tempdir().unwrap();
        let store = VapidStore::new(dir.path().join("vapid.json"));
        let first = store.load_or_generate(DEFAULT_SUBJECT).unwrap();
        let second = store.load_or_generate(DEFAULT_SUBJECT).unwrap();
        assert_eq!(first.public, second.public);
    }

    #[test]
    fn a_corrupt_identity_is_reported_never_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vapid.json");
        std::fs::write(&path, "{ not json").unwrap();
        let err = VapidStore::new(&path)
            .load_or_generate(DEFAULT_SUBJECT)
            .expect_err("a bad key file must not be silently replaced");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(path.exists(), "and the file is still there to look at");
    }

    #[test]
    fn a_payload_says_which_session_and_nothing_about_it() {
        // A lock screen is read by whoever is standing there.
        let payload = PushPayload {
            session_id: "s1".into(),
            kind: "gate".into(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert_eq!(json, r#"{"session_id":"s1","kind":"gate"}"#);
        assert!(!json.contains("prompt"));
    }
}
