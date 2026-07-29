//! Registry for tracking proxies and active requests.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

/// Information about a connected proxy.
#[derive(Debug, Clone)]
pub struct ProxyInfo {
    /// Unique proxy identifier.
    pub proxy_id: String,
    /// Process ID of the proxy.
    pub pid: u32,
    /// Time of last heartbeat.
    pub last_heartbeat: Instant,
    /// Time when proxy registered.
    pub registered_at: Instant,
}

impl ProxyInfo {
    /// Create a new proxy info.
    pub fn new(proxy_id: impl Into<String>, pid: u32) -> Self {
        let now = Instant::now();
        Self {
            proxy_id: proxy_id.into(),
            pid,
            last_heartbeat: now,
            registered_at: now,
        }
    }

    /// Update the heartbeat timestamp.
    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
    }

    /// Check if the proxy has timed out.
    pub fn is_timed_out(&self, timeout: std::time::Duration) -> bool {
        self.last_heartbeat.elapsed() > timeout
    }
}

/// Registry for tracking connected proxies.
pub struct ProxyRegistry {
    proxies: RwLock<HashMap<String, ProxyInfo>>,
}

impl Default for ProxyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProxyRegistry {
    /// Create a new proxy registry.
    pub fn new() -> Self {
        Self {
            proxies: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new proxy.
    pub fn register(&self, proxy_id: impl Into<String>, pid: u32) {
        let proxy_id = proxy_id.into();
        let info = ProxyInfo::new(&proxy_id, pid);
        self.proxies.write().unwrap().insert(proxy_id, info);
    }

    /// Unregister a proxy.
    pub fn unregister(&self, proxy_id: &str) -> Option<ProxyInfo> {
        self.proxies.write().unwrap().remove(proxy_id)
    }

    /// Check if a proxy is registered.
    pub fn is_registered(&self, proxy_id: &str) -> bool {
        self.proxies.read().unwrap().contains_key(proxy_id)
    }

    /// Get proxy info.
    pub fn get(&self, proxy_id: &str) -> Option<ProxyInfo> {
        self.proxies.read().unwrap().get(proxy_id).cloned()
    }

    /// Update heartbeat for a proxy.
    pub fn heartbeat(&self, proxy_id: &str) -> bool {
        if let Some(info) = self.proxies.write().unwrap().get_mut(proxy_id) {
            info.update_heartbeat();
            true
        } else {
            false
        }
    }

    /// Get all registered proxy IDs.
    pub fn all_proxy_ids(&self) -> Vec<String> {
        self.proxies.read().unwrap().keys().cloned().collect()
    }

    /// Get count of active proxies.
    pub fn active_count(&self) -> usize {
        self.proxies.read().unwrap().len()
    }

    /// Remove timed out proxies.
    pub fn remove_timed_out(&self, timeout: std::time::Duration) -> Vec<String> {
        let mut removed = Vec::new();
        let mut proxies = self.proxies.write().unwrap();

        proxies.retain(|id, info| {
            if info.is_timed_out(timeout) {
                removed.push(id.clone());
                false
            } else {
                true
            }
        });

        removed
    }
}

/// A registered browser connection (declared `role:"browser"` at registration),
/// addressable for `browser_*` tool routing independently of who sent the chat.
/// In the headless automation model there is exactly one browser per daemon.
#[derive(Debug, Clone)]
pub struct BrowserEntry {
    /// Proxy ID of the browser's connection.
    pub proxy_id: String,
    /// Routing identity bytes (proxy_id as UTF-8).
    pub client_identity: Vec<u8>,
    /// When the browser registered.
    pub registered_at: Instant,
    /// Last heartbeat.
    pub last_heartbeat: Instant,
}

/// Error resolving a browser binding.
#[derive(Debug, thiserror::Error)]
pub enum BrowserBindError {
    /// No browser has registered.
    #[error("no browser registered")]
    NoBrowser,
    /// More than one browser is registered (headless model expects exactly one).
    #[error("ambiguous browser binding: {0} browsers registered")]
    Ambiguous(usize),
    /// Timed out waiting for a browser to register.
    #[error("timed out waiting for a browser to register")]
    Timeout,
}

/// Registry of connections that declared `role:"browser"`. Distinct from
/// [`ProxyRegistry`] (which tracks all proxies) and [`SessionProxyTracker`]
/// (the `/loop` borrow hack): this is the explicit routing target for
/// `browser_*` tools in headless/automation sessions.
pub struct BrowserRegistry {
    browsers: RwLock<HashMap<String, BrowserEntry>>,
}

impl Default for BrowserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowserRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            browsers: RwLock::new(HashMap::new()),
        }
    }

    /// Record a browser connection.
    pub fn register(&self, proxy_id: impl Into<String>, client_identity: Vec<u8>) {
        let proxy_id = proxy_id.into();
        let now = Instant::now();
        self.browsers.write().unwrap().insert(
            proxy_id.clone(),
            BrowserEntry {
                proxy_id,
                client_identity,
                registered_at: now,
                last_heartbeat: now,
            },
        );
    }

    /// Remove a browser connection (on disconnect).
    pub fn unregister(&self, proxy_id: &str) -> Option<BrowserEntry> {
        self.browsers.write().unwrap().remove(proxy_id)
    }

    /// Number of registered browsers.
    pub fn count(&self) -> usize {
        self.browsers.read().unwrap().len()
    }

    /// Resolve the single registered browser (headless model: exactly one).
    pub fn single(&self) -> Result<BrowserEntry, BrowserBindError> {
        let map = self.browsers.read().unwrap();
        match map.len() {
            0 => Err(BrowserBindError::NoBrowser),
            1 => Ok(map.values().next().unwrap().clone()),
            n => Err(BrowserBindError::Ambiguous(n)),
        }
    }

    /// Wait until at least one browser is registered, then resolve it.
    /// Returns [`BrowserBindError::Timeout`] if none registers in `timeout`.
    pub async fn wait_for_browser(
        &self,
        timeout: std::time::Duration,
    ) -> Result<BrowserEntry, BrowserBindError> {
        let start = Instant::now();
        loop {
            match self.single() {
                Ok(e) => return Ok(e),
                Err(BrowserBindError::NoBrowser) if start.elapsed() < timeout => {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
                Err(BrowserBindError::NoBrowser) => return Err(BrowserBindError::Timeout),
                Err(other) => return Err(other),
            }
        }
    }

    /// Snapshot of currently-registered proxy ids. Used by headless launch to
    /// remember which browsers were already live *before* spawning a new one,
    /// so the new instance can be picked out from the crowd once it registers.
    pub fn proxy_ids(&self) -> std::collections::HashSet<String> {
        self.browsers.read().unwrap().keys().cloned().collect()
    }

    /// Any browser currently registered, regardless of how many (live-policy
    /// availability check — unlike [`Self::single`], multiple registrants are
    /// not an error here). Returns the first entry found, or `None`.
    pub fn any(&self) -> Option<BrowserEntry> {
        self.browsers.read().unwrap().values().next().cloned()
    }

    /// Wait until a browser whose proxy_id is NOT in `exclude` registers, then
    /// resolve it. Solves the race `wait_for_browser` can't: when a live
    /// browser is already registered (in `exclude`) and a *new* (e.g.
    /// headless) instance is expected, `wait_for_browser`/`single` would
    /// either bind the wrong (already-registered) browser instantly or return
    /// `Ambiguous` once both are present. Polls every 50ms like
    /// [`Self::wait_for_browser`]. If multiple non-excluded browsers are
    /// registered, returns the one with the earliest `registered_at`, breaking
    /// exact `Instant` ties on `proxy_id` so the pick is fully deterministic
    /// (two browsers registered in the same instant would otherwise resolve
    /// non-deterministically) rather than erroring.
    pub async fn wait_for_new_browser(
        &self,
        exclude: &std::collections::HashSet<String>,
        timeout: std::time::Duration,
    ) -> Result<BrowserEntry, BrowserBindError> {
        let start = Instant::now();
        loop {
            let candidate = {
                let map = self.browsers.read().unwrap();
                map.values()
                    .filter(|e| !exclude.contains(&e.proxy_id))
                    .min_by_key(|e| (e.registered_at, e.proxy_id.clone()))
                    .cloned()
            };
            if let Some(entry) = candidate {
                return Ok(entry);
            }
            if start.elapsed() < timeout {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            } else {
                return Err(BrowserBindError::Timeout);
            }
        }
    }
}

/// Process-global handle to the daemon's [`BrowserRegistry`], set once at
/// startup so the automation session runner (which runs off the task path,
/// not the connection handler) can resolve the bound browser. Mirrors
/// `crate::loops::CURRENT_LOOP_MANAGER`.
pub static CURRENT_BROWSER_REGISTRY: std::sync::OnceLock<std::sync::Arc<BrowserRegistry>> =
    std::sync::OnceLock::new();

/// Sessions that have a browser of their own.
///
/// `agent_exec` already states the priority for unattended runs: an explicit
/// binding beats borrowing the session's most recent sidebar, and without
/// either the run has `proxy_id=""` and its `browser_*` requests are dropped
/// at the writer lookup. The chat path had only the second rule, because a
/// chat turn has always had a sender worth routing back to. A headless head
/// does not — its sender is a synthetic proxy with nothing behind it — so it
/// registers here and the chat path follows the same order.
///
/// Empty on the desktop, where the sender already *is* the browser.
#[derive(Default)]
pub struct SessionBrowserBindings {
    bindings: RwLock<HashMap<String, BrowserEntry>>,
}

impl SessionBrowserBindings {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Point `session_id` at `entry`, replacing any previous binding — which
    /// is exactly what a restarted browser needs.
    pub fn bind(&self, session_id: &str, entry: BrowserEntry) {
        if let Ok(mut m) = self.bindings.write() {
            m.insert(session_id.to_string(), entry);
        }
    }

    /// The browser bound to `session_id`, if any.
    pub fn get(&self, session_id: &str) -> Option<BrowserEntry> {
        self.bindings.read().ok()?.get(session_id).cloned()
    }

    /// Forget `session_id`'s binding.
    pub fn unbind(&self, session_id: &str) {
        if let Ok(mut m) = self.bindings.write() {
            m.remove(session_id);
        }
    }
}

/// Process-global session→browser bindings, set once at startup by whoever
/// creates them. Mirrors [`CURRENT_BROWSER_REGISTRY`]. Unset on the desktop,
/// where the chat path then skips the lookup entirely.
pub static CURRENT_SESSION_BINDINGS: std::sync::OnceLock<
    std::sync::Arc<SessionBrowserBindings>,
> = std::sync::OnceLock::new();

/// Role a connecting proxy declares in its registration frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterRole {
    /// The headless browser (its `browser_*` tools are the automation target).
    Browser,
    /// A control/sidebar client (the default when `role` is absent).
    Control,
}

/// Parse a registration frame into `(proxy_id, role)`. Absent `role` ⇒ Control,
/// preserving headed behavior. Returns `None` for a non-register / malformed frame.
pub fn parse_register_frame(data: &[u8]) -> Option<(String, RegisterRole)> {
    let val: serde_json::Value = serde_json::from_slice(data).ok()?;
    if val.get("type").and_then(|t| t.as_str()) != Some("register") {
        return None;
    }
    let proxy_id = val.get("proxy_id").and_then(|v| v.as_str())?.to_string();
    let role = match val.get("role").and_then(|v| v.as_str()) {
        Some("browser") => RegisterRole::Browser,
        _ => RegisterRole::Control,
    };
    Some((proxy_id, role))
}

/// Tracks the most-recently-active sidebar proxy per session_id, so that
/// `/loop` iterations can borrow a connected sidebar to fulfill `browser_*`
/// tool calls. Iterations themselves have `proxy_id=""` (no inbound chat
/// connection), and without this tracker their browser requests get dropped
/// at the `No writer for proxy ""` check in `server.rs::browser request handler`.
///
/// Updated by `server.rs` on every Chat-channel message arrival; read by
/// `IterationExecutor` at iteration start.
pub struct SessionProxyTracker {
    map: RwLock<HashMap<String, SessionProxyEntry>>,
}

#[derive(Clone, Debug)]
pub struct SessionProxyEntry {
    pub proxy_id: String,
    pub client_identity: Vec<u8>,
    pub last_seen: Instant,
}

impl Default for SessionProxyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionProxyTracker {
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }

    /// Record that `session_id` was most recently seen on `proxy_id`.
    /// Empty proxy_id is ignored (iteration-internal messages can't borrow
    /// from themselves).
    pub fn note(&self, session_id: &str, proxy_id: &str, client_identity: &[u8]) {
        if session_id.is_empty() || proxy_id.is_empty() {
            return;
        }
        let entry = SessionProxyEntry {
            proxy_id: proxy_id.to_string(),
            client_identity: client_identity.to_vec(),
            last_seen: Instant::now(),
        };
        self.map
            .write()
            .unwrap()
            .insert(session_id.to_string(), entry);
    }

    /// Return the latest sidebar proxy info for `session_id`, if any.
    pub fn latest(&self, session_id: &str) -> Option<SessionProxyEntry> {
        self.map.read().unwrap().get(session_id).cloned()
    }
}

/// Information about an active request.
#[derive(Debug, Clone)]
pub struct ActiveRequest {
    /// Request ID.
    pub request_id: String,
    /// Proxy ID that initiated the request.
    pub proxy_id: String,
    /// Session ID for the request.
    pub session_id: String,
    /// Time when request started.
    pub started_at: Instant,
}

impl ActiveRequest {
    /// Create a new active request.
    pub fn new(
        request_id: impl Into<String>,
        proxy_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            proxy_id: proxy_id.into(),
            session_id: session_id.into(),
            started_at: Instant::now(),
        }
    }

    /// Get elapsed time since request started.
    pub fn elapsed(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }
}

/// Registry for tracking active requests.
pub struct RequestRegistry {
    /// Active requests by request ID.
    requests: RwLock<HashMap<String, ActiveRequest>>,
    /// Request IDs by session ID (for checking if session is busy).
    sessions: RwLock<HashMap<String, String>>,
}

impl Default for RequestRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestRegistry {
    /// Create a new request registry.
    pub fn new() -> Self {
        Self {
            requests: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new active request.
    ///
    /// Returns false if the session already has an active request.
    pub fn register(
        &self,
        request_id: impl Into<String>,
        proxy_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> bool {
        let request_id = request_id.into();
        let proxy_id = proxy_id.into();
        let session_id = session_id.into();

        // Check if session is busy
        {
            let sessions = self.sessions.read().unwrap();
            if sessions.contains_key(&session_id) {
                return false;
            }
        }

        let request = ActiveRequest::new(&request_id, proxy_id, &session_id);

        self.requests
            .write()
            .unwrap()
            .insert(request_id.clone(), request);
        self.sessions
            .write()
            .unwrap()
            .insert(session_id, request_id);

        true
    }

    /// Complete and remove an active request.
    pub fn complete(&self, request_id: &str) -> Option<ActiveRequest> {
        let request = self.requests.write().unwrap().remove(request_id)?;
        self.sessions.write().unwrap().remove(&request.session_id);
        Some(request)
    }

    /// Get an active request.
    pub fn get(&self, request_id: &str) -> Option<ActiveRequest> {
        self.requests.read().unwrap().get(request_id).cloned()
    }

    /// Check if a session has an active request.
    pub fn is_session_busy(&self, session_id: &str) -> bool {
        self.sessions.read().unwrap().contains_key(session_id)
    }

    /// Get the active request ID for a session.
    pub fn get_request_for_session(&self, session_id: &str) -> Option<String> {
        self.sessions.read().unwrap().get(session_id).cloned()
    }

    /// Get count of active requests.
    pub fn active_count(&self) -> usize {
        self.requests.read().unwrap().len()
    }

    /// Remove all requests for a proxy (when proxy disconnects).
    pub fn remove_for_proxy(&self, proxy_id: &str) -> Vec<ActiveRequest> {
        let mut removed = Vec::new();
        let mut requests = self.requests.write().unwrap();
        let mut sessions = self.sessions.write().unwrap();

        let to_remove: Vec<String> = requests
            .iter()
            .filter(|(_, req)| req.proxy_id == proxy_id)
            .map(|(id, _)| id.clone())
            .collect();

        for request_id in to_remove {
            if let Some(request) = requests.remove(&request_id) {
                sessions.remove(&request.session_id);
                removed.push(request);
            }
        }

        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_proxy_info_new() {
        let info = ProxyInfo::new("proxy-001", 12345);

        assert_eq!(info.proxy_id, "proxy-001");
        assert_eq!(info.pid, 12345);
    }

    #[test]
    fn test_proxy_info_heartbeat() {
        let mut info = ProxyInfo::new("proxy-001", 12345);
        let old_heartbeat = info.last_heartbeat;

        std::thread::sleep(Duration::from_millis(10));
        info.update_heartbeat();

        assert!(info.last_heartbeat > old_heartbeat);
    }

    #[test]
    fn test_proxy_info_timeout() {
        let info = ProxyInfo::new("proxy-001", 12345);

        // Should not be timed out immediately
        assert!(!info.is_timed_out(Duration::from_secs(30)));

        // Should be timed out with zero timeout
        assert!(info.is_timed_out(Duration::ZERO));
    }

    #[test]
    fn test_proxy_registry_register() {
        let registry = ProxyRegistry::new();

        registry.register("proxy-001", 12345);

        assert!(registry.is_registered("proxy-001"));
        assert!(!registry.is_registered("proxy-002"));
        assert_eq!(registry.active_count(), 1);
    }

    #[test]
    fn test_proxy_registry_unregister() {
        let registry = ProxyRegistry::new();

        registry.register("proxy-001", 12345);
        let info = registry.unregister("proxy-001");

        assert!(info.is_some());
        assert!(!registry.is_registered("proxy-001"));
    }

    #[test]
    fn test_proxy_registry_heartbeat() {
        let registry = ProxyRegistry::new();

        registry.register("proxy-001", 12345);

        assert!(registry.heartbeat("proxy-001"));
        assert!(!registry.heartbeat("proxy-002"));
    }

    #[test]
    fn test_proxy_registry_all_proxy_ids() {
        let registry = ProxyRegistry::new();

        registry.register("proxy-001", 1);
        registry.register("proxy-002", 2);
        registry.register("proxy-003", 3);

        let ids = registry.all_proxy_ids();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_active_request_new() {
        let request = ActiveRequest::new("req-001", "proxy-001", "session-001");

        assert_eq!(request.request_id, "req-001");
        assert_eq!(request.proxy_id, "proxy-001");
        assert_eq!(request.session_id, "session-001");
    }

    #[test]
    fn test_request_registry_register() {
        let registry = RequestRegistry::new();

        assert!(registry.register("req-001", "proxy-001", "session-001"));

        let request = registry.get("req-001");
        assert!(request.is_some());
        assert_eq!(request.unwrap().proxy_id, "proxy-001");
    }

    #[test]
    fn test_request_registry_session_busy() {
        let registry = RequestRegistry::new();

        // First request should succeed
        assert!(registry.register("req-001", "proxy-001", "session-001"));

        // Second request for same session should fail
        assert!(!registry.register("req-002", "proxy-001", "session-001"));

        assert!(registry.is_session_busy("session-001"));
    }

    #[test]
    fn test_request_registry_complete() {
        let registry = RequestRegistry::new();

        registry.register("req-001", "proxy-001", "session-001");

        let completed = registry.complete("req-001");
        assert!(completed.is_some());

        // Session should no longer be busy
        assert!(!registry.is_session_busy("session-001"));

        // Request should no longer exist
        assert!(registry.get("req-001").is_none());
    }

    #[test]
    fn test_request_registry_remove_for_proxy() {
        let registry = RequestRegistry::new();

        registry.register("req-001", "proxy-001", "session-001");
        registry.register("req-002", "proxy-001", "session-002");
        registry.register("req-003", "proxy-002", "session-003");

        let removed = registry.remove_for_proxy("proxy-001");

        assert_eq!(removed.len(), 2);
        assert_eq!(registry.active_count(), 1);
    }

    #[test]
    fn browser_registry_single_ok_and_ambiguous() {
        let r = BrowserRegistry::new();
        assert!(matches!(r.single(), Err(BrowserBindError::NoBrowser)));
        r.register("proxy-b1", b"proxy-b1".to_vec());
        let e = r.single().expect("one browser");
        assert_eq!(e.proxy_id, "proxy-b1");
        assert_eq!(e.client_identity, b"proxy-b1".to_vec());
        r.register("proxy-b2", b"proxy-b2".to_vec());
        assert!(matches!(r.single(), Err(BrowserBindError::Ambiguous(2))));
        r.unregister("proxy-b2");
        assert_eq!(r.single().unwrap().proxy_id, "proxy-b1");
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn browser_registry_proxy_ids_snapshot() {
        let r = BrowserRegistry::new();
        assert!(r.proxy_ids().is_empty());
        r.register("proxy-b1", b"proxy-b1".to_vec());
        r.register("proxy-b2", b"proxy-b2".to_vec());
        let ids = r.proxy_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("proxy-b1"));
        assert!(ids.contains("proxy-b2"));
    }

    #[test]
    fn browser_registry_any_empty_and_nonempty() {
        let r = BrowserRegistry::new();
        assert!(r.any().is_none());
        r.register("proxy-b1", b"proxy-b1".to_vec());
        let e = r.any().expect("one browser registered");
        assert_eq!(e.proxy_id, "proxy-b1");
        // Doesn't error with multiple registered, unlike `single`.
        r.register("proxy-b2", b"proxy-b2".to_vec());
        assert!(r.any().is_some());
    }

    #[tokio::test]
    async fn wait_for_new_browser_immediate_hit_when_non_excluded_present() {
        let r = BrowserRegistry::new();
        r.register("proxy-old", b"proxy-old".to_vec());
        let exclude: std::collections::HashSet<String> = std::collections::HashSet::new();
        let e = r
            .wait_for_new_browser(&exclude, Duration::from_millis(200))
            .await
            .expect("should resolve immediately");
        assert_eq!(e.proxy_id, "proxy-old");
    }

    #[tokio::test]
    async fn wait_for_new_browser_excludes_already_registered() {
        let r = std::sync::Arc::new(BrowserRegistry::new());
        r.register("proxy-live", b"proxy-live".to_vec());
        let mut exclude = std::collections::HashSet::new();
        exclude.insert("proxy-live".to_string());

        let r2 = r.clone();
        let handle = tokio::spawn(async move {
            r2.wait_for_new_browser(&exclude, Duration::from_secs(2))
                .await
        });

        // Give the waiter a moment to start polling, then register the
        // genuinely new (headless) browser.
        tokio::time::sleep(Duration::from_millis(120)).await;
        r.register("proxy-headless", b"proxy-headless".to_vec());

        let e = handle
            .await
            .unwrap()
            .expect("should resolve to the new browser");
        assert_eq!(e.proxy_id, "proxy-headless");
    }

    #[tokio::test]
    async fn wait_for_new_browser_times_out_when_only_excluded_present() {
        let r = BrowserRegistry::new();
        r.register("proxy-live", b"proxy-live".to_vec());
        let mut exclude = std::collections::HashSet::new();
        exclude.insert("proxy-live".to_string());

        let result = r
            .wait_for_new_browser(&exclude, Duration::from_millis(120))
            .await;
        assert!(matches!(result, Err(BrowserBindError::Timeout)));
    }

    #[tokio::test]
    async fn wait_for_new_browser_returns_earliest_registered_when_multiple_new() {
        let r = BrowserRegistry::new();
        let exclude: std::collections::HashSet<String> = std::collections::HashSet::new();

        r.register("proxy-first", b"proxy-first".to_vec());
        tokio::time::sleep(Duration::from_millis(10)).await;
        r.register("proxy-second", b"proxy-second".to_vec());

        let e = r
            .wait_for_new_browser(&exclude, Duration::from_millis(200))
            .await
            .expect("should resolve");
        assert_eq!(e.proxy_id, "proxy-first");
    }

    #[test]
    fn register_frame_role_parse() {
        let browser = br#"{"type":"register","proxy_id":"p1","role":"browser"}"#;
        let control = br#"{"type":"register","proxy_id":"p2"}"#;
        assert_eq!(
            parse_register_frame(browser),
            Some(("p1".to_string(), RegisterRole::Browser))
        );
        assert_eq!(
            parse_register_frame(control),
            Some(("p2".to_string(), RegisterRole::Control))
        );
        assert_eq!(parse_register_frame(b"not json"), None);
        assert_eq!(
            parse_register_frame(br#"{"type":"hello","proxy_id":"p3"}"#),
            None
        );
    }

    fn browser_entry(id: &str) -> BrowserEntry {
        BrowserEntry {
            proxy_id: id.to_string(),
            client_identity: id.as_bytes().to_vec(),
            registered_at: Instant::now(),
            last_heartbeat: Instant::now(),
        }
    }

    #[test]
    fn a_bound_session_resolves_to_its_browser() {
        let b = SessionBrowserBindings::new();
        b.bind("sess-1", browser_entry("browser-a"));
        assert_eq!(b.get("sess-1").unwrap().proxy_id, "browser-a");
    }

    #[test]
    fn an_unbound_session_resolves_to_nothing() {
        // This is the desktop case and it must stay the desktop case: an empty
        // table means the chat path behaves exactly as it did before.
        let b = SessionBrowserBindings::new();
        b.bind("sess-1", browser_entry("browser-a"));
        assert!(b.get("sess-2").is_none());
        assert!(SessionBrowserBindings::new().get("sess-1").is_none());
    }

    #[test]
    fn rebinding_replaces_rather_than_accumulates() {
        // The headless service rebinds after it restarts a crashed browser;
        // routing to the dead one would be worse than not routing at all.
        let b = SessionBrowserBindings::new();
        b.bind("sess-1", browser_entry("browser-a"));
        b.bind("sess-1", browser_entry("browser-b"));
        assert_eq!(b.get("sess-1").unwrap().proxy_id, "browser-b");
    }

    #[test]
    fn unbinding_leaves_nothing_behind() {
        let b = SessionBrowserBindings::new();
        b.bind("sess-1", browser_entry("browser-a"));
        b.unbind("sess-1");
        assert!(b.get("sess-1").is_none());
    }

    #[test]
    fn the_binding_carries_the_routing_identity_too() {
        // `with_bound_browser` sets both fields from this entry; a binding that
        // carried only the proxy_id would route to a writer that is never found.
        let b = SessionBrowserBindings::new();
        b.bind("sess-1", browser_entry("browser-a"));
        let got = b.get("sess-1").unwrap();
        assert_eq!(got.client_identity, b"browser-a".to_vec());
    }
}
