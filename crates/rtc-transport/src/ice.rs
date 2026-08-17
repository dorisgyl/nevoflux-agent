//! Finding a path when the direct one does not exist.
//!
//! Two peers behind ordinary home routers usually reach each other after a STUN
//! lookup tells each what the other will see. Two peers behind *symmetric* NAT,
//! or carrier-grade NAT, or a corporate firewall that drops UDP, never do — and
//! that is roughly one connection in five, which is far too many to treat as an
//! edge case. Those need a relay: a TURN server both ends can reach, forwarding
//! for them.
//!
//! # No provider is chosen here
//!
//! Which TURN service to use, what it costs and how credentials are issued are
//! deployment questions, and answering them in code would be answering them for
//! every deployment. This module is the shape of the answer: a list of servers
//! from configuration, validated, turned into candidates. An empty list is
//! legal and means host candidates only — right for a LAN, and the honest
//! default before anyone has decided.

use std::net::SocketAddr;

/// One STUN or TURN server.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IceServer {
    /// `stun:host:port` or `turn:host:port`, optionally `?transport=tcp`.
    pub url: String,
    /// TURN only. STUN needs no credentials and must not be given any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<String>,
}

/// What kind of server a URL names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerKind {
    /// Tells you your own public address. Costs nothing to use and relays
    /// nothing.
    Stun,
    /// Forwards traffic for peers that cannot reach each other. Costs bandwidth
    /// and needs credentials.
    Turn,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IceConfigError {
    #[error("{0:?} is not a stun: or turn: URL")]
    UnknownScheme(String),
    #[error("{0} has no host")]
    NoHost(String),
    #[error("TURN server {0} needs a username and a credential")]
    TurnNeedsCredentials(String),
}

/// Check a configured server and say what it is.
///
/// Validated when the configuration is read rather than when a call is placed.
/// A typo'd TURN URL discovered at connect time is a session that silently
/// fails to connect for the one user whose network needed it — the hardest
/// possible case to reproduce.
pub fn classify(server: &IceServer) -> Result<ServerKind, IceConfigError> {
    let url = server.url.trim();
    let kind = if url.starts_with("stun:") || url.starts_with("stuns:") {
        ServerKind::Stun
    } else if url.starts_with("turn:") || url.starts_with("turns:") {
        ServerKind::Turn
    } else {
        return Err(IceConfigError::UnknownScheme(url.to_string()));
    };

    let rest = url.split_once(':').map(|(_, r)| r).unwrap_or_default();
    let host = rest.split(['?', '/']).next().unwrap_or_default();
    // A bare port, or nothing at all, names no server.
    if host.is_empty() || host.starts_with(':') {
        return Err(IceConfigError::NoHost(url.to_string()));
    }

    if kind == ServerKind::Turn && (server.username.is_none() || server.credential.is_none()) {
        return Err(IceConfigError::TurnNeedsCredentials(url.to_string()));
    }
    Ok(kind)
}

/// Validate a whole configured list.
///
/// Reports every problem rather than the first: an operator fixing a config
/// should see all of it at once, not discover the second typo after redeploying
/// to fix the first.
pub fn validate(servers: &[IceServer]) -> Vec<IceConfigError> {
    servers.iter().filter_map(|s| classify(s).err()).collect()
}

/// Whether this configuration can serve a peer with no direct path.
///
/// STUN alone cannot: it only reports an address, and a symmetric NAT makes
/// that address useless to anyone else. Worth surfacing, because a deployment
/// with STUN and no TURN works perfectly in testing and fails for a fifth of
/// real users.
pub fn can_relay(servers: &[IceServer]) -> bool {
    servers.iter().any(|s| classify(s) == Ok(ServerKind::Turn))
}

/// A server-reflexive candidate: this end as the outside world sees it.
///
/// Kept separate from the host candidate because they are discovered
/// differently — a host address is known immediately, this one only after a
/// STUN round trip — and trickled separately for the same reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflexiveCandidate {
    pub local: SocketAddr,
    pub public: SocketAddr,
}

impl ReflexiveCandidate {
    /// Whether this actually told us anything.
    ///
    /// A machine with a public address sees the same address back, and offering
    /// it a second time as srflx just gives the far end a duplicate to check.
    pub fn is_useful(&self) -> bool {
        self.local != self.public
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(url: &str) -> IceServer {
        IceServer {
            url: url.into(),
            username: Some("u".into()),
            credential: Some("p".into()),
        }
    }

    fn stun(url: &str) -> IceServer {
        IceServer {
            url: url.into(),
            username: None,
            credential: None,
        }
    }

    #[test]
    fn recognises_both_schemes_and_their_secure_forms() {
        assert_eq!(
            classify(&stun("stun:stun.example.com:3478")),
            Ok(ServerKind::Stun)
        );
        assert_eq!(
            classify(&stun("stuns:stun.example.com:5349")),
            Ok(ServerKind::Stun)
        );
        assert_eq!(
            classify(&turn("turn:turn.example.com:3478")),
            Ok(ServerKind::Turn)
        );
        assert_eq!(
            classify(&turn("turns:turn.example.com:5349")),
            Ok(ServerKind::Turn)
        );
    }

    #[test]
    fn a_turn_url_with_a_transport_hint_is_still_a_turn_url() {
        // TCP TURN is what gets through a firewall that drops UDP outright —
        // precisely the deployment that needs TURN most.
        assert_eq!(
            classify(&turn("turn:turn.example.com:3478?transport=tcp")),
            Ok(ServerKind::Turn)
        );
    }

    #[test]
    fn turn_without_credentials_is_refused_at_config_time() {
        // Otherwise this is discovered when a call fails to connect, for the one
        // user whose network needed the relay — the hardest case to reproduce.
        let s = IceServer {
            url: "turn:turn.example.com:3478".into(),
            username: Some("u".into()),
            credential: None,
        };
        assert_eq!(
            classify(&s),
            Err(IceConfigError::TurnNeedsCredentials(
                "turn:turn.example.com:3478".into()
            ))
        );
    }

    #[test]
    fn stun_needs_no_credentials() {
        assert_eq!(
            classify(&stun("stun:stun.example.com:3478")),
            Ok(ServerKind::Stun)
        );
    }

    #[test]
    fn anything_that_is_not_stun_or_turn_is_refused() {
        for url in ["https://example.com", "example.com:3478", "", "turnx:h:1"] {
            assert!(
                matches!(classify(&stun(url)), Err(IceConfigError::UnknownScheme(_))),
                "accepted {url:?}"
            );
        }
    }

    #[test]
    fn a_url_naming_no_host_is_refused() {
        assert!(matches!(
            classify(&stun("stun:")),
            Err(IceConfigError::NoHost(_))
        ));
    }

    #[test]
    fn validation_reports_every_problem_at_once() {
        // An operator fixing a config should see all of it, not discover the
        // second typo after redeploying to fix the first.
        let servers = vec![
            stun("stun:good.example.com:3478"),
            stun("nonsense"),
            IceServer {
                url: "turn:needs.creds:3478".into(),
                username: None,
                credential: None,
            },
        ];
        assert_eq!(validate(&servers).len(), 2);
    }

    #[test]
    fn an_empty_configuration_is_legal_and_relays_nothing() {
        // Host candidates only. Right for a LAN, and the honest default before
        // anyone has picked a provider.
        assert!(validate(&[]).is_empty());
        assert!(!can_relay(&[]));
    }

    #[test]
    fn stun_alone_cannot_relay() {
        // The trap: a deployment with STUN and no TURN works perfectly in
        // testing and fails for roughly a fifth of real users.
        assert!(!can_relay(&[stun("stun:stun.example.com:3478")]));
        assert!(can_relay(&[
            stun("stun:stun.example.com:3478"),
            turn("turn:turn.example.com:3478"),
        ]));
    }

    #[test]
    fn a_misconfigured_turn_server_does_not_count_as_relay_capable() {
        // It would connect in testing behind an easy NAT and fail exactly where
        // the relay was the point.
        let broken = IceServer {
            url: "turn:turn.example.com:3478".into(),
            username: None,
            credential: None,
        };
        assert!(!can_relay(&[broken]));
    }

    #[test]
    fn a_reflexive_candidate_that_repeats_the_host_address_is_not_worth_sending() {
        let a: SocketAddr = "203.0.113.5:5000".parse().unwrap();
        let b: SocketAddr = "192.0.2.9:41234".parse().unwrap();
        assert!(!ReflexiveCandidate {
            local: a,
            public: a
        }
        .is_useful());
        assert!(ReflexiveCandidate {
            local: a,
            public: b
        }
        .is_useful());
    }
}
