//! What the headless head is set to, and where that came from.
//!
//! Precedence is env > `config.toml` > default, resolved per setting rather
//! than as a block: overriding the mode at `docker run` time must not quietly
//! reset the execution tier to its default.

use crate::config::RemoteControlConfig;

/// The snapshot the service runs with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedControl {
    /// Chat mode for every remote turn.
    pub mode: String,
    /// Agent-execution tier written onto this head's session at startup.
    pub execution_tier: String,
}

/// A value counts as set only if it is present and not blank — `-e VAR=` in a
/// container leaves an empty string behind, and that means "I did not set it".
fn present(v: Option<String>) -> Option<String> {
    v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Resolve mode + tier. `env` is injected so precedence is testable without
/// mutating the process environment, which races across parallel tests.
pub fn resolve(cfg: &RemoteControlConfig, env: &dyn Fn(&str) -> Option<String>) -> ResolvedControl {
    let mode = present(env("NEVOFLUX_REMOTE_MODE"))
        .or_else(|| present(cfg.mode.clone()))
        .map(|m| match m.as_str() {
            "chat" | "browser" | "agent" => m,
            // An unrecognized mode resolves to the least capable one. Falling
            // back to the `agent` default here would turn a typo into a wider
            // grant than anyone asked for.
            _ => "chat".to_string(),
        })
        .unwrap_or_else(|| "agent".to_string());

    let execution_tier = present(env("NEVOFLUX_REMOTE_EXECUTION_TIER"))
        .or_else(|| present(cfg.execution_tier.clone()))
        // `from_setting` maps anything unrecognized to the safest tier, and
        // `as_setting` gives back its canonical spelling — so the value stored
        // on the session is always one the permission gate will recognise.
        .map(|t| {
            nevoflux_protocol::ExecutionTier::from_setting(&t)
                .as_setting()
                .to_string()
        })
        .unwrap_or_else(|| {
            nevoflux_protocol::ExecutionTier::default()
                .as_setting()
                .to_string()
        });

    ResolvedControl {
        mode,
        execution_tier,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn defaults_when_nothing_is_set() {
        let r = resolve(&RemoteControlConfig::default(), &no_env);
        assert_eq!(r.mode, "agent");
        assert_eq!(
            r.execution_tier,
            nevoflux_protocol::ExecutionTier::default().as_setting()
        );
    }

    #[test]
    fn toml_beats_the_default() {
        let cfg = RemoteControlConfig {
            mode: Some("browser".into()),
            execution_tier: Some("browser-auto".into()),
            ..Default::default()
        };
        let r = resolve(&cfg, &no_env);
        assert_eq!(r.mode, "browser");
        assert_eq!(r.execution_tier, "browser-auto");
    }

    #[test]
    fn env_beats_toml() {
        let cfg = RemoteControlConfig {
            mode: Some("browser".into()),
            execution_tier: Some("browser-auto".into()),
            ..Default::default()
        };
        let env = |k: &str| match k {
            "NEVOFLUX_REMOTE_MODE" => Some("chat".to_string()),
            "NEVOFLUX_REMOTE_EXECUTION_TIER" => Some("full-auto".to_string()),
            _ => None,
        };
        let r = resolve(&cfg, &env);
        assert_eq!(r.mode, "chat");
        assert_eq!(r.execution_tier, "full-auto");
    }

    #[test]
    fn each_setting_resolves_independently() {
        // Setting only the mode env var must not drag the tier along with it.
        let cfg = RemoteControlConfig {
            mode: Some("browser".into()),
            execution_tier: Some("browser-auto".into()),
            ..Default::default()
        };
        let env = |k: &str| (k == "NEVOFLUX_REMOTE_MODE").then(|| "agent".to_string());
        let r = resolve(&cfg, &env);
        assert_eq!(r.mode, "agent");
        assert_eq!(r.execution_tier, "browser-auto");
    }

    #[test]
    fn an_unknown_mode_falls_back_rather_than_granting_more() {
        let cfg = RemoteControlConfig {
            mode: Some("supervisor".into()),
            execution_tier: None,
            ..Default::default()
        };
        assert_eq!(resolve(&cfg, &no_env).mode, "chat");
    }

    #[test]
    fn an_unknown_tier_falls_back_to_the_safest() {
        let cfg = RemoteControlConfig {
            mode: None,
            execution_tier: Some("yolo".into()),
            ..Default::default()
        };
        assert_eq!(resolve(&cfg, &no_env).execution_tier, "read-only");
    }

    #[test]
    fn blank_values_are_treated_as_unset() {
        // `docker run -e NEVOFLUX_REMOTE_MODE=` sets it to the empty string;
        // reading that as a mode would resolve it to the fallback and quietly
        // demote a head whose config.toml said otherwise.
        let cfg = RemoteControlConfig {
            mode: Some("browser".into()),
            execution_tier: None,
            ..Default::default()
        };
        let env = |k: &str| (k == "NEVOFLUX_REMOTE_MODE").then(String::new);
        assert_eq!(resolve(&cfg, &env).mode, "browser");
    }
}
