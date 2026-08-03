//! Which soul answers this turn.
//!
//! Resolution runs once per user message and is a pure function of three inputs:
//! an explicit `@`-mention, the session's sticky override, and the current tab's
//! container. Nothing is remembered between turns except the override, so moving
//! to another container follows along on its own.
//!
//! The resolver never writes: it returns the slug to use plus what should happen
//! to the override, and the caller persists that. This keeps the decision itself
//! testable without a database.

use serde::{Deserialize, Serialize};

use super::roles::AgentRoleRegistry;
use super::space_souls::SpaceSoulBindings;

/// A soul pinned by `@`-mention, and the container it was pinned in.
///
/// The container travels with the pin so that leaving it can clear the pin:
/// a soul is portable, a container is not (a pinned soul must not carry one
/// container's assistant into another container's memory).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoulOverride {
    /// Role directory name.
    pub slug: String,
    /// The cookieStoreId the pin was made in.
    pub container: String,
}

/// Key under which the override lives in `Session.metadata`.
pub const OVERRIDE_METADATA_KEY: &str = "soul_override";

/// What the caller should do with the stored override after resolving.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverrideAction {
    /// Leave the stored override as it is.
    Keep,
    /// Store this override (a new pin).
    Set(SoulOverride),
    /// Remove the stored override.
    Clear,
}

/// The outcome of one resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    /// The role slug that answers this turn, or `None` for "no soul": the
    /// assistant behaves exactly as it did before souls existed.
    pub slug: Option<String>,
    /// What to do with the session's stored override.
    pub action: OverrideAction,
}

impl Resolution {
    fn none(action: OverrideAction) -> Self {
        Self { slug: None, action }
    }

    fn some(slug: impl Into<String>, action: OverrideAction) -> Self {
        Self {
            slug: Some(slug.into()),
            action,
        }
    }
}

/// What the user's `@` said this turn.
///
/// Distinguishing "said nothing" from "said go back to normal" matters: the first
/// leaves a pin alone, the second removes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mention<'a> {
    /// The user did not mention a soul.
    None,
    /// The user picked one (a slug, or a display name).
    Soul(&'a str),
    /// The user asked to go back to this container's own soul.
    Clear,
}

/// Decide which soul answers this turn.
///
/// Order (§7.2): an explicit mention wins, then a sticky pin made in this same
/// container, then the container's binding, then nothing.
///
/// A mentioned soul the registry does not know is treated as ordinary text rather
/// than an error: the user may simply have typed an email address.
pub fn resolve(
    mention: Mention<'_>,
    override_state: Option<&SoulOverride>,
    cookie_store_id: &str,
    bindings: &SpaceSoulBindings,
    registry: &AgentRoleRegistry,
) -> Resolution {
    let bound = binding_slug(cookie_store_id, bindings, registry);

    // 0. "Back to normal": drop the pin and use whatever this container uses.
    if mention == Mention::Clear {
        return match bound {
            Some(slug) => Resolution::some(slug, OverrideAction::Clear),
            None => Resolution::none(OverrideAction::Clear),
        };
    }

    let mention = match mention {
        Mention::Soul(m) => Some(m),
        _ => None,
    };

    // 1. An explicit @-mention this turn.
    if let Some(mentioned) = mention.and_then(|m| registry.resolve_slug(m)) {
        // Pinning the soul this container already uses would leave a pin that
        // says nothing; treat it as "go back to the default" instead, so the UI
        // never has to distinguish "pinned to X" from "X is simply the default".
        if bound.as_deref() == Some(mentioned.as_str()) {
            return Resolution::some(mentioned, OverrideAction::Clear);
        }
        let pin = SoulOverride {
            slug: mentioned.clone(),
            container: cookie_store_id.to_string(),
        };
        return Resolution::some(mentioned, OverrideAction::Set(pin));
    }

    // 2. A sticky pin, but only while we are still in the container it was made in.
    if let Some(pin) = override_state {
        if pin.container == cookie_store_id {
            if let Some(slug) = registry.resolve_slug(&pin.slug) {
                return Resolution::some(slug, OverrideAction::Keep);
            }
            // The pinned soul is gone (deleted since the pin was made).
            tracing::warn!(
                "Session pinned soul '{}', which no longer exists; falling back to the \
                 container's binding",
                pin.slug
            );
            return match bound {
                Some(slug) => Resolution::some(slug, OverrideAction::Clear),
                None => Resolution::none(OverrideAction::Clear),
            };
        }
        // 2'. We have moved to another container: the pin does not travel.
        return match bound {
            Some(slug) => Resolution::some(slug, OverrideAction::Clear),
            None => Resolution::none(OverrideAction::Clear),
        };
    }

    // 3. This container's binding, or 4. nothing.
    match bound {
        Some(slug) => Resolution::some(slug, OverrideAction::Keep),
        None => Resolution::none(OverrideAction::Keep),
    }
}

/// The slug bound to this container, if it still names a registered role.
///
/// A binding left pointing at a deleted role reads as "unbound" rather than as a
/// failure: the user gets the default assistant plus a warning in the log.
fn binding_slug(
    cookie_store_id: &str,
    bindings: &SpaceSoulBindings,
    registry: &AgentRoleRegistry,
) -> Option<String> {
    let slug = bindings.get(cookie_store_id)?;
    match registry.resolve_slug(slug) {
        Some(resolved) => Some(resolved),
        None => {
            tracing::warn!(
                "Container '{}' is bound to soul '{}', which does not exist. \
                 Using the default assistant.",
                cookie_store_id,
                slug
            );
            None
        }
    }
}

/// Read an override out of a session's metadata.
pub fn override_from_metadata(
    metadata: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Option<SoulOverride> {
    let raw = metadata?.get(OVERRIDE_METADATA_KEY)?;
    match serde_json::from_value::<SoulOverride>(raw.clone()) {
        Ok(pin) => Some(pin),
        Err(e) => {
            tracing::warn!(
                "Ignoring unreadable soul_override in session metadata: {}",
                e
            );
            None
        }
    }
}

/// Apply an [`OverrideAction`] to a session's metadata map.
///
/// Returns `true` when the map changed and needs persisting.
pub fn apply_override_action(
    metadata: &mut std::collections::HashMap<String, serde_json::Value>,
    action: &OverrideAction,
) -> bool {
    match action {
        OverrideAction::Keep => false,
        OverrideAction::Clear => metadata.remove(OVERRIDE_METADATA_KEY).is_some(),
        OverrideAction::Set(pin) => match serde_json::to_value(pin) {
            Ok(value) => {
                let changed = metadata.get(OVERRIDE_METADATA_KEY) != Some(&value);
                metadata.insert(OVERRIDE_METADATA_KEY.to_string(), value);
                changed
            }
            Err(e) => {
                tracing::warn!("Could not store soul override: {}", e);
                false
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::Path;

    /// A registry holding the given slugs, each with `name` = slug unless given.
    fn registry_with(roles: &[(&str, &str)], dir: &Path) -> AgentRoleRegistry {
        std::fs::create_dir_all(dir).unwrap();
        for (slug, name) in roles {
            let role_dir = dir.join(slug);
            std::fs::create_dir_all(&role_dir).unwrap();
            std::fs::write(
                role_dir.join("IDENTITY.md"),
                format!("---\nname: {}\ndescription: test\n---\n", name),
            )
            .unwrap();
            std::fs::write(role_dir.join("SOUL.md"), format!("I am {}.", name)).unwrap();
        }
        let mut registry = AgentRoleRegistry::with_builtin_sources(dir.to_path_buf(), Vec::new());
        registry.scan().unwrap();
        registry
    }

    fn bindings_of(pairs: &[(&str, &str)]) -> SpaceSoulBindings {
        SpaceSoulBindings::from_map(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn pin(slug: &str, container: &str) -> SoulOverride {
        SoulOverride {
            slug: slug.to_string(),
            container: container.to_string(),
        }
    }

    #[test]
    fn resolve_mention_wins() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex"), ("engineer", "nova")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(
            Mention::Soul("engineer"),
            None,
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug.as_deref(), Some("engineer"));
        assert_eq!(
            r.action,
            OverrideAction::Set(pin("engineer", "firefox-container-1"))
        );
    }

    /// Mentioning the soul this container already uses means "back to normal",
    /// not "pin the thing that was already true".
    #[test]
    fn resolve_mention_equal_to_binding_clears() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(
            Mention::Soul("research"),
            Some(&pin("research", "firefox-container-1")),
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug.as_deref(), Some("research"));
        assert_eq!(r.action, OverrideAction::Clear);
    }

    #[test]
    fn resolve_sticky_same_container() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex"), ("engineer", "nova")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(
            Mention::None,
            Some(&pin("engineer", "firefox-container-1")),
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug.as_deref(), Some("engineer"));
        assert_eq!(r.action, OverrideAction::Keep);
    }

    /// A pin is scoped to the container it was made in: a soul is portable, a
    /// container is not.
    #[test]
    fn resolve_override_cleared_on_container_change() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex"), ("engineer", "nova")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-2", "research")]);

        let r = resolve(
            Mention::None,
            Some(&pin("engineer", "firefox-container-1")),
            "firefox-container-2",
            &bindings,
            &registry,
        );

        assert_eq!(
            r.slug.as_deref(),
            Some("research"),
            "the new container's own soul"
        );
        assert_eq!(r.action, OverrideAction::Clear);
    }

    #[test]
    fn resolve_override_cleared_on_move_to_unbound_container() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("engineer", "nova")], tmp.path());
        let bindings = bindings_of(&[]);

        let r = resolve(
            Mention::None,
            Some(&pin("engineer", "firefox-container-1")),
            "firefox-default",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug, None);
        assert_eq!(r.action, OverrideAction::Clear);
    }

    /// "Back to normal" drops the pin and hands the container's own soul back.
    #[test]
    fn resolve_explicit_clear_drops_the_pin() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex"), ("engineer", "nova")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(
            Mention::Clear,
            Some(&pin("engineer", "firefox-container-1")),
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug.as_deref(), Some("research"));
        assert_eq!(r.action, OverrideAction::Clear);
    }

    /// Clearing in a container that has no soul of its own leaves no soul.
    #[test]
    fn resolve_explicit_clear_in_unbound_container_is_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("engineer", "nova")], tmp.path());
        let bindings = bindings_of(&[]);

        let r = resolve(
            Mention::Clear,
            Some(&pin("engineer", "firefox-default")),
            "firefox-default",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug, None);
        assert_eq!(r.action, OverrideAction::Clear);
    }

    /// Saying nothing is not the same as saying "back to normal": a pin survives
    /// a turn the user did not mention souls in.
    #[test]
    fn resolve_silence_keeps_the_pin() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex"), ("engineer", "nova")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(
            Mention::None,
            Some(&pin("engineer", "firefox-container-1")),
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug.as_deref(), Some("engineer"), "the pin still stands");
        assert_eq!(r.action, OverrideAction::Keep);
    }

    #[test]
    fn resolve_binding_hit() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(
            Mention::None,
            None,
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug.as_deref(), Some("research"));
        assert_eq!(r.action, OverrideAction::Keep);
    }

    /// An unbound container is the pre-souls assistant, unchanged.
    #[test]
    fn resolve_unbound_is_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(Mention::None, None, "firefox-default", &bindings, &registry);

        assert_eq!(r.slug, None);
        assert_eq!(r.action, OverrideAction::Keep);
    }

    /// A binding left pointing at a deleted soul must not break the turn.
    #[test]
    fn resolve_orphan_binding_is_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "deleted-soul")]);

        let r = resolve(
            Mention::None,
            None,
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug, None);
        assert_eq!(r.action, OverrideAction::Keep);
    }

    /// An `@` that names nothing is just text.
    #[test]
    fn resolve_accepts_only_known_mentions() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(
            Mention::Soul("someone@example.com"),
            None,
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(
            r.slug.as_deref(),
            Some("research"),
            "falls through to the binding"
        );
        assert_eq!(r.action, OverrideAction::Keep);
    }

    /// The sidebar sends a slug, but a mention resolved by display name must work
    /// too, so the daemon stays usable from clients that only know names.
    #[test]
    fn resolve_mention_accepts_display_name() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex")], tmp.path());
        let bindings = bindings_of(&[]);

        let r = resolve(
            Mention::Soul("alex"),
            None,
            "firefox-default",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug.as_deref(), Some("research"), "resolved to the slug");
    }

    /// A pin whose soul was deleted mid-session falls back rather than erroring.
    #[test]
    fn resolve_pin_to_deleted_soul_falls_back() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = registry_with(&[("research", "alex")], tmp.path());
        let bindings = bindings_of(&[("firefox-container-1", "research")]);

        let r = resolve(
            Mention::None,
            Some(&pin("gone", "firefox-container-1")),
            "firefox-container-1",
            &bindings,
            &registry,
        );

        assert_eq!(r.slug.as_deref(), Some("research"));
        assert_eq!(r.action, OverrideAction::Clear);
    }

    // ── metadata round-trip ────────────────────────────────────────────

    #[test]
    fn override_survives_a_metadata_round_trip() {
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        let action = OverrideAction::Set(pin("engineer", "firefox-container-1"));

        assert!(apply_override_action(&mut meta, &action));
        let read_back = override_from_metadata(Some(&meta));

        assert_eq!(read_back, Some(pin("engineer", "firefox-container-1")));
    }

    #[test]
    fn clear_removes_the_override() {
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        apply_override_action(&mut meta, &OverrideAction::Set(pin("a", "c1")));

        assert!(apply_override_action(&mut meta, &OverrideAction::Clear));
        assert_eq!(override_from_metadata(Some(&meta)), None);
        assert!(!meta.contains_key(OVERRIDE_METADATA_KEY));
    }

    #[test]
    fn keep_does_not_touch_metadata() {
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        apply_override_action(&mut meta, &OverrideAction::Set(pin("a", "c1")));

        assert!(!apply_override_action(&mut meta, &OverrideAction::Keep));
        assert_eq!(override_from_metadata(Some(&meta)), Some(pin("a", "c1")));
    }

    #[test]
    fn setting_the_same_pin_twice_reports_no_change() {
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        let action = OverrideAction::Set(pin("a", "c1"));

        assert!(apply_override_action(&mut meta, &action));
        assert!(
            !apply_override_action(&mut meta, &action),
            "an unchanged pin should not trigger a session write"
        );
    }

    /// Sessions written before souls existed have no override; and a corrupt one
    /// must not poison the turn.
    #[test]
    fn unreadable_or_absent_override_reads_as_none() {
        assert_eq!(override_from_metadata(None), None);
        assert_eq!(override_from_metadata(Some(&HashMap::new())), None);

        let mut junk: HashMap<String, serde_json::Value> = HashMap::new();
        junk.insert(
            OVERRIDE_METADATA_KEY.to_string(),
            serde_json::json!("not an object"),
        );
        assert_eq!(override_from_metadata(Some(&junk)), None);
    }
}
