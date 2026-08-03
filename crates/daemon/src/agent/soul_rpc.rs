//! System commands behind the Space Souls settings UI.
//!
//! Four thin wrappers over the role registry and the bindings file, so the UI can
//! list souls and bind them to containers without going through the LLM.
//!
//! The bindings file stays the source of truth: these write it, the watcher
//! reloads it, and a user editing it by hand sees the same result.

use std::path::PathBuf;

use serde_json::json;

use super::space_souls::{is_valid_container_id, SpaceSoulBindings};
use crate::wasm::HostServices;

/// Build the `system_response` envelope every command replies with.
fn respond(
    request_id: &str,
    command: &str,
    result: Result<serde_json::Value, (&str, String)>,
) -> serde_json::Value {
    match result {
        Ok(data) => json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": command,
                "success": true,
                "data": data,
            }
        }),
        Err((code, message)) => json!({
            "type": "system_response",
            "payload": {
                "request_id": request_id,
                "command": command,
                "success": false,
                "error": { "code": code, "message": message },
            }
        }),
    }
}

fn request_id_of(params: &serde_json::Value) -> String {
    params
        .get("request_id")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string()
}

fn config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("nevoflux"))
}

/// `soul.list` — every soul the user can bind or mention.
pub async fn handle_soul_list(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let Some(registry) = services.role_registry() else {
        return respond(
            &request_id,
            "soul.list",
            Err(("REGISTRY_UNAVAILABLE", "Role registry not available".into())),
        );
    };

    let mut souls: Vec<serde_json::Value> = registry
        .list()
        .into_iter()
        // Subagent workers share the role directory format but are not Space
        // assistants; they belong in the delegation list, not here.
        .filter(|s| s.kind != crate::agent::roles::ROLE_KIND_SUBAGENT)
        .map(|s| {
            // The avatar is loaded per soul rather than in `list()` because it
            // means reading (and possibly re-encoding) a file.
            let avatar = registry
                .get(&s.slug)
                .ok()
                .and_then(|def| soul_avatar_data_uri(&def));

            json!({
                "slug": s.slug,
                "name": s.name,
                "description": s.description,
                "avatar": avatar,
            })
        })
        .collect();

    // Stable order: the UI lists these, and a set's iteration order is not one.
    souls.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or_default()
            .cmp(b["name"].as_str().unwrap_or_default())
    });

    respond(&request_id, "soul.list", Ok(json!({ "souls": souls })))
}

/// A soul's avatar, ready for an `<img>`, or `None` if it has none.
pub fn soul_avatar_data_uri(soul: &crate::agent::roles::AgentRoleDefinition) -> Option<String> {
    avatar_data_uri(&soul.slug, soul.avatar.as_deref()?)
}

/// Resolve a soul's `avatar` frontmatter value into something an `<img>` can use.
///
/// A `data:` URI is passed through. A relative path is read from the soul's own
/// directory and inlined, because the sidebar and the floating avatar are both
/// web contexts that cannot open a file path.
fn avatar_data_uri(slug: &str, avatar: &str) -> Option<String> {
    if avatar.starts_with("data:") {
        return Some(avatar.to_string());
    }

    let dir = config_dir()?.join("agents").join(slug);
    let path = dir.join(avatar.trim_start_matches("./"));

    // Stay inside the soul's own directory: the value is user-authored.
    let canonical_dir = dir.canonicalize().ok()?;
    let canonical = path.canonicalize().ok()?;
    if !canonical.starts_with(&canonical_dir) {
        tracing::warn!(
            "Soul '{}' points its avatar outside its own directory; ignoring",
            slug
        );
        return None;
    }

    let bytes = std::fs::read(&canonical).ok()?;
    let mime = match canonical
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        other => {
            tracing::warn!(
                "Soul '{}' has an avatar of unsupported type {:?}",
                slug,
                other
            );
            return None;
        }
    };

    use base64::Engine;
    Some(format!(
        "data:{};base64,{}",
        mime,
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

/// `soul.bindings` — which soul is bound to which container.
pub async fn handle_soul_bindings(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let Some(bindings) = services.space_soul_bindings.as_ref() else {
        return respond(
            &request_id,
            "soul.bindings",
            Err(("BINDINGS_UNAVAILABLE", "Bindings not available".into())),
        );
    };

    let map: serde_json::Map<String, serde_json::Value> = match bindings.read() {
        Ok(guard) => guard
            .iter()
            .map(|(container, slug)| (container.to_string(), json!(slug)))
            .collect(),
        Err(e) => {
            return respond(
                &request_id,
                "soul.bindings",
                Err((
                    "BINDINGS_UNAVAILABLE",
                    format!("Bindings lock poisoned: {}", e),
                )),
            )
        }
    };

    respond(&request_id, "soul.bindings", Ok(json!({ "bindings": map })))
}

/// `soul.bind` — give a container a soul.
pub async fn handle_soul_bind(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let container = params
        .get("container")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let slug = params.get("slug").and_then(|s| s.as_str()).unwrap_or("");

    if !is_valid_container_id(container) {
        return respond(
            &request_id,
            "soul.bind",
            Err((
                "INVALID_CONTAINER",
                format!(
                    "'{}' is not a container id. Expected 'firefox-default' or 'firefox-container-N'.",
                    container
                ),
            )),
        );
    }

    // A binding to a soul that does not exist would silently do nothing at chat
    // time, so it is refused at write time instead.
    match services.role_registry() {
        Some(registry) if registry.resolve_slug(slug).is_some() => {}
        Some(_) => {
            return respond(
                &request_id,
                "soul.bind",
                Err(("UNKNOWN_SOUL", format!("No soul named '{}'", slug))),
            )
        }
        None => {
            return respond(
                &request_id,
                "soul.bind",
                Err(("REGISTRY_UNAVAILABLE", "Role registry not available".into())),
            )
        }
    }

    mutate_bindings(services, &request_id, "soul.bind", |b| {
        b.set(container, slug);
    })
}

/// `soul.unbind` — take a container's soul away.
pub async fn handle_soul_unbind(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let container = params
        .get("container")
        .and_then(|c| c.as_str())
        .unwrap_or("");

    if !is_valid_container_id(container) {
        return respond(
            &request_id,
            "soul.unbind",
            Err((
                "INVALID_CONTAINER",
                format!("'{}' is not a container id.", container),
            )),
        );
    }

    mutate_bindings(services, &request_id, "soul.unbind", |b| {
        b.remove(container);
    })
}

/// Apply `change` to the shared bindings and write them out.
///
/// The in-memory copy is updated too rather than waiting for the file watcher, so
/// the next chat sees the new binding even if the watcher is slow or absent.
fn mutate_bindings(
    services: &HostServices,
    request_id: &str,
    command: &str,
    change: impl FnOnce(&mut SpaceSoulBindings),
) -> serde_json::Value {
    let Some(shared) = services.space_soul_bindings.as_ref() else {
        return respond(
            request_id,
            command,
            Err(("BINDINGS_UNAVAILABLE", "Bindings not available".into())),
        );
    };
    let Some(dir) = config_dir() else {
        return respond(
            request_id,
            command,
            Err((
                "CONFIG_ERROR",
                "Could not determine config directory".into(),
            )),
        );
    };

    let mut guard = match shared.write() {
        Ok(g) => g,
        Err(e) => {
            return respond(
                request_id,
                command,
                Err((
                    "BINDINGS_UNAVAILABLE",
                    format!("Bindings lock poisoned: {}", e),
                )),
            )
        }
    };

    let previous = guard.clone();
    change(&mut guard);

    if let Err(e) = guard.save(&dir) {
        // Leave memory matching disk rather than drifting from it.
        *guard = previous;
        return respond(request_id, command, Err(("WRITE_FAILED", e)));
    }

    let map: serde_json::Map<String, serde_json::Value> = guard
        .iter()
        .map(|(container, slug)| (container.to_string(), json!(slug)))
        .collect();

    respond(request_id, command, Ok(json!({ "bindings": map })))
}

// ── Authoring commands ─────────────────────────────────────────────
//
// The editor never assembles YAML: it sends fields, the daemon writes the file.
// A frontmatter typo would make a soul vanish from the list, so the one place
// that can produce that mistake is the one place that is tested.

/// `soul.read` — everything the editor needs to show a soul.
pub async fn handle_soul_read(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let slug = params.get("slug").and_then(|s| s.as_str()).unwrap_or("");

    let Some(registry) = services.role_registry() else {
        return respond(
            &request_id,
            "soul.read",
            Err(("REGISTRY_UNAVAILABLE", "Role registry not available".into())),
        );
    };

    match registry.get(slug) {
        Ok(def) => respond(
            &request_id,
            "soul.read",
            Ok(json!({
                "slug": def.slug,
                "name": def.name,
                "description": def.description,
                "avatar": soul_avatar_data_uri(&def),
                "allowed_tools": def.allowed_tools_list(),
                "identity": def.identity,
                "soul": def.system_prompt,
                "tools": def.tools_doc,
                "agents": def.agents_doc,
                // A built-in has no file of its own yet; saving makes a copy.
                "is_builtin": !registry.user_role_dir(&def.slug).exists(),
            })),
        ),
        Err(e) => respond(&request_id, "soul.read", Err(("UNKNOWN_SOUL", e))),
    }
}

/// `soul.create` — scaffold a new soul from a name.
///
/// Only the two required files are written: an empty TOOLS.md would mean "this
/// soul overrides the global tool guidance with nothing", which is not what a new
/// soul wants.
pub async fn handle_soul_create(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");

    let Some(registry) = services.role_registry() else {
        return respond(
            &request_id,
            "soul.create",
            Err(("REGISTRY_UNAVAILABLE", "Role registry not available".into())),
        );
    };

    if let Some(reason) = crate::agent::roles::name_rejection(name) {
        return respond(&request_id, "soul.create", Err(("INVALID_NAME", reason)));
    }

    let slug = match crate::agent::roles::slug_from_name(name, &registry.known_slugs()) {
        Ok(s) => s,
        Err(e) => return respond(&request_id, "soul.create", Err(("INVALID_NAME", e))),
    };

    let meta = crate::agent::roles::AgentRoleMetadata {
        name: name.trim().to_string(),
        ..Default::default()
    };
    let bodies = crate::agent::roles::RoleBodies {
        identity: format!("{} — an assistant.", name.trim()),
        soul: format!("You are {}.", name.trim()),
        ..Default::default()
    };

    match registry.write_role(&slug, &meta, &bodies) {
        Ok(()) => {
            rescan(registry);
            respond(&request_id, "soul.create", Ok(json!({ "slug": slug })))
        }
        Err(e) => respond(&request_id, "soul.create", Err(("WRITE_FAILED", e))),
    }
}

/// `soul.write` — save the editor's fields.
pub async fn handle_soul_write(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let slug = params.get("slug").and_then(|s| s.as_str()).unwrap_or("");

    let Some(registry) = services.role_registry() else {
        return respond(
            &request_id,
            "soul.write",
            Err(("REGISTRY_UNAVAILABLE", "Role registry not available".into())),
        );
    };

    // Start from what is on disk so fields the editor does not show (mode,
    // provider, max_iterations — the subagent knobs) survive a save.
    let mut meta = match registry.get(slug) {
        Ok(def) => def.into_metadata(),
        Err(_) => crate::agent::roles::AgentRoleMetadata::default(),
    };

    if let Some(name) = params.get("name").and_then(|v| v.as_str()) {
        meta.name = name.trim().to_string();
    }
    if let Some(description) = params.get("description").and_then(|v| v.as_str()) {
        meta.description = description.trim().to_string();
    }
    if let Some(list) = params.get("allowed_tools") {
        meta.allowed_tools = string_list(list);
    }
    // A soul the user creates is a soul, never a subagent worker.
    meta.kind = crate::agent::roles::ROLE_KIND_SOUL.to_string();

    let bodies = crate::agent::roles::RoleBodies {
        identity: params
            .get("identity")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        soul: params
            .get("soul")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        tools: params
            .get("tools")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        agents: params
            .get("agents")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    };

    if bodies.soul.trim().is_empty() {
        return respond(
            &request_id,
            "soul.write",
            Err((
                "EMPTY_SOUL",
                "A soul needs a personality: SOUL.md cannot be empty.".into(),
            )),
        );
    }

    match registry.write_role(slug, &meta, &bodies) {
        Ok(()) => {
            rescan(registry);
            respond(&request_id, "soul.write", Ok(json!({ "slug": slug })))
        }
        Err(e) => respond(&request_id, "soul.write", Err(("WRITE_FAILED", e))),
    }
}

/// `soul.delete` — remove a soul and any binding that pointed at it.
///
/// Leaving the binding would make a Space silently fall back to the default
/// assistant with nothing in the UI to explain why.
pub async fn handle_soul_delete(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let slug = params.get("slug").and_then(|s| s.as_str()).unwrap_or("");

    let Some(registry) = services.role_registry() else {
        return respond(
            &request_id,
            "soul.delete",
            Err(("REGISTRY_UNAVAILABLE", "Role registry not available".into())),
        );
    };

    if let Err(e) = registry.delete_role(slug) {
        return respond(&request_id, "soul.delete", Err(("DELETE_FAILED", e)));
    }
    rescan(registry);

    let orphaned: Vec<String> = match services.space_soul_bindings.as_ref() {
        Some(shared) => match shared.read() {
            Ok(guard) => guard
                .iter()
                .filter(|(_, bound)| *bound == slug)
                .map(|(container, _)| container.to_string())
                .collect(),
            Err(_) => vec![],
        },
        None => vec![],
    };

    if orphaned.is_empty() {
        return respond(&request_id, "soul.delete", Ok(json!({ "slug": slug })));
    }

    mutate_bindings(services, &request_id, "soul.delete", |b| {
        for container in &orphaned {
            b.remove(container);
        }
    })
}

/// `soul.set_avatar` — give a soul a face.
///
/// The image is written next to the soul as a file, and the frontmatter points at
/// it. Inlining base64 would work, but it would make IDENTITY.md unreadable to the
/// person who has to hand-edit it.
pub async fn handle_soul_set_avatar(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);
    let slug = params.get("slug").and_then(|s| s.as_str()).unwrap_or("");
    let data_uri = params
        .get("data_uri")
        .and_then(|d| d.as_str())
        .unwrap_or("");

    let Some(registry) = services.role_registry() else {
        return respond(
            &request_id,
            "soul.set_avatar",
            Err(("REGISTRY_UNAVAILABLE", "Role registry not available".into())),
        );
    };

    let bytes = match decode_image_data_uri(data_uri) {
        Ok(b) => b,
        Err(e) => return respond(&request_id, "soul.set_avatar", Err(("INVALID_IMAGE", e))),
    };

    let def = match registry.get(slug) {
        Ok(d) => d,
        Err(e) => return respond(&request_id, "soul.set_avatar", Err(("UNKNOWN_SOUL", e))),
    };

    let dir = registry.user_role_dir(&def.slug);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return respond(
            &request_id,
            "soul.set_avatar",
            Err((
                "WRITE_FAILED",
                format!("Could not create {}: {}", dir.display(), e),
            )),
        );
    }
    if let Err(e) = std::fs::write(dir.join(AVATAR_FILE), &bytes) {
        return respond(
            &request_id,
            "soul.set_avatar",
            Err((
                "WRITE_FAILED",
                format!("Could not write {}: {}", AVATAR_FILE, e),
            )),
        );
    }

    let mut meta = def.clone().into_metadata();
    meta.avatar = Some(format!("./{}", AVATAR_FILE));
    let bodies = def.into_bodies();

    match registry.write_role(slug, &meta, &bodies) {
        Ok(()) => {
            rescan(registry);
            respond(
                &request_id,
                "soul.set_avatar",
                Ok(json!({ "avatar": data_uri })),
            )
        }
        Err(e) => respond(&request_id, "soul.set_avatar", Err(("WRITE_FAILED", e))),
    }
}

/// The file a soul's avatar is written to.
pub const AVATAR_FILE: &str = "avatar.png";

/// Decode an image `data:` URI, refusing anything that is not an image.
///
/// The value comes from a file the user picked, but it arrives as a string, so
/// the type is checked rather than trusted.
fn decode_image_data_uri(data_uri: &str) -> Result<Vec<u8>, String> {
    let rest = data_uri
        .strip_prefix("data:")
        .ok_or_else(|| "That is not an image.".to_string())?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| "That image could not be read.".to_string())?;

    if !meta.starts_with("image/") {
        return Err("That file is not an image.".into());
    }
    if !meta.ends_with(";base64") {
        return Err("That image could not be read.".into());
    }

    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|_| "That image could not be read.".to_string())
}

/// Read a JSON value as a list of non-empty strings.
///
/// Accepts an array or a newline-separated string, because the editor shows these
/// as a textarea with one entry per line.
fn string_list(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        serde_json::Value::String(text) => text
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => vec![],
    }
}

/// Make a write visible to the next chat.
///
/// The soul-directory watcher does not recurse into `agents/`, so an edit is only
/// picked up because we ask for it here.
fn rescan(registry: &crate::agent::roles::AgentRoleRegistry) {
    if let Err(e) = registry.scan() {
        tracing::warn!("Could not reload souls after writing: {}", e);
    }
}

// ── AI Draft ───────────────────────────────────────────────────────
//
// Writing a personality from a blank page is the hard part of making a soul, so
// the model can offer a first draft. This is a single call with no tools: not an
// agent run, not a tool loop, nothing to authorize. The result only ever lands in
// a textarea — saving it is still the user's move.

/// Which file a draft is for.
///
/// Each has a different job, so each gets its own instructions; a draft that read
/// like the wrong file would be worse than no draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DraftTarget {
    Soul,
    Identity,
    Tools,
    Agents,
}

impl DraftTarget {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "soul" => Some(Self::Soul),
            "identity" => Some(Self::Identity),
            "tools" => Some(Self::Tools),
            "agents" => Some(Self::Agents),
            _ => None,
        }
    }

    /// What this file is and the exact shape it must take, with a worked
    /// example. The example matters: without it the model writes plausible prose
    /// that ignores the file's format (an identity line that reads like ad copy
    /// instead of "Name — …").
    fn brief(self) -> &'static str {
        match self {
            Self::Soul => {
                "SOUL.md — the assistant's personality, written as instructions addressed to \
                 the assistant itself. It replaces the user's global personality while this \
                 soul is answering. Write ONLY the persona: do NOT open with a 'You are \
                 <name> …' line and do NOT restate who or what the assistant is — its identity \
                 already lives in IDENTITY.md, and repeating it here just wastes the context. \
                 Start directly at the first heading. Follow this shape exactly, keeping the \
                 four sections:\n\n\
                 ## Core values\n\
                 - <what it fundamentally cares about and refuses to compromise on>\n\n\
                 ## How it thinks\n\
                 - <the lens it approaches problems through>\n\n\
                 ## Communication style\n\
                 - <how it talks: tone, directness, when it pushes back>\n\n\
                 ## Red lines & banned words\n\
                 - Never <phrases or behaviours it must never produce>\n\n\
                 Put 2–4 concrete bullets under each heading — every bullet must change how the \
                 assistant behaves, no filler. Translate the headings only if you are writing \
                 the body in another language."
            }
            Self::Identity => {
                "IDENTITY.md body — ONE short line: the assistant introducing itself, in the \
                 first person, with a bit of character. Keep it to a single sentence (roughly \
                 12–25 words), because this is the fixed identity line and the real task needs \
                 the rest of the context. Give it a personality — a stance, an attitude — while \
                 staying honest that it is an AI, and name the role it plays and where it works. \
                 Its name and avatar are set elsewhere, so add no emoji and no 'Name:' label. Do \
                 NOT list its values, thinking, style, or rules — that all lives in SOUL.md. \
                 Shape:\n\n\
                 I'm <Name> — <a personable one-line self-introduction: what I am, the role I \
                 play, and where I work>.\n\n\
                 Example: \"I'm IronMan — an AI that lives in your browser and treats every real \
                 machine, file, and account I touch as mine to look after.\""
            }
            Self::Tools => {
                "TOOLS.md — short guidance on which tools this assistant reaches for and when, \
                 as bullets under a heading. It replaces the user's global tool guidance while \
                 this soul is answering. Follow this shape:\n\n\
                 ## Tool preferences\n\n\
                 - <when to prefer one tool over another>\n\
                 - <a habit worth enforcing>\n\n\
                 Only write it if this soul really works differently from the default; otherwise \
                 an empty file inherits the global guidance."
            }
            Self::Agents => {
                "AGENTS.md — short guidance on what work this assistant hands to other assistants \
                 and what it keeps, as bullets under a heading. It replaces the user's global \
                 delegation guidance while this soul is answering. Follow this shape:\n\n\
                 ## Delegation\n\n\
                 - <what to hand off, and to whom>\n\
                 - <what to keep and do itself>"
            }
        }
    }
}

/// `soul.generate` — draft one of a soul's files.
pub async fn handle_soul_generate(
    services: &HostServices,
    params: &serde_json::Value,
) -> serde_json::Value {
    let request_id = request_id_of(params);

    let Some(target) = params
        .get("target")
        .and_then(|t| t.as_str())
        .and_then(DraftTarget::parse)
    else {
        return respond(
            &request_id,
            "soul.generate",
            Err((
                "INVALID_TARGET",
                "Expected target to be one of: soul, identity, tools, agents.".into(),
            )),
        );
    };

    let Some(llm) = services.llm_config.as_ref() else {
        return respond(
            &request_id,
            "soul.generate",
            Err((
                "LLM_NOT_CONFIGURED",
                "Set up a model in Settings → AI Models first.".into(),
            )),
        );
    };

    let prompt = build_draft_prompt(services, params, target);

    // No tools: this is a single call that returns text, not an agent run.
    let request = crate::wasm::llm::LlmChatRequest {
        messages: vec![crate::wasm::llm::LlmMessage {
            role: "user".to_string(),
            content: prompt,
            tool_calls: None,
            tool_call_id: None,
            attachments: vec![],
            reasoning: None,
        }],
        system: Some(DRAFT_SYSTEM_PROMPT.to_string()),
        temperature: None,
        max_tokens: None,
        tools: None,
    };

    // Stream and collect: ACP providers (claude-code, antigravity, …) reject the
    // one-shot chat path, so a draft has to go through the streaming path even
    // though it wants a single string back.
    match crate::wasm::llm::execute_llm_text(
        llm.provider,
        &llm.api_key,
        &llm.model,
        request,
        llm.base_url.as_deref(),
        Some(services.clone()),
    )
    .await
    {
        Ok(response) => {
            let content = strip_code_fence(&response);
            if content.trim().is_empty() {
                return respond(
                    &request_id,
                    "soul.generate",
                    Err(("EMPTY_DRAFT", "The model returned nothing to use.".into())),
                );
            }
            respond(
                &request_id,
                "soul.generate",
                Ok(json!({ "content": content })),
            )
        }
        Err(e) => respond(
            &request_id,
            "soul.generate",
            Err(("GENERATE_FAILED", format!("{}", e))),
        ),
    }
}

const DRAFT_SYSTEM_PROMPT: &str = "\
You write configuration files that define an assistant's persona. The assistant \
can be anything — a code reviewer, a writer, a researcher, a browser operator — so \
take its role only from the name, purpose, and existing files you are given; do not \
assume it works with the browser or any other particular domain. Write in English by \
default; use another language only when the user's request or the file's existing \
content is written in that language. Return the file's contents and nothing else: no \
preamble, no explanation, no code fence. Write plainly and concretely — every line \
should change how the assistant behaves. Keep it short enough to read in one sitting.";

/// Assemble what the model needs to write a useful draft.
///
/// The other files come along because a soul is one thing described four ways: a
/// personality drafted without seeing the identity would contradict it.
fn build_draft_prompt(
    services: &HostServices,
    params: &serde_json::Value,
    target: DraftTarget,
) -> String {
    let field = |key: &str| {
        params
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
    };
    let target_key = target_field_key(target);

    let mut prompt = format!(
        "Write the {}.\n\n{}\n\n",
        target_label(target),
        target.brief()
    );

    let name = field("name");
    let description = field("description");
    if !name.is_empty() {
        prompt.push_str(&format!("The assistant is called '{}'.\n", name));
    }
    if !description.is_empty() {
        prompt.push_str(&format!("Its purpose: {}\n", description));
    }

    // The field the user is drafting is the starting point, not something to
    // ignore: if they have already written some of it, rewrite from there rather
    // than replacing it wholesale; if it is empty, write a first version from the
    // name, purpose, and the other files.
    let current = field(target_key);
    if current.is_empty() {
        prompt.push_str(
            "\nThere is no draft of this file yet. Write a first version from the assistant's \
             name and purpose, and keep it consistent with its other files below.\n",
        );
    } else {
        prompt.push_str(&format!(
            "\nHere is the current {} — improve it: keep what works, apply the request below, \
             and make it match the format above.\n\n--- current {} ---\n{}\n",
            target_label(target),
            target_label(target),
            current
        ));
    }

    // The other files come along so the draft stays consistent with them — a soul
    // is one assistant described several ways.
    let others: Vec<(&str, &str)> = [
        ("soul", "Personality (SOUL.md)", field("soul")),
        ("identity", "Identity (IDENTITY.md)", field("identity")),
        ("tools", "Tool guidance (TOOLS.md)", field("tools")),
        ("agents", "Delegation guidance (AGENTS.md)", field("agents")),
    ]
    .into_iter()
    .filter(|(key, _, content)| *key != target_key && !content.is_empty())
    .map(|(_, label, content)| (label, content))
    .collect();

    if !others.is_empty() {
        prompt.push_str("\nIts other files already say (for consistency, do not copy):\n");
        for (label, content) in others {
            prompt.push_str(&format!("\n--- {} ---\n{}\n", label, content));
        }
    }

    let brief = field("brief");
    if !brief.is_empty() {
        prompt.push_str(&format!("\nWhat the user asked for: {}\n", brief));
    }

    // The user's own global files show the house style this soul should sound
    // like a variation of, not a stranger to.
    if let Some(retriever) = services.knowledge_retriever.as_ref() {
        let cache = retriever.soul_cache();
        if matches!(target, DraftTarget::Soul) && !cache.soul_raw.trim().is_empty() {
            prompt.push_str(&format!(
                "\nFor tone, the user's own default assistant is described like this \
                 (do not copy it — this soul is a different assistant):\n\n{}\n",
                cache.soul_raw.trim()
            ));
        }
    }

    prompt
}

/// The params key the frontend sends a given file's current content under.
fn target_field_key(target: DraftTarget) -> &'static str {
    match target {
        DraftTarget::Soul => "soul",
        DraftTarget::Identity => "identity",
        DraftTarget::Tools => "tools",
        DraftTarget::Agents => "agents",
    }
}

fn target_label(target: DraftTarget) -> &'static str {
    match target {
        DraftTarget::Soul => "SOUL.md",
        DraftTarget::Identity => "IDENTITY.md",
        DraftTarget::Tools => "TOOLS.md",
        DraftTarget::Agents => "AGENTS.md",
    }
}

/// Unwrap a fenced block, if the model wrapped the file in one despite being told
/// not to.
///
/// Models do this often enough that stripping it is cheaper than a retry, and a
/// stray ``` in a personality file would be visible to the user forever.
fn strip_code_fence(content: &str) -> String {
    let trimmed = content.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed.to_string();
    };
    // Drop the language tag on the opening fence, if any.
    let rest = rest.split_once('\n').map(|(_, body)| body).unwrap_or("");
    rest.trim_end()
        .strip_suffix("```")
        .unwrap_or(rest)
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The response shape the sidebar parses is part of the contract.
    #[test]
    fn responses_carry_request_id_and_command() {
        let ok = respond("req-1", "soul.list", Ok(json!({ "souls": [] })));
        assert_eq!(ok["type"], "system_response");
        assert_eq!(ok["payload"]["request_id"], "req-1");
        assert_eq!(ok["payload"]["command"], "soul.list");
        assert_eq!(ok["payload"]["success"], true);

        let err = respond("req-2", "soul.bind", Err(("UNKNOWN_SOUL", "nope".into())));
        assert_eq!(err["payload"]["success"], false);
        assert_eq!(err["payload"]["error"]["code"], "UNKNOWN_SOUL");
        assert_eq!(err["payload"]["error"]["message"], "nope");
    }

    // ── AI Draft ───────────────────────────────────────────────────────

    #[test]
    fn only_the_four_files_can_be_drafted() {
        assert_eq!(DraftTarget::parse("soul"), Some(DraftTarget::Soul));
        assert_eq!(DraftTarget::parse("identity"), Some(DraftTarget::Identity));
        assert_eq!(DraftTarget::parse("tools"), Some(DraftTarget::Tools));
        assert_eq!(DraftTarget::parse("agents"), Some(DraftTarget::Agents));

        assert_eq!(DraftTarget::parse("SOUL"), None);
        assert_eq!(DraftTarget::parse("../etc/passwd"), None);
        assert_eq!(DraftTarget::parse(""), None);
    }

    /// Models wrap files in fences despite being asked not to; a stray ``` in a
    /// personality file would stay there forever.
    #[test]
    fn a_fenced_draft_is_unwrapped() {
        assert_eq!(
            strip_code_fence(
                "```
You are alex.
```"
            ),
            "You are alex."
        );
        assert_eq!(
            strip_code_fence(
                "```markdown
You are alex.
```"
            ),
            "You are alex.",
            "the language tag goes too"
        );
    }

    #[test]
    fn an_unfenced_draft_is_left_alone() {
        assert_eq!(strip_code_fence("You are alex."), "You are alex.");
        assert_eq!(
            strip_code_fence(
                "  You are alex.

"
            ),
            "You are alex.",
            "but stray whitespace goes"
        );
    }

    /// A fenced block inside the draft is content, not a wrapper.
    #[test]
    fn a_draft_keeps_fences_that_are_not_the_wrapper() {
        let with_example = "You are alex.

```rust
fn main() {}
```";
        assert_eq!(strip_code_fence(with_example), with_example);
    }

    // ── Authoring ──────────────────────────────────────────────────────

    /// A user pasting a file that is not an image must be told so, not have it
    /// written to disk as one.
    #[test]
    fn only_images_are_accepted_as_avatars() {
        assert!(decode_image_data_uri("data:image/png;base64,AAAA").is_ok());

        for bad in [
            "data:text/html;base64,PGh0bWw+",
            "data:application/octet-stream;base64,AAAA",
            "not a data uri",
            "data:image/png,raw-not-base64",
            "data:image/png;base64,!!!not-base64!!!",
        ] {
            assert!(
                decode_image_data_uri(bad).is_err(),
                "'{}' should be refused",
                bad
            );
        }
    }

    #[test]
    fn avatar_bytes_survive_decoding() {
        // "hi" in base64
        let decoded = decode_image_data_uri("data:image/png;base64,aGk=").unwrap();
        assert_eq!(decoded, b"hi");
    }

    /// The editor shows these as a textarea, one entry per line; tests and other
    /// callers may send an array. Both mean the same list.
    #[test]
    fn a_tool_list_reads_from_lines_or_an_array() {
        let from_lines = string_list(&json!(
            "web_search
  brain_*  

"
        ));
        assert_eq!(from_lines, vec!["web_search", "brain_*"]);

        let from_array = string_list(&json!(["web_search", "  brain_*  ", ""]));
        assert_eq!(from_array, vec!["web_search", "brain_*"]);
    }

    #[test]
    fn a_missing_tool_list_is_empty_not_an_error() {
        assert!(string_list(&json!(null)).is_empty());
        assert!(string_list(&json!("")).is_empty());
        assert!(string_list(&json!(42)).is_empty());
    }

    #[test]
    fn inline_data_uris_pass_through() {
        let uri = "data:image/png;base64,AAAA";
        assert_eq!(avatar_data_uri("any", uri), Some(uri.to_string()));
    }

    /// A soul's avatar must not be a way to read arbitrary files: the value is
    /// user-authored text.
    #[test]
    fn avatar_paths_outside_the_soul_directory_are_refused() {
        assert_eq!(
            avatar_data_uri("nonexistent-soul", "../../secrets.png"),
            None
        );
        assert_eq!(avatar_data_uri("nonexistent-soul", "/etc/passwd"), None);
    }
}
