//! Agent role definitions and registry.
//!
//! A role (also called a soul) is a directory:
//!
//! ```text
//! <slug>/IDENTITY.md   required — YAML frontmatter (machine config) + identity text
//! <slug>/SOUL.md       required — the personality prompt
//! <slug>/TOOLS.md      optional — overrides the global tool guidance
//! <slug>/AGENTS.md     optional — overrides the global subagent guidance
//! ```
//!
//! The directory name is the **slug**: the stable key used for lookups, bindings
//! and whitelists. The **name** is what users see and type; it comes from
//! IDENTITY.md's frontmatter and falls back to the slug.
//!
//! Built-in roles are compiled into the binary and act as a read-only base layer;
//! user roles in `user_dir` override built-ins with the same slug. The registry
//! scans at startup for L1 (summary) loading and composes full definitions on
//! demand (L2).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use nevoflux_protocol::subagent::{AgentRoleSummary, ToolsConfig};

/// Frontmatter delimiter for role definition files.
const FRONTMATTER_DELIMITER: &str = "---";

/// Required: frontmatter (machine config) plus the role's identity text.
pub const IDENTITY_FILE: &str = "IDENTITY.md";
/// Required: the personality prompt. Subagent runs use this as their system prompt.
pub const SOUL_FILE: &str = "SOUL.md";
/// Optional: overrides the global tool guidance section.
pub const TOOLS_FILE: &str = "TOOLS.md";
/// Optional: overrides the global subagent guidance section.
pub const AGENTS_FILE: &str = "AGENTS.md";

/// Which layer a role was loaded from. User roles win over built-ins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoleLayer {
    Builtin,
    User,
}

/// YAML frontmatter metadata from a role's IDENTITY.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRoleMetadata {
    /// Display name; `@`-mentions and the UI use this. Empty means "fall back to the slug".
    #[serde(default)]
    pub name: String,
    /// What this role is for: a `"soul"` (a Space's main-session assistant, which
    /// users create and bind to containers) or a `"subagent"` (a fixed worker the
    /// main assistant dispatches for a clean-context subtask). They share one
    /// directory format but never appear in each other's lists: souls in the Space
    /// bindings, subagents in the delegation targets.
    #[serde(default = "default_kind")]
    pub kind: String,
    /// Human-readable description
    #[serde(default)]
    pub description: String,
    /// Avatar: a path relative to the role directory, or an inline `data:` URI
    #[serde(default)]
    pub avatar: Option<String>,
    /// Agent mode: "chat", "browser", or "agent".
    ///
    /// Only the subagent spawn path reads this; a bound main session always takes
    /// its mode from the user's own mode switch.
    #[serde(default = "default_mode")]
    pub mode: String,
    /// LLM provider name (e.g. "anthropic", "openai"). Subagent path only.
    #[serde(default)]
    pub provider: Option<String>,
    /// Model name override. Subagent path only.
    #[serde(default)]
    pub model: Option<String>,
    /// Tool allowlist patterns (e.g. ["browser_*", "read_file"]). Empty means
    /// "inherit the mode's full tool set".
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    /// Tools kept in every request; the rest stay reachable through tool search.
    /// Empty means "advertise everything allowed".
    #[serde(default)]
    pub advertised_tools: Vec<String>,
    /// Skills suggested to this role each turn. Empty means "all skills".
    #[serde(default)]
    pub skills: Vec<String>,
    /// Tool access mode; only valid value is "none" to disable all tools
    #[serde(default)]
    pub tools: Option<String>,
    /// Maximum iterations before timeout
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
}

impl Default for AgentRoleMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            kind: default_kind(),
            description: String::new(),
            avatar: None,
            mode: default_mode(),
            provider: None,
            model: None,
            allowed_tools: Vec::new(),
            advertised_tools: Vec::new(),
            skills: Vec::new(),
            tools: None,
            max_iterations: default_max_iterations(),
        }
    }
}

fn default_mode() -> String {
    "agent".to_string()
}

/// A role with no declared `kind` is a soul. Subagent workers opt in explicitly,
/// so the four built-ins carry `kind: subagent` and everything a user creates is a
/// soul by default.
fn default_kind() -> String {
    ROLE_KIND_SOUL.to_string()
}

/// A Space's main-session assistant.
pub const ROLE_KIND_SOUL: &str = "soul";
/// A fixed worker dispatched for a clean-context subtask.
pub const ROLE_KIND_SUBAGENT: &str = "subagent";

fn default_max_iterations() -> u32 {
    10
}

/// The Markdown bodies that make up a role directory.
#[derive(Debug, Clone, Default)]
pub struct RoleBodies {
    /// IDENTITY.md body (everything after the frontmatter)
    pub identity: String,
    /// SOUL.md in full
    pub soul: String,
    /// TOOLS.md, when the role overrides the global tool guidance
    pub tools: Option<String>,
    /// AGENTS.md, when the role overrides the global subagent guidance
    pub agents: Option<String>,
}

/// Raw file contents of a role directory, before parsing.
#[derive(Debug, Clone)]
struct RoleSource {
    /// IDENTITY.md in full, frontmatter included
    identity_raw: String,
    soul: String,
    tools: Option<String>,
    agents: Option<String>,
}

impl RoleSource {
    /// Read a role directory from disk.
    ///
    /// Missing IDENTITY.md or SOUL.md is an error: the role does not register.
    fn from_dir(dir: &Path) -> Result<Self, String> {
        let read_required = |file: &str| -> Result<String, String> {
            std::fs::read_to_string(dir.join(file))
                .map_err(|e| format!("Failed to read {}: {}", file, e))
        };
        let read_optional = |file: &str| -> Option<String> {
            let path = dir.join(file);
            if !path.exists() {
                return None;
            }
            match std::fs::read_to_string(&path) {
                Ok(content) => Some(content),
                Err(e) => {
                    tracing::warn!("Failed to read {}: {}", path.display(), e);
                    None
                }
            }
        };

        Ok(Self {
            identity_raw: read_required(IDENTITY_FILE)?,
            soul: read_required(SOUL_FILE)?,
            tools: read_optional(TOOLS_FILE),
            agents: read_optional(AGENTS_FILE),
        })
    }

    /// Assemble a role from its compiled-in `(filename, content)` pairs.
    fn from_embedded(files: &[(String, String)]) -> Result<Self, String> {
        let find = |name: &str| {
            files
                .iter()
                .find(|(f, _)| f == name)
                .map(|(_, c)| c.clone())
        };

        Ok(Self {
            identity_raw: find(IDENTITY_FILE)
                .ok_or_else(|| format!("Missing {}", IDENTITY_FILE))?,
            soul: find(SOUL_FILE).ok_or_else(|| format!("Missing {}", SOUL_FILE))?,
            tools: find(TOOLS_FILE),
            agents: find(AGENTS_FILE),
        })
    }

    /// Parse the frontmatter and split out the bodies.
    ///
    /// `slug` fills in `name` when IDENTITY.md leaves it unset.
    fn parse(self, slug: &str) -> Result<(AgentRoleMetadata, RoleBodies), String> {
        let (mut metadata, identity_body) = parse_role_frontmatter(&self.identity_raw)?;
        if metadata.name.trim().is_empty() {
            metadata.name = slug.to_string();
        }

        let bodies = RoleBodies {
            identity: identity_body,
            soul: self.soul.trim().to_string(),
            tools: self.tools.map(|t| t.trim().to_string()),
            agents: self.agents.map(|a| a.trim().to_string()),
        };

        Ok((metadata, bodies))
    }
}

/// Full agent role definition with parsed configuration.
#[derive(Debug, Clone)]
pub struct AgentRoleDefinition {
    /// Directory name: the stable key for lookups, bindings and whitelists
    pub slug: String,
    /// Display name (frontmatter `name`, defaulting to the slug)
    pub name: String,
    /// `"soul"` (a Space assistant) or `"subagent"` (a dispatchable worker).
    pub kind: String,
    /// Human-readable description
    pub description: String,
    /// Avatar path (relative to the role directory) or inline `data:` URI
    pub avatar: Option<String>,
    /// SOUL.md — the personality prompt. Subagent runs use this as their system prompt.
    pub system_prompt: String,
    /// IDENTITY.md body: the role's identity text
    pub identity: String,
    /// TOOLS.md, when the role overrides the global tool guidance
    pub tools_doc: Option<String>,
    /// AGENTS.md, when the role overrides the global subagent guidance
    pub agents_doc: Option<String>,
    /// Agent mode: "chat", "browser", or "agent". Subagent path only.
    pub mode: String,
    /// LLM provider name. Subagent path only.
    pub provider: Option<String>,
    /// Model name override. Subagent path only.
    pub model: Option<String>,
    /// Tool access configuration; None means inherit mode's full tool set
    pub tools_config: Option<ToolsConfig>,
    /// Tools kept in every request; empty means "advertise everything allowed"
    pub advertised_tools: Vec<String>,
    /// Skills suggested each turn; empty means "all skills"
    pub skills: Vec<String>,
    /// Maximum iterations before timeout
    pub max_iterations: u32,
}

impl AgentRoleDefinition {
    /// Build a definition from a slug, its parsed frontmatter and its bodies.
    ///
    /// # Validation rules
    /// - `tools: "none"` and non-empty `allowed_tools` are mutually exclusive
    /// - `model` requires `provider` to be set
    /// - `tools: "none"` forces `max_iterations` to 1
    pub fn from_parts(
        slug: &str,
        meta: AgentRoleMetadata,
        bodies: RoleBodies,
    ) -> Result<Self, String> {
        // Validate: tools=none and allowed_tools are mutually exclusive
        if meta.tools.as_deref() == Some("none") && !meta.allowed_tools.is_empty() {
            return Err("Cannot specify both 'tools: none' and 'allowed_tools'".to_string());
        }

        // Validate: model requires provider
        if meta.model.is_some() && meta.provider.is_none() {
            return Err("Specifying 'model' requires 'provider' to also be set".to_string());
        }

        // Compute tools_config and max_iterations
        let (tools_config, max_iterations) = if meta.tools.as_deref() == Some("none") {
            // tools: none disables all tools and forces single iteration
            (Some(ToolsConfig::None), 1)
        } else if !meta.allowed_tools.is_empty() {
            (
                Some(ToolsConfig::Allow(meta.allowed_tools)),
                meta.max_iterations,
            )
        } else {
            // No tool restrictions specified: inherit mode's defaults
            (None, meta.max_iterations)
        };

        let name = if meta.name.trim().is_empty() {
            slug.to_string()
        } else {
            meta.name
        };

        Ok(Self {
            slug: slug.to_string(),
            name,
            kind: meta.kind,
            description: meta.description,
            avatar: meta.avatar,
            system_prompt: bodies.soul,
            identity: bodies.identity,
            tools_doc: bodies.tools,
            agents_doc: bodies.agents,
            mode: meta.mode,
            provider: meta.provider,
            model: meta.model,
            tools_config,
            advertised_tools: meta.advertised_tools,
            skills: meta.skills,
            max_iterations,
        })
    }

    /// Whether this role is a Space assistant (as opposed to a subagent worker).
    pub fn is_soul(&self) -> bool {
        self.kind != ROLE_KIND_SUBAGENT
    }

    /// Whether this role is a dispatchable subagent worker.
    pub fn is_subagent(&self) -> bool {
        self.kind == ROLE_KIND_SUBAGENT
    }
}

impl AgentRoleDefinition {
    /// The allowlist as the editor shows it: one tool per entry.
    ///
    /// `tools_config` folds "no restriction" and "no tools" into the same type;
    /// this only reports an explicit allowlist.
    pub fn allowed_tools_list(&self) -> Vec<String> {
        match &self.tools_config {
            Some(ToolsConfig::Allow(list)) => list.clone(),
            _ => Vec::new(),
        }
    }

    /// Rebuild the frontmatter this definition came from.
    ///
    /// Editing a soul is read-modify-write: fields the editor does not show (the
    /// subagent knobs — mode, provider, model, max_iterations) have to survive a
    /// save, so they round-trip through here rather than being reset to defaults.
    pub fn into_metadata(self) -> AgentRoleMetadata {
        let allowed_tools = self.allowed_tools_list();
        let tools =
            matches!(self.tools_config, Some(ToolsConfig::None)).then(|| "none".to_string());
        AgentRoleMetadata {
            name: self.name,
            kind: self.kind,
            description: self.description,
            avatar: self.avatar,
            mode: self.mode,
            provider: self.provider,
            model: self.model,
            allowed_tools,
            advertised_tools: self.advertised_tools,
            skills: self.skills,
            tools,
            max_iterations: self.max_iterations,
        }
    }

    /// The Markdown bodies this definition came from.
    pub fn into_bodies(self) -> RoleBodies {
        RoleBodies {
            identity: self.identity,
            soul: self.system_prompt,
            tools: self.tools_doc,
            agents: self.agents_doc,
        }
    }
}

/// Split a document into its raw YAML frontmatter and the body that follows.
///
/// The frontmatter is returned verbatim so callers that rewrite files can
/// preserve the author's formatting.
fn split_frontmatter(content: &str) -> Result<(&str, &str), String> {
    let content = content.trim();

    if !content.starts_with(FRONTMATTER_DELIMITER) {
        return Err("Missing frontmatter delimiter".into());
    }

    let after_start = &content[FRONTMATTER_DELIMITER.len()..];
    let end_pos = after_start
        .find(&format!("\n{}", FRONTMATTER_DELIMITER))
        .ok_or("Missing closing frontmatter delimiter")?;

    let yaml = after_start[..end_pos].trim();

    let body_start = FRONTMATTER_DELIMITER.len() + end_pos + 1 + FRONTMATTER_DELIMITER.len();
    let body = if body_start < content.len() {
        content[body_start..].trim()
    } else {
        ""
    };

    Ok((yaml, body))
}

/// Parse YAML frontmatter from a role's IDENTITY.md.
///
/// Returns the parsed metadata and the body content (the identity text).
/// The file format is:
/// ```text
/// ---
/// name: role-name
/// description: A brief description
/// ---
///
/// Identity text here.
/// ```
pub fn parse_role_frontmatter(content: &str) -> Result<(AgentRoleMetadata, String), String> {
    let (yaml, body) = split_frontmatter(content)?;
    let metadata: AgentRoleMetadata =
        serde_yaml::from_str(yaml).map_err(|e| format!("YAML parse error: {}", e))?;
    Ok((metadata, body.to_string()))
}

/// Registry for agent role definitions.
///
/// Supports two-layer loading:
/// - L1 (scan): Parse IDENTITY.md frontmatter only, to build summaries
/// - L2 (get): Compose the full definition on demand, with caching
///
/// Built-in roles are compiled into the binary and act as a read-only base
/// layer; user-defined roles from `user_dir` override built-ins with the same
/// slug.
pub struct AgentRoleRegistry {
    /// L1 summaries, keyed by slug.
    ///
    /// Behind a lock like the L2 cache below: souls are created and edited while
    /// the daemon runs, and every holder of the registry has only an `Arc`.
    summaries: RwLock<HashMap<String, AgentRoleSummary>>,
    /// Display name -> slug, for `@name` resolution
    by_name: RwLock<HashMap<String, String>>,
    /// L2 cached full definitions, keyed by slug (RwLock for interior mutability through &self)
    definitions: RwLock<HashMap<String, AgentRoleDefinition>>,
    /// User role definitions directory
    user_dir: PathBuf,
    /// Built-in role sources as `(slug, [(filename, content)])`.
    ///
    /// These are compiled into the binary rather than read from disk, so they
    /// resolve on an installed machine where no source tree is present.
    builtin: Vec<(String, Vec<(String, String)>)>,
}

impl AgentRoleRegistry {
    /// Create a registry over `user_dir`, backed by the compiled-in built-in roles.
    pub fn new(user_dir: PathBuf) -> Self {
        Self::with_builtin_sources(
            user_dir,
            nevoflux_builtin_wasm::BUILTIN_AGENT_ROLES
                .iter()
                .map(|(slug, files)| {
                    (
                        slug.to_string(),
                        files
                            .iter()
                            .map(|(name, content)| (name.to_string(), content.to_string()))
                            .collect(),
                    )
                })
                .collect(),
        )
    }

    /// Create a registry with an explicit built-in layer.
    ///
    /// Tests use this to exercise the fallback behavior against synthetic roles
    /// instead of whichever roles happen to ship in the binary.
    pub fn with_builtin_sources(
        user_dir: PathBuf,
        builtin: Vec<(String, Vec<(String, String)>)>,
    ) -> Self {
        Self {
            summaries: RwLock::new(HashMap::new()),
            by_name: RwLock::new(HashMap::new()),
            definitions: RwLock::new(HashMap::new()),
            user_dir,
            builtin,
        }
    }

    /// Collect role summaries from both layers (L1 loading).
    ///
    /// Migrates any legacy flat `{slug}.md` files first, then reads the built-in
    /// layer and the user directory. User roles override built-ins with the same
    /// slug. Returns the total number of distinct roles found.
    pub fn scan(&self) -> Result<usize, String> {
        self.summaries.write().unwrap().clear();
        self.by_name.write().unwrap().clear();
        self.definitions.write().unwrap().clear();

        migrate_flat_roles(&self.user_dir);

        // (slug, name, layer) in load order; used to resolve name collisions.
        let mut claims: Vec<(String, String, RoleLayer)> = Vec::new();

        // Built-in layer first
        let builtin = self.builtin.clone();
        for (slug, files) in &builtin {
            let source = match RoleSource::from_embedded(files) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Built-in role '{}' is incomplete: {}", slug, e);
                    continue;
                }
            };
            if let Some(name) = self.insert_summary(slug, source, &format!("<builtin>/{}", slug)) {
                claims.push((slug.clone(), name, RoleLayer::Builtin));
            }
        }

        // User directory overrides built-ins with the same slug
        claims.extend(self.scan_directory(&self.user_dir.clone())?);

        self.build_name_index(claims);

        let count = self.summaries.read().unwrap().len();
        Ok(count)
    }

    /// List all available role summaries.
    pub fn list(&self) -> Vec<AgentRoleSummary> {
        self.summaries.read().unwrap().values().cloned().collect()
    }

    /// Resolve a slug or display name to a slug.
    ///
    /// Slugs win over names, so a role is always reachable by its directory name.
    /// Falls back to probing the layers directly, so `get()` works on a registry
    /// that has not been scanned.
    pub fn resolve_slug(&self, name_or_slug: &str) -> Option<String> {
        if self.summaries.read().unwrap().contains_key(name_or_slug) {
            return Some(name_or_slug.to_string());
        }
        if let Some(slug) = self.by_name.read().unwrap().get(name_or_slug) {
            return Some(slug.clone());
        }
        // Not scanned (or scanned before the role appeared): probe by slug.
        if self
            .user_dir
            .join(name_or_slug)
            .join(IDENTITY_FILE)
            .exists()
        {
            return Some(name_or_slug.to_string());
        }
        if self.builtin.iter().any(|(slug, _)| slug == name_or_slug) {
            return Some(name_or_slug.to_string());
        }
        None
    }

    /// The slug a display name resolves to, if any.
    ///
    /// `resolve_slug` prefers slugs; this asks only about names, which is what
    /// `@name` completion needs.
    pub fn slug_for_name(&self, name: &str) -> Option<String> {
        self.by_name.read().unwrap().get(name).cloned()
    }

    /// Get a full role definition by slug or display name (L2 loading with caching).
    ///
    /// Checks the definition cache first. On cache miss, composes from the user
    /// directory, falling back to the built-in layer. Returns a cloned definition
    /// to avoid holding the lock.
    pub fn get(&self, name_or_slug: &str) -> Result<AgentRoleDefinition, String> {
        let slug = self
            .resolve_slug(name_or_slug)
            .ok_or_else(|| format!("Role '{}' not found", name_or_slug))?;

        // Check cache first
        {
            let cache = self.definitions.read().unwrap();
            if let Some(def) = cache.get(&slug) {
                return Ok(def.clone());
            }
        }

        let definition = self.load_definition(&slug)?;

        // Cache it
        {
            let mut cache = self.definitions.write().unwrap();
            cache.insert(slug, definition.clone());
        }

        Ok(definition)
    }

    /// Scan the user directory for role subdirectories.
    ///
    /// Returns each registered role's `(slug, name, layer)` claim.
    fn scan_directory(&self, dir: &Path) -> Result<Vec<(String, String, RoleLayer)>, String> {
        let mut claims = Vec::new();
        if !dir.exists() {
            return Ok(claims);
        }

        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }
            let Some(slug) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };

            let source = match RoleSource::from_dir(&path) {
                Ok(s) => s,
                Err(e) => {
                    // A directory without IDENTITY.md/SOUL.md is not a role: skip
                    // it quietly enough to not spam, loudly enough to debug.
                    tracing::warn!("Skipping role directory {}: {}", path.display(), e);
                    continue;
                }
            };

            let slug = slug.to_string();
            if let Some(name) = self.insert_summary(&slug, source, &path.display().to_string()) {
                claims.push((slug, name, RoleLayer::User));
            }
        }

        Ok(claims)
    }

    /// Parse `source` and record its L1 summary under `slug`, overwriting any
    /// earlier entry for the same slug. `origin` labels the source in warnings.
    ///
    /// Returns the role's display name when a summary was recorded.
    fn insert_summary(&self, slug: &str, source: RoleSource, origin: &str) -> Option<String> {
        let (metadata, _bodies) = match source.parse(slug) {
            Ok(parsed) => parsed,
            Err(e) => {
                tracing::warn!("Failed to parse role {}: {}", origin, e);
                return None;
            }
        };

        if metadata.description.is_empty() {
            tracing::warn!("Role {} has empty description", origin);
        }

        self.summaries.write().unwrap().insert(
            slug.to_string(),
            AgentRoleSummary {
                slug: slug.to_string(),
                kind: metadata.kind,
                name: metadata.name.clone(),
                description: metadata.description,
            },
        );
        Some(metadata.name)
    }

    /// Build the name -> slug index, refusing to silently pick a winner.
    ///
    /// A user role may take a built-in's name (that is an override). Two roles in
    /// the same layer sharing a name is ambiguous: both are dropped and the
    /// conflict is reported, so `@name` never resolves to a coin flip.
    fn build_name_index(&self, claims: Vec<(String, String, RoleLayer)>) {
        // name -> claims that want it, deduplicated by slug (user overrides builtin)
        let mut wanted: HashMap<String, Vec<(String, RoleLayer)>> = HashMap::new();
        let mut layer_of: HashMap<String, RoleLayer> = HashMap::new();

        for (slug, name, layer) in claims {
            layer_of.insert(slug.clone(), layer);
            let entry = wanted.entry(name).or_default();
            if let Some(existing) = entry.iter_mut().find(|(s, _)| *s == slug) {
                // Same slug claimed twice: the later (user) layer wins.
                existing.1 = layer;
            } else {
                entry.push((slug, layer));
            }
        }

        for (name, mut owners) in wanted {
            // Drop claims whose slug lost its summary to a same-slug override in
            // another layer under a different name.
            owners.retain(|(slug, layer)| layer_of.get(slug) == Some(layer));
            if owners.len() <= 1 {
                if let Some((slug, _)) = owners.first() {
                    self.by_name.write().unwrap().insert(name, slug.clone());
                }
                continue;
            }

            let user_owners: Vec<_> = owners
                .iter()
                .filter(|(_, layer)| *layer == RoleLayer::User)
                .collect();

            match user_owners.len() {
                // A user role shadows built-ins that share its name.
                1 => {
                    self.by_name
                        .write()
                        .unwrap()
                        .insert(name, user_owners[0].0.clone());
                }
                // Ambiguous within a layer: register neither.
                _ => {
                    let slugs: Vec<&str> = owners.iter().map(|(s, _)| s.as_str()).collect();
                    tracing::error!(
                        "Duplicate role name '{}' claimed by {}; none of them will \
                         be reachable by name. Give each role a unique 'name' in {}.",
                        name,
                        slugs.join(", "),
                        IDENTITY_FILE
                    );
                    for (slug, _) in &owners {
                        self.summaries.write().unwrap().remove(slug);
                    }
                }
            }
        }
    }

    /// Compose a full role definition for `slug`.
    ///
    /// Checks the user directory first, then falls back to the compiled-in
    /// built-in layer.
    fn load_definition(&self, slug: &str) -> Result<AgentRoleDefinition, String> {
        let user_path = self.user_dir.join(slug);
        if user_path.join(IDENTITY_FILE).exists() {
            let source = RoleSource::from_dir(&user_path)?;
            let (metadata, bodies) = source.parse(slug)?;
            return AgentRoleDefinition::from_parts(slug, metadata, bodies);
        }

        if let Some((_, files)) = self.builtin.iter().find(|(s, _)| s == slug) {
            let source = RoleSource::from_embedded(files)?;
            let (metadata, bodies) = source.parse(slug)?;
            return AgentRoleDefinition::from_parts(slug, metadata, bodies);
        }

        Err(format!("Role '{}' not found", slug))
    }
}

/// Migrate legacy flat `{slug}.md` role files into `{slug}/` directories.
///
/// The frontmatter becomes IDENTITY.md and the body becomes SOUL.md; the
/// original is kept as `{slug}.md.bak`. Best-effort by design: an existing
/// directory is never overwritten, and any failure leaves the original file
/// untouched so a user's role can always be recovered by hand.
fn migrate_flat_roles(dir: &Path) {
    if !dir.exists() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(slug) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string)
        else {
            continue;
        };

        let target = dir.join(&slug);
        if target.exists() {
            continue; // Already migrated, or a directory owns this slug: leave both alone.
        }

        if let Err(e) = migrate_one_flat_role(&path, &target) {
            tracing::warn!(
                "Could not migrate legacy role {} to a directory: {}. Leaving it as is.",
                path.display(),
                e
            );
        } else {
            tracing::info!(
                "Migrated legacy role {} to {}/",
                path.display(),
                target.display()
            );
        }
    }
}

/// Split one flat role file into a role directory. See [`migrate_flat_roles`].
fn migrate_one_flat_role(path: &Path, target: &Path) -> Result<(), String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let (yaml, body) = split_frontmatter(&content)?;

    std::fs::create_dir_all(target).map_err(|e| e.to_string())?;

    let identity = format!("{d}\n{yaml}\n{d}\n", d = FRONTMATTER_DELIMITER, yaml = yaml);
    std::fs::write(target.join(IDENTITY_FILE), identity).map_err(|e| e.to_string())?;
    std::fs::write(target.join(SOUL_FILE), format!("{}\n", body.trim()))
        .map_err(|e| e.to_string())?;

    // Rename rather than delete: the original stays recoverable.
    std::fs::rename(path, path.with_extension("md.bak")).map_err(|e| e.to_string())?;
    Ok(())
}

// ── Authoring ──────────────────────────────────────────────────────
//
// Creating and editing souls from the UI. The files stay the source of truth and
// stay hand-editable, so everything written here is what a person would have
// typed — and everything typed by a person parses back through the same code.

/// The most a soul's name may be. Long enough for a name, short enough to sit in
/// a chip.
pub const MAX_NAME_LEN: usize = 32;

/// Turn a name into a directory that is safe to put on a filesystem and in a
/// config key.
///
/// `taken` are slugs that already exist; a clash gets a numbered suffix rather
/// than silently overwriting someone else's soul.
pub fn slug_from_name(name: &str, taken: &[String]) -> Result<String, String> {
    let mut slug = String::new();
    let mut last_was_dash = false;
    for c in name.trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
            last_was_dash = false;
        } else if !last_was_dash && !slug.is_empty() {
            slug.push('-');
            last_was_dash = true;
        }
    }
    let base = slug
        .trim_matches('-')
        .chars()
        .take(MAX_NAME_LEN)
        .collect::<String>();
    let base = base.trim_matches('-').to_string();

    if base.is_empty() {
        return Err(format!(
            "'{}' has no letters or numbers to make a folder name from",
            name
        ));
    }

    if !taken.iter().any(|t| t == &base) {
        return Ok(base);
    }
    for n in 2..1000 {
        let candidate = format!("{}-{}", base, n);
        if !taken.iter().any(|t| t == &candidate) {
            return Ok(candidate);
        }
    }
    Err(format!(
        "Too many souls already named something like '{}'",
        base
    ))
}

/// Whether `slug` is a name this daemon will create a directory for.
///
/// Slugs arrive from the UI and become paths, so they are checked rather than
/// trusted.
pub fn is_valid_slug(slug: &str) -> bool {
    !slug.is_empty()
        && slug.len() <= MAX_NAME_LEN
        && slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !slug.starts_with('-')
        && !slug.ends_with('-')
}

/// Why `name` cannot be a soul's display name, if it cannot.
///
/// Names are typed after `@`, so a name with a space in it could never be
/// mentioned — the mention parser stops at the space.
pub fn name_rejection(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Some("A soul needs a name.".into());
    }
    if trimmed.chars().count() > MAX_NAME_LEN {
        return Some(format!("Names are at most {} characters.", MAX_NAME_LEN));
    }
    if trimmed.chars().any(char::is_whitespace) {
        return Some("Names are one word, so they can be typed after @.".into());
    }
    if trimmed.contains('@') || trimmed.contains('#') {
        return Some("Names cannot contain @ or #.".into());
    }
    None
}

impl AgentRoleRegistry {
    /// The directory a user-authored soul lives in.
    pub fn user_role_dir(&self, slug: &str) -> PathBuf {
        self.user_dir.join(slug)
    }

    /// Every slug the registry knows, across both layers.
    pub fn known_slugs(&self) -> Vec<String> {
        let mut slugs: Vec<String> = self.summaries.read().unwrap().keys().cloned().collect();
        for (slug, _) in &self.builtin {
            if !slugs.contains(slug) {
                slugs.push(slug.clone());
            }
        }
        slugs
    }

    /// Whether another soul already answers to `name`.
    fn name_taken_by_other(&self, name: &str, slug: &str) -> bool {
        self.summaries
            .read()
            .unwrap()
            .values()
            .any(|s| s.slug != slug && s.name.eq_ignore_ascii_case(name))
    }

    /// Write a soul to disk.
    ///
    /// Editing a built-in copies it into the user directory: the embedded layer is
    /// read-only, and a user's edit should not vanish on the next release.
    ///
    /// Callers must [`scan`](Self::scan) afterwards for the change to be visible.
    pub fn write_role(
        &self,
        slug: &str,
        meta: &AgentRoleMetadata,
        bodies: &RoleBodies,
    ) -> Result<(), String> {
        if !is_valid_slug(slug) {
            return Err(format!(
                "'{}' is not a folder name: use lowercase letters, numbers and dashes.",
                slug
            ));
        }
        if let Some(reason) = name_rejection(&meta.name) {
            return Err(reason);
        }
        if self.name_taken_by_other(&meta.name, slug) {
            return Err(format!(
                "Another soul is already called '{}'. Names have to be unique so @{} \
                 means one thing.",
                meta.name, meta.name
            ));
        }

        let dir = self.user_role_dir(slug);
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("Could not create {}: {}", dir.display(), e))?;

        let identity = format!(
            "{d}\n{yaml}{d}\n\n{body}\n",
            d = FRONTMATTER_DELIMITER,
            yaml = serde_yaml::to_string(meta)
                .map_err(|e| format!("Could not write the frontmatter: {}", e))?,
            body = bodies.identity.trim()
        );
        write_file(&dir, IDENTITY_FILE, &identity)?;
        write_file(&dir, SOUL_FILE, &format!("{}\n", bodies.soul.trim()))?;

        // An empty optional file and an absent one mean the same thing — the
        // global section applies — so an emptied field removes the file rather
        // than leaving a blank one behind.
        write_optional(&dir, TOOLS_FILE, bodies.tools.as_deref())?;
        write_optional(&dir, AGENTS_FILE, bodies.agents.as_deref())?;

        Ok(())
    }

    /// Delete a user-authored soul's directory.
    ///
    /// A built-in has nothing to delete in the user directory; deleting a copied
    /// built-in restores the embedded original, which is the useful behaviour.
    pub fn delete_role(&self, slug: &str) -> Result<(), String> {
        if !is_valid_slug(slug) {
            return Err(format!("'{}' is not a folder name.", slug));
        }
        let dir = self.user_role_dir(slug);
        if !dir.exists() {
            return Ok(());
        }
        std::fs::remove_dir_all(&dir)
            .map_err(|e| format!("Could not delete {}: {}", dir.display(), e))
    }
}

fn write_file(dir: &Path, name: &str, contents: &str) -> Result<(), String> {
    std::fs::write(dir.join(name), contents).map_err(|e| format!("Could not write {}: {}", name, e))
}

fn write_optional(dir: &Path, name: &str, contents: Option<&str>) -> Result<(), String> {
    let path = dir.join(name);
    match contents.map(str::trim).filter(|c| !c.is_empty()) {
        Some(c) => std::fs::write(&path, format!("{}\n", c))
            .map_err(|e| format!("Could not write {}: {}", name, e)),
        None => match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("Could not remove {}: {}", name, e)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an embedded built-in layer entry from IDENTITY/SOUL contents.
    fn builtin_role(slug: &str, identity: &str, soul: &str) -> (String, Vec<(String, String)>) {
        (
            slug.to_string(),
            vec![
                (IDENTITY_FILE.to_string(), identity.to_string()),
                (SOUL_FILE.to_string(), soul.to_string()),
            ],
        )
    }

    /// Write a role directory under `base`.
    fn write_role_dir(base: &Path, slug: &str, identity: &str, soul: &str) {
        let dir = base.join(slug);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(IDENTITY_FILE), identity).unwrap();
        std::fs::write(dir.join(SOUL_FILE), soul).unwrap();
    }

    fn bodies(soul: &str) -> RoleBodies {
        RoleBodies {
            soul: soul.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_parse_role_frontmatter_basic() {
        let content = r#"---
name: researcher
description: A role for web research tasks
mode: browser
provider: anthropic
model: claude-sonnet-4-20250514
allowed_tools:
  - browser_*
  - read_file
max_iterations: 20
---

alex — a research assistant.
"#;

        let (meta, body) = parse_role_frontmatter(content).unwrap();
        assert_eq!(meta.name, "researcher");
        assert_eq!(meta.description, "A role for web research tasks");
        assert_eq!(meta.mode, "browser");
        assert_eq!(meta.provider, Some("anthropic".to_string()));
        assert_eq!(meta.model, Some("claude-sonnet-4-20250514".to_string()));
        assert_eq!(meta.allowed_tools, vec!["browser_*", "read_file"]);
        assert_eq!(meta.max_iterations, 20);
        assert!(body.contains("research assistant"));
    }

    #[test]
    fn test_parse_role_frontmatter_minimal() {
        let content = r#"---
name: simple
description: A simple role
---

Just an identity line.
"#;

        let (meta, body) = parse_role_frontmatter(content).unwrap();
        assert_eq!(meta.name, "simple");
        assert_eq!(meta.description, "A simple role");
        assert_eq!(meta.mode, "agent"); // default
        assert_eq!(meta.max_iterations, 10); // default
        assert!(meta.provider.is_none());
        assert!(meta.model.is_none());
        assert!(meta.allowed_tools.is_empty());
        assert!(meta.advertised_tools.is_empty());
        assert!(meta.skills.is_empty());
        assert_eq!(meta.kind, "soul"); // default
        assert!(meta.avatar.is_none());
        assert!(meta.tools.is_none());
        assert!(body.contains("identity line"));
    }

    #[test]
    fn test_parse_role_frontmatter_missing_delimiter() {
        let content = "name: broken\nno frontmatter here";
        let result = parse_role_frontmatter(content);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Missing frontmatter delimiter"));
    }

    #[test]
    fn test_parse_role_frontmatter_invalid_yaml() {
        let content = r#"---
name: [invalid yaml
---

Content
"#;

        let result = parse_role_frontmatter(content);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("YAML parse error"));
    }

    #[test]
    fn test_definition_model_without_provider() {
        let meta = AgentRoleMetadata {
            name: "test".to_string(),
            model: Some("gpt-4o".to_string()),
            ..Default::default()
        };

        let result = AgentRoleDefinition::from_parts("test", meta, RoleBodies::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("provider"));
    }

    #[test]
    fn test_definition_tools_none_and_allowed_tools() {
        let meta = AgentRoleMetadata {
            name: "test".to_string(),
            allowed_tools: vec!["read_file".to_string()],
            tools: Some("none".to_string()),
            ..Default::default()
        };

        let result = AgentRoleDefinition::from_parts("test", meta, RoleBodies::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("tools: none"));
    }

    #[test]
    fn test_definition_tools_none_forces_max_iterations_1() {
        let meta = AgentRoleMetadata {
            name: "analyzer".to_string(),
            description: "Pure analysis".to_string(),
            mode: "chat".to_string(),
            tools: Some("none".to_string()),
            max_iterations: 20, // will be forced to 1
            ..Default::default()
        };

        let def =
            AgentRoleDefinition::from_parts("analyzer", meta, bodies("Analyze this.")).unwrap();
        assert_eq!(def.max_iterations, 1);
        assert_eq!(def.tools_config, Some(ToolsConfig::None));
    }

    #[test]
    fn test_definition_tools_config_allow() {
        let meta = AgentRoleMetadata {
            name: "restricted".to_string(),
            description: "Restricted tools".to_string(),
            allowed_tools: vec!["browser_*".to_string(), "read_file".to_string()],
            max_iterations: 15,
            ..Default::default()
        };

        let def =
            AgentRoleDefinition::from_parts("restricted", meta, RoleBodies::default()).unwrap();
        assert_eq!(
            def.tools_config,
            Some(ToolsConfig::Allow(vec![
                "browser_*".to_string(),
                "read_file".to_string()
            ]))
        );
        assert_eq!(def.max_iterations, 15);
    }

    #[test]
    fn test_definition_tools_config_none() {
        let meta = AgentRoleMetadata {
            name: "no-tools".to_string(),
            description: "No tools".to_string(),
            mode: "chat".to_string(),
            tools: Some("none".to_string()),
            ..Default::default()
        };

        let def = AgentRoleDefinition::from_parts("no-tools", meta, RoleBodies::default()).unwrap();
        assert_eq!(def.tools_config, Some(ToolsConfig::None));
        assert_eq!(def.max_iterations, 1);
    }

    #[test]
    fn test_definition_tools_config_inherit() {
        let meta = AgentRoleMetadata {
            name: "inherit".to_string(),
            description: "Inherits tools".to_string(),
            ..Default::default()
        };

        let def = AgentRoleDefinition::from_parts("inherit", meta, RoleBodies::default()).unwrap();
        assert_eq!(def.tools_config, None); // None = inherit
    }

    #[test]
    fn test_definition_with_provider_and_model() {
        let meta = AgentRoleMetadata {
            name: "custom".to_string(),
            description: "Custom model".to_string(),
            mode: "chat".to_string(),
            provider: Some("openai".to_string()),
            model: Some("gpt-4o".to_string()),
            max_iterations: 5,
            ..Default::default()
        };

        let def = AgentRoleDefinition::from_parts("custom", meta, bodies("Use GPT-4o.")).unwrap();
        assert_eq!(def.provider, Some("openai".to_string()));
        assert_eq!(def.model, Some("gpt-4o".to_string()));
        assert_eq!(def.max_iterations, 5);
    }

    #[test]
    fn test_parse_frontmatter_empty_body() {
        let content = r#"---
name: empty
description: No body
---
"#;

        let (meta, body) = parse_role_frontmatter(content).unwrap();
        assert_eq!(meta.name, "empty");
        assert!(body.is_empty());
    }

    #[test]
    fn test_soul_body_keeps_code_blocks() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "coder",
            "---\nname: coder\ndescription: Code helper\n---\n\ncoder identity.\n",
            r#"Write code following these patterns:

```rust
fn main() {
    println!("Hello");
}
```

Always add tests.
"#,
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let def = registry.get("coder").unwrap();
        assert!(def.system_prompt.contains("```rust"));
        assert!(def.system_prompt.contains("fn main()"));
        assert!(def.system_prompt.contains("Always add tests."));
    }

    #[test]
    fn test_registry_scan_and_list() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let roles_dir = temp_dir.path().join("roles");
        std::fs::create_dir_all(&roles_dir).unwrap();

        write_role_dir(
            &roles_dir,
            "researcher",
            "---\nname: researcher\ndescription: Web research role\nmode: browser\n---\n",
            "You are a researcher.",
        );
        write_role_dir(
            &roles_dir,
            "coder",
            "---\nname: coder\ndescription: Code writing role\n---\n",
            "You write clean code.",
        );

        // A loose file is not a role directory and must be ignored.
        std::fs::write(roles_dir.join("notes.txt"), "not a role").unwrap();

        let registry = AgentRoleRegistry::with_builtin_sources(roles_dir, Vec::new());
        let count = registry.scan().unwrap();
        assert_eq!(count, 2);

        let summaries = registry.list();
        assert_eq!(summaries.len(), 2);

        let names: Vec<&str> = summaries.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"researcher"));
        assert!(names.contains(&"coder"));
    }

    #[test]
    fn test_registry_user_overrides_builtin() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        let builtin = vec![builtin_role(
            "writer",
            "---\nname: writer\ndescription: Built-in writer role\n---\n",
            "Built-in prompt.",
        )];

        // User role with the same slug
        write_role_dir(
            &user_dir,
            "writer",
            "---\nname: writer\ndescription: Custom writer role\n---\n",
            "Custom prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, builtin);
        let count = registry.scan().unwrap();
        assert_eq!(count, 1); // Same slug, so only 1 summary

        let summaries = registry.list();
        assert_eq!(summaries.len(), 1);
        // User description should override builtin
        assert_eq!(summaries[0].description, "Custom writer role");
    }

    #[test]
    fn test_registry_get_caches() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "tester",
            "---\nname: tester\ndescription: Test role\nmode: agent\n---\n",
            "You test things.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        // First get composes from disk
        let def1 = registry.get("tester").unwrap();
        assert_eq!(def1.name, "tester");
        assert_eq!(def1.mode, "agent");

        // Second get should return the cached definition
        let def2 = registry.get("tester").unwrap();
        assert_eq!(def2.name, "tester");

        // Verify it's in the cache, keyed by slug
        assert!(registry.definitions.read().unwrap().contains_key("tester"));
    }

    #[test]
    fn test_registry_get_not_found() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let result = registry.get("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_registry_get_prefers_user_over_builtin() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        let builtin = vec![builtin_role(
            "helper",
            "---\nname: helper\ndescription: Built-in helper\n---\n",
            "Built-in helper prompt.",
        )];

        write_role_dir(
            &user_dir,
            "helper",
            "---\nname: helper\ndescription: User helper\n---\n",
            "User helper prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, builtin);
        registry.scan().unwrap();

        let def = registry.get("helper").unwrap();
        assert_eq!(def.description, "User helper");
        assert!(def.system_prompt.contains("User helper prompt"));
    }

    #[test]
    fn test_registry_scan_nonexistent_directory() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("nonexistent_user");

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        let count = registry.scan().unwrap();
        assert_eq!(count, 0);
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_registry_scan_skips_invalid_roles() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "valid",
            "---\nname: valid\ndescription: Valid role\n---\n",
            "Content.",
        );

        // Invalid YAML in IDENTITY.md
        write_role_dir(&user_dir, "broken", "---\nname: [broken\n---\n", "Content.");

        // IDENTITY.md without frontmatter
        write_role_dir(&user_dir, "nofm", "No frontmatter here", "Content.");

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        let count = registry.scan().unwrap();
        assert_eq!(count, 1);
        assert_eq!(registry.list().len(), 1);
        assert_eq!(registry.list()[0].name, "valid");
    }

    // ── Directory structure ────────────────────────────────────────────

    /// IDENTITY.md and SOUL.md are both required: a directory missing either is
    /// not a role, and must not take the process down with it.
    #[test]
    fn test_missing_identity_or_soul_is_not_registered() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");

        // SOUL.md only
        let soul_only = user_dir.join("soulonly");
        std::fs::create_dir_all(&soul_only).unwrap();
        std::fs::write(soul_only.join(SOUL_FILE), "Prompt with no identity.").unwrap();

        // IDENTITY.md only
        let identity_only = user_dir.join("identityonly");
        std::fs::create_dir_all(&identity_only).unwrap();
        std::fs::write(
            identity_only.join(IDENTITY_FILE),
            "---\nname: identityonly\ndescription: No soul\n---\n",
        )
        .unwrap();

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        let count = registry.scan().unwrap();

        assert_eq!(count, 0, "Neither incomplete directory should register");
        assert!(registry.get("soulonly").is_err());
        assert!(registry.get("identityonly").is_err());
    }

    /// The slug is the identity of last resort: a role with no `name` is still
    /// addressable.
    #[test]
    fn test_name_defaults_to_slug() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "anon",
            "---\ndescription: No name given\n---\n",
            "Prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        assert_eq!(registry.list()[0].name, "anon");
        assert_eq!(registry.get("anon").unwrap().name, "anon");
    }

    /// `@name` must never resolve to a coin flip: same-layer name collisions
    /// unregister both claimants rather than picking one.
    #[test]
    fn test_duplicate_name_fails_fast() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "first",
            "---\nname: alex\ndescription: First alex\n---\n",
            "First prompt.",
        );
        write_role_dir(
            &user_dir,
            "second",
            "---\nname: alex\ndescription: Second alex\n---\n",
            "Second prompt.",
        );
        write_role_dir(
            &user_dir,
            "unaffected",
            "---\nname: nova\ndescription: Fine\n---\n",
            "Nova prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        let count = registry.scan().unwrap();

        assert_eq!(count, 1, "Only the non-conflicting role should survive");
        assert_eq!(registry.list()[0].slug, "unaffected");
        assert!(registry.slug_for_name("alex").is_none());
        assert_eq!(
            registry.slug_for_name("nova").as_deref(),
            Some("unaffected")
        );
    }

    /// A user role may deliberately take a built-in's name; that is an override,
    /// not a conflict, and the name resolves to the user's role.
    #[test]
    fn test_user_name_shadows_builtin_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        let builtin = vec![builtin_role(
            "builtin-helper",
            "---\nname: helper\ndescription: Built-in\n---\n",
            "Built-in prompt.",
        )];

        write_role_dir(
            &user_dir,
            "my-helper",
            "---\nname: helper\ndescription: Mine\n---\n",
            "My prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, builtin);
        registry.scan().unwrap();

        // Both slugs stay listed; the shared name resolves to the user's role.
        assert_eq!(registry.list().len(), 2);
        assert_eq!(
            registry.slug_for_name("helper").as_deref(),
            Some("my-helper")
        );
        assert_eq!(registry.get("helper").unwrap().description, "Mine");
        // The built-in remains reachable by its own slug.
        assert_eq!(
            registry.get("builtin-helper").unwrap().description,
            "Built-in"
        );
    }

    /// When one role's name equals another role's slug, the slug wins: a role is
    /// always reachable by its directory name, and no user role can hijack
    /// lookups of a built-in by naming itself after it.
    #[test]
    fn test_slug_wins_over_name_on_lookup() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        let builtin = vec![builtin_role(
            "researcher",
            "---\nname: researcher\ndescription: Built-in\n---\n",
            "Built-in prompt.",
        )];

        write_role_dir(
            &user_dir,
            "mine",
            "---\nname: researcher\ndescription: Mine\n---\n",
            "My prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, builtin);
        registry.scan().unwrap();

        // "researcher" is both a slug (built-in) and a name (the user's role).
        // The slug wins.
        assert_eq!(registry.get("researcher").unwrap().slug, "researcher");
        assert_eq!(registry.get("researcher").unwrap().description, "Built-in");
        // The user's role is reachable by its own slug...
        assert_eq!(registry.get("mine").unwrap().description, "Mine");
        // ...and the name index still points at it, so `@researcher` in the UI
        // resolves to the user's role before any lookup happens.
        assert_eq!(
            registry.slug_for_name("researcher").as_deref(),
            Some("mine")
        );
    }

    /// Subagent spawn passes a role name; bindings and whitelists pass a slug.
    /// Both must reach the same definition.
    #[test]
    fn test_get_accepts_slug_or_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "research",
            "---\nname: alex\ndescription: Research copilot\n---\n",
            "You are alex.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let by_slug = registry.get("research").unwrap();
        let by_name = registry.get("alex").unwrap();

        assert_eq!(by_slug.slug, "research");
        assert_eq!(by_slug.name, "alex");
        assert_eq!(by_slug.slug, by_name.slug);
        assert_eq!(by_slug.system_prompt, by_name.system_prompt);
    }

    /// No TOOLS.md means "inherit", which is the pre-existing behaviour for a
    /// role that lists no allowed_tools.
    #[test]
    fn test_tools_md_absent_keeps_none() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "plain",
            "---\nname: plain\ndescription: Plain role\n---\n",
            "Prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let def = registry.get("plain").unwrap();
        assert_eq!(def.tools_config, None);
        assert!(def.tools_doc.is_none());
        assert!(def.agents_doc.is_none());
    }

    /// The optional overlays are read when present.
    #[test]
    fn test_optional_overlay_files_are_read() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        let dir = user_dir.join("full");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(IDENTITY_FILE),
            "---\nname: full\ndescription: Everything\navatar: ./avatar.png\n---\n\nIdentity text.\n",
        )
        .unwrap();
        std::fs::write(dir.join(SOUL_FILE), "Soul text.").unwrap();
        std::fs::write(dir.join(TOOLS_FILE), "Tool guidance.").unwrap();
        std::fs::write(dir.join(AGENTS_FILE), "Delegation guidance.").unwrap();

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let def = registry.get("full").unwrap();
        assert_eq!(def.identity, "Identity text.");
        assert_eq!(def.system_prompt, "Soul text.");
        assert_eq!(def.tools_doc.as_deref(), Some("Tool guidance."));
        assert_eq!(def.agents_doc.as_deref(), Some("Delegation guidance."));
        assert_eq!(def.avatar.as_deref(), Some("./avatar.png"));
    }

    /// The subagent spawn path reads mode/provider/model/max_iterations off the
    /// definition. Moving to a directory layout must not drop any of them.
    #[test]
    fn test_subagent_frontmatter_fields_preserved() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "legacyish",
            "---\nname: legacyish\ndescription: Keeps subagent knobs\nmode: browser\n\
             provider: anthropic\nmodel: claude-sonnet-4-20250514\nmax_iterations: 20\n\
             allowed_tools:\n  - \"browser_*\"\n---\n",
            "Prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let def = registry.get("legacyish").unwrap();
        assert_eq!(def.mode, "browser");
        assert_eq!(def.provider, Some("anthropic".to_string()));
        assert_eq!(def.model, Some("claude-sonnet-4-20250514".to_string()));
        assert_eq!(def.max_iterations, 20);
        assert_eq!(
            def.tools_config,
            Some(ToolsConfig::Allow(vec!["browser_*".to_string()]))
        );
    }

    /// Phase-1 parses the new frontmatter keys but nothing consumes them yet.
    #[test]
    fn test_new_frontmatter_keys_are_parsed() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "kitted",
            "---\nname: kitted\ndescription: All keys\nadvertised_tools:\n  - web_search\n\
             skills:\n  - research\n---\n",
            "Prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let def = registry.get("kitted").unwrap();
        assert_eq!(def.advertised_tools, vec!["web_search"]);
        assert_eq!(def.skills, vec!["research"]);
    }

    /// A role's `kind` defaults to soul; the built-ins opt into subagent.
    #[test]
    fn test_kind_defaults_to_soul_and_can_be_subagent() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "mine",
            "---\nname: mine\ndescription: A soul\n---\n",
            "Prompt.",
        );
        write_role_dir(
            &user_dir,
            "helper",
            "---\nname: helper\nkind: subagent\ndescription: A worker\n---\n",
            "Prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let mine = registry.get("mine").unwrap();
        assert!(mine.is_soul());
        assert!(!mine.is_subagent());

        let helper = registry.get("helper").unwrap();
        assert!(helper.is_subagent());
        assert!(!helper.is_soul());

        // The summaries carry kind too, so callers can filter without a full load.
        let by_slug: std::collections::HashMap<_, _> = registry
            .list()
            .into_iter()
            .map(|s| (s.slug, s.kind))
            .collect();
        assert_eq!(by_slug["mine"], "soul");
        assert_eq!(by_slug["helper"], "subagent");
    }

    // ── Legacy flat-file migration ─────────────────────────────────────

    #[test]
    fn test_flat_md_migrates_to_directory() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        std::fs::write(
            user_dir.join("foo.md"),
            "---\nname: foo\ndescription: Legacy role\nmode: browser\nmax_iterations: 7\n---\n\nLegacy prompt body.\n",
        )
        .unwrap();

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir.clone(), Vec::new());
        let count = registry.scan().unwrap();
        assert_eq!(count, 1);

        assert!(user_dir.join("foo").join(IDENTITY_FILE).exists());
        assert!(user_dir.join("foo").join(SOUL_FILE).exists());
        assert!(user_dir.join("foo.md.bak").exists());
        assert!(!user_dir.join("foo.md").exists());

        // Semantics survive the move.
        let def = registry.get("foo").unwrap();
        assert_eq!(def.name, "foo");
        assert_eq!(def.description, "Legacy role");
        assert_eq!(def.mode, "browser");
        assert_eq!(def.max_iterations, 7);
        assert_eq!(def.system_prompt, "Legacy prompt body.");
    }

    #[test]
    fn test_flat_migration_skips_when_dir_exists() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        write_role_dir(
            &user_dir,
            "foo",
            "---\nname: foo\ndescription: The directory one\n---\n",
            "Directory prompt.",
        );
        std::fs::write(
            user_dir.join("foo.md"),
            "---\nname: foo\ndescription: The flat one\n---\n\nFlat prompt.\n",
        )
        .unwrap();

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir.clone(), Vec::new());
        registry.scan().unwrap();

        // The directory is untouched and no backup was made.
        assert!(user_dir.join("foo.md").exists());
        assert!(!user_dir.join("foo.md.bak").exists());
        assert_eq!(
            registry.get("foo").unwrap().description,
            "The directory one"
        );
    }

    /// A file that cannot be split is left exactly where it is.
    #[test]
    fn test_flat_migration_leaves_unparseable_file_alone() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();

        std::fs::write(user_dir.join("bad.md"), "no frontmatter at all").unwrap();

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir.clone(), Vec::new());
        let count = registry.scan().unwrap();

        assert_eq!(count, 0);
        assert!(user_dir.join("bad.md").exists(), "original must survive");
        assert!(!user_dir.join("bad.md.bak").exists());
    }

    // ── Authoring ──────────────────────────────────────────────────────

    #[test]
    fn slug_from_name_is_safe_to_put_on_a_filesystem() {
        assert_eq!(slug_from_name("My Soul!", &[]).unwrap(), "my-soul");
        assert_eq!(slug_from_name("alex", &[]).unwrap(), "alex");
        assert_eq!(
            slug_from_name("  Research   Copilot  ", &[]).unwrap(),
            "research-copilot"
        );
        assert_eq!(
            slug_from_name("A---B", &[]).unwrap(),
            "a-b",
            "runs of punctuation collapse to one dash"
        );
        assert_eq!(
            slug_from_name("!!!weird!!!", &[]).unwrap(),
            "weird",
            "no leading or trailing dashes"
        );
    }

    /// Two souls must never share a folder, whatever the user calls them.
    #[test]
    fn slug_conflicts_get_a_suffix() {
        let taken = vec!["alex".to_string()];
        assert_eq!(slug_from_name("Alex", &taken).unwrap(), "alex-2");

        let taken = vec!["alex".to_string(), "alex-2".to_string()];
        assert_eq!(slug_from_name("alex", &taken).unwrap(), "alex-3");
    }

    /// A name with nothing to slugify cannot become a folder.
    #[test]
    fn a_name_with_no_letters_has_no_slug() {
        assert!(slug_from_name("!!!", &[]).is_err());
        assert!(slug_from_name("   ", &[]).is_err());
    }

    #[test]
    fn slug_validation_matches_what_we_generate() {
        assert!(is_valid_slug("alex"));
        assert!(is_valid_slug("my-soul-2"));

        for bad in [
            "",
            "-alex",
            "alex-",
            "Alex",
            "my soul",
            "../evil",
            "a/b",
            "a".repeat(33).as_str(),
        ] {
            assert!(!is_valid_slug(bad), "'{}' should not be a valid slug", bad);
        }
    }

    /// A name with a space could never be typed after `@`: the mention parser
    /// stops at the space.
    #[test]
    fn names_are_one_word() {
        assert!(name_rejection("alex").is_none());
        assert!(name_rejection("my soul").is_some());
        assert!(name_rejection("").is_some());
        assert!(name_rejection("   ").is_some());
        assert!(name_rejection(&"a".repeat(33)).is_some());
        assert!(name_rejection("al@x").is_some());
        assert!(name_rejection("al#x").is_some());
    }

    /// What the UI writes must be what the loader reads.
    #[test]
    fn write_role_round_trips_through_the_loader() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());

        let meta = AgentRoleMetadata {
            name: "alex".into(),
            description: "Research copilot".into(),
            avatar: Some("./avatar.png".into()),
            allowed_tools: vec!["web_search".into()],
            ..Default::default()
        };
        let bodies = RoleBodies {
            identity: "alex — research copilot.".into(),
            soul: "You are alex.".into(),
            tools: Some("Prefer web_search.".into()),
            agents: None,
        };

        registry.write_role("research", &meta, &bodies).unwrap();
        registry.scan().unwrap();

        let def = registry.get("research").unwrap();
        assert_eq!(def.name, "alex");
        assert_eq!(def.description, "Research copilot");
        assert_eq!(def.avatar.as_deref(), Some("./avatar.png"));
        assert_eq!(def.identity, "alex — research copilot.");
        assert_eq!(def.system_prompt, "You are alex.");
        assert_eq!(def.tools_doc.as_deref(), Some("Prefer web_search."));
        assert!(def.agents_doc.is_none(), "an absent optional stays absent");
        assert!(def.is_soul(), "a user-written role is a soul by default");
        assert_eq!(
            def.tools_config,
            Some(ToolsConfig::Allow(vec!["web_search".into()]))
        );
    }

    /// The file the UI writes must still be one a person can open and edit.
    #[test]
    fn write_role_produces_parseable_frontmatter() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        let registry = AgentRoleRegistry::with_builtin_sources(user_dir.clone(), Vec::new());

        let meta = AgentRoleMetadata {
            name: "alex".into(),
            ..Default::default()
        };
        registry
            .write_role(
                "research",
                &meta,
                &RoleBodies {
                    soul: "Hi.".into(),
                    ..Default::default()
                },
            )
            .unwrap();

        let raw = std::fs::read_to_string(user_dir.join("research").join(IDENTITY_FILE)).unwrap();
        let (parsed, _) = parse_role_frontmatter(&raw).expect("the loader must read what we write");
        assert_eq!(parsed.name, "alex");
    }

    /// Emptying an optional field removes the file: an empty TOOLS.md and no
    /// TOOLS.md have to keep meaning the same thing.
    #[test]
    fn emptying_an_optional_file_removes_it() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        let registry = AgentRoleRegistry::with_builtin_sources(user_dir.clone(), Vec::new());

        let meta = AgentRoleMetadata {
            name: "alex".into(),
            ..Default::default()
        };
        let with_tools = RoleBodies {
            soul: "Hi.".into(),
            tools: Some("Some guidance.".into()),
            ..Default::default()
        };
        registry.write_role("research", &meta, &with_tools).unwrap();
        assert!(user_dir.join("research").join(TOOLS_FILE).exists());

        let without = RoleBodies {
            soul: "Hi.".into(),
            tools: Some("   ".into()),
            ..Default::default()
        };
        registry.write_role("research", &meta, &without).unwrap();
        assert!(
            !user_dir.join("research").join(TOOLS_FILE).exists(),
            "a blank field should not leave an empty file behind"
        );
    }

    #[test]
    fn write_role_refuses_a_bad_slug() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        let registry = AgentRoleRegistry::with_builtin_sources(user_dir.clone(), Vec::new());
        let meta = AgentRoleMetadata {
            name: "alex".into(),
            ..Default::default()
        };

        for bad in ["../evil", "a/b", "UPPER", ""] {
            assert!(registry
                .write_role(bad, &meta, &RoleBodies::default())
                .is_err());
        }
        assert!(
            !user_dir.exists() || std::fs::read_dir(&user_dir).unwrap().next().is_none(),
            "nothing should have been created"
        );
    }

    #[test]
    fn write_role_refuses_a_bad_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let registry =
            AgentRoleRegistry::with_builtin_sources(temp_dir.path().join("user"), Vec::new());

        let bad = AgentRoleMetadata {
            name: "two words".into(),
            ..Default::default()
        };
        assert!(registry
            .write_role("research", &bad, &RoleBodies::default())
            .is_err());
    }

    /// `@name` has to mean one thing.
    #[test]
    fn write_role_refuses_a_name_another_soul_has() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();
        write_role_dir(
            &user_dir,
            "first",
            "---\nname: alex\ndescription: The first\n---\n",
            "Prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let clash = AgentRoleMetadata {
            name: "alex".into(),
            ..Default::default()
        };
        let err = registry
            .write_role(
                "second",
                &clash,
                &RoleBodies {
                    soul: "Hi.".into(),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(
            err.contains("alex"),
            "the message should name the clash: {}",
            err
        );
    }

    /// Renaming a soul is not a clash with itself.
    #[test]
    fn write_role_allows_a_soul_to_keep_its_own_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        std::fs::create_dir_all(&user_dir).unwrap();
        write_role_dir(
            &user_dir,
            "research",
            "---\nname: alex\ndescription: Mine\n---\n",
            "Prompt.",
        );

        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, Vec::new());
        registry.scan().unwrap();

        let same = AgentRoleMetadata {
            name: "alex".into(),
            description: "Edited".into(),
            ..Default::default()
        };
        assert!(registry
            .write_role(
                "research",
                &same,
                &RoleBodies {
                    soul: "New prompt.".into(),
                    ..Default::default()
                }
            )
            .is_ok());
    }

    /// Editing a built-in copies it into the user directory: the embedded layer
    /// is read-only, and the edit must survive the next release.
    #[test]
    fn editing_a_builtin_copies_it_to_the_user_dir() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        let builtin = vec![builtin_role(
            "researcher",
            "---\nname: researcher\ndescription: Built-in\n---\n",
            "Built-in prompt.",
        )];
        let registry = AgentRoleRegistry::with_builtin_sources(user_dir.clone(), builtin);
        registry.scan().unwrap();

        let meta = AgentRoleMetadata {
            name: "researcher".into(),
            description: "Mine now".into(),
            ..Default::default()
        };
        registry
            .write_role(
                "researcher",
                &meta,
                &RoleBodies {
                    soul: "My prompt.".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        registry.scan().unwrap();

        assert!(user_dir.join("researcher").join(IDENTITY_FILE).exists());
        let def = registry.get("researcher").unwrap();
        assert_eq!(def.description, "Mine now");
        assert_eq!(def.system_prompt, "My prompt.");
    }

    /// Deleting a copied built-in restores the embedded original rather than
    /// leaving a hole.
    #[test]
    fn deleting_a_copied_builtin_restores_the_original() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        let builtin = vec![builtin_role(
            "researcher",
            "---\nname: researcher\ndescription: Built-in\n---\n",
            "Built-in prompt.",
        )];
        let registry = AgentRoleRegistry::with_builtin_sources(user_dir, builtin);

        let meta = AgentRoleMetadata {
            name: "researcher".into(),
            description: "Mine".into(),
            ..Default::default()
        };
        registry
            .write_role(
                "researcher",
                &meta,
                &RoleBodies {
                    soul: "Mine.".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        registry.scan().unwrap();
        assert_eq!(registry.get("researcher").unwrap().description, "Mine");

        registry.delete_role("researcher").unwrap();
        registry.scan().unwrap();
        assert_eq!(
            registry.get("researcher").unwrap().description,
            "Built-in",
            "the embedded original comes back"
        );
    }

    #[test]
    fn deleting_a_soul_removes_its_directory() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let user_dir = temp_dir.path().join("user");
        let registry = AgentRoleRegistry::with_builtin_sources(user_dir.clone(), Vec::new());

        let meta = AgentRoleMetadata {
            name: "alex".into(),
            ..Default::default()
        };
        registry
            .write_role(
                "research",
                &meta,
                &RoleBodies {
                    soul: "Hi.".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(user_dir.join("research").exists());

        registry.delete_role("research").unwrap();
        assert!(!user_dir.join("research").exists());
        assert!(
            registry.delete_role("research").is_ok(),
            "deleting twice is not an error"
        );
    }

    #[test]
    fn delete_refuses_a_bad_slug() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let registry =
            AgentRoleRegistry::with_builtin_sources(temp_dir.path().join("user"), Vec::new());
        assert!(registry.delete_role("../evil").is_err());
    }

    /// The registry is shared as an `Arc`, so a soul created while the daemon runs
    /// has to become visible without a restart.
    #[test]
    fn a_soul_written_at_runtime_is_visible_after_a_rescan() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let registry =
            AgentRoleRegistry::with_builtin_sources(temp_dir.path().join("user"), Vec::new());
        registry.scan().unwrap();
        assert!(registry.list().is_empty());

        let meta = AgentRoleMetadata {
            name: "alex".into(),
            ..Default::default()
        };
        registry
            .write_role(
                "research",
                &meta,
                &RoleBodies {
                    soul: "Hi.".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        registry.scan().unwrap();

        assert_eq!(registry.list().len(), 1);
        assert_eq!(registry.slug_for_name("alex").as_deref(), Some("research"));
    }

    // ── Built-in layer ─────────────────────────────────────────────────

    /// Look up a role's embedded files by its catalog key.
    fn embedded_role(slug: &str) -> &'static [(&'static str, &'static str)] {
        nevoflux_builtin_wasm::BUILTIN_AGENT_ROLES
            .iter()
            .find(|(s, _)| *s == slug)
            .unwrap_or_else(|| panic!("Built-in role not embedded: {}", slug))
            .1
    }

    fn embedded_identity(slug: &str) -> &'static str {
        embedded_role(slug)
            .iter()
            .find(|(f, _)| *f == IDENTITY_FILE)
            .unwrap_or_else(|| panic!("Built-in role '{}' has no {}", slug, IDENTITY_FILE))
            .1
    }

    #[test]
    fn test_builtin_roles_parse() {
        let expected_roles = vec![
            ("explorer", "browser", 10),
            ("researcher", "browser", 20),
            ("worker", "agent", 15),
            ("reader", "agent", 10),
        ];

        for (slug, mode, max_iter) in expected_roles {
            let files: Vec<(String, String)> = embedded_role(slug)
                .iter()
                .map(|(f, c)| (f.to_string(), c.to_string()))
                .collect();
            let source = RoleSource::from_embedded(&files)
                .unwrap_or_else(|e| panic!("Built-in role '{}' incomplete: {}", slug, e));
            let (meta, bodies) = source
                .parse(slug)
                .unwrap_or_else(|e| panic!("Built-in role '{}' failed to parse: {}", slug, e));

            assert_eq!(meta.name, slug, "Name mismatch for {}", slug);
            assert!(
                !meta.description.is_empty(),
                "Description empty for {}",
                slug
            );
            assert_eq!(meta.mode, mode, "Mode mismatch for {}", slug);
            assert_eq!(
                meta.max_iterations, max_iter,
                "max_iterations mismatch for {}",
                slug
            );
            assert!(!bodies.soul.is_empty(), "SOUL.md empty for {}", slug);

            // Verify it validates as a definition too
            let def = AgentRoleDefinition::from_parts(slug, meta, bodies).unwrap();
            assert_eq!(def.slug, slug);
        }
    }

    /// `load_definition` looks built-ins up by their catalog key, while `scan`
    /// advertises the frontmatter `name`. If the two ever diverged, a role that
    /// `list()` reports would be unreachable by the name shown to users.
    #[test]
    fn test_builtin_catalog_key_matches_frontmatter_name() {
        for (slug, _) in nevoflux_builtin_wasm::BUILTIN_AGENT_ROLES {
            let (meta, _) = parse_role_frontmatter(embedded_identity(slug))
                .unwrap_or_else(|e| panic!("Built-in role '{}' failed to parse: {}", slug, e));
            assert_eq!(
                &meta.name, slug,
                "Built-in role catalog key '{}' does not match its frontmatter name '{}'",
                slug, meta.name
            );
        }
    }

    /// Every built-in must carry both required files, or it silently vanishes on
    /// an installed binary.
    #[test]
    fn test_builtin_roles_have_required_files() {
        for (slug, files) in nevoflux_builtin_wasm::BUILTIN_AGENT_ROLES {
            for required in [IDENTITY_FILE, SOUL_FILE] {
                assert!(
                    files
                        .iter()
                        .any(|(f, c)| *f == required && !c.trim().is_empty()),
                    "Built-in role '{}' is missing a non-empty {}",
                    slug,
                    required
                );
            }
        }
    }

    /// The built-in roles must resolve with no role directory on disk anywhere.
    /// This is what an installed binary sees: the source tree it was built from
    /// is gone, and the user has never created `<config>/nevoflux/agents`.
    #[test]
    fn test_builtin_roles_resolve_without_any_directory() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let missing_user_dir = temp_dir.path().join("no_such_agents_dir");
        assert!(!missing_user_dir.exists());

        let registry = AgentRoleRegistry::new(missing_user_dir);
        let count = registry.scan().unwrap();
        assert_eq!(count, 4, "Expected the 4 embedded built-in roles");

        let names: Vec<String> = registry.list().into_iter().map(|s| s.name).collect();
        for expected in ["explorer", "researcher", "worker", "reader"] {
            assert!(
                names.contains(&expected.to_string()),
                "Missing {}",
                expected
            );

            // L2 load must work too, not just the L1 summary.
            let def = registry.get(expected).unwrap();
            assert_eq!(def.slug, expected);
            assert!(!def.system_prompt.is_empty());
        }
    }
}
