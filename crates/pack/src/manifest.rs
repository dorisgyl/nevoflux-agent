//! `pack.toml` model, parsing, and parse-time field validation.

use semver::Version;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub pack: PackMeta,
    #[serde(default)]
    pub components: Components,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackMeta {
    pub name: String,
    pub version: Version,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub authors: Vec<String>,
    pub protocol: String,
    pub min_nevoflux: Version,
    #[serde(default)]
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Components {
    pub skills: Option<SkillsComponent>,
    pub canvas_tools: Option<CanvasToolsComponent>,
    #[serde(default)]
    pub seed: Vec<SeedComponent>,
    pub knowledge: Option<KnowledgeComponent>,
    pub dashboard: Option<DashboardComponent>,
    pub protected: Option<ProtectedComponent>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SkillsComponent {
    pub dir: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CanvasToolsComponent {
    pub files: Vec<String>,
    #[serde(default)]
    pub external_binaries: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SeedComponent {
    pub slug: String,
    pub from: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KnowledgeComponent {
    pub from: String,
    pub source_name: Option<String>,
    pub trust: String,
    pub unlock: UnlockSpec,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum UnlockSpec {
    Key { key: String },
    Password { password: String },
}

#[derive(Debug, Clone, Deserialize)]
pub struct DashboardComponent {
    pub artifact_id: String,
    pub content_type: String,
    pub files_from: String,
    pub entry: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProtectedComponent {
    #[serde(default)]
    pub slugs: Vec<String>,
    #[serde(default)]
    pub prefixes: Vec<String>,
}

pub const SUPPORTED_PROTOCOLS: &[&str] = &["pack-protocol/0.1"];

impl Manifest {
    /// Parse and run parse-time field validation. Capability/namespace checks
    /// live in `capability::validate` and run later against `ResolvedPaths`.
    pub fn parse(toml_src: &str) -> Result<Manifest, String> {
        let m: Manifest = toml::from_str(toml_src).map_err(|e| e.to_string())?;
        m.validate_fields()?;
        Ok(m)
    }

    /// The GBrain namespace prefix: explicit override, else pack name.
    pub fn namespace(&self) -> &str {
        self.pack.namespace.as_deref().unwrap_or(&self.pack.name)
    }

    fn validate_fields(&self) -> Result<(), String> {
        // name: [a-z0-9-]+
        if self.pack.name.is_empty()
            || !self
                .pack
                .name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(format!(
                "invalid pack.name '{}': must match [a-z0-9-]+",
                self.pack.name
            ));
        }
        // protocol supported
        if !SUPPORTED_PROTOCOLS.contains(&self.pack.protocol.as_str()) {
            return Err(format!("unsupported protocol '{}'", self.pack.protocol));
        }
        // knowledge.trust must be read-only in v1
        if let Some(k) = &self.components.knowledge {
            if k.trust != "read-only" {
                return Err(format!(
                    "knowledge.trust '{}' unsupported in v1 (only 'read-only')",
                    k.trust
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
        [pack]
        name = "hello-pack"
        version = "0.1.0"
        protocol = "pack-protocol/0.1"
        min_nevoflux = "0.3.0"

        [components.skills]
        dir = "skills"
    "#;

    #[test]
    fn parses_minimal_manifest() {
        let m = Manifest::parse(MINIMAL).unwrap();
        assert_eq!(m.pack.name, "hello-pack");
        assert_eq!(m.namespace(), "hello-pack");
        assert_eq!(m.components.skills.unwrap().dir, "skills");
    }

    #[test]
    fn namespace_override_wins() {
        let src = MINIMAL.replace(
            "name = \"hello-pack\"",
            "name = \"career-pack\"\nnamespace = \"career\"",
        );
        let m = Manifest::parse(&src).unwrap();
        assert_eq!(m.namespace(), "career");
    }

    #[test]
    fn rejects_bad_name() {
        let src = MINIMAL.replace("hello-pack", "Hello_Pack");
        assert!(Manifest::parse(&src).unwrap_err().contains("pack.name"));
    }

    #[test]
    fn rejects_unsupported_protocol() {
        let src = MINIMAL.replace("pack-protocol/0.1", "pack-protocol/9.9");
        assert!(Manifest::parse(&src)
            .unwrap_err()
            .contains("unsupported protocol"));
    }

    #[test]
    fn rejects_non_readonly_knowledge_trust() {
        let src = format!(
            "{MINIMAL}\n[components.knowledge]\nfrom=\"kb.nbrain\"\ntrust=\"full-merge\"\nunlock={{ password = \"x\" }}\n"
        );
        assert!(Manifest::parse(&src).unwrap_err().contains("read-only"));
    }
}
