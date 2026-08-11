/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Stable id allocation for user-defined LLM providers.
//!
//! The id is derived from the display name once, at creation, and never
//! changes again — renaming a provider edits only its `display_name`, so the
//! active-provider pointer and any stored `custom:<id>` string stay valid.

/// Longest id we will mint. Long enough to stay readable in `config.toml`,
/// short enough that the disambiguating suffix always fits.
const MAX_SLUG_LEN: usize = 48;

/// Slugify `name` into the id charset: `[a-z0-9-]`, no leading/trailing or
/// repeated hyphens. Returns an empty string when nothing survives (an all-CJK
/// name, pure punctuation, or an empty input).
fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut pending_hyphen = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_hyphen && !out.is_empty() {
                out.push('-');
            }
            pending_hyphen = false;
            out.push(ch.to_ascii_lowercase());
            if out.len() >= MAX_SLUG_LEN {
                break;
            }
        } else {
            pending_hyphen = true;
        }
    }
    out
}

/// Allocate a stable, unused id for a custom provider named `display_name`.
///
/// `is_taken` reports whether a candidate id already exists. Falls back to
/// `custom-N` when the name yields no usable slug.
pub fn allocate_custom_id<F: Fn(&str) -> bool>(display_name: &str, is_taken: F) -> String {
    let base = slugify(display_name);
    if base.is_empty() {
        let mut n = 1usize;
        loop {
            let candidate = format!("custom-{n}");
            if !is_taken(&candidate) {
                return candidate;
            }
            n += 1;
        }
    }

    if !is_taken(&base) {
        return base;
    }
    let mut n = 2usize;
    loop {
        let candidate = format!("{base}-{n}");
        if !is_taken(&candidate) {
            return candidate;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn taken(set: &BTreeSet<String>) -> impl Fn(&str) -> bool + '_ {
        move |id: &str| set.contains(id)
    }

    #[test]
    fn slugifies_a_plain_name() {
        let none = BTreeSet::new();
        assert_eq!(allocate_custom_id("My LLM", taken(&none)), "my-llm");
    }

    #[test]
    fn collapses_punctuation_and_trims() {
        let none = BTreeSet::new();
        assert_eq!(
            allocate_custom_id("  Acme // Gateway (v2)!  ", taken(&none)),
            "acme-gateway-v2"
        );
    }

    #[test]
    fn disambiguates_collisions() {
        let mut set = BTreeSet::new();
        set.insert("my-llm".to_string());
        assert_eq!(allocate_custom_id("My LLM", taken(&set)), "my-llm-2");
        set.insert("my-llm-2".to_string());
        assert_eq!(allocate_custom_id("My LLM", taken(&set)), "my-llm-3");
    }

    #[test]
    fn falls_back_when_slug_is_empty() {
        let none = BTreeSet::new();
        assert_eq!(
            allocate_custom_id("\u{672c}\u{5730}\u{7ad9}", taken(&none)),
            "custom-1"
        );
        assert_eq!(allocate_custom_id("!!!", taken(&none)), "custom-1");
        assert_eq!(allocate_custom_id("", taken(&none)), "custom-1");
    }

    #[test]
    fn disambiguates_the_fallback_too() {
        let mut set = BTreeSet::new();
        set.insert("custom-1".to_string());
        assert_eq!(
            allocate_custom_id("\u{672c}\u{5730}\u{7ad9}", taken(&set)),
            "custom-2"
        );
    }

    #[test]
    fn keeps_digits_and_existing_hyphens() {
        let none = BTreeSet::new();
        assert_eq!(
            allocate_custom_id("gpt-4o-proxy", taken(&none)),
            "gpt-4o-proxy"
        );
    }

    #[test]
    fn result_always_matches_the_id_charset() {
        let none = BTreeSet::new();
        let long = "a".repeat(200);
        for name in [
            "My LLM",
            "\u{672c}\u{5730}\u{7ad9}",
            "\u{DC}n\u{EF}c\u{F8}d\u{E9}",
            long.as_str(),
        ] {
            let id = allocate_custom_id(name, taken(&none));
            assert!(!id.is_empty());
            assert!(
                id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "bad id: {id}"
            );
            assert!(!id.starts_with('-') && !id.ends_with('-'), "bad id: {id}");
            assert!(id.len() <= 48, "id too long: {id}");
        }
    }
}
