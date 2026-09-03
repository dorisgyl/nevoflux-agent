//! CLI module with clap-based argument parsing.
//!
//! Provides subcommands for configuration management and setup.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// NevoFlux Agent - AI-powered browser assistant
#[derive(Parser, Debug)]
#[command(name = "nevoflux-agent")]
#[command(version)]
#[command(about = "NevoFlux Agent - AI-powered browser assistant")]
#[command(
    long_about = "NevoFlux is an AI-powered browser assistant that provides \
    intelligent automation and assistance through browser extensions and MCP integration."
)]
// Allow external subcommands (Firefox passes manifest path and extension ID)
#[command(allow_external_subcommands = true)]
pub struct Cli {
    /// Run as MCP server (stdio bridge for Claude Code integration)
    #[arg(long)]
    pub mcp: bool,

    /// Run as daemon (core processing server)
    #[arg(long)]
    pub daemon: bool,

    /// Check daemon status
    #[arg(long)]
    pub status: bool,

    /// Stop the running daemon
    #[arg(long)]
    pub stop: bool,

    /// Config file path (overrides default location)
    #[arg(long, short)]
    pub config: Option<PathBuf>,

    /// Enable verbose output
    #[arg(long, short)]
    pub verbose: bool,

    /// Enable trace output for debugging (writes JSONL to data dir)
    #[arg(long)]
    pub trace: bool,

    /// Run proxy in dev mode (connect to manually-started daemon on port 19500)
    #[arg(long)]
    pub dev: bool,

    /// Override daemon port range start (used when proxy spawns a daemon)
    #[arg(long, hide = true)]
    pub port_start: Option<u16>,

    /// Override daemon port range end (used when proxy spawns a daemon)
    #[arg(long, hide = true)]
    pub port_end: Option<u16>,

    /// Bind to this exact port (used by proxy in managed mode)
    #[arg(long, hide = true)]
    pub port: Option<u16>,

    /// Daemon self-terminates on idle (set by proxy when spawning)
    #[arg(long, hide = true)]
    pub managed: bool,

    /// Run the daemon in headless automation mode: spawn a browser and serve the
    /// task HTTP API. Requires --daemon. (P4)
    #[arg(long)]
    pub headless: bool,

    /// Run the remote-control service: keep one browser and one conversation
    /// alive and drive them from a phone through nevoflux.app. Prints a connect
    /// block (link + pairing code) on startup and serves no HTTP at all — the
    /// relay socket is the only outward face. Requires --daemon.
    #[arg(long)]
    pub remote_control: bool,

    /// HTTP bind address for the task API in headless mode (e.g. 0.0.0.0:8080).
    /// Also serves the OpenAI-compatible /v1/chat/completions on this port.
    #[arg(long)]
    pub http_addr: Option<std::net::SocketAddr>,

    /// Serve the OpenAI-compatible API (/v1/chat/completions) on a DEDICATED port
    /// (in addition to --http-addr). e.g. 0.0.0.0:8081
    #[arg(long)]
    pub openai_addr: Option<std::net::SocketAddr>,

    /// Serve the MCP-over-HTTP server (POST /mcp) on this port. e.g. 0.0.0.0:8082
    #[arg(long)]
    pub mcp_addr: Option<std::net::SocketAddr>,

    /// Serve the ACP-over-HTTP endpoint (POST /acp) on this port. e.g. 0.0.0.0:8083
    #[arg(long)]
    pub acp_addr: Option<std::net::SocketAddr>,

    /// Serve the Anthropic Messages API (POST /v1/messages) on a DEDICATED
    /// port. e.g. 0.0.0.0:8085
    ///
    /// Also served on --http-addr. It gets its own flag rather than riding on
    /// --openai-addr because the headers (`x-api-key`, `anthropic-version`)
    /// and the error envelope are Anthropic's, not OpenAI's.
    #[arg(long)]
    pub anthropic_addr: Option<std::net::SocketAddr>,

    /// Serve the A2A endpoints on this port. e.g. 0.0.0.0:8084
    ///
    /// Mounts three paths: `GET /.well-known/agent-card.json` (discovery),
    /// `POST /a2a` (protocol 0.3.0) and `POST /a2a/v1` (protocol 1.0). Each
    /// POST path speaks exactly one protocol version — that is how the two are
    /// advertised in the card, and it keeps the "empty A2A-Version means 0.3"
    /// rule from ever mattering.
    #[arg(long)]
    pub a2a_addr: Option<std::net::SocketAddr>,

    /// Serve the token-protected admin API on this port. e.g. 127.0.0.1:8084
    ///
    /// Requires NEVOFLUX_ADMIN_TOKEN; without it the surface is not mounted.
    /// Keep this on a private interface: it deploys code to the server.
    #[arg(long)]
    pub admin_addr: Option<std::net::SocketAddr>,

    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
#[allow(dead_code)] // External variant's Vec<String> is used in tests but not in main binary
pub enum Commands {
    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Pack install protocol
    Pack {
        #[command(subcommand)]
        action: PackAction,
    },
    /// nevoflux.app account: sign in (for --remote-control), status, sign out
    Account {
        #[command(subcommand)]
        action: AccountAction,
    },
    /// Run interactive setup wizard
    Setup,
    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
    /// External subcommand (catches Firefox native messaging args)
    #[command(external_subcommand)]
    External(Vec<String>),
}

/// Configuration subcommand actions.
#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Show current configuration
    Show,
    /// Initialize default configuration
    Init,
    /// Get a configuration value by key
    Get {
        /// The configuration key (e.g., "app.theme")
        key: String,
    },
    /// Set a configuration value
    Set {
        /// The configuration key (e.g., "app.theme")
        key: String,
        /// The value to set (JSON format for complex values)
        value: String,
    },
    /// Delete a configuration value
    Delete {
        /// The configuration key to delete
        key: String,
    },
    /// List configuration values by prefix
    List {
        /// Optional prefix to filter keys (e.g., "app.")
        #[arg(default_value = "")]
        prefix: String,
    },
}

/// Pack subcommand actions.
#[derive(Subcommand, Debug)]
pub enum PackAction {
    /// Validate a pack manifest without installing (capability check)
    Validate {
        /// manifest path or github:user/repo[/sub][@ref]
        source: String,
    },
    /// Fetch + preview a pack (local path or github:user/repo[/sub][@ref]) without installing
    Inspect {
        /// manifest path or github:user/repo[/sub][@ref]
        source: String,
    },
    /// Install a pack from a local pack.toml or a github source
    Install {
        /// manifest path or github:user/repo[/sub][@ref]
        source: String,
        /// Overwrite an existing installation
        #[arg(long)]
        force: bool,
        /// Skip the remote-source preview/confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Uninstall a pack by name
    Uninstall {
        /// Name of the installed pack
        name: String,
        /// Also remove pack-owned data (e.g. seeded pages)
        #[arg(long)]
        purge_data: bool,
        /// Force removal even if integrity checks fail
        #[arg(long)]
        force: bool,
    },
    /// Update an installed pack from a local pack.toml or a github source
    Update {
        /// manifest path or github:user/repo[/sub][@ref]
        source: String,
    },
    /// List installed packs
    List,
    /// Show status of one pack
    Status {
        /// Name of the installed pack
        name: String,
    },
}

/// Account subcommand actions.
///
/// `login` runs the RFC 8628 device authorization grant against nevoflux.app
/// entirely in-process — no daemon required — so a container can mint its own
/// account token before `--remote-control` starts. The resulting token is the
/// same credential `NEVOFLUX_SERVICE_TOKEN` supplies; `--print-token` emits it
/// on stdout for capture into a secret store.
#[derive(Subcommand, Debug)]
pub enum AccountAction {
    /// Sign in to nevoflux.app (device grant). Prints a URL + code to approve on
    /// any device, then stores the account token used by `--remote-control`.
    Login {
        /// Also print the raw token as the final stdout line (prompts/status go
        /// to stderr, so stdout carries only the token — capture it with
        /// `TOKEN=$(nevoflux account login --print-token)`).
        #[arg(long)]
        print_token: bool,
        /// Do not write <data_dir>/account-token; only emit the token on stdout
        /// (implies --print-token). Use when the token lives solely in a secret.
        #[arg(long)]
        no_save: bool,
    },
    /// Report whether an account token is present (env override or on disk).
    Status,
    /// Remove the stored account-token file.
    Logout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_no_args() {
        let cli = Cli::try_parse_from(["nevoflux"]).unwrap();
        assert!(!cli.mcp);
        assert!(!cli.daemon);
        assert!(!cli.status);
        assert!(!cli.stop);
        assert!(cli.config.is_none());
        assert!(!cli.verbose);
        assert!(!cli.dev);
        assert!(cli.port_start.is_none());
        assert!(cli.port_end.is_none());
        assert!(!cli.managed);
        assert!(cli.command.is_none());
    }

    #[test]
    fn test_cli_parse_mcp_flag() {
        let cli = Cli::try_parse_from(["nevoflux", "--mcp"]).unwrap();
        assert!(cli.mcp);
    }

    #[test]
    fn test_cli_parse_daemon_flag() {
        let cli = Cli::try_parse_from(["nevoflux", "--daemon"]).unwrap();
        assert!(cli.daemon);
    }

    #[test]
    fn test_cli_parse_status_flag() {
        let cli = Cli::try_parse_from(["nevoflux", "--status"]).unwrap();
        assert!(cli.status);
    }

    #[test]
    fn test_cli_parse_stop_flag() {
        let cli = Cli::try_parse_from(["nevoflux", "--stop"]).unwrap();
        assert!(cli.stop);
    }

    #[test]
    fn test_cli_parse_verbose_flag() {
        let cli = Cli::try_parse_from(["nevoflux", "-v"]).unwrap();
        assert!(cli.verbose);

        let cli = Cli::try_parse_from(["nevoflux", "--verbose"]).unwrap();
        assert!(cli.verbose);
    }

    #[test]
    fn test_cli_parse_config_path() {
        let cli = Cli::try_parse_from(["nevoflux", "-c", "/path/to/config.toml"]).unwrap();
        assert_eq!(cli.config, Some(PathBuf::from("/path/to/config.toml")));

        let cli = Cli::try_parse_from(["nevoflux", "--config", "/other/path.toml"]).unwrap();
        assert_eq!(cli.config, Some(PathBuf::from("/other/path.toml")));
    }

    #[test]
    fn test_cli_parse_config_show() {
        let cli = Cli::try_parse_from(["nevoflux", "config", "show"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Config {
                action: ConfigAction::Show
            })
        ));
    }

    #[test]
    fn test_cli_parse_config_init() {
        let cli = Cli::try_parse_from(["nevoflux", "config", "init"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Config {
                action: ConfigAction::Init
            })
        ));
    }

    #[test]
    fn test_cli_parse_config_get() {
        let cli = Cli::try_parse_from(["nevoflux", "config", "get", "app.theme"]).unwrap();
        match cli.command {
            Some(Commands::Config {
                action: ConfigAction::Get { key },
            }) => {
                assert_eq!(key, "app.theme");
            }
            _ => panic!("Expected Config Get command"),
        }
    }

    #[test]
    fn test_cli_parse_config_set() {
        let cli = Cli::try_parse_from(["nevoflux", "config", "set", "app.theme", "dark"]).unwrap();
        match cli.command {
            Some(Commands::Config {
                action: ConfigAction::Set { key, value },
            }) => {
                assert_eq!(key, "app.theme");
                assert_eq!(value, "dark");
            }
            _ => panic!("Expected Config Set command"),
        }
    }

    #[test]
    fn test_cli_parse_config_set_json_value() {
        let cli = Cli::try_parse_from([
            "nevoflux",
            "config",
            "set",
            "app.settings",
            r#"{"theme":"dark","font_size":14}"#,
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Config {
                action: ConfigAction::Set { key, value },
            }) => {
                assert_eq!(key, "app.settings");
                assert_eq!(value, r#"{"theme":"dark","font_size":14}"#);
            }
            _ => panic!("Expected Config Set command"),
        }
    }

    #[test]
    fn test_cli_parse_config_delete() {
        let cli = Cli::try_parse_from(["nevoflux", "config", "delete", "app.theme"]).unwrap();
        match cli.command {
            Some(Commands::Config {
                action: ConfigAction::Delete { key },
            }) => {
                assert_eq!(key, "app.theme");
            }
            _ => panic!("Expected Config Delete command"),
        }
    }

    #[test]
    fn test_cli_parse_config_list() {
        let cli = Cli::try_parse_from(["nevoflux", "config", "list"]).unwrap();
        match cli.command {
            Some(Commands::Config {
                action: ConfigAction::List { prefix },
            }) => {
                assert_eq!(prefix, "");
            }
            _ => panic!("Expected Config List command"),
        }
    }

    #[test]
    fn test_cli_parse_config_list_with_prefix() {
        let cli = Cli::try_parse_from(["nevoflux", "config", "list", "app."]).unwrap();
        match cli.command {
            Some(Commands::Config {
                action: ConfigAction::List { prefix },
            }) => {
                assert_eq!(prefix, "app.");
            }
            _ => panic!("Expected Config List command"),
        }
    }

    #[test]
    fn test_cli_parse_setup() {
        let cli = Cli::try_parse_from(["nevoflux", "setup"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Setup)));
    }

    #[test]
    fn test_cli_parse_combined_flags() {
        let cli = Cli::try_parse_from(["nevoflux", "--verbose", "--daemon"]).unwrap();
        assert!(cli.verbose);
        assert!(cli.daemon);
    }

    #[test]
    fn test_cli_parse_trace_flag() {
        let cli = Cli::try_parse_from(["nevoflux", "--trace", "--daemon"]).unwrap();
        assert!(cli.trace);
        assert!(cli.daemon);
    }

    #[test]
    fn test_cli_parse_dev_flag() {
        let cli = Cli::try_parse_from(["nevoflux", "--dev"]).unwrap();
        assert!(cli.dev);
    }

    #[test]
    fn test_cli_parse_port_start_end() {
        let cli = Cli::try_parse_from(["nevoflux", "--port-start", "19501", "--port-end", "19600"])
            .unwrap();
        assert_eq!(cli.port_start, Some(19501));
        assert_eq!(cli.port_end, Some(19600));
    }

    #[test]
    fn test_cli_parse_managed_flag() {
        let cli = Cli::try_parse_from(["nevoflux", "--daemon", "--managed"]).unwrap();
        assert!(cli.daemon);
        assert!(cli.managed);
    }

    #[test]
    fn test_cli_parse_port_flag() {
        let cli =
            Cli::try_parse_from(["nevoflux", "--daemon", "--managed", "--port", "19523"]).unwrap();
        assert!(cli.daemon);
        assert!(cli.managed);
        assert_eq!(cli.port, Some(19523));
    }

    #[test]
    fn test_cli_parse_firefox_native_messaging_args() {
        // Firefox passes manifest path and extension ID as arguments
        let cli = Cli::try_parse_from([
            "nevoflux",
            "/home/user/.mozilla/native-messaging-hosts/com.nevoflux.agent.json",
            "agent@nevoflux.com",
        ])
        .unwrap();

        // These should be captured as external subcommand, not cause an error
        match cli.command {
            Some(Commands::External(args)) => {
                assert_eq!(args.len(), 2);
                assert!(args[0].contains("native-messaging-hosts"));
                assert_eq!(args[1], "agent@nevoflux.com");
            }
            _ => panic!("Expected External command for Firefox args"),
        }
    }

    #[test]
    fn test_cli_parse_account_login() {
        let cli = Cli::try_parse_from(["nevoflux", "account", "login"]).unwrap();
        match cli.command {
            Some(Commands::Account {
                action:
                    AccountAction::Login {
                        print_token,
                        no_save,
                    },
            }) => {
                assert!(!print_token);
                assert!(!no_save);
            }
            _ => panic!("Expected Account::Login"),
        }
    }

    #[test]
    fn test_cli_parse_account_login_flags() {
        let cli =
            Cli::try_parse_from(["nevoflux", "account", "login", "--print-token", "--no-save"])
                .unwrap();
        match cli.command {
            Some(Commands::Account {
                action:
                    AccountAction::Login {
                        print_token,
                        no_save,
                    },
            }) => {
                assert!(print_token);
                assert!(no_save);
            }
            _ => panic!("Expected Account::Login with flags"),
        }
    }

    #[test]
    fn test_cli_parse_account_status_and_logout() {
        assert!(matches!(
            Cli::try_parse_from(["nevoflux", "account", "status"])
                .unwrap()
                .command,
            Some(Commands::Account {
                action: AccountAction::Status
            })
        ));
        assert!(matches!(
            Cli::try_parse_from(["nevoflux", "account", "logout"])
                .unwrap()
                .command,
            Some(Commands::Account {
                action: AccountAction::Logout
            })
        ));
    }
}
