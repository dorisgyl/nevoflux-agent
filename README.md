# NevoFlux Agent

AI-powered browser assistant with native computer control capabilities.

## Features

- **Browser Automation**: Full browser control via extension - navigate, click, type, fill, screenshot, JavaScript execution, accessibility tree snapshots
- **Computer Control**: Screenshots, mouse, keyboard support for Linux (X11), macOS, and Windows
- **MCP Integration**: Works with Claude Code via Model Context Protocol (stdio transport)
- **MCP Manager**: Connect to external MCP servers with auto-reconnection and health monitoring
- **Native Messaging**: Secure bridge between browser extension and daemon via ZeroMQ
- **LLM Integration**: Anthropic, OpenAI, OpenRouter, DeepSeek, Qwen, Gemini, Groq, Ollama, Mistral, xAI, Cohere, Perplexity, Together — plus subprocess/ACP agents (Claude Code, Gemini CLI, OpenClaw)
- **WASM Skills**: Extensible skill system using WebAssembly modules with host functions
- **Session Management**: Persistent session storage with SQLite, automatic cleanup policies
- **Hot Config Reload**: Configuration file watching with automatic updates
- **Tool Search**: BM25 text search for discovering tools across registries
- **API Key Management**: Layered lookup (environment, keyring, config file)

## Installation

### Prerequisites

- Rust 1.91.0 (pinned via `rust-toolchain.toml`)
- Linux: X11 display server (Wayland not yet supported)
- macOS: Accessibility permissions required for computer control
- Windows: Administrator privileges for native messaging host installation

### Build from Source

```bash
git clone https://github.com/nevoflux-browser/nevoflux-agent
cd nevoflux-agent
cargo build --release
```

The binary will be available at `target/release/nevoflux-agent`.

> The executable is named **`nevoflux-agent`**. The examples below use `nevoflux`
> for brevity — either alias it (`alias nevoflux=nevoflux-agent`) or type the full name.

### Using justfile

Development tasks are automated via `just`:

```bash
just build          # Build all crates
just release        # Build release version
just test           # Run all tests
just ci             # Full CI (fmt check + clippy + tests)
just daemon         # Run daemon with verbose logging
just mcp            # Run MCP server
just status         # Check daemon status
just stop           # Stop running daemon
just fmt            # Format code
just lint           # Run clippy
just doc            # Generate documentation
just clean          # Clean build artifacts
```

### Install Native Messaging Host

```bash
# Linux/macOS
./install/native-host/setup.sh $(pwd)/target/release/nevoflux-agent YOUR_EXTENSION_ID chrome

# Windows (PowerShell as Administrator)
.\install\native-host\setup.ps1 -BinaryPath "$PWD\target\release\nevoflux-agent.exe" -ExtensionId "YOUR_EXTENSION_ID" -Browser chrome
```

## Usage

### Start Daemon

```bash
nevoflux-agent --daemon           # Start daemon
nevoflux-agent --daemon --verbose # With verbose logging
```

### Check Status

```bash
nevoflux-agent --status
```

### Stop Daemon

```bash
nevoflux-agent --stop
```

### MCP Mode (for Claude Code)

```bash
nevoflux-agent --mcp
```

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "nevoflux-agent": {
      "command": "/path/to/nevoflux-agent",
      "args": ["--mcp"]
    }
  }
}
```

### Configuration Management

```bash
nevoflux-agent config init              # Create default config
nevoflux-agent config show              # Show current config
nevoflux-agent config get llm.provider  # Get specific value
nevoflux-agent config set llm.provider qwen  # Set value
nevoflux-agent config list daemon.      # List by prefix
nevoflux-agent config delete key        # Delete value
```

### Interactive Setup

```bash
nevoflux setup
```

### Other Commands

```bash
nevoflux pack list              # Manage capability packs (validate/inspect/install/…)
nevoflux completions bash       # Generate shell completions
```

Server/deployment flags — `--headless`, `--http-addr`, `--openai-addr`, `--mcp-addr`,
`--acp-addr` — are covered under [Headless Deployment](#headless-deployment-docker).

## Headless Deployment (Docker)

Run NevoFlux Agent as a **containerized service** — no GUI needed. The
`nevoflux/agent` image packages the daemon plus a headless browser and serves an
HTTP **task API**, an **OpenAI-compatible** `/v1/chat/completions` endpoint, and
optional **MCP** / **ACP** interfaces. Each task drives a real browser
(automation + computer use + shell/fs tools) with the built-in knowledge base —
so "headless" is only the deployment mode, not the extent of what it does.

The image is built from `deploy/headless/` and **downloads prebuilt releases from
GitHub — no local compilation**:

```bash
docker build -t nevoflux/agent:latest deploy/headless      # amd64 or arm64
cd deploy/headless && docker compose up -d                 # long-running service

# submit a browser task
curl -s localhost:8080/tasks -H 'Content-Type: application/json' \
  -d '{"task":"open example.com and report the title","mode":"browser"}'
# OpenAI-compatible endpoint: POST localhost:8080/v1/chat/completions
```

Notable modes (full guide in [`deploy/headless/README.md`](deploy/headless/README.md)):

- **Session mode** (`NEVOFLUX_SESSION_MODE=1`): reuse ONE warm browser + profile
  clone across a task-flow (soft-reset between tasks); end with `{"end_session":true}`
  on a task or `POST /session/close`. Persist the profile back to a base profile
  with the `save_profile` flag / `POST /session/close {"save":true}`.
- **Fixed-script mode** (`NEVOFLUX_HEADLESS_SCRIPT=…`): run a deterministic Python
  flow instead of the LLM agent loop (see `deploy/headless/examples/`).
- **One task per container** (untrusted workloads): `docker compose --profile oneshot run …`.
- **Live view**: noVNC at `http://localhost:6080/vnc.html` with `NEVOFLUX_VNC=1`.

Versions default to the latest release; pin with `--build-arg AGENT_VERSION=vX.Y.Z
--build-arg BROWSER_VERSION=X.Y.Z`. To build from local/unreleased binaries, use
`deploy/headless/Dockerfile.local` (COPY-based). Helper scripts for the task API
and session mode live in `deploy/headless/examples/`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NevoFlux Agent                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────────────────┐   │
│  │  Claude   │    │  Browser  │    │      Main Binary      │   │
│  │   Code    │    │ Extension │    │      (nevoflux)       │   │
│  └─────┬─────┘    └─────┬─────┘    └───────────┬───────────┘   │
│        │                │                      │               │
│        │ stdio          │ Native               │               │
│        │                │ Messaging            │               │
│        ▼                ▼                      ▼               │
│  ┌───────────┐    ┌───────────┐    ┌───────────────────────┐   │
│  │    MCP    │    │  Bridge   │    │       Daemon          │   │
│  │  Server   │◄──►│  (ZeroMQ) │◄──►│    (Core Engine)      │   │
│  └───────────┘    └───────────┘    └───────────┬───────────┘   │
│                                                │               │
│                                    ┌───────────┴───────────┐   │
│                                    │                       │   │
│                              ┌─────▼─────┐  ┌─────────────┐   │
│                              │   WASM    │  │   Computer  │   │
│                              │  Skills   │  │   Control   │   │
│                              └───────────┘  └─────────────┘   │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │                    Shared Crates                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐  │   │
│  │  │Protocol │ │ Storage │ │   LLM   │ │   Testing   │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘  │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Crate Structure

| Crate | Description |
|-------|-------------|
| `nevoflux` | Main binary with CLI interface |
| `nevoflux-daemon` | Core daemon: sessions, WASM runtime, agent runner, permissions |
| `nevoflux-mcp` | MCP server/client, tools, external server manager |
| `nevoflux-bridge` | Native messaging bridge between browser and daemon |
| `nevoflux-protocol` | Shared message types and serialization (JSON/MessagePack) |
| `nevoflux-storage` | SQLite-based persistent storage |
| `nevoflux-llm` | LLM provider abstraction (Anthropic, OpenAI, Gemini, DeepSeek, Qwen, … + ACP/CLI agents) |
| `nevoflux-computer` | Cross-platform computer control (screenshot, mouse, keyboard) |
| `nevoflux-skills` | WASM skill loading and management |
| `nevoflux-builtin-wasm` | Built-in WASM skill modules |
| `nevoflux-testing` | Testing utilities, mocks, and fixtures |
| `nevoflux-brain` | Knowledge/memory brain engine |
| `nevoflux-llm-gateway` | In-process OpenAI-compatible LLM + embeddings gateway |
| `nevoflux-pack` | Pack (capability bundle) install + lifecycle |

## MCP Tools

### Browser Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `browser_navigate` | Navigate to URL | `url` |
| `browser_click` | Click element by selector | `selector` |
| `browser_type` | Type text into element | `selector`, `text`, `clear?` |
| `browser_fill` | Fill form field (clears first) | `selector`, `value` |
| `browser_screenshot` | Capture page screenshot | `full_page?` |
| `browser_get_content` | Get page/element text | `selector?` |
| `browser_eval_js` | Execute JavaScript | `script` |
| `browser_wait_for` | Wait for element | `selector`, `timeout_ms?` |
| `browser_scroll` | Scroll page | `direction`, `amount?` |
| `browser_get_element` | Get element info | `selector` |
| `browser_query_all` | Query all matching elements | `selector` |
| `browser_get_elements` | Get info for all matching elements | `selector` |
| `browser_click_by_id` | Click by snapshot ID | `element_id` |
| `browser_fill_by_id` | Fill by snapshot ID | `element_id`, `value` |
| `browser_type_by_id` | Type by snapshot ID | `element_id`, `text` |
| `browser_get_markdown` | Extract page as Markdown | - |

### Computer Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `computer_screenshot` | Capture screen | `monitor?` |
| `computer_mouse_move` | Move cursor (no click) | `x`, `y` |
| `computer_click` | Click at position | `x`, `y`, `button?`, `click_type?` |
| `computer_type_text` | Type text | `text`, `delay_ms?` |
| `computer_key` | Press keyboard keys | `key`, `modifiers?`, `repeat?` |
| `computer_scroll` | Scroll at position | `x`, `y`, `direction`, `amount?` |
| `computer_drag` | Drag between positions | `start_x`, `start_y`, `end_x`, `end_y`, `button?` |
| `computer_cursor_position` | Get cursor position | - |
| `computer_mouse_down` | Press and hold button | `x`, `y`, `button?` |
| `computer_mouse_up` | Release button | `x`, `y`, `button?` |
| `computer_hold_key` | Hold key for duration | `key`, `duration_ms`, `modifiers?` |
| `computer_wait` | Wait for duration | `ms` |

### Agent Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `agent_chat` | Send message to AI | `message`, `context?` |

### Built-in Agent Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write content to file |
| `list_files` | List directory contents |

## Configuration

Configuration file: `~/.config/nevoflux/config.toml`

### Example Configuration

```toml
[daemon]
port_range_start = 19500
port_range_end = 19600
idle_timeout_secs = 1800
heartbeat_timeout_secs = 30
heartbeat_interval_secs = 10
max_concurrent_requests = 100
keep_alive_for_mcp = true

[daemon.session]
max_sessions = 500
inactive_days = 90
max_storage_mb = 500
auto_create = true

[daemon.context]
system_prompt_reserve = 2000
safety_margin = 500
max_history_messages = 50
include_memory = true
include_current_page = true

[llm]
default_provider = "anthropic"
default_model = "claude-sonnet-4-20250514"
max_tokens = 32768
temperature = 0.7
timeout_secs = 120
max_retries = 3

[storage]
max_size_mb = 1024
wal_mode = true
vacuum_on_startup = false

[logging]
level = "info"
stdout = true
json_format = false
```

### Environment Variables

```bash
# LLM Provider API Keys
ANTHROPIC_API_KEY     # Anthropic Claude
OPENAI_API_KEY        # OpenAI GPT
DASHSCOPE_API_KEY     # Qwen (Alibaba)
DEEPSEEK_API_KEY      # DeepSeek
OPENROUTER_API_KEY    # OpenRouter

# NevoFlux Settings
NEVOFLUX_DATA_DIR     # Override data directory
RUST_LOG              # Log level / filter (e.g. info, debug, nevoflux_daemon=trace)
NEVOFLUX_SESSION_MODE # Headless: reuse ONE browser across a task-flow (see Headless Deployment)
ORT_DYLIB_PATH        # Path to libonnxruntime (embedding feature, load-dynamic builds)
```

## Development

### Run Tests

```bash
just test                        # All tests
cargo test -p nevoflux-daemon    # Single crate
cargo test test_name -- --nocapture  # Specific test
```

### Test Infrastructure

The `nevoflux-testing` crate provides:

- `MockLlmProvider` - Fake LLM responses for testing
- `MockMcpClient` - Fake MCP server for testing
- `MockPermissionChecker` - Configurable permission responses
- `TestDaemonBuilder` - Build test daemon instances
- Property-based tests with `proptest` for protocol layer

### Linting & Formatting

```bash
just lint    # Run clippy
just fmt     # Format code
just ci      # Full CI check
```

### Generate Documentation

```bash
just doc     # Generate and open docs
```

### Build for Release

```bash
just release
```

## Platform Notes

### Linux

- Requires X11 display server for computer control
- Wayland support is not yet implemented
- Install `xdotool` for enhanced input simulation (optional)

### macOS

- Requires Accessibility permissions for mouse/keyboard control
- Grant via System Preferences > Security & Privacy > Privacy > Accessibility

### Windows

- Administrator privileges required for native messaging host registration
- Uses Windows API for screen capture and input simulation

## Troubleshooting

### Daemon Issues

```bash
nevoflux-agent --status    # Check current status
nevoflux-agent --stop      # Clean up stale files
```

### WASM Errors

Ensure wasmtime compatibility (version 27 required).

### Linux Display Issues

Computer control requires X11. Check with:

```bash
echo $XDG_SESSION_TYPE  # Should be "x11"
```

## License

AGPL-3.0
