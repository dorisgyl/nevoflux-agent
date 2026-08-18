# NevoFlux Headless — Deploy (P7)

Runs the automation daemon; it spawns a headless browser per task and serves a
task HTTP API. Verified end-to-end (a real `POST /tasks` "navigate example.com,
report title" returned **Example Domain** via a real agent + headless browser).

## Isolation is the container — and granularity matters

The product does **not** build its own sandbox; it assumes it runs **inside** an
isolated environment. **Security depends on the *granularity*, not just presence:**

- **Untrusted workloads → one disposable container per task.** `bash`/`run_command`
  run in the daemon process, so a long-lived container serving many untrusted tasks
  re-opens cross-task poisoning. Use the one-shot `run --task` form, one container
  per task, torn down after.
- **Long-lived service mode** (`--daemon --headless --http-addr`) → **trusted tasks
  only**, or accept the in-container cross-task risk.

The container's **network egress policy** (allowlist to LLM API + task domains) and
**no host credential mounts** (only `/work` + read-only base-profiles) are the hard
boundary. The product adds in-process defense-in-depth (tool allowlist, fs-sandbox,
`SENSITIVE_PATHS`, domain checks) as layers, not the boundary.

## Build

The image DOWNLOADS the official prebuilt releases from GitHub — **no local
compilation**. Just:

```bash
docker build -t nevoflux/agent:latest deploy/headless      # amd64 or arm64
```

- **Versions:** default to the latest release; pin for reproducible builds:
  `--build-arg AGENT_VERSION=v0.3.9 --build-arg BROWSER_VERSION=0.3.9`
  (agent tags carry a leading `v`, browser tags do not). `latest` builds are not
  reproducible and may pair a newer agent with a newer browser.
- **Multi-arch:** `TARGETARCH` selects the `x86_64`/`aarch64` assets; on arm64
  hosts (or `docker buildx --platform linux/arm64`) it downloads the arm build.

### Building from local (unreleased) binaries — `Dockerfile.local`

Developers testing locally-built or unreleased daemon/browser changes (a feature
not yet in a release) stage the four artifacts and build the COPY-based variant:

```bash
# stage into deploy/headless/: dist/nevoflux/ (./mach package tree),
# nevoflux-agent (built daemon), libonnxruntime.so (v1.24.x), models/fastembed/
docker build -f deploy/headless/Dockerfile.local -t nevoflux/agent:local deploy/headless
```
(Or in compose: `build.dockerfile: Dockerfile.local`.)

## Bundled packs (build-time clone → startup install)

Pack install goes over the daemon's RPC — there is no offline installer — so we
**clone** the pack repos into the image at build time (no daemon needed), and the
`entrypoint.sh` **installs** them once the daemon is up at startup. This avoids
running a daemon during `docker build` and works even with an ephemeral (tmpfs)
data dir — packs are re-applied each start from the baked-in clones.

```bash
docker build -t nevoflux/agent:latest \
  --build-arg PACK_REPOS="owner/pack-repo https://github.com/x/y.git" \
  deploy/headless
# → cloned to /opt/packs/<name>; entrypoint runs `pack install /opt/packs/<name>[/pack.toml] --yes --force`
#   after the daemon is listening. No packs are installed if PACK_REPOS is empty.
```

`PACK_REPOS` is space-separated: a github `owner/repo` shorthand, or a full git
URL. Only applies to daemon-mode startup (one-shot `run --task` runs directly).

## GBrain (knowledge brain) — installed + enabled by default

The image installs **GBrain** at build time and enables it:

- Installs `bun` (via `bun.sh/install`) and `bun add github:garrytan/gbrain#<pin>`
  into `~/.nevoflux/brain-tool`; creates the brain dir `~/.gbrain`. No daemon is
  needed at build — the daemon spawns gbrain at startup and auto-resolves these paths.
- Writes `~/.config/nevoflux/config.toml` with `[knowledge_base.brain] enabled = true`
  and `[llm] provider = <LLM_PROVIDER>` (default `anthropic`). GBrain's LLM calls
  route through the in-process gateway to that provider, so **no separate gbrain key**
  is needed — just the provider's `*_API_KEY` (e.g. `ANTHROPIC_API_KEY`) at runtime.

```bash
# defaults: gbrain on, provider anthropic (needs ANTHROPIC_API_KEY at runtime)
docker build -t nevoflux/agent:latest deploy/headless

# other provider / pin / disable:
docker build --build-arg LLM_PROVIDER=openai \                # then set OPENAI_API_KEY at run
             --build-arg GBRAIN_PIN=github:garrytan/gbrain#<sha> \
             --build-arg INSTALL_GBRAIN=0 \                    # skip gbrain entirely
             -t nevoflux/agent:latest deploy/headless
```

> If you **mount your own** `config.toml`, keep `[knowledge_base.brain] enabled = true`
> (and your `[llm]`) or gbrain won't start. Without a working `[llm]` provider, browser
> tasks + brain storage still work, but gbrain's LLM synthesis (`brain_think`) won't.

## Run — one task per container (untrusted; recommended)

```bash
docker run --rm \
  --read-only --tmpfs /tmp --tmpfs /var/nevoflux/data \
  --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 512 --memory 4g --cpus 2 \
  -v "$PWD/out:/work" \
  -v nevoflux-base-profiles:/base-profiles:ro \
  -e ANTHROPIC_API_KEY="$SHORT_LIVED_KEY" \
  -e HTTP_PROXY="$EGRESS_PROXY" -e HTTPS_PROXY="$EGRESS_PROXY" \
  nevoflux/agent:latest \
  run --task "open example.com and report the title" --profile base1 \
      --policy browser-only --wall-clock 300s --token-budget 200k
# result + debug bundle drained to ./out (result.json, debug-bundle/)
```

## Run — service mode (trusted) + observe

```bash
docker run --rm -p 8080:8080 -p 6080:6080 -p 5900:5900 \
  -e NEVOFLUX_VNC=1 -e NEVOFLUX_VNC_PASSWD=/etc/nevoflux/vncpasswd \
  nevoflux/agent:latest
# submit:  curl -X POST localhost:8080/tasks -d '{"task":"...","mode":"browser"}'
# watch:   curl -N localhost:8080/tasks/<id>/events        (live task events)
#          http://localhost:6080/vnc.html                  (live browser in ANY web browser, via noVNC)
#          VNC localhost:5900                               (or a native VNC client)
# metrics: curl localhost:8080/metrics
```

**Live browser in a web browser (noVNC):** with `NEVOFLUX_VNC=1`, the image runs
`x11vnc` (raw VNC on `:5900`) *and* `websockify` serving noVNC on `:6080`, so you
can watch at `http://<host>:6080/vnc.html` with no client install — it just needs
the VNC password. Add `NEVOFLUX_VNC_VIEWONLY=1` to watch without being able to
click (avoids interfering with the automation). Keep both ports **off in prod** or
behind an authenticated HTTPS proxy / SSH tunnel — a raw VNC/noVNC port is a
remote-control surface.

## Task API reference

Served on `--http-addr`. All endpoints are JSON. Verified live (OpenAI + MCP +
SSE each drove a real browser task to "Example Domain").

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/tasks` | submit a task → `{"id":"task-N"}` |
| `GET` | `/tasks/:id` | status / result snapshot |
| `GET` | `/tasks/:id/events` | **SSE** — `status` frames until terminal |
| `DELETE` | `/tasks/:id` | cancel (marks `failed`) → `true`/`false` |
| `GET` | `/metrics` | Prometheus (`nevoflux_tasks_total` / `_failed`) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible (also on `--openai-addr`) |

**`POST /tasks` body:**
```jsonc
{ "task": "open example.com and report the title",  // required
  "mode": "browser",         // default "browser"
  "profile": "base1",         // optional — clones this base-profile (login state)
  "policy": { "allow_shell": false, "allow_fs_write": false,
              "allow_upload": false, "domain_allowlist": [] },   // default: all false / empty
  "wall_clock_secs": 300, "token_budget": 200000,                // optional caps
  "idempotent": false, "no_retry": false }                       // retry controls
```
**status / result** (`GET /tasks/:id`):
```jsonc
{ "id": "task-0", "status": "succeeded",   // queued | running | succeeded | failed
  "attempts": 1, "output": "The title is Example Domain.", "error": null, "artifacts": [] }
```
The result + a debug bundle are also drained to the task workspace under `/work`.

### SSE `GET /tasks/:id/events`
```bash
curl -N localhost:8080/tasks/task-0/events
# event: status
# data: {"id":"task-0","status":"running","attempts":0,...}
# event: status
# data: {"id":"task-0","status":"succeeded","output":"...Example Domain","attempts":1,...}
```
Emits a `status` frame on each change (plus the terminal one); keep-alive comments
in between. The stream ends when the task reaches `succeeded`/`failed`.

## Session mode (task-flow: reuse one browser)

By default every task is fully isolated: clone a fresh profile → launch a browser
→ run → kill the browser + delete the clone. Set `NEVOFLUX_SESSION_MODE=1` to run
a **task-flow** instead: the first task launches ONE browser + profile clone, and
subsequent tasks **reuse** it (the active tab is soft-reset to `about:blank`
between tasks) until you end the flow.

```bash
# enable: NEVOFLUX_SESSION_MODE=1 in the environment
curl -X POST localhost:8080/tasks -d '{"task":"log in to example.com"}'          # launches the browser
curl -X POST localhost:8080/tasks -d '{"task":"go to the dashboard"}'            # reuses it (shared login)
curl -X POST localhost:8080/tasks -d '{"task":"export the report","end_session":true}'  # runs, then tears down
# or end out-of-band at any time:
curl -X POST localhost:8080/session/close        # → {"closed":true|false}
```

- **Drive tasks sequentially** (submit → wait for it to finish → submit the next).
  The shared browser is serialized by a lock, but task ORDER is the caller's job.
- **Shared state within a flow:** tasks share cookies / localStorage / login. A
  prompt-injected task can affect later tasks in the SAME flow — use session mode
  only for **trusted, sequential** flows. Cross-flow isolation is preserved (the
  next flow clones a fresh base profile).
- If the shared browser crashes mid-flow, the next task relaunches it against the
  same clone (cookies persist on disk, so login survives).

## Remote control from a phone

One container, one browser, one long conversation, driven from your phone
through [nevoflux.app](https://nevoflux.app). Unlike every mode above there is
no interface to call: `--remote-control` serves no HTTP at all, so there is
nothing to publish and no port to guard. The relay socket is the whole
outward face.

```bash
# 1. Put a nevoflux account token where the container can read it. Sign in on a
#    desktop NevoFlux, then copy `account-token` out of its data dir.
cp ~/.local/share/nevoflux/account-token ./account-token

# 2. Start it.
docker compose --profile remote up remote

# 3. Read the connect block it printed and paste the WHOLE thing into the
#    portal on your phone — it finds the link and the code for you.
docker compose --profile remote logs remote | head -40
```

**The pairing code is the secret, and the account is the second lock.** Two
things have to be true to reach this head: the code, and the account it signed
in as. The relay binds a channel to the first account that opens it and answers
any other with 403, so a leaked code alone reaches nothing — and someone signed
in as you still cannot read a word without it, since the channel key is derived
from the code and the relay never holds one (a forged frame fails AES-GCM and
is dropped).

The channel id is unguessable but not confidential, and it no longer has to be:
knowing it used to be enough to take up one of the two places on a channel,
which put a working head off the air.

**`remote-data` must be a real volume, never a tmpfs.** It holds
`remote-control.json` — the channel and its pairing code — plus the browser
profile and the conversation. On a tmpfs the identity dies with the container
and the paired phone can only report that it stopped connecting. This is why
the `remote` service in `docker-compose.yml` does *not* list
`/var/nevoflux/data` under `tmpfs`, while the task services do.

**Mode and execution tier are fixed at startup**, resolved from
`NEVOFLUX_REMOTE_MODE` / `NEVOFLUX_REMOTE_EXECUTION_TIER`, else
`[remote_control]` in `config.toml`, else `agent` / `read-only`. Each resolves
on its own, so setting one does not reset the other. The phone shows both and
can change neither: a container has no local user standing by to take a grant
back, so a phone that could raise `chat` to `agent` would make the setting a
suggestion.

```toml
# ~/.config/nevoflux/config.toml
[remote_control]
mode = "agent"                    # chat | browser | agent
execution_tier = "browser-auto"   # read-only | browser-auto |
                                  # browser-auto-local-read | full-auto
```

**Startup is all-or-nothing.** A missing account token, an unreachable account
service, an unwritable data dir, or a browser that will not come up each exit
non-zero with the reason — a container that half-starts reads as healthy to an
orchestrator and is useless to the person holding the phone.

**A corrupt `remote-control.json` refuses to start** rather than generating a
new identity. Regenerating would strand the paired phone, which would show up
only as "it will not connect any more". Delete the file deliberately if you
mean to re-pair.

**If the browser dies it is relaunched** and rebound to the same conversation,
and the phone gets a notice saying so — open pages are gone, signed-in sites
are not. Three consecutive failures to bring it back exit the container so the
orchestrator recreates it.

## Alternative interfaces: OpenAI / MCP / ACP

Three thin front-ends map a prompt to a task (all reduce to the same runner).
Each is available on the main port and can also bind a **dedicated port**:

| Interface | Endpoint | Dedicated-port flag |
|---|---|---|
| OpenAI-compatible | `POST /v1/chat/completions` | `--openai-addr` (also on `--http-addr`) |
| MCP (JSON-RPC 2.0) | `POST /mcp` | `--mcp-addr` |
| ACP (JSON-RPC 2.0) | `POST /acp` | `--acp-addr` |

```bash
# each interface on its own port
nevoflux-agent --daemon --headless \
  --http-addr 0.0.0.0:8080 --openai-addr 0.0.0.0:8081 \
  --mcp-addr 0.0.0.0:8082 --acp-addr 0.0.0.0:8083
```

**OpenAI** — the last `user` message becomes the task:
```bash
curl -X POST localhost:8081/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"open example.com, report title"}]}'
# → {"object":"chat.completion","choices":[{"message":{"role":"assistant",
#      "content":"...Example Domain"},"finish_reason":"stop"}], ...}
```
**MCP** — one tool `run_browser_task` (plus `initialize`, `tools/list`):
```bash
curl -X POST localhost:8082/mcp -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"run_browser_task","arguments":{"task":"open example.com, report title"}}}'
# → {"result":{"content":[{"type":"text","text":"...Example Domain"}],"isError":false}}
```
**ACP** — a prompt turn (plus `initialize`, `session/new`):
```bash
curl -X POST localhost:8083/acp -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"session/prompt",
       "params":{"prompt":[{"type":"text","text":"open example.com, report title"}]}}'
# → {"result":{"stopReason":"end_turn","content":[{"type":"text","text":"...Example Domain"}]}}
```

### Env-var overrides for the thin interfaces

OpenAI/MCP/ACP requests carry only a prompt — so `mode` / `profile` / `policy` /
caps come from the environment (`TaskRequest::from_env`). Set these on the container:

| env var | field | default |
|---|---|---|
| `NEVOFLUX_TASK_MODE` | mode | `browser` |
| `NEVOFLUX_TASK_PROFILE` | profile | none |
| `NEVOFLUX_POLICY_ALLOW_SHELL` | policy.allow_shell | `false` |
| `NEVOFLUX_POLICY_ALLOW_FS_WRITE` | policy.allow_fs_write | `false` |
| `NEVOFLUX_POLICY_ALLOW_UPLOAD` | policy.allow_upload | `false` |
| `NEVOFLUX_POLICY_DOMAIN_ALLOWLIST` | policy.domain_allowlist | empty (comma-separated) |
| `NEVOFLUX_WALL_CLOCK_SECS` | wall_clock_secs | none |
| `NEVOFLUX_TOKEN_BUDGET` | token_budget | none |
| `NEVOFLUX_IDEMPOTENT` | idempotent | `false` |
| `NEVOFLUX_NO_RETRY` | no_retry | `false` |

Booleans accept `1` / `true` / `yes`. `POST /tasks` still takes all of these per
request in its JSON body — the env vars are only the defaults for the interfaces
that can't carry them.

> **Scope note:** these front-ends are intentionally minimal — single-tool MCP;
> request/response ACP without streaming `session/update` notifications;
> non-streaming OpenAI. Enough to drive a headless task from an OpenAI / MCP / ACP
> client, not full protocol implementations.

## Fixed-script mode (no LLM)

For a **deterministic** browser-use pipeline that needs **no LLM provider**, point
the daemon at a Python script:

```bash
NEVOFLUX_HEADLESS_SCRIPT=/opt/nevoflux/fixed-flow.py
```

When set, **every headless task runs that script instead of the LLM agent loop**.
The script defines `def run(task): ...`; the daemon calls it with the interface's
task string (POST /tasks `task`, OpenAI last user message, MCP `task` arg, ACP
prompt). Whatever `run` **returns** (or prints) becomes the interface `output`; a
raised exception → `status:"failed"` + the error. It uses the same browser tools as
the agent (in the sandboxed Monty interpreter), driving the *bound* headless
browser — but with **zero LLM calls and no API key**. Templates:
[`examples/fixed-flow.py`](examples/fixed-flow.py) (basic navigate→fill→click→read)
and [`examples/fixed-flow-advanced.py`](examples/fixed-flow-advanced.py) (multi-step
pagination + `try/except` + always returns a structured `{ok, ...}` dict).

```python
def run(task):
    nav = browser_navigate(url="https://example.com/search")
    tab = nav["tab_id"]                              # navigate opens a NEW tab
    browser_fill(selector="#q", value=task, tab_id=tab)
    browser_click(selector="button[type=submit]", tab_id=tab)
    browser_wait_for(selector="#results", tab_id=tab, timeout_ms=15000)
    return browser_get_markdown(tab_id=tab)["markdown"]
```

Verified live (no LLM key set): `POST /tasks {"task":"read the page"}` ran the
script and returned the real page markdown (`# Example Domain …`).

**Two gotchas** (also in the example's comments):
1. `browser_navigate` opens a **new, inactive** tab and returns `{"tab_id": N}` —
   thread that `tab_id` into every later call, or tools hit "No active web tab found".
2. Tool results are **structured** dicts, not strings:
   `browser_get_markdown(...)` → `{"markdown","title","url","success"}` — index the field.

For this mode the image needs **no `ANTHROPIC_API_KEY` and no GBrain** — pure
deterministic pipeline. Mount your script + set the env var:
```yaml
# docker-compose.yml (headless service)
environment:
  NEVOFLUX_HEADLESS_SCRIPT: /opt/nevoflux/fixed-flow.py
volumes:
  - ./fixed-flow.py:/opt/nevoflux/fixed-flow.py:ro
```

## Docker Compose (`docker-compose.yml`)

The compose file packages the two `docker run` invocations above as reusable,
hardened services so you don't retype the flags. It defines:

- **`headless`** — long-running **service mode** (`--daemon --headless --http-addr 0.0.0.0:8080`).
  Publishes the task API on `:8080`. **Trusted tasks only** (one shared daemon over time).
- **`oneshot`** — **one task per container** (untrusted; recommended). Gated behind the
  `oneshot` compose profile so `up` doesn't start it; you invoke it per task with
  `docker compose run`, which passes your `run --task …` args to the entrypoint.

Both apply the container hardening (`read_only`, `cap_drop: ALL`,
`no-new-privileges`, `pids_limit`, cpu/mem limits, tmpfs data dir) and the same
volumes (`./out:/work` for drained results, `base-profiles:/base-profiles:ro`).

### Use it

```bash
# 0. one-time: put an API key + (optionally) an egress proxy in the environment
export ANTHROPIC_API_KEY=sk-...            # or leave a front proxy to inject it
export EGRESS_PROXY=http://egress:3128

# 1. build the image
docker compose build

# 2a. service mode (trusted): serve the task API
docker compose up
#   curl -X POST localhost:8080/tasks -d '{"task":"open example.com, report title","mode":"browser"}'
#   curl localhost:8080/metrics

# 2b. one task per container (untrusted): runs, drains ./out, exits
docker compose --profile oneshot run --rm oneshot \
  run --task "open example.com and report the title" \
      --profile base1 --policy browser-only --wall-clock 300s --token-budget 200k
```

### What to change for your setup

| Where | Change |
|---|---|
| `environment.ANTHROPIC_API_KEY` | your key — better: leave it empty and inject via `HTTP_PROXY` so a prompt-injected agent can't read it from env |
| `environment.HTTP_PROXY/HTTPS_PROXY` (`EGRESS_PROXY`) | point at an egress proxy that **allowlists** the LLM API + your task domains (this is the hard network boundary; compose can't express it itself) |
| `volumes: base-profiles` | populate this named volume once with pre-authenticated login profiles (a human logs in), or set `external: true` to reuse one; cloned per task |
| `volumes: ./out:/work` | host dir where `result.json` + `debug-bundle/` are drained |
| `ports` / VNC | uncomment `5900:5900` **and** set `NEVOFLUX_VNC=1` (+ a password file) only to watch a run live; keep off in prod |
| `deploy.resources.limits`, `pids_limit` | tune cpu/memory/pids per host |
| `security_opt: seccomp` | add a tuned seccomp profile once available |
| `networks` | attach `headless` to an egress-restricted network (external firewall / proxy sidecar) — the comment in the file marks where |

> **Note:** the `Dockerfile` `COPY`s expect `dist/nevoflux/` (Linux Gecko build) and
> the `nevoflux-agent` Linux binary staged into `deploy/headless/` before `build`.

## k8s (one Job per task)

`restartPolicy: Never`, `activeDeadlineSeconds` = wall-clock, `securityContext`
runAsNonRoot/readOnlyRootFilesystem/seccomp RuntimeDefault/drop ALL, `emptyDir`
for `/work` + data, a `NetworkPolicy` for egress, base-profiles as a RO volume.
Concurrency/warm capacity is the platform's job (the product runs one browser).

## Files
- `Dockerfile` — hardened image (non-root, tini, Xvfb, Gecko libs).
- `docker-compose.yml` — `headless` (service) + `oneshot` (per-task) + `remote`
  (phone-driven, no HTTP) services, pre-hardened.
- `entrypoint.sh` — dbus → Xvfb → (optional VNC) → daemon.
- `native-host/com.nevoflux.agent.json` — native-messaging manifest (verify path via `about:support`).
