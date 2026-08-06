#!/usr/bin/env bash
# NevoFlux headless entrypoint (P7): dbus → Xvfb → (optional VNC) → daemon.
# On Linux the browser needs a display; Xvfb provides a virtual one. (On
# Windows the daemon uses native -headless and needs no Xvfb — see the E2E.)
set -euo pipefail

# 1) dbus session (Firefox needs it)
eval "$(dbus-launch --sh-syntax)"

# 2) Xvfb virtual display
Xvfb "${DISPLAY}" -screen 0 1280x1024x24 -nolisten tcp &
for _ in $(seq 1 100); do xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1 && break; sleep 0.1; done

# 3) optional live VNC (default off; NEVOFLUX_VNC=1 + a password file to enable).
#    Raw VNC on :5900 (native clients) AND noVNC on :6080 (any web browser at
#    http://<host>:6080/vnc.html). NEVOFLUX_VNC_VIEWONLY=1 to watch without input.
if [ "${NEVOFLUX_VNC:-0}" = "1" ]; then
  vnc_ro=""
  [ "${NEVOFLUX_VNC_VIEWONLY:-0}" = "1" ] && vnc_ro="-viewonly"
  x11vnc -display "${DISPLAY}" -rfbauth "${NEVOFLUX_VNC_PASSWD:-/etc/nevoflux/vncpasswd}" \
         -forever -shared -bg -rfbport 5900 ${vnc_ro}
  # noVNC: bridge the browser's WebSocket to the VNC TCP port (5900).
  websockify --web=/usr/share/novnc/ 6080 localhost:5900 &
fi

# Install packs cloned into the image at build time (Dockerfile ARG PACK_REPOS).
# Pack install goes over the daemon's RPC, so this can only run once a daemon is
# up — hence startup, not build. Idempotent each start (works with a tmpfs data
# dir; --force re-applies on a persistent one).
install_bundled_packs() {
  [ -d /opt/packs ] && [ -n "$(ls -A /opt/packs 2>/dev/null)" ] || return 0
  for _ in $(seq 1 60); do [ -f "${NEVOFLUX_DATA_DIR}/daemon.port" ] && break; sleep 1; done
  for p in /opt/packs/*/; do
    [ -d "$p" ] || continue
    src="$p"; [ -f "${p}pack.toml" ] && src="${p}pack.toml"
    echo "installing bundled pack: $src"
    nevoflux-agent pack install "$src" --yes --force || echo "WARN: pack install failed: $src"
  done
}

# Initialize the GBrain repo BEFORE the daemon starts. gbrain's dir (~/.gbrain)
# is an ephemeral tmpfs, so it is empty on every start and `gbrain serve` (spawned
# by the daemon at boot) would crash with "No brain configured" → the daemon's MCP
# init times out after 120s and brain is disabled. This mirrors the daemon install
# wizard's init (embedding via the in-process gateway, zero-padded to dim 512).
init_gbrain_brain() {
  local bun="$HOME/.bun/bin/bun"
  local cli="$HOME/.nevoflux/brain-tool/node_modules/gbrain/src/cli.ts"
  local brain_dir="${GBRAIN_BRAIN_DIR:-$HOME/.gbrain}"
  # NEVOFLUX_BRAIN_ENABLED=0 stops the daemon spawning `gbrain serve`, but this
  # init runs before the daemon exists and would still pay its startup cost —
  # up to 40s of waiting for a brain nothing will use. Only an explicit off
  # value skips it: the authoritative setting lives in config.toml, which this
  # script cannot read, so anything else must keep today's behaviour rather
  # than disable a brain the daemon is about to want. Mirrors the accepted
  # values in `init_brain::brain_enabled`.
  case "$(printf '%s' "${NEVOFLUX_BRAIN_ENABLED:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')" in
    0|false|no)
      echo "NEVOFLUX_BRAIN_ENABLED=${NEVOFLUX_BRAIN_ENABLED}; skipping gbrain brain init"
      return 0
      ;;
  esac
  [ -x "$bun" ] && [ -f "$cli" ] || return 0            # gbrain not installed → skip
  [ -f "$brain_dir/config.json" ] && return 0           # already initialized → skip
  mkdir -p "$brain_dir"
  echo "initializing gbrain brain at $brain_dir (one-time per start; tmpfs) ..."
  # init only PROBES the gateway (liveness); placeholder OPENAI_* is fine here — the
  # real gateway is up once the daemon spawns `gbrain serve`. init prints its --json
  # success line then lingers (bun does not exit), so wait for config.json then stop it.
  ( cd "$brain_dir" && \
    OPENAI_BASE_URL="http://127.0.0.1:1/v1" OPENAI_API_KEY="x" \
    OPENROUTER_BASE_URL="http://127.0.0.1:1/v1" OPENROUTER_API_KEY="x" \
    GBRAIN_BRAIN_DIR="$brain_dir" \
    "$bun" run "$cli" init --pglite --json --embedding-dimensions 512 \
      --embedding-model openai:text-embedding-3-small </dev/null >/tmp/gbrain-init.log 2>&1 ) &
  local ipid=$!
  for _ in $(seq 1 80); do [ -f "$brain_dir/config.json" ] && break; sleep 0.5; done
  kill "$ipid" 2>/dev/null || true
  if [ -f "$brain_dir/config.json" ]; then
    echo "gbrain brain initialized."
  else
    echo "WARN: gbrain init did not complete in time; brain will be disabled (see /tmp/gbrain-init.log)"
  fi
}

case " $* " in
  *" --daemon "*)
    # Daemon mode: start in background so we can install bundled packs once it's
    # up, then hand over. Forward SIGTERM/SIGINT for a clean shutdown.
    init_gbrain_brain    # ready the brain before the daemon spawns `gbrain serve`
    nevoflux-agent "$@" &
    DAEMON_PID=$!
    trap 'kill -TERM "$DAEMON_PID" 2>/dev/null || true' TERM INT
    install_bundled_packs
    wait "$DAEMON_PID"
    ;;
  *)
    # Non-daemon (e.g. one-shot `run --task`, `pack`, `config`): run directly.
    exec nevoflux-agent "$@"
    ;;
esac
