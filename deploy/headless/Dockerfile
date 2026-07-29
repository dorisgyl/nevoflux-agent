# NevoFlux headless automation image (P7).
# Runs the automation daemon, which spawns a headless browser per task.
# Isolation is the CONTAINER (per v2 §0 — the product does not build its own
# sandbox); for untrusted workloads run ONE disposable container per task.
#
# This image DOWNLOADS the official prebuilt releases from GitHub — no local
# compilation. To build from locally-staged artifacts instead, use Dockerfile.local.
FROM debian:bookworm-slim

# --- system deps: Xvfb / x11vnc / dbus / Gecko libs + xz for .tar.xz ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb x11vnc novnc websockify dbus-x11 xauth x11-utils tini ca-certificates git curl unzip xz-utils \
      libgtk-3-0 libdbus-glib-1-2 libxt6 libasound2 libx11-xcb1 libxtst6 libpci3 \
      fonts-liberation fonts-noto-color-emoji fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# --- non-root user + dirs ---
# A tmpfs mounted over a pre-existing dir inherits that dir's MODE but resets the owner
# to root. Make the ephemeral data dir 1777 so the non-root daemon (uid 10001) can still
# write daemon.lock/.port/db when /var/nevoflux/data is a tmpfs.
RUN useradd -m -u 10001 nevo \
 && mkdir -p /var/nevoflux/data /work /base-profiles /opt/packs \
 && chown -R nevo:nevo /var/nevoflux /work /base-profiles /opt/packs \
 && chmod 1777 /var/nevoflux/data

# --- Browser (Gecko build) from the browser repo's release tar.xz -> /opt/nevoflux/ ---
# Default: latest release. Pin with --build-arg BROWSER_VERSION=0.3.9 (NO leading "v").
ARG BROWSER_VERSION=latest
ARG TARGETARCH
RUN set -eu; \
    arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$arch" in amd64) A=x86_64;; arm64) A=aarch64;; *) echo "unsupported arch: $arch" >&2; exit 1;; esac; \
    if [ "$BROWSER_VERSION" = latest ]; then \
      u="https://github.com/nevoflux-browser/nevoflux/releases/latest/download/nevoflux.linux-${A}.tar.xz"; \
    else \
      u="https://github.com/nevoflux-browser/nevoflux/releases/download/${BROWSER_VERSION}/nevoflux.linux-${A}.tar.xz"; \
    fi; \
    echo "downloading browser: $u"; \
    curl -fSL "$u" | tar -xJ -C /opt; \
    chown -R nevo:nevo /opt/nevoflux

# --- Agent (daemon + bundled ONNX Runtime + e5 model) from the agent repo's tar.gz ---
# The tarball's top level is exactly the daemon exe_dir layout:
#   nevoflux-agent, lib/libonnxruntime.so, models/fastembed/<e5>
# so it extracts straight into /usr/local/bin. Checksum-verified.
# Default: latest release. Pin with --build-arg AGENT_VERSION=v0.3.9 (WITH leading "v").
ARG AGENT_VERSION=latest
RUN set -eu; \
    arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$arch" in amd64) A=x86_64;; arm64) A=aarch64;; *) echo "unsupported arch: $arch" >&2; exit 1;; esac; \
    if [ "$AGENT_VERSION" = latest ]; then \
      b="https://github.com/nevoflux-browser/nevoflux-agent/releases/latest/download"; \
    else \
      b="https://github.com/nevoflux-browser/nevoflux-agent/releases/download/${AGENT_VERSION}"; \
    fi; \
    f="nevoflux-agent-linux-${A}.tar.gz"; \
    cd /tmp; \
    echo "downloading agent: $b/$f"; \
    curl -fSL -o "$f" "$b/$f"; \
    curl -fSL -o sums.txt "$b/checksums-sha256.txt"; \
    grep "  $f\$" sums.txt | sha256sum -c -; \
    tar -xzf "$f" -C /usr/local/bin; \
    rm -f "$f" sums.txt; \
    chown -R nevo:nevo /usr/local/bin/lib /usr/local/bin/models; \
    chmod 0755 /usr/local/bin/nevoflux-agent

# --- entrypoint + native-messaging host manifest ---
COPY --chmod=0755 entrypoint.sh                  /usr/local/bin/entrypoint.sh
# NOTE: verify the search path for this fork via about:support → Native Messaging.
COPY --chown=nevo:nevo native-host/com.nevoflux.agent.json \
     /home/nevo/.mozilla/native-messaging-hosts/com.nevoflux.agent.json

# --- env: pin data dir + display; models are bundled so run HuggingFace-offline ---
ENV NEVOFLUX_DATA_DIR=/var/nevoflux/data \
    DISPLAY=:99 \
    HOME=/home/nevo \
    NEVOFLUX_BROWSER_BIN=/opt/nevoflux/nevoflux \
    NEVOFLUX_BASE_PROFILES=/base-profiles \
    HF_HUB_OFFLINE=1 \
    MOZ_DISABLE_CONTENT_SANDBOX=1

USER nevo
WORKDIR /work

# --- (optional) bundle packs at build time -------------------------------------
ARG PACK_REPOS=""
RUN if [ -n "$PACK_REPOS" ]; then set -eu; \
      for repo in $PACK_REPOS; do \
        case "$repo" in *://*|*@*) url="$repo";; *) url="https://github.com/$repo.git";; esac; \
        name="$(basename "${repo%.git}")"; \
        echo "cloning pack repo: $url -> /opt/packs/$name"; \
        git clone --depth 1 "$url" "/opt/packs/$name"; \
      done; \
    fi

# --- GBrain (knowledge/memory brain): installed + enabled by default -----------
ARG INSTALL_GBRAIN=1
ARG GBRAIN_PIN="github:garrytan/gbrain#090bb53"
ARG LLM_PROVIDER="anthropic"
RUN if [ "$INSTALL_GBRAIN" = "1" ]; then set -eu; \
      curl -fsSL https://bun.sh/install | bash; \
      BUN="$HOME/.bun/bin/bun"; \
      mkdir -p "$HOME/.nevoflux/brain-tool" "$HOME/.gbrain" "$HOME/.config/nevoflux"; \
      chmod 1777 "$HOME/.gbrain"; \
      ( cd "$HOME/.nevoflux/brain-tool" && "$BUN" add "$GBRAIN_PIN" ); \
      printf '[llm]\nprovider = "%s"\n\n[knowledge_base]\nenabled = true\n\n[knowledge_base.brain]\nenabled = true\n' \
        "$LLM_PROVIDER" > "$HOME/.config/nevoflux/config.toml"; \
      echo "gbrain installed ($GBRAIN_PIN); brain enabled; [llm].provider=$LLM_PROVIDER (key from *_API_KEY env at runtime)"; \
    fi

# /work drains task results. /var/nevoflux/data holds the daemon database and —
# for --remote-control — the channel identity (pairing code included) plus that
# head's browser profile. Mount it on a real volume for that mode: on a tmpfs,
# as the task service below does, the pairing is destroyed every time the
# container stops, and a phone that was paired can only report that it no
# longer connects.
VOLUME ["/work", "/var/nevoflux/data"]

ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint.sh"]
# Long-running service (serves the task API); for untrusted use `run --task` one-shot.
CMD ["--daemon","--headless","--http-addr","0.0.0.0:8080"]
