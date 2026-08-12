# NevoFlux Agent Development Commands
# See https://github.com/casey/just for installation

# Default recipe - show available commands
default:
    @just --list

# Build the project in debug mode
build:
    cargo build

# Build the project in release mode
release:
    cargo build --release

# Run all tests
test:
    cargo test --all

# Run unit tests only (lib tests in each crate)
test-unit:
    cargo test --all --lib

# Run integration tests only
test-integration:
    cargo test --all --test '*'

# Run tests with output shown
test-verbose:
    cargo test --all -- --nocapture

# Run a specific test by name
test-one NAME:
    cargo test {{NAME}} -- --nocapture

# Run tests for a specific crate
test-crate CRATE:
    cargo test -p {{CRATE}}

# Run clippy linter
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Run clippy and fix issues automatically
lint-fix:
    cargo clippy --all-targets --all-features --fix --allow-dirty

# Format code
fmt:
    cargo fmt --all

# Check if code is formatted
fmt-check:
    cargo fmt --all -- --check

# Run cargo check (fast compilation check)
check:
    cargo check --all

# Generate documentation
doc:
    cargo doc --all --no-deps

# Generate and open documentation
doc-open:
    cargo doc --all --no-deps --open

# Clean build artifacts
clean:
    cargo clean

# Run the daemon
daemon:
    cargo run -- --daemon

# Run in MCP server mode
mcp:
    cargo run -- --mcp

# Check daemon status
status:
    cargo run -- --status

# Stop the daemon
stop:
    cargo run -- --stop

# Run all quality checks (fmt, lint, test)
ci: fmt-check lint test

# Watch for changes and run tests
watch:
    cargo watch -x test

# Watch for changes and run a specific test
watch-test NAME:
    cargo watch -x "test {{NAME}} -- --nocapture"

# Count lines of code
loc:
    @echo "Lines of Rust code:"
    @find . -name "*.rs" -not -path "./target/*" | xargs wc -l | tail -1

# Show test count
test-count:
    @cargo test --all 2>&1 | grep -E "^test result:" | awk '{sum += $4} END {print "Total tests:", sum}'

# Run benchmarks (if any)
bench:
    cargo bench

# Update dependencies
update:
    cargo update

# Check for outdated dependencies
outdated:
    cargo outdated

# Audit dependencies for security vulnerabilities
audit:
    cargo audit

# Generate coverage report (requires cargo-tarpaulin)
coverage:
    cargo tarpaulin --out Html --output-dir target/coverage

# Run the application with arguments
run *ARGS:
    cargo run -- {{ARGS}}

# Download embedding model for bundled deployment
download-model:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Downloading multilingual-e5-small embedding model..."
    pip install -q huggingface_hub
    python3 -c "
    from huggingface_hub import snapshot_download
    import os
    cache_dir = 'models/fastembed'
    os.makedirs(cache_dir, exist_ok=True)
    snapshot_download('intfloat/multilingual-e5-small', cache_dir=cache_dir, allow_patterns=['onnx/model_O4.onnx', 'onnx/tokenizer.json', 'onnx/config.json', 'onnx/special_tokens_map.json', 'onnx/tokenizer_config.json', 'onnx/sentencepiece.bpe.model'])
    import glob, shutil
    for f in glob.glob(os.path.join(cache_dir, '**', 'model_O4.onnx'), recursive=True):
        target = os.path.join(os.path.dirname(f), 'model.onnx')
        os.rename(f, target)
        print(f'Renamed {f} -> {target}')
    # Resolve symlinks and remove blobs to avoid duplicate data
    model_dir = os.path.join(cache_dir, 'models--intfloat--multilingual-e5-small')
    for root, dirs, files in os.walk(os.path.join(model_dir, 'snapshots')):
        for fname in files:
            fpath = os.path.join(root, fname)
            if os.path.islink(fpath):
                real = os.path.realpath(fpath)
                os.remove(fpath)
                shutil.copy2(real, fpath)
    blobs = os.path.join(model_dir, 'blobs')
    if os.path.isdir(blobs):
        shutil.rmtree(blobs)
    print(f'Model downloaded to {cache_dir}')
    "

# Fetch SenseVoice + VAD models for local transcription
fetch-asr-models:
    #!/usr/bin/env bash
    set -euo pipefail
    # The local filenames below are this project's convention, not the upstream
    # ones -- the various exports disagree with each other, and daemon-side path
    # resolution (crates/daemon/src/tts/asr.rs, following the kokoro.rs
    # precedent) depends on them staying put. Rename here, rename there.
    #
    # The daemon never fetches anything itself. Adding network behaviour to a
    # native-messaging host expected to start in under a second is a bad trade,
    # and pulling gigabytes at a moment nobody chose is a worse one.
    DEST="${HOME}/.cache/nevoflux/models"
    mkdir -p "$DEST"
    pip install -q huggingface_hub
    python3 - "$DEST" <<'PY'
    import os, shutil, sys
    from huggingface_hub import hf_hub_download

    dest = sys.argv[1]
    # (repo_id, filename in repo, local name this project uses)
    wanted = [
        ("FunAudioLLM/SenseVoiceSmall", "model_quant.onnx", "sensevoice-small.int8.onnx"),
        ("FunAudioLLM/SenseVoiceSmall", "tokens.json",      "sensevoice-tokens.json"),
        ("funasr/fsmn-vad",             "model.onnx",       "fsmn-vad.onnx"),
    ]
    for repo, remote, local in wanted:
        src = hf_hub_download(repo_id=repo, filename=remote)
        target = os.path.join(dest, local)
        shutil.copyfile(src, target)
        print(f"  {repo}/{remote} -> {target}")
    PY
    echo "ASR models are in $DEST"
    echo "Whisper is optional and only needed for --features asr-whisper: just fetch-whisper-model"

# Fetch Whisper weights (only needed for --features asr-whisper)
fetch-whisper-model SIZE="large-v3-turbo":
    #!/usr/bin/env bash
    set -euo pipefail
    # Separate from fetch-asr-models on purpose: the default build has no
    # Whisper engine, so most people never need this and should not be made to
    # download a gigabyte to find that out.
    DEST="${HOME}/.cache/nevoflux/models"
    mkdir -p "$DEST"
    pip install -q huggingface_hub
    python3 - "$DEST" "{{SIZE}}" <<'PY'
    import os, shutil, sys
    from huggingface_hub import hf_hub_download

    dest, size = sys.argv[1], sys.argv[2]
    src = hf_hub_download(repo_id="ggerganov/whisper.cpp", filename=f"ggml-{size}.bin")
    target = os.path.join(dest, f"whisper-{size}.q8.gguf")
    shutil.copyfile(src, target)
    print(f"  {target}")
    PY

# ONNX Runtime version for load-dynamic builds. Keep in lockstep with
# EXPECTED_ORT_VERSION in crates/llm/src/embedding.rs (fastembed 5 -> ORT 1.24.x).
ort_version := "1.24.2"

# Copy a local ONNX Runtime shared library into target/<profile>/lib so that
# `--features ort-load-dynamic` builds load it without setting ORT_DYLIB_PATH
# (the build resolves <exe_dir>/lib/<libonnxruntime> at startup).
# Usage:
#   just ort-dylib ~/onnxruntime/lib/libonnxruntime.so.1.20.0
#   just ort-dylib ~/onnxruntime/lib/libonnxruntime.so.1.20.0 release
ort-dylib SRC PROFILE="debug":
    #!/usr/bin/env bash
    set -euo pipefail
    src="{{SRC}}"
    [ -f "$src" ] || { echo "error: not a file: $src" >&2; exit 1; }
    case "$(uname -s)" in
      Darwin)               name="libonnxruntime.dylib" ;;
      MINGW*|MSYS*|CYGWIN*) name="onnxruntime.dll" ;;
      *)                    name="libonnxruntime.so" ;;
    esac
    dst="target/{{PROFILE}}/lib"
    mkdir -p "$dst"
    cp -f "$src" "$dst/$name"
    echo "linked $src -> $dst/$name"

# Download the ONNX Runtime matching {{ort_version}} for this platform and copy
# it into target/<profile>/lib (see `ort-dylib`). On networks with a broken
# default route, pass a proxy, e.g. socks5h://127.0.0.1:1080.
# Usage:
#   just ort-fetch
#   just ort-fetch release
#   just ort-fetch debug socks5h://127.0.0.1:1080
ort-fetch PROFILE="debug" PROXY="":
    #!/usr/bin/env bash
    set -euo pipefail
    v="{{ort_version}}"
    case "$(uname -s)-$(uname -m)" in
      Linux-x86_64)  pkg="onnxruntime-linux-x64-$v";     lib="lib/libonnxruntime.so.$v" ;;
      Linux-aarch64) pkg="onnxruntime-linux-aarch64-$v"; lib="lib/libonnxruntime.so.$v" ;;
      Darwin-arm64)  pkg="onnxruntime-osx-arm64-$v";     lib="lib/libonnxruntime.$v.dylib" ;;
      Darwin-x86_64) pkg="onnxruntime-osx-x86_64-$v";    lib="lib/libonnxruntime.$v.dylib" ;;
      *) echo "error: unsupported platform; fetch manually and use 'just ort-dylib'" >&2; exit 1 ;;
    esac
    url="https://github.com/microsoft/onnxruntime/releases/download/v$v/$pkg.tgz"
    proxy=(); [ -n "{{PROXY}}" ] && proxy=(--proxy "{{PROXY}}")
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    echo "downloading $url"
    curl -fsSL "${proxy[@]}" "$url" -o "$tmp/ort.tgz"
    tar xzf "$tmp/ort.tgz" -C "$tmp"
    just ort-dylib "$tmp/$pkg/$lib" "{{PROFILE}}"

# Build and install locally
install:
    cargo install --path .

# Uninstall local installation
uninstall:
    cargo uninstall nevoflux-agent
