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

# Fetch SenseVoice weights for local transcription
fetch-asr-models:
    #!/usr/bin/env bash
    set -euo pipefail
    # Weights come from the sherpa-onnx exports, not FunAudioLLM's own repo,
    # which ships only model.pt -- there is no official ONNX. These are
    # maintained by sherpa-onnx's author and, importantly, carry the whole
    # preprocessing contract inside the ONNX metadata (LFR window, CMVN
    # vectors, language ids). That is why no am.mvn is fetched: reading those
    # numbers from a side file would be a second source of truth for values
    # the model already states. `just dump-asr-model` prints them.
    #
    # 2024-07-17 is the original iic/SenseVoiceSmall export. Do not "upgrade"
    # to the 2025-09-09 one without checking its metadata first: despite the
    # newer date and identical tensor contract, it is the ASLP-lab WSYue
    # Cantonese fine-tune, which is not what a Mandarin-first default wants.
    # The `comment` and `url` metadata keys are what tell the two apart.
    #
    # curl rather than huggingface_hub: this needs no Python environment to be
    # correct, and `pip install` into whichever interpreter happens to be first
    # on PATH is a step that fails quietly on machines with a broken site-
    # packages. Honour *_PROXY as curl already does.
    #
    # Local filenames are this project's convention. Daemon-side resolution
    # (crates/daemon/src/tts/asr.rs, following kokoro.rs) depends on them
    # staying put -- rename here, rename there.
    #
    # No VAD yet: it is only needed past the 30 s single-pass limit, and that
    # path lands with the segmentation work.
    #
    # The daemon fetches nothing itself. Network behaviour in a native-
    # messaging host expected to start in under a second is a bad trade, and
    # pulling hundreds of megabytes at a moment nobody chose is a worse one.
    DEST="${HOME}/.cache/nevoflux/models"
    REPO="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    mkdir -p "$DEST"
    fetch() {
        local remote="$1" local_name="$2"
        if [ -s "$DEST/$local_name" ]; then
            echo "  $local_name (already present)"
            return
        fi
        echo "  $local_name ..."
        curl -fL --retry 3 --progress-bar \
            "https://huggingface.co/$REPO/resolve/main/$remote" \
            -o "$DEST/$local_name.part"
        mv "$DEST/$local_name.part" "$DEST/$local_name"
    }
    fetch "model.int8.onnx" "sensevoice-small.int8.onnx"
    fetch "tokens.txt"      "sensevoice-tokens.txt"
    echo "ASR models are in $DEST"
    echo "Whisper is optional, only for --features asr-whisper: just fetch-whisper-model"

# Print an ASR model's ONNX I/O signature and metadata
dump-asr-model MODEL="~/.cache/nevoflux/models/sensevoice-small.int8.onnx" PROFILE="debug":
    #!/usr/bin/env bash
    set -euo pipefail
    # The SenseVoice exports state their own preprocessing contract in ONNX
    # metadata. Read it from the file rather than from a blog post, and read it
    # again whenever the model version moves -- two exports can share a tensor
    # contract and still be different models (see the note in fetch-asr-models).
    ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$PWD/target/{{PROFILE}}/lib/libonnxruntime.so}" \
    cargo run -q -p nevoflux-asr --example dump_model \
        --features sensevoice,ort-load-dynamic -- "$(eval echo {{MODEL}})"

# Fetch Whisper weights (only needed for --features asr-whisper)
fetch-whisper-model SIZE="large-v3-turbo":
    #!/usr/bin/env bash
    set -euo pipefail
    # Separate from fetch-asr-models on purpose: the default build has no
    # Whisper engine, so most people never need this and should not be made to
    # download a gigabyte to find that out.
    DEST="${HOME}/.cache/nevoflux/models"
    mkdir -p "$DEST"
    TARGET="$DEST/whisper-{{SIZE}}.ggml.bin"
    if [ -s "$TARGET" ]; then echo "  already present: $TARGET"; exit 0; fi
    curl -fL --retry 3 --progress-bar \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{{SIZE}}.bin" \
        -o "$TARGET.part"
    mv "$TARGET.part" "$TARGET"
    echo "  $TARGET"

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
