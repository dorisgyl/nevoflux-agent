#!/bin/sh
# Vendor rtc-mdns with the one unstable call replaced, and enable the patch.
#
# rtc-mdns 0.20.2 calls `Ipv4Addr::from_octets`, which is still unstable behind
# the `ip_from` feature (rust#131360), so the crate does not build on stable.
# `Ipv4Addr::from` is the stable equivalent for a `[u8; 4]`.
#
# POSIX sh: run under bash/zsh on Linux and macOS, and under Git Bash on
# Windows. Idempotent — re-running it is a no-op.
set -eu

cd "$(dirname "$0")"

if [ -d vendor/rtc-mdns ]; then
    echo "vendor/rtc-mdns already present; nothing to do"
    exit 0
fi

# Make sure the source is unpacked in the registry before copying it.
cargo fetch >/dev/null 2>&1 || true
SRC=$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" -maxdepth 2 -type d -name 'rtc-mdns-0.20.2' | head -1)

if [ -z "$SRC" ]; then
    echo "error: rtc-mdns-0.20.2 not unpacked in the cargo registry." >&2
    echo "Run 'cargo fetch' in this directory first, then re-run setup.sh." >&2
    exit 1
fi

mkdir -p vendor
cp -r "$SRC" vendor/rtc-mdns
chmod -R u+w vendor/rtc-mdns
# A vendored copy keeps the registry checksum file, which cargo rejects on a
# path dependency.
rm -f vendor/rtc-mdns/.cargo-checksum.json

sed -i.bak 's/Ipv4Addr::from_octets(a\.a)/Ipv4Addr::from(a.a)/' \
    vendor/rtc-mdns/src/proto/mod.rs
rm -f vendor/rtc-mdns/src/proto/mod.rs.bak

# Nothing tracked is edited: Cargo.toml already points [patch.crates-io] here,
# so until this script runs the build simply fails on a missing path — which is
# a clearer thing to hit than a silently different dependency.
echo "patched. now: cargo run --release"
