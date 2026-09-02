#!/usr/bin/env bash
# Drive the headless agent over A2A **v0.3.0** — the legacy tier.
#
# Every difference from the v1 script is a v1.0 breaking change: the RPC names,
# the `kind` discriminators on messages and parts, the lowercase state enum.
set -euo pipefail

BASE="${1:-http://localhost:8084}"
AUTH=()
[ -n "${NEVOFLUX_A2A_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $NEVOFLUX_A2A_TOKEN")

echo "== agent card =="
# The card is served in the 1.0 shape; the 0.3.0 entry is one of its interfaces.
curl -sS "${AUTH[@]}" "$BASE/.well-known/agent-card.json" \
  | jq '.supportedInterfaces[] | select(.protocolVersion == "0.3.0")'

echo "== message/send (blocking) =="
curl -sS "${AUTH[@]}" -X POST "$BASE/a2a" \
  -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{
        "kind":"message","messageId":"m1","role":"user","contextId":"demo",
        "parts":[{"kind":"text","text":"open example.com and report the title"}]}}}' \
  | tee /dev/stderr \
  | jq -e '.result.status.state == "completed"' >/dev/null \
  || { echo "task did not complete" >&2; exit 1; }
echo "completed"

echo "== message/stream (SSE) =="
curl -sSN "${AUTH[@]}" -X POST "$BASE/a2a" \
  -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"message/stream","params":{"message":{
        "kind":"message","messageId":"m2","role":"user","contextId":"demo",
        "parts":[{"kind":"text","text":"now report the first paragraph"}]}}}'
