#!/usr/bin/env bash
# Drive the headless agent over A2A **v1.0**.
#
# Discovery first: the card tells you which versions this deployment speaks and
# at which URLs. Then one blocking call, then one streaming call.
set -euo pipefail

BASE="${1:-http://localhost:8084}"
AUTH=()
[ -n "${NEVOFLUX_A2A_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $NEVOFLUX_A2A_TOKEN")

echo "== agent card =="
curl -sS "${AUTH[@]}" "$BASE/.well-known/agent-card.json" \
  | jq '.name, .supportedInterfaces, [.skills[].id]'

echo "== sendMessage (blocking) =="
curl -sS "${AUTH[@]}" -X POST "$BASE/a2a/v1" \
  -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"sendMessage","params":{"message":{
        "messageId":"m1","role":"ROLE_USER","contextId":"demo",
        "parts":[{"text":"open example.com and report the title"}]}}}' \
  | jq '.result.status.state, .result.status.message.parts[0].text'

echo "== sendStreamingMessage (SSE) =="
curl -sSN "${AUTH[@]}" -X POST "$BASE/a2a/v1" \
  -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"sendStreamingMessage","params":{"message":{
        "messageId":"m2","role":"ROLE_USER","contextId":"demo",
        "parts":[{"text":"now report the first paragraph"}]}}}'
