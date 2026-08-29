#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$ROOT/.env" ]; then
  # shellcheck disable=SC1091
  set -a
  source "$ROOT/.env"
  set +a
fi

URL="${VOICEBOX_API_URL:-http://127.0.0.1:17493}"
MODE="${VOICE_ENGINE_MODE:-embedded}"

if curl -sf "$URL/health" >/dev/null; then
  echo "Voice engine OK ($URL, mode=$MODE)"
  exit 0
fi

echo "Voice engine not reachable at $URL (mode=$MODE)"
if [ "$MODE" = "embedded" ]; then
  echo "Run: ./scripts/setup.sh && ./scripts/dev.sh"
else
  echo "Start your external engine or switch VOICE_ENGINE_MODE=embedded"
fi
exit 1
