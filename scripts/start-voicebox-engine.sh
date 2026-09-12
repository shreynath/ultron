#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VB_ROOT="$ROOT/backend/vendor/voicebox"
VENV="$VB_ROOT/backend/venv"
HOST="${ULTRON_VOICEBOX_HOST:-127.0.0.1}"
PORT="${ULTRON_VOICEBOX_PORT:-17493}"

if [ -f "$ROOT/.env" ]; then
  # shellcheck disable=SC1091
  set -a
  source "$ROOT/.env"
  set +a
fi

if [ -z "${ULTRON_VOICEBOX_DATA_DIR:-}" ]; then
  DEFAULT_APP_DATA="$HOME/Library/Application Support/sh.voicebox.app"
  if [ -d "$DEFAULT_APP_DATA" ]; then
    ULTRON_VOICEBOX_DATA_DIR="$DEFAULT_APP_DATA"
  else
    ULTRON_VOICEBOX_DATA_DIR="$ROOT/data/voicebox"
  fi
fi

if [ ! -d "$VENV" ]; then
  echo "Engine venv missing. Run: ./scripts/setup-voicebox-engine.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$VB_ROOT"

export PYTHONPATH="$VB_ROOT:${PYTHONPATH:-}"
mkdir -p "$ULTRON_VOICEBOX_DATA_DIR"

echo "Starting embedded Voicebox engine on ${HOST}:${PORT}"
echo "  data-dir: $ULTRON_VOICEBOX_DATA_DIR"

exec "$VENV/bin/python" -m backend.main \
  --host "$HOST" \
  --port "$PORT" \
  --data-dir "$ULTRON_VOICEBOX_DATA_DIR"
