#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Ultron setup"

if ! command -v ffmpeg >/dev/null; then
  echo "Missing ffmpeg. Install with: brew install ffmpeg"
  exit 1
fi

if ! command -v yt-dlp >/dev/null; then
  echo "Missing yt-dlp. Install with: brew install yt-dlp"
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

echo "==> Backend dependencies"
cd backend
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -e . >/dev/null

echo "==> Embedded voice engine (vendored Voicebox)"
if [ ! -d vendor/voicebox/.git ]; then
  echo "Cloning Voicebox source..."
  git clone --depth 1 https://github.com/jamiepine/voicebox.git vendor/voicebox
fi
"$ROOT/scripts/setup-voicebox-engine.sh"

echo "==> Frontend dependencies"
cd "$ROOT/frontend"
npm install >/dev/null

echo ""
echo "Setup complete."
echo "  Dev:      ./scripts/dev.sh"
echo "  UI:       http://localhost:3000"
echo ""
echo "Ultron now bundles the Voicebox engine — you do NOT need the Voicebox desktop app."
