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

echo "==> Frontend dependencies"
cd "$ROOT/frontend"
npm install >/dev/null

echo "==> Checking Voicebox"
if curl -sf http://127.0.0.1:17493/health >/dev/null; then
  echo "Voicebox: OK"
else
  echo "Voicebox: NOT RUNNING"
  echo "  Download and launch Voicebox from https://voicebox.sh/download/mac-arm"
fi

echo ""
echo "Setup complete."
echo "  Backend:  cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000"
echo "  Frontend: cd frontend && npm run dev"
echo "  UI:       http://localhost:3000"
