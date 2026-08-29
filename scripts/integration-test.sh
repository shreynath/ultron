#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_URL="${ULTRON_API_URL:-http://127.0.0.1:8000}"
VOICEBOX_URL="${VOICEBOX_API_URL:-http://127.0.0.1:17493}"

echo "==> Ultron integration test"
echo "API:       $API_URL"
echo "Voicebox:  $VOICEBOX_URL"
echo ""

pass=0
fail=0

check() {
  local name="$1"
  local cmd="$2"
  if eval "$cmd" >/dev/null 2>&1; then
    echo "  OK   $name"
    pass=$((pass + 1))
  else
    echo "  FAIL $name"
    fail=$((fail + 1))
  fi
}

echo "Prerequisites"
check "ffmpeg installed" "command -v ffmpeg"
check "yt-dlp installed" "command -v yt-dlp"
check "Voicebox running" "curl -sf $VOICEBOX_URL/health"
check "Ultron API running" "curl -sf $API_URL/api/health"

echo ""
echo "API endpoints"
check "GET /api/health" "curl -sf $API_URL/api/health"
check "GET /api/voices" "curl -sf $API_URL/api/voices"
check "GET /api/voices/tools/status" "curl -sf $API_URL/api/voices/tools/status"

echo ""
echo "Unit tests"
cd backend
source .venv/bin/activate
if pytest -q; then
  echo "  OK   backend pytest"
  pass=$((pass + 1))
else
  echo "  FAIL backend pytest"
  fail=$((fail + 1))
fi

echo ""
echo "Frontend build"
cd "$ROOT/frontend"
if npm run build >/dev/null 2>&1; then
  echo "  OK   frontend build"
  pass=$((pass + 1))
else
  echo "  FAIL frontend build"
  fail=$((fail + 1))
fi

echo ""
echo "Git safety"
if git check-ignore -q .env 2>/dev/null; then
  echo "  OK   .env is gitignored"
  pass=$((pass + 1))
else
  echo "  FAIL .env is not gitignored"
  fail=$((fail + 1))
fi

echo ""
echo "Results: $pass passed, $fail failed"

if [ "$fail" -gt 0 ]; then
  echo ""
  echo "Some checks failed. Common fixes:"
  echo "  - Launch Voicebox: https://voicebox.sh/download/mac-arm"
  echo "  - Start Ultron API: ./scripts/dev.sh"
  echo "  - Install tools: brew install ffmpeg yt-dlp"
  exit 1
fi

echo "All integration checks passed."
