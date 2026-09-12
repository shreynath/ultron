#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$ROOT/.env" ]; then
  # shellcheck disable=SC1091
  set -a
  source "$ROOT/.env"
  set +a
fi

VOICE_ENGINE_MODE="${VOICE_ENGINE_MODE:-embedded}"
VOICE_PORT="${ULTRON_VOICEBOX_PORT:-17493}"
ENGINE_VENV="$ROOT/backend/vendor/voicebox/backend/venv"
PERFORMANCE_PROFILE="${ULTRON_PERFORMANCE_PROFILE:-maximum}"

if [ "$PERFORMANCE_PROFILE" = "maximum" ]; then
  CPU_THREADS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
  export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$CPU_THREADS}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
  BACKEND_WORKERS="${ULTRON_BACKEND_WORKERS:-$CPU_THREADS}"
else
  BACKEND_WORKERS="${ULTRON_BACKEND_WORKERS:-1}"
fi

free_port() {
  local port="$1"
  local pids
  pids=$(lsof -ti ":$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "Stopping existing process on port $port..."
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 2
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
      sleep 1
    fi
  fi
}

wait_for_engine() {
  local url="http://127.0.0.1:${VOICE_PORT}/health"
  local i
  for i in $(seq 1 90); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "Voice engine healthy."
      return 0
    fi
    if [ -n "${ENGINE_PID:-}" ] && ! kill -0 "$ENGINE_PID" 2>/dev/null; then
      echo "Voice engine process exited early. Run ./scripts/setup-voicebox-engine.sh"
      return 1
    fi
    sleep 1
  done
  echo "Voice engine did not become healthy within 90s."
  return 1
}

cleanup() {
  if [ -n "${ENGINE_PID:-}" ]; then kill "$ENGINE_PID" 2>/dev/null || true; fi
  if [ -n "${BACKEND_PID:-}" ]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
  if [ -n "${FRONTEND_PID:-}" ]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

free_port 8000
free_port 3000

if [ "$VOICE_ENGINE_MODE" = "embedded" ]; then
  if [ ! -x "$ENGINE_VENV/bin/python" ]; then
    echo "Embedded voice engine not installed."
    echo "Run: ./scripts/setup.sh"
    exit 1
  fi
  if ! "$ENGINE_VENV/bin/python" -c "import uvicorn" >/dev/null 2>&1; then
    echo "Engine venv is broken. Run: ./scripts/setup-voicebox-engine.sh"
    exit 1
  fi
  free_port "$VOICE_PORT"
  echo "Starting embedded voice engine..."
  "$ROOT/scripts/start-voicebox-engine.sh" &
  ENGINE_PID=$!
  wait_for_engine || exit 1
fi

cd "$ROOT/backend"
source .venv/bin/activate
if [ "$PERFORMANCE_PROFILE" = "maximum" ]; then
  echo "Performance profile: maximum (CPU threads: $OMP_NUM_THREADS, backend workers: $BACKEND_WORKERS)"
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers "$BACKEND_WORKERS" &
else
  uvicorn app.main:app --reload --port 8000 &
fi
BACKEND_PID=$!

cd "$ROOT/frontend"
npm run dev -- --port 3000 &
FRONTEND_PID=$!

echo ""
echo "Ultron running:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  Voice engine mode: $VOICE_ENGINE_MODE"
if [ "$VOICE_ENGINE_MODE" = "embedded" ]; then
  echo "  Voice engine: http://127.0.0.1:${VOICE_PORT}"
elif [ "$VOICE_ENGINE_MODE" = "external" ]; then
  if curl -sf "http://127.0.0.1:${VOICE_PORT}/health" >/dev/null 2>&1; then
    echo "  Voice engine: OK (external)"
  else
    echo "  Voice engine: NOT RUNNING (external mode)"
  fi
fi

wait
