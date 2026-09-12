#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VB_ROOT="$ROOT/backend/vendor/voicebox"
BACKEND="$VB_ROOT/backend"
VENV="$VB_ROOT/backend/venv"
SLIM_REQ="$ROOT/scripts/voice-engine-requirements-mlx.txt"

if [ ! -d "$VB_ROOT/.git" ]; then
  echo "Voicebox source missing. Run: git clone --depth 1 https://github.com/jamiepine/voicebox.git backend/vendor/voicebox"
  exit 1
fi

python_candidates() {
  for candidate in \
    python3.12 \
    /opt/homebrew/bin/python3.12 \
    /usr/local/bin/python3.12 \
    python3.11 \
    /opt/homebrew/bin/python3.11 \
    python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
    fi
  done
}

pick_python() {
  local candidate minor major
  for candidate in $(python_candidates); do
    major="$("$candidate" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)"
    minor="$("$candidate" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
      # Prefer 3.12 for best ML wheel compatibility
      if [ "$minor" -eq 12 ]; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  for candidate in $(python_candidates); do
    major="$("$candidate" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)"
    minor="$("$candidate" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

ensure_python() {
  if pick_python >/dev/null; then
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "Python 3.12 not found. Installing via Homebrew..."
    brew install python@3.12
    return 0
  fi
  return 1
}

if ! ensure_python; then
  echo "Python 3.11–3.13 required. Install Python 3.12:"
  echo "  brew install python@3.12"
  exit 1
fi

SYSTEM_PYTHON="$(pick_python)"
PY_VERSION="$("$SYSTEM_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "==> Embedded Voicebox engine setup"
echo "    Python: $($SYSTEM_PYTHON --version)"
if [ "$PY_VERSION" = "3.13" ]; then
  echo "    Note: Python 3.13 works with Ultron's slim MLX engine set."
fi

if [ "$(uname -m)" != "arm64" ] || [ "$(uname)" != "Darwin" ]; then
  echo "Ultron embedded engine currently targets Apple Silicon (MLX)."
  echo "On other platforms, set VOICE_ENGINE_MODE=external and run Voicebox separately."
  exit 1
fi

recreate_venv=false
if [ -d "$VENV" ]; then
  if [ ! -x "$VENV/bin/python" ]; then
    recreate_venv=true
  else
    venv_version="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$venv_version" != "$PY_VERSION" ]; then
      echo "Recreating engine venv (was Python $venv_version, using $PY_VERSION)..."
      recreate_venv=true
    elif ! "$VENV/bin/python" -c "import uvicorn" >/dev/null 2>&1; then
      echo "Engine venv incomplete (missing uvicorn). Reinstalling..."
      recreate_venv=true
    fi
  fi
fi

if [ "$recreate_venv" = true ]; then
  rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
  echo "Creating engine virtualenv at backend/vendor/voicebox/backend/venv"
  "$SYSTEM_PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip -q

echo "Installing Ultron MLX voice engine dependencies (slim set)..."
pip install -r "$SLIM_REQ"
pip install --no-deps mlx-lm==0.31.1
pip install --no-deps mlx-audio==0.4.1

"$VENV/bin/python" -c "import uvicorn, mlx, fastapi; print('Engine deps OK')"

echo ""
echo "Embedded voice engine ready."
echo "  Data dir: \${ULTRON_VOICEBOX_DATA_DIR:-~/Library/Application Support/sh.voicebox.app}"
echo "  Start with: ./scripts/dev.sh"
