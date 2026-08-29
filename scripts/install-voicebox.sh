#!/usr/bin/env bash
set -euo pipefail

VOICEBOX_APP="/Applications/Voicebox.app"
DMG_PATH="/tmp/Voicebox-install.dmg"

get_download_url() {
  python3 - <<'PY'
import json, urllib.request
data = json.load(urllib.request.urlopen("https://api.github.com/repos/jamiepine/voicebox/releases/latest"))
for asset in data["assets"]:
    if asset["name"].endswith("_aarch64.dmg"):
        print(asset["browser_download_url"])
        break
PY
}

echo "==> Voicebox installer for Apple Silicon Mac"
echo ""

if [ -d "$VOICEBOX_APP" ]; then
  echo "Voicebox is already installed at $VOICEBOX_APP"
  echo "Launching..."
  open -a Voicebox
  sleep 3
  exec "$(cd "$(dirname "$0")" && pwd)/verify-voicebox.sh"
fi

DOWNLOAD_URL="$(get_download_url)"
if [ -z "$DOWNLOAD_URL" ]; then
  echo "Could not find Voicebox download URL. Install manually:"
  echo "  https://voicebox.sh/download/mac-arm"
  exit 1
fi

echo "Voicebox is not installed."
echo "Download URL: $DOWNLOAD_URL"
echo ""
echo "Option A — automatic download (recommended):"
read -r -p "Download and install Voicebox now? [y/N] " answer
if [[ ! "$answer" =~ ^[Yy]$ ]]; then
  echo ""
  echo "Manual install:"
  echo "  1. Open https://voicebox.sh/download/mac-arm in your browser"
  echo "  2. Drag Voicebox to Applications"
  echo "  3. Launch Voicebox from Applications"
  echo "  4. Run: ./scripts/verify-voicebox.sh"
  exit 0
fi

echo "Downloading Voicebox (~480 MB)..."
curl -L --progress-bar -o "$DMG_PATH" "$DOWNLOAD_URL"

echo "Mounting DMG..."
MOUNT_POINT=$(hdiutil attach "$DMG_PATH" -nobrowse | tail -1 | awk '{$1=$2=""; print $0}' | xargs)

echo "Installing to Applications..."
cp -R "$MOUNT_POINT/Voicebox.app" /Applications/

hdiutil detach "$MOUNT_POINT" -quiet || true
rm -f "$DMG_PATH"

# Fix Gatekeeper if needed
xattr -cr "$VOICEBOX_APP" 2>/dev/null || true

echo "Launching Voicebox..."
open -a Voicebox

echo ""
echo "Waiting for Voicebox API to start (up to 60s)..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:17493/health >/dev/null 2>&1; then
    echo "Voicebox is running!"
    exit 0
  fi
  sleep 2
done

echo ""
echo "Voicebox was installed but the API is not responding yet."
echo "Open Voicebox from Applications and wait for it to finish starting,"
echo "then run: ./scripts/verify-voicebox.sh"
