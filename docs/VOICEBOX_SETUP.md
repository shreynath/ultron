# Voicebox Setup (macOS Apple Silicon)

Ultron uses [Voicebox](https://voicebox.sh) as its local voice engine. On your Mac, run the **native app** — not Docker — for MLX/Metal acceleration.

## Install

1. Download Voicebox for Apple Silicon: https://voicebox.sh/download/mac-arm
2. Open the DMG and drag Voicebox to Applications
3. Launch Voicebox from Applications
4. Grant microphone/accessibility permissions if prompted

## Verify

```bash
./scripts/verify-voicebox.sh
```

Expected output:

```
Voicebox health: OK
API URL: http://127.0.0.1:17493
```

Or manually:

```bash
curl http://127.0.0.1:17493/health
```

## First-time model download

The first time you clone a voice or generate speech, Voicebox downloads the TTS model automatically.

| Engine | Size | Recommended for |
|---|---|---|
| Qwen3-TTS 1.7B | ~3.5 GB | Best cloning quality (default in Ultron) |
| Chatterbox Multilingual | ~1 GB | 23 languages |
| LuxTTS | ~1 GB | Fast CPU inference |
| Kokoro | ~350 MB | Preset voices only (no cloning) |

Ultron defaults to **Qwen3-TTS** for voice cloning. You can switch engines in the Voices upload form or when generating speech in Studio.

## Whisper (dictation)

Ultron uses Voicebox's Whisper integration for transcription. Default model: `whisper-turbo` (~8x faster than large).

## Troubleshooting

| Issue | Fix |
|---|---|
| `Voicebox: NOT RUNNING` | Launch the Voicebox app |
| Clone fails with 502 | Ensure Voicebox API is reachable at port 17493 |
| Slow first generation | Normal — model is downloading |
| Poor clone quality | Use clearer source audio; try a different engine |

## Remote / Docker (optional)

If you later run Voicebox on a GPU server, set in `.env`:

```
VOICEBOX_API_URL=http://your-server:17493
```

See `docker-compose.yml` for a headless Voicebox container setup (best on Linux with NVIDIA GPU, not macOS Docker).
