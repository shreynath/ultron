# Embedded Voice Engine

Ultron bundles the open-source [Voicebox](https://github.com/jamiepine/voicebox) Python engine directly. You do **not** need the Voicebox desktop app running alongside Ultron.

## How it works

```
Next.js UI  →  Ultron FastAPI  →  Embedded Voicebox engine (backend/vendor/voicebox)
```

- `backend/vendor/voicebox/` — vendored Voicebox source (MIT)
- `scripts/setup-voicebox-engine.sh` — installs engine Python deps into `backend/vendor/voicebox/backend/venv`
- `scripts/dev.sh` — starts engine + Ultron backend + frontend in one command

## Modes (`VOICE_ENGINE_MODE`)

| Mode | Description |
|------|-------------|
| `embedded` (default) | Ultron launches the vendored Voicebox backend as a managed subprocess |
| `direct` | Experimental in-process engine (requires Voicebox deps in the Ultron Python env) |
| `external` | Legacy: point `VOICEBOX_API_URL` at Voicebox.app or a remote server |

## Data directory

By default Ultron reuses existing Voicebox app data if present:

`~/Library/Application Support/sh.voicebox.app`

Override with `ULTRON_VOICEBOX_DATA_DIR`. Voice profiles and downloaded models persist here.

## Ultron customizations

Patches in `backend/vendor/voicebox/` (documented here, not upstream):

- **MLX instruct forwarding** — passes delivery/emotion `instruct` to mlx-audio when supported
- **Qwen ICL reference selection** — Qwen Base uses one clean, self-contained
  profile sample rather than concatenating every uploaded clip into one
  conditioning prompt. This avoids blending other speakers and scene audio into
  the clone. Other Voicebox engines retain their native multi-sample behavior.
- **Cinematic preset** — `backend/app/services/voice_presets.py` applies pitch, reverb, chorus, compression
- **Post-processing** — speed + bass boost via ffmpeg in `audio_postprocess.py`

## First run

```bash
./scripts/setup.sh    # installs Ultron + embedded engine (large download)
./scripts/dev.sh      # starts everything
```

First speech generation downloads the Qwen3-TTS 1.7B MLX model (~3.5GB).

## Future: deeper fork

The embedded engine is the foundation for deeper Ultron-specific TTS work:

- Native emotion/prosody controls in the generation path (not just post-FX)
- Custom Ultron voice pipeline replacing HTTP hops entirely
- Fine-tuned delivery models for cinematic characters
