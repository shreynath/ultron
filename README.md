# Ultron Voice

Ultron is a conversation-first cognitive assistant with integrated cloned voice delivery. It uses the open-source [Voicebox](https://github.com/jamiepine/voicebox) engine **embedded directly** — no separate desktop app required.

## What it does

- Hold a text or voice conversation through the Executive and Cognitive Runtime
- Hear normal assistant responses automatically in a cloned voice when voice output is enabled
- Clone any voice from a YouTube URL, MP3, MP4, or WAV
- Use advanced dictation and standalone TTS tools when needed

## Architecture

```
Ultron Web UI (Next.js :3000)
        ↓
Conversation API (FastAPI :8000)
        ↓
Executive → planning/retrieval → authorized capability → observation → verification → learning
        ↓
Optional Voicebox delivery
        ↓
Embedded Voicebox engine (backend/vendor/voicebox, MLX on Apple Silicon :17493)
```

See [docs/VOICE_ENGINE.md](docs/VOICE_ENGINE.md) for engine modes and customization.

## Prerequisites

1. **Apple Silicon Mac** recommended (MLX acceleration for Qwen3-TTS + Whisper)
2. **Homebrew tools**:
   ```bash
   brew install ffmpeg yt-dlp
   ```
3. **Python 3.12+** and **Node.js 20+**
4. **~5GB disk** for voice models (downloaded on first use)

## Setup

```bash
# From repo root
cp .env.example .env
# Edit .env and add your OLLAMA_API_KEY (optional)

./scripts/setup.sh
```

### Run everything

```bash
./scripts/dev.sh
```

### Run backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

### Run frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Verify & test

```bash
./scripts/verify-voicebox.sh    # Check Voicebox is running
./scripts/integration-test.sh   # Run full test suite
```

## Usage

1. Start Ultron backend and frontend.
2. Open **Conversation** and type or use the microphone.
3. Open **Voice** settings to select a cloned profile for automatic playback.
4. Use **Voices** to create profiles and **Advanced** for standalone voice tools.

## Environment variables

| Variable | Description |
|---|---|
| `OLLAMA_API_KEY` | Ollama Cloud API key (optional, for transcript refinement) |
| `OLLAMA_HOST` | Default: `https://ollama.com` |
| `VOICEBOX_API_URL` | Default: `http://127.0.0.1:17493` |
| `NEXT_PUBLIC_API_URL` | Default: `http://127.0.0.1:8000` |

**Never commit `.env`** — it is gitignored.

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System health check |
| `/api/voices` | GET | List voice profiles |
| `/api/voices/clone` | POST | Clone from URL |
| `/api/voices/clone/upload` | POST | Clone from file upload |
| `/api/transcribe` | POST | Transcribe audio |
| `/api/transcribe/refine` | POST | Refine transcript via Ollama |
| `/api/speak` | POST | Generate speech |

## Security note

Rotate your Ollama API key if it was ever shared in chat or committed by mistake.

## Learning architecture

Every completed or failed cognitive run is recorded as a redacted immutable
experience in `BRAIN_LEARNING_DB` (default `data/brain_learning.sqlite3`).
Only claims backed by Step 7 evidence verification are promoted to reusable
knowledge. Future planning receives retrieved knowledge as typed, untrusted
reference data with provenance and confidence; it never becomes authority or
an executable command. See [docs/BRAIN_LEARNING.md](docs/BRAIN_LEARNING.md).

## Public-web research

Ultron can conduct bounded, capability-authorized public-web research with
source/claim/evidence separation, corroboration, contradiction retention, and
freshness-aware world knowledge. The deployment and safety model is documented
in [docs/BRAIN_RESEARCH.md](docs/BRAIN_RESEARCH.md). Set
`BRAIN_RESEARCH_API_TOKEN` before exposing the research control API.
