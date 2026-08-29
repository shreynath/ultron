# Ultron Voice

Phase 1 of Ultron: a custom voice studio for **dictation** and **zero-shot voice cloning**, powered by the free [Voicebox](https://voicebox.sh) engine running locally on your Mac.

## What it does

- Clone any voice from a YouTube URL, MP3, MP4, or WAV
- Extract the best speech clips automatically from long source audio
- Dictate with your microphone (Whisper via Voicebox)
- Generate speech in cloned voices for arbitrary new text
- Optionally refine transcripts with Ollama Cloud

## Architecture

```
Ultron Web UI (Next.js :3000)
        ↓
Ultron API (FastAPI :8000)
        ↓
Voicebox native app (:17493) — MLX on Apple Silicon
```

## Prerequisites

1. **Voicebox** (macOS Apple Silicon): https://voicebox.sh/download/mac-arm
2. **Homebrew tools**:
   ```bash
   brew install ffmpeg yt-dlp
   ```
3. **Python 3.11+** and **Node.js 20+**

## Setup

```bash
# From repo root
cp .env.example .env
# Edit .env and add your OLLAMA_API_KEY (optional)

./scripts/setup.sh
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

## Usage

1. Launch **Voicebox** and wait for it to start (API on `http://127.0.0.1:17493`)
2. Start Ultron backend and frontend
3. Go to **Voices** → paste a YouTube URL or upload media → clone
4. Go to **Studio** → dictate or generate speech with your cloned voice

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

## Roadmap

- Phase 2: Conversational agent with cloned voice output
- Phase 3: Higher-fidelity fine-tuning on longer datasets
- Phase 4+: Broader AGI capabilities beyond voice
