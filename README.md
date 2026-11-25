# Real-Time Multilingual Speech-to-Text & TTS Dashboard

A lightweight FastAPI + Vanilla JS web application for real-time multilingual Speech-to-Text (e.g., Hindi + English code-switching), Text-to-Speech playback, and a relational database dashboard with full CRUD history.

## Features

- **Real-Time Multilingual STT**:
  - **Browser Web Speech API** (instant, zero-config, supports `hi-IN` Hinglish, `en-IN`, `en-US`, etc.)
  - **Gemini Live API** via `/ws/transcribe?engine=gemini-live` (streams 16kHz PCM16 audio over WebSocket)
  - **Self-Hosted `faster-whisper`** via `/ws/transcribe?engine=faster-whisper` (optional open-source server model)
- **Text-to-Speech (TTS)**: Built-in browser `speechSynthesis` playback on live text and every saved history entry.
- **Relational Database + CRUD Dashboard**:
  - Supports **PostgreSQL** (`DATABASE_URL`) and falls back to **SQLite** (`stt.db`) for zero-config local runs.
  - Create, search/list, inline edit, and delete transcriptions.

## Quick Start (Local)

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Quick Start (Docker + PostgreSQL)

```bash
docker compose up --build
```

## API Endpoints

- `GET /api/health` — Database and STT engine status
- `GET /api/transcriptions?q=&language=` — List/search transcriptions
- `POST /api/transcriptions` — Create transcription
- `GET /api/transcriptions/{id}` — Get transcription by ID
- `PATCH /api/transcriptions/{id}` — Update title, transcript, or language
- `DELETE /api/transcriptions/{id}` — Delete transcription
- `WS /ws/transcribe` — Real-time 16kHz PCM16 audio streaming
