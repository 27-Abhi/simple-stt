import asyncio
import base64
import json
import logging
import os
import sqlite3
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SQLITE_PATH = os.getenv("SQLITE_PATH", "stt.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_LIVE_MODEL = os.getenv(
    "GEMINI_LIVE_MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025"
)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

logger = logging.getLogger("stt")

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

_whisper_model = None


def _is_postgres() -> bool:
    return DATABASE_URL.startswith(("postgres://", "postgresql://"))


@contextmanager
def get_db():
    """Yield a DB connection (PostgreSQL when DATABASE_URL is set, else SQLite)."""
    if _is_postgres():
        if psycopg is None:
            raise RuntimeError("psycopg is required when DATABASE_URL is PostgreSQL")
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _q(sql: str) -> str:
    """Convert ? placeholders to %s for PostgreSQL."""
    return sql.replace("?", "%s") if _is_postgres() else sql


def init_db() -> None:
    pk = "SERIAL PRIMARY KEY" if _is_postgres() else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with get_db() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id {pk},
                title TEXT NOT NULL,
                transcript TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'hi-IN,en-IN',
                engine TEXT NOT NULL DEFAULT 'web-speech',
                duration_seconds REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _get_whisper():
    global _whisper_model
    if _whisper_model is None and WhisperModel is not None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


class TranscriptionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    transcript: str = Field(..., min_length=1, max_length=50000)
    language: str = Field(default="hi-IN,en-IN", max_length=64)
    engine: str = Field(default="web-speech", max_length=64)
    duration_seconds: float = Field(default=0.0, ge=0.0)


class TranscriptionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    transcript: str | None = Field(default=None, min_length=1, max_length=50000)
    language: str | None = Field(default=None, max_length=64)


app = FastAPI(title="Multilingual Real-Time STT & TTS Service", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def serve_index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "database": "postgresql" if _is_postgres() else "sqlite",
        "gemini_live_configured": bool(GEMINI_API_KEY),
        "faster_whisper_available": WhisperModel is not None,
    }


@app.get("/api/transcriptions")
def list_transcriptions(
    q: str | None = Query(default=None, max_length=200),
    language: str | None = Query(default=None, max_length=64),
):
    clauses = []
    params: list = []
    if q:
        clauses.append("(LOWER(title) LIKE ? OR LOWER(transcript) LIKE ?)")
        like = f"%{q.lower()}%"
        params.extend([like, like])
    if language:
        clauses.append("language = ?")
        params.append(language)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = _q(f"SELECT * FROM transcriptions {where} ORDER BY id DESC")
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/transcriptions", status_code=201)
def create_transcription(payload: TranscriptionCreate):
    now = _now_iso()
    title = (payload.title or "").strip()
    if not title:
        snippet = payload.transcript.strip().splitlines()[0]
        title = snippet[:60] + ("..." if len(snippet) > 60 else "")

    with get_db() as conn:
        if _is_postgres():
            row = conn.execute(
                """
                INSERT INTO transcriptions (title, transcript, language, engine, duration_seconds, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (title, payload.transcript.strip(), payload.language, payload.engine, payload.duration_seconds, now, now),
            ).fetchone()
        else:
            cur = conn.execute(
                """
                INSERT INTO transcriptions (title, transcript, language, engine, duration_seconds, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (title, payload.transcript.strip(), payload.language, payload.engine, payload.duration_seconds, now, now),
            )
            row = conn.execute("SELECT * FROM transcriptions WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


@app.get("/api/transcriptions/{item_id}")
def get_transcription(item_id: int):
    with get_db() as conn:
        row = conn.execute(_q("SELECT * FROM transcriptions WHERE id = ?"), (item_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return dict(row)


@app.patch("/api/transcriptions/{item_id}")
def update_transcription(item_id: int, payload: TranscriptionUpdate):
    updates = {}
    if payload.title is not None:
        updates["title"] = payload.title.strip()
    if payload.transcript is not None:
        updates["transcript"] = payload.transcript.strip()
    if payload.language is not None:
        updates["language"] = payload.language.strip()
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updates["updated_at"] = _now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = [*updates.values(), item_id]

    with get_db() as conn:
        cur = conn.execute(_q(f"UPDATE transcriptions SET {set_clause} WHERE id = ?"), params)
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Transcription not found")
        row = conn.execute(_q("SELECT * FROM transcriptions WHERE id = ?"), (item_id,)).fetchone()
    return dict(row)


@app.delete("/api/transcriptions/{item_id}", status_code=204)
def delete_transcription(item_id: int):
    with get_db() as conn:
        cur = conn.execute(_q("DELETE FROM transcriptions WHERE id = ?"), (item_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Transcription not found")


async def _relay_gemini_live(ws: WebSocket, api_key: str, language: str) -> None:
    """Bridge browser PCM16 audio chunks to Gemini Live API and stream text deltas back."""
    import websockets
    from websockets.exceptions import InvalidHandshake, WebSocketException

    uri = (
        "wss://generativelanguage.googleapis.com/ws/"
        f"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"
    )
    try:
        gemini_ws = await websockets.connect(uri, max_size=2**22, open_timeout=10)
    except InvalidHandshake as exc:
        logger.warning("Gemini Live handshake failed: %s", exc)
        await ws.send_json(
            {
                "type": "error",
                "detail": f"Gemini Live connection rejected ({exc}). Check GEMINI_API_KEY and GEMINI_LIVE_MODEL ({GEMINI_LIVE_MODEL}).",
            }
        )
        return
    except (OSError, TimeoutError, WebSocketException) as exc:
        logger.warning("Gemini Live network error: %s", exc)
        await ws.send_json({"type": "error", "detail": f"Could not reach Gemini Live API: {exc}"})
        return

    async with gemini_ws:
        setup_msg = {
            "setup": {
                "model": GEMINI_LIVE_MODEL,
                "generationConfig": {"responseModalities": ["TEXT"]},
                "systemInstruction": {
                    "parts": [
                        {
                            "text": (
                                "You are a verbatim multilingual speech-to-text transcriber. "
                                f"Preferred languages: {language}. "
                                "Transcribe the user's spoken audio accurately in real time, preserving "
                                "code-switching (for example Hindi and English in the same sentence). "
                                "Output ONLY the transcribed words, no commentary or answers."
                            )
                        }
                    ]
                },
            }
        }
        await gemini_ws.send(json.dumps(setup_msg))

        # Wait for Gemini's setupComplete (or immediate error/close) before streaming audio
        try:
            first_raw = await asyncio.wait_for(gemini_ws.recv(), timeout=10)
        except TimeoutError:
            await ws.send_json({"type": "error", "detail": "Gemini Live setup timed out."})
            return
        except WebSocketException as exc:
            logger.warning("Gemini Live closed during setup: %s", exc)
            await ws.send_json({"type": "error", "detail": f"Gemini Live setup rejected: {exc}"})
            return

        first_payload = json.loads(first_raw)
        if first_payload.get("error"):
            err = first_payload["error"]
            detail = err.get("message") if isinstance(err, dict) else str(err)
            await ws.send_json({"type": "error", "detail": f"Gemini API error: {detail}"})
            return

        await ws.send_json({"type": "ready", "engine": "gemini-live"})

        async def client_to_gemini():
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("bytes") is not None:
                        b64 = base64.b64encode(msg["bytes"]).decode("ascii")
                        await gemini_ws.send(
                            json.dumps(
                                {
                                    "realtimeInput": {
                                        "mediaChunks": [
                                            {"mimeType": "audio/pcm;rate=16000", "data": b64}
                                        ]
                                    }
                                }
                            )
                        )
                    elif msg.get("text"):
                        data = json.loads(msg["text"])
                        if data.get("type") == "end_of_speech":
                            await gemini_ws.send(
                                json.dumps({"realtimeInput": {"audioStreamEnd": True}})
                            )
                        elif data.get("type") == "stop":
                            break
            except WebSocketException as exc:
                logger.warning("Gemini Live send failed: %s", exc)
                with suppress(Exception):
                    await ws.send_json(
                        {"type": "error", "detail": f"Gemini Live send failed: {exc}"}
                    )

        async def gemini_to_client():
            try:
                async for raw in gemini_ws:
                    payload = json.loads(raw)
                    if payload.get("error"):
                        err = payload["error"]
                        detail = err.get("message") if isinstance(err, dict) else str(err)
                        await ws.send_json({"type": "error", "detail": f"Gemini API error: {detail}"})
                        return
                    server_content = payload.get("serverContent") or {}
                    model_turn = server_content.get("modelTurn") or {}
                    for part in model_turn.get("parts", []):
                        if part.get("text"):
                            await ws.send_json(
                                {"type": "transcript", "text": part["text"], "is_final": False}
                            )
                    if server_content.get("turnComplete"):
                        await ws.send_json({"type": "turn_complete"})
            except WebSocketException as exc:
                logger.warning("Gemini Live stream closed: %s", exc)
                with suppress(Exception):
                    await ws.send_json(
                        {"type": "error", "detail": f"Gemini Live stream closed: {exc}"}
                    )

        tasks = [
            asyncio.create_task(client_to_gemini()),
            asyncio.create_task(gemini_to_client()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                raise exc


async def _stream_whisper(ws: WebSocket, language: str) -> None:
    """Buffer incoming 16kHz PCM16 audio and transcribe chunks with faster-whisper."""
    import numpy as np

    model = _get_whisper()
    if model is None:
        await ws.send_json({"type": "error", "detail": "faster-whisper is not installed on server"})
        return

    await ws.send_json({"type": "ready", "engine": "faster-whisper"})
    pcm_buffer = bytearray()
    lang_hint = None if "," in language or language == "auto" else language.split("-")[0]

    while True:
        msg = await ws.receive()
        if msg.get("bytes") is not None:
            pcm_buffer.extend(msg["bytes"])
            # Transcribe every ~2 seconds of 16kHz mono PCM16 (64,000 bytes)
            if len(pcm_buffer) >= 64000:
                audio = np.frombuffer(bytes(pcm_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                segments, _ = model.transcribe(audio, language=lang_hint, vad_filter=True)
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if text:
                    await ws.send_json({"type": "transcript", "text": text, "is_final": True})
                pcm_buffer.clear()
        elif msg.get("text"):
            data = json.loads(msg["text"])
            msg_type = data.get("type")
            if msg_type in ("end_of_speech", "stop"):
                if len(pcm_buffer) > 3200:
                    audio = np.frombuffer(bytes(pcm_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = model.transcribe(audio, language=lang_hint, vad_filter=True)
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                    if text:
                        await ws.send_json({"type": "transcript", "text": text, "is_final": True})
                    pcm_buffer.clear()
                if msg_type == "stop":
                    break


@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    ws: WebSocket,
    engine: str = Query(default="auto"),
    language: str = Query(default="hi-IN,en-IN"),
    api_key: str = Query(default=""),
):
    await ws.accept()
    key = api_key.strip() or GEMINI_API_KEY
    chosen = engine
    if chosen == "auto":
        if key:
            chosen = "gemini-live"
        elif WhisperModel is not None:
            chosen = "faster-whisper"
        else:
            chosen = "none"

    try:
        if chosen == "gemini-live":
            if not key:
                await ws.send_json({"type": "error", "detail": "GEMINI_API_KEY not configured"})
            else:
                await _relay_gemini_live(ws, key, language)
        elif chosen == "faster-whisper":
            await _stream_whisper(ws, language)
        else:
            await ws.send_json(
                {
                    "type": "error",
                    "detail": "No server STT engine active. Use Browser Web Speech mode or configure GEMINI_API_KEY / faster-whisper.",
                }
            )
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            await ws.send_json({"type": "error", "detail": str(exc)})
