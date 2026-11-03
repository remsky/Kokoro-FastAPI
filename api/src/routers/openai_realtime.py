"""OpenAI Realtime-compatible websocket TTS: input text in, PCM audio out."""

import asyncio
import base64
import json
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from loguru import logger

from ..core.config import settings
from ..services.speech_context import (
    MAX_CONTEXTS,
    MODEL_RATE,
    SpeechContext,
    Words,
    check_websocket_access,
    parse_speed,
)
from .openai_compatible import get_tts_service, process_and_validate_voices

router = APIRouter(tags=["OpenAI Realtime Compatible TTS"])

PCM_FORMAT = {"type": "audio/pcm", "rate": MODEL_RATE}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def input_text(items: list) -> str:
    """The input_text parts of the user messages in a response input."""
    return " ".join(
        part["text"]
        for item in items
        if item.get("type") == "message" and item.get("role") == "user"
        for part in item.get("content") or []
        if part.get("type") == "input_text" and part.get("text")
    )


def voice_name(voice) -> str:
    return str(voice["id"] if isinstance(voice, dict) else voice)


def merge_output(output: dict, update: dict) -> dict:
    merged = {**output, **((update.get("audio") or {}).get("output") or {})}
    fmt = merged["format"]
    if fmt.get("type") != "audio/pcm" or fmt.get("rate", MODEL_RATE) != MODEL_RATE:
        raise ValueError(f"Only audio/pcm at {MODEL_RATE} Hz is supported")
    parse_speed(merged["speed"])
    return merged


def response_body(response_id: str, status: str, **fields) -> dict:
    return {
        "id": response_id,
        "object": "realtime.response",
        "status": status,
        "output_modalities": ["audio"],
        **fields,
    }


@router.websocket("/v1/realtime", dependencies=[Depends(check_websocket_access)])
async def realtime(websocket: WebSocket, model: str = "kokoro"):
    await websocket.accept()
    tts_service = await get_tts_service()
    session_id = new_id("sess")
    output = {"format": PCM_FORMAT, "voice": settings.default_voice, "speed": 1.0}
    contexts: Dict[str, SpeechContext] = {}
    send_lock = asyncio.Lock()

    async def send(event_type: str, **fields) -> None:
        message = {"type": event_type, "event_id": new_id("event"), **fields}
        async with send_lock:
            try:
                await websocket.send_text(json.dumps(message))
            except (WebSocketDisconnect, RuntimeError):
                pass

    def session() -> dict:
        return {
            "id": session_id,
            "object": "realtime.session",
            "type": "realtime",
            "model": model,
            "output_modalities": ["audio"],
            "audio": {"output": output},
        }

    async def update_session(update: dict) -> None:
        nonlocal output
        merged = merge_output(output, update)
        await process_and_validate_voices(voice_name(merged["voice"]), tts_service)
        output = merged
        await send("session.updated", session=session())

    async def create_response(params: dict) -> None:
        text = input_text(params.get("input") or [])
        if not text.strip():
            raise ValueError(
                "response.create needs input text, this server only speaks the input it is given"
            )
        if len(text) > settings.max_input_length:
            raise ValueError(f"Input exceeds {settings.max_input_length} characters")
        if len(contexts) >= MAX_CONTEXTS:
            raise ValueError(f"More than {MAX_CONTEXTS} responses in progress")
        chosen = merge_output(output, params)
        voice = await process_and_validate_voices(
            voice_name(chosen["voice"]), tts_service
        )

        response_id, item_id = new_id("resp"), new_id("item")
        ids = {
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
        }
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_audio", "transcript": text}],
        }
        part = item["content"][0]

        async def on_audio(data: bytes, _words: Words, _start: float, _dur: float):
            await send(
                "response.output_audio.delta",
                **ids,
                delta=base64.b64encode(data).decode(),
            )

        async def on_done() -> None:
            contexts.pop(response_id, None)
            await send("response.output_audio.done", **ids)
            await send("response.output_audio_transcript.done", **ids, transcript=text)
            await send("response.content_part.done", **ids, part=part)
            await send(
                "response.output_item.done",
                response_id=response_id,
                output_index=0,
                item=item,
            )
            await send(
                "response.done",
                response=response_body(response_id, "completed", output=[item]),
            )

        async def on_error(error: str) -> None:
            logger.error(f"Realtime response {response_id} failed: {error}")
            contexts.pop(response_id, None)
            details = {
                "type": "failed",
                "error": {"type": "server_error", "message": error},
            }
            await send(
                "response.done",
                response=response_body(response_id, "failed", status_details=details),
            )

        await send(
            "response.created", response=response_body(response_id, "in_progress")
        )
        await send(
            "response.output_item.added",
            response_id=response_id,
            output_index=0,
            item={**item, "status": "in_progress", "content": []},
        )
        await send(
            "response.content_part.added",
            **ids,
            part={"type": "output_audio", "transcript": ""},
        )
        await send("response.output_audio_transcript.delta", **ids, delta=text)
        context = contexts[response_id] = SpeechContext(
            tts_service,
            voice,
            parse_speed(chosen["speed"]),
            "pcm",
            MODEL_RATE,
            False,
            on_audio,
            on_done,
            on_error,
        )
        context.push(text)
        context.close()

    async def cancel_response(response_id: Optional[str]) -> None:
        targets = [response_id] if response_id else list(contexts)
        if not any(rid in contexts for rid in targets):
            raise ValueError("Cancellation failed: no active response found")
        for rid in targets:
            context = contexts.pop(rid, None)
            if context is None:
                continue
            context.cancel()
            details = {"type": "cancelled", "reason": "client_cancelled"}
            await send(
                "response.done",
                response=response_body(rid, "cancelled", status_details=details),
            )

    await send("session.created", session=session())
    try:
        while True:
            message = await websocket.receive_text()
            event: dict = {}
            try:
                parsed = json.loads(message)
                if not isinstance(parsed, dict):
                    raise ValueError("Events must be JSON objects")
                event = parsed
                event_type = event.get("type")
                if event_type == "session.update":
                    await update_session(event.get("session") or {})
                elif event_type == "response.create":
                    await create_response(event.get("response") or {})
                elif event_type == "response.cancel":
                    await cancel_response(event.get("response_id"))
                else:
                    raise ValueError(
                        f"Unsupported event type '{event_type}', this server only speaks response.create input text"
                    )
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                error = {
                    "type": "invalid_request_error",
                    "code": None,
                    "message": str(e),
                    "param": None,
                    "event_id": event.get("event_id"),
                }
                await send("error", error=error)
    except WebSocketDisconnect:
        pass
    except (ValueError, KeyError, AttributeError) as e:
        await websocket.close(code=1008, reason=str(e)[:120])
    finally:
        for context in contexts.values():
            context.cancel()
