"""Cartesia-compatible websocket TTS."""

import asyncio
import base64
import json
from typing import Dict

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from loguru import logger

from ..services.speech_context import (
    MAX_CONTEXTS,
    SpeechContext,
    Words,
    check_websocket_access,
    parse_speed,
)
from .elevenlabs_compatible import SAMPLE_RATES
from .openai_compatible import get_tts_service, process_and_validate_voices

router = APIRouter(tags=["Cartesia Compatible TTS"])


def parse_output_format(output_format: dict) -> int:
    """Return the sample rate of a raw pcm_s16le output_format."""
    if output_format.get("container", "raw") != "raw":
        raise ValueError("Only the raw container is supported")
    if output_format.get("encoding", "pcm_s16le") != "pcm_s16le":
        raise ValueError("Only pcm_s16le encoding is supported")
    rate = output_format.get("sample_rate", 24000)
    if rate not in SAMPLE_RATES:
        raise ValueError(f"Unsupported sample rate: {rate}")
    return int(rate)


@router.websocket("/tts/websocket", dependencies=[Depends(check_websocket_access)])
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    tts_service = await get_tts_service()
    contexts: Dict[str, SpeechContext] = {}
    send_lock = asyncio.Lock()

    async def send(message: dict) -> None:
        async with send_lock:
            try:
                await websocket.send_text(json.dumps(message))
            except (WebSocketDisconnect, RuntimeError):
                pass

    async def send_error(context_id: str, error: str) -> None:
        contexts.pop(context_id, None)
        await send(
            {
                "type": "error",
                "error": error,
                "done": True,
                "status_code": 500,
                "context_id": context_id,
            }
        )

    async def open_context(context_id: str, message: dict) -> SpeechContext:
        voice = await process_and_validate_voices(message["voice"]["id"], tts_service)
        sample_rate = parse_output_format(message.get("output_format") or {})
        speed = (message.get("generation_config") or {}).get("speed", 1.0)

        async def on_audio(
            data: bytes, words: Words, start: float, _duration: float
        ) -> None:
            await send(
                {
                    "type": "chunk",
                    "data": base64.b64encode(data).decode(),
                    "done": False,
                    "status_code": 206,
                    "context_id": context_id,
                }
            )
            if words:
                await send(
                    {
                        "type": "timestamps",
                        "word_timestamps": {
                            "words": [w for w, _, _ in words],
                            "start": [round(start + s, 3) for _, s, _ in words],
                            "end": [round(start + e, 3) for _, _, e in words],
                        },
                        "done": False,
                        "status_code": 206,
                        "context_id": context_id,
                    }
                )

        async def on_done() -> None:
            contexts.pop(context_id, None)
            await send(
                {
                    "type": "done",
                    "done": True,
                    "status_code": 200,
                    "context_id": context_id,
                }
            )

        async def on_error(error: str) -> None:
            logger.error(f"Websocket context {context_id} failed: {error}")
            await send_error(context_id, error)

        return SpeechContext(
            tts_service,
            voice,
            parse_speed(speed),
            "pcm",
            sample_rate,
            message.get("max_buffer_delay_ms") == 0,
            on_audio,
            on_done,
            on_error,
        )

    try:
        while True:
            message = json.loads(await websocket.receive_text())
            context_id = str(message.get("context_id", ""))
            context = contexts.get(context_id)
            if message.get("cancel"):
                if context:
                    contexts.pop(context_id).cancel()
                continue
            try:
                if context is None:
                    if len(contexts) >= MAX_CONTEXTS:
                        raise ValueError(f"More than {MAX_CONTEXTS} open contexts")
                    context = contexts[context_id] = await open_context(
                        context_id, message
                    )
                context.push(message.get("transcript") or "")
                if not message.get("continue"):
                    context.close()
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                if context:
                    context.cancel()
                await send_error(context_id, str(e))
    except WebSocketDisconnect:
        pass
    except (ValueError, KeyError, AttributeError) as e:
        await websocket.close(code=1008, reason=str(e)[:120])
    finally:
        for context in contexts.values():
            context.cancel()
