"""ElevenLabs-compatible websocket TTS (multi-stream-input)."""

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
from .openai_compatible import get_tts_service, process_and_validate_voices

router = APIRouter(tags=["ElevenLabs Compatible TTS"])

FORMATS = {"pcm", "mp3"}
SAMPLE_RATES = {8000, 16000, 22050, 24000, 32000, 44100, 48000}


def parse_output_format(output_format: str) -> tuple[str, int]:
    """Split an ElevenLabs output_format like pcm_24000 or mp3_44100_128."""
    parts = output_format.split("_")
    if len(parts) < 2 or parts[0] not in FORMATS or not parts[1].isdigit():
        raise ValueError(f"Unsupported output_format: {output_format}")
    rate = int(parts[1])
    if rate not in SAMPLE_RATES:
        raise ValueError(f"Unsupported sample rate: {rate}")
    return parts[0], rate


def char_alignment(words: Words, duration: float, offset: float) -> dict:
    """Spread each word's timing evenly over its characters, in ms from offset.

    Each word is followed by a space lasting until the next word, the last one
    until the end of the audio, as clients advance their clock by each
    message's alignment span.
    """
    chars, starts, durations = [], [], []
    next_starts = [max(start, 0.0) for _, start, _ in words[1:]] + [duration]
    for (word, start, end), next_start in zip(words, next_starts):
        start = max(start, 0.0)
        step = max(end - start, 0) / len(word)
        for j, char in enumerate(word):
            chars.append(char)
            starts.append(round((offset + start + j * step) * 1000))
            durations.append(round(step * 1000))
        chars.append(" ")
        starts.append(round((offset + end) * 1000))
        durations.append(round(max(next_start - end, 0) * 1000))
    return {"chars": chars, "charStartTimesMs": starts, "charDurationsMs": durations}


@router.websocket(
    "/v1/text-to-speech/{voice_id}/multi-stream-input",
    dependencies=[Depends(check_websocket_access)],
)
async def multi_stream_input(
    websocket: WebSocket,
    voice_id: str,
    output_format: str = "mp3_44100_128",
    auto_mode: bool = False,
    sync_alignment: bool = False,
):
    await websocket.accept()
    try:
        audio_format, sample_rate = parse_output_format(output_format)
        tts_service = await get_tts_service()
        voice = await process_and_validate_voices(voice_id, tts_service)
    except ValueError as e:
        await websocket.close(code=1008, reason=str(e)[:120])
        return

    contexts: Dict[str, SpeechContext] = {}
    send_lock = asyncio.Lock()

    async def send(message: dict) -> None:
        async with send_lock:
            try:
                await websocket.send_text(json.dumps(message))
            except (WebSocketDisconnect, RuntimeError):
                pass

    def open_context(context_id: str, message: dict) -> SpeechContext:
        voice_settings = message.get("voice_settings") or {}

        async def on_audio(
            data: bytes, words: Words, start: float, duration: float
        ) -> None:
            reply = {
                "audio": base64.b64encode(data).decode(),
                "isFinal": None,
                "contextId": context_id,
            }
            if words:
                reply["alignment"] = reply["normalizedAlignment"] = char_alignment(
                    words, duration, start if sync_alignment else 0.0
                )
            await send(reply)

        async def on_done() -> None:
            contexts.pop(context_id, None)
            await send({"isFinal": True, "contextId": context_id})

        async def on_error(error: str) -> None:
            logger.error(f"Websocket context {context_id} failed: {error}")
            await send({"error": error, "contextId": context_id})
            await on_done()

        return SpeechContext(
            tts_service,
            voice,
            parse_speed(voice_settings.get("speed", 1.0)),
            audio_format,
            sample_rate,
            auto_mode,
            on_audio,
            on_done,
            on_error,
        )

    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if message.get("close_socket"):
                break
            context_id = str(message.get("context_id", ""))
            text = message.get("text") or ""
            flush = bool(message.get("flush"))
            context = contexts.get(context_id)
            if message.get("close_context"):
                if context:
                    context.close(flush=False)
                continue
            if context is None:
                if not text and not flush:
                    continue
                if len(contexts) >= MAX_CONTEXTS:
                    raise ValueError(f"More than {MAX_CONTEXTS} open contexts")
                context = contexts[context_id] = open_context(context_id, message)
            context.push(text, flush=flush)
    except WebSocketDisconnect:
        pass
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        await websocket.close(code=1008, reason=str(e)[:120])
    finally:
        for context in contexts.values():
            context.cancel()
