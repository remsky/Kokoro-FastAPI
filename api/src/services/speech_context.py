"""Incremental text-in, audio-out speech for the websocket TTS routes."""

import asyncio
import math
import re
from math import gcd
from typing import Awaitable, Callable, List, Optional, Tuple

import numpy as np
from fastapi import WebSocket, WebSocketException
from scipy.signal import resample_poly

from ..core.config import settings
from ..structures.schemas import clamp_rate
from .streaming_audio_writer import StreamingAudioWriter
from .tts_service import TTSService

MODEL_RATE = 24000
MAX_CONTEXTS = 16
SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")

Words = List[Tuple[str, float, float]]
EmitAudio = Callable[[bytes, Words, float, float], Awaitable[None]]


def resample(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == MODEL_RATE or len(audio) == 0:
        return audio
    g = gcd(rate, MODEL_RATE)
    out = resample_poly(audio.astype(np.float32), rate // g, MODEL_RATE // g)
    return np.clip(out, -32768, 32767).astype(np.int16)


def check_websocket_access(websocket: WebSocket) -> None:
    """Kill switch, then CORS_ORIGINS for browsers, which CORS doesn't cover."""
    if not settings.enable_websocket_tts:
        raise WebSocketException(code=1008, reason="Websocket TTS is disabled")
    origin = websocket.headers.get("origin")
    if origin is None:
        return
    origins = settings.cors_origins
    if not (settings.cors_enabled and ("*" in origins or origin in origins)):
        raise WebSocketException(code=1008, reason="Origin not allowed")


def parse_speed(value) -> float:
    speed = float(value)
    if not math.isfinite(speed):
        raise ValueError(f"Unsupported speed: {value}")
    return clamp_rate(speed)


class SpeechContext:
    """Speaks text fragments in arrival order for one context.

    With immediate set every fragment is spoken as it arrives, otherwise text is
    held until a sentence boundary or flush. emit_audio gets the encoded bytes,
    word timings relative to those bytes, their start within the context and
    their duration, in seconds.
    """

    def __init__(
        self,
        tts_service: TTSService,
        voice: str,
        speed: float,
        audio_format: str,
        sample_rate: int,
        immediate: bool,
        emit_audio: EmitAudio,
        emit_done: Callable[[], Awaitable[None]],
        emit_error: Callable[[str], Awaitable[None]],
    ):
        self.tts_service = tts_service
        self.voice = voice
        self.speed = speed
        self.writer = StreamingAudioWriter(audio_format, sample_rate)
        self.sample_rate = sample_rate
        self.immediate = immediate
        self.emit_audio = emit_audio
        self.emit_done = emit_done
        self.emit_error = emit_error
        self.buffer = ""
        self.received = 0
        self.unsent_words: Words = []
        self.unsent_start = 0.0
        self.closed = False
        self.queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.task = asyncio.create_task(self._run())

    def push(self, text: str, flush: bool = False) -> None:
        if self.closed:
            return
        self.received += len(text)
        if self.received > settings.max_input_length:
            raise ValueError(
                f"Context text exceeds {settings.max_input_length} characters"
            )
        start = max(len(self.buffer) - 1, 0)
        self.buffer += text
        cut = len(self.buffer) if flush or self.immediate else 0
        if not cut:
            for match in SENTENCE_END.finditer(self.buffer, start):
                cut = match.end()
        if cut:
            if self.buffer[:cut].strip():
                self.queue.put_nowait(self.buffer[:cut])
            self.buffer = self.buffer[cut:]

    def close(self, flush: bool = True) -> None:
        if self.closed:
            return
        if flush:
            self.push("", flush=True)
        self.closed = True
        self.queue.put_nowait(None)

    def cancel(self) -> None:
        self.closed = True
        self.task.cancel()

    async def _run(self) -> None:
        offset = 0.0
        try:
            while (text := await self.queue.get()) is not None:
                offset = await self._speak(text, offset)
            tail = self.writer.write_chunk(finalize=True)
            if tail:
                await self._emit(tail, offset)
            await self.emit_done()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.emit_error(str(e))
        finally:
            self.writer.close()

    async def _emit(self, data: bytes, end: float) -> None:
        """Send encoded bytes with the words of all audio not yet sent.

        Encoders like mp3 hold audio back, so bytes can lag the chunks they
        were written from.
        """
        start = self.unsent_start
        words = [(w, s - start, e - start) for w, s, e in self.unsent_words]
        self.unsent_words, self.unsent_start = [], end
        await self.emit_audio(data, words, start, end - start)

    async def _speak(self, text: str, offset: float) -> float:
        call_start = offset
        async for chunk in self.tts_service.generate_audio_stream(
            text,
            self.voice,
            StreamingAudioWriter("pcm", MODEL_RATE),
            speed=self.speed,
            output_format="pcm",
            return_timestamps=True,
        ):
            if not chunk.output:
                continue
            audio = np.frombuffer(chunk.output, dtype=np.int16)
            data = self.writer.write_chunk(resample(audio, self.sample_rate))
            words = self.unsent_words
            for w in chunk.word_timestamps or []:
                start, end = call_start + w.start_time, call_start + w.end_time
                if words and not any(c.isalnum() for c in w.word):
                    words[-1] = (words[-1][0] + w.word, words[-1][1], end)
                else:
                    words.append((w.word, start, end))
            offset += len(audio) / MODEL_RATE
            if data:
                await self._emit(data, offset)
        return offset
