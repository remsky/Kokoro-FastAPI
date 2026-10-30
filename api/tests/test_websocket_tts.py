import asyncio
import base64
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.src.core.config import settings
from api.src.inference.base import AudioChunk
from api.src.main import app
from api.src.services.speech_context import MAX_CONTEXTS, SENTENCE_END, resample
from api.src.structures.schemas import WordTimestamp

client = TestClient(app)
SAMPLES = 2400


class FakeService:
    def __init__(self, delay: float = 0):
        self.spoken = []
        self.delay = delay
        self.samples = SAMPLES
        self.fail = False

    async def generate_audio_stream(self, text, voice, writer, **kwargs):
        self.spoken.append(text)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("Model failed")
        yield AudioChunk(
            np.array([], np.int16),
            word_timestamps=[
                WordTimestamp(word="Hi", start_time=0.0, end_time=0.04),
                WordTimestamp(word=".", start_time=0.04, end_time=0.05),
                WordTimestamp(word="There", start_time=0.05, end_time=0.1),
            ],
            output=np.ones(self.samples, np.int16).tobytes(),
        )


@pytest.fixture
def service():
    fake = FakeService()
    voices = AsyncMock(side_effect=lambda voice, _service: voice)
    with (
        patch(
            "api.src.routers.elevenlabs_compatible.get_tts_service",
            AsyncMock(return_value=fake),
        ),
        patch(
            "api.src.routers.cartesia_compatible.get_tts_service",
            AsyncMock(return_value=fake),
        ),
        patch(
            "api.src.routers.elevenlabs_compatible.process_and_validate_voices", voices
        ),
        patch(
            "api.src.routers.cartesia_compatible.process_and_validate_voices", voices
        ),
    ):
        yield fake


def el_url(query: str = "output_format=pcm_16000&auto_mode=true") -> str:
    return f"/v1/text-to-speech/af_heart/multi-stream-input?{query}"


def receive_until(ws, done) -> list:
    messages = []
    while True:
        messages.append(ws.receive_json())
        if done(messages[-1]):
            return messages


def cartesia_message(**overrides) -> dict:
    return {
        "context_id": "k1",
        "voice": {"mode": "id", "id": "am_michael"},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
        "max_buffer_delay_ms": 0,
        "transcript": "",
        "continue": True,
        **overrides,
    }


def test_elevenlabs_turn(service):
    with client.websocket_connect(el_url()) as ws:
        ws.send_json({"text": " ", "context_id": "c1", "voice_settings": {}})
        ws.send_json({"text": "Hello there. ", "context_id": "c1"})
        ws.send_json({"text": "Bye now.", "context_id": "c1"})
        ws.send_json({"context_id": "c1", "flush": True})
        ws.send_json({"context_id": "c1", "close_context": True})
        messages = receive_until(ws, lambda m: m.get("isFinal"))

    audio = [m for m in messages if m.get("audio")]
    assert service.spoken == ["Hello there. ", "Bye now."]
    assert all(m["contextId"] == "c1" for m in messages)
    assert len(base64.b64decode(audio[0]["audio"])) == SAMPLES * 2 * 16000 // 24000
    for message in audio:
        alignment = message["normalizedAlignment"]
        assert "".join(alignment["chars"]) == "Hi. There "
        assert alignment["charStartTimesMs"][0] == 0
        assert (
            alignment["charStartTimesMs"][-1] + alignment["charDurationsMs"][-1] == 100
        )


def test_elevenlabs_sync_alignment_is_cumulative(service):
    with client.websocket_connect(el_url() + "&sync_alignment=true") as ws:
        ws.send_json({"text": "Hello there. ", "context_id": "c1"})
        ws.send_json({"text": "Bye now.", "context_id": "c1"})
        ws.send_json({"context_id": "c1", "close_context": True})
        messages = receive_until(ws, lambda m: m.get("isFinal"))
    starts = [m["alignment"]["charStartTimesMs"] for m in messages if m.get("audio")]
    assert starts[0][0] == 0
    assert starts[1][0] == 100


def test_elevenlabs_buffers_to_sentences_without_auto_mode(service):
    with client.websocket_connect(el_url("output_format=pcm_24000")) as ws:
        ws.send_json({"text": "Hello there. Ho", "context_id": "c1"})
        ws.send_json({"text": "w are you", "context_id": "c1"})
        ws.send_json({"text": "", "context_id": "c1", "flush": True})
        ws.send_json({"context_id": "c1", "close_context": True})
        receive_until(ws, lambda m: m.get("isFinal"))
    assert service.spoken == ["Hello there. ", "How are you"]


def test_elevenlabs_mp3_keeps_alignment_of_buffered_audio(service):
    service.samples = 240
    with client.websocket_connect(
        el_url("output_format=mp3_44100_128&auto_mode=true")
    ) as ws:
        ws.send_json({"text": "Hello there. ", "context_id": "c1"})
        ws.send_json({"text": "Bye now.", "context_id": "c1"})
        ws.send_json({"context_id": "c1", "close_context": True})
        messages = receive_until(ws, lambda m: m.get("isFinal"))
    audio = [m for m in messages if m.get("audio")]
    data = b"".join(base64.b64decode(m["audio"]) for m in audio)
    assert data[:3] == b"ID3" or data[0] == 0xFF
    chars = "".join(c for m in audio for c in m.get("alignment", {}).get("chars", []))
    assert chars == "Hi. There Hi. There "


@pytest.mark.parametrize("output_format", ["ulaw_8000", "pcm_12345", "pcm"])
def test_elevenlabs_rejects_unsupported_format(service, output_format):
    with client.websocket_connect(el_url(f"output_format={output_format}")) as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 1008


def test_rejects_non_finite_speed(service):
    with client.websocket_connect(el_url()) as ws:
        ws.send_json(
            {
                "text": "Hi.",
                "context_id": "c1",
                "voice_settings": {"speed": float("nan")},
            }
        )
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 1008

    with client.websocket_connect("/tts/websocket") as ws:
        ws.send_json(cartesia_message(generation_config={"speed": float("nan")}))
        error = ws.receive_json()
    assert error["type"] == "error"
    assert service.spoken == []


def test_binary_frame_closes_cleanly(service):
    for url in (el_url(), "/tts/websocket"):
        with client.websocket_connect(url) as ws:
            ws.send_bytes(b"hi")
            with pytest.raises(WebSocketDisconnect) as e:
                ws.receive_json()
        assert e.value.code == 1008


def test_kill_switch(service, monkeypatch):
    monkeypatch.setattr(settings, "enable_websocket_tts", False)
    for url in (el_url(), "/tts/websocket"):
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect(url) as ws:
                ws.receive_json()
        assert e.value.code == 1008


@pytest.mark.parametrize(
    "origin, allowed",
    [
        (None, True),
        ("http://testserver", False),
        ("https://app.example", True),
        ("https://other.example", False),
    ],
)
def test_origin_follows_cors_origins(service, monkeypatch, origin, allowed):
    monkeypatch.setattr(settings, "cors_origins", ["https://app.example"])
    headers = {"origin": origin} if origin else {}
    for url in (el_url(), "/tts/websocket"):
        if allowed:
            with client.websocket_connect(url, headers=headers) as ws:
                ws.send_json({"context_id": "c1", "close_context": True})
            continue
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect(url, headers=headers):
                pass
        assert e.value.code == 1008


def test_browsers_refused_with_cors_disabled(service, monkeypatch):
    monkeypatch.setattr(settings, "cors_enabled", False)
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect(
            el_url(), headers={"origin": "http://testserver"}
        ):
            pass
    assert e.value.code == 1008
    with client.websocket_connect(el_url()) as ws:
        ws.send_json({"context_id": "c1", "close_context": True})


def test_cartesia_turn(service):
    rate = 8000
    output_format = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": rate}
    with client.websocket_connect("/tts/websocket") as ws:
        ws.send_json(
            cartesia_message(transcript="Hello there. ", output_format=output_format)
        )
        ws.send_json(
            cartesia_message(transcript="Bye now.", output_format=output_format)
        )
        ws.send_json(
            cartesia_message(output_format=output_format, **{"continue": False})
        )
        messages = receive_until(ws, lambda m: m["type"] == "done")

    assert service.spoken == ["Hello there. ", "Bye now."]
    assert all(m["context_id"] == "k1" for m in messages)
    chunks = [m for m in messages if m["type"] == "chunk"]
    assert all(not m["done"] for m in chunks)
    assert len(base64.b64decode(chunks[0]["data"])) == SAMPLES * 2 * rate // 24000
    stamps = [m["word_timestamps"] for m in messages if m["type"] == "timestamps"]
    assert stamps[0]["words"] == ["Hi.", "There"]
    assert stamps[1]["start"][0] == pytest.approx(SAMPLES / 24000)
    assert messages[-1]["done"] is True


def test_cartesia_cancel_keeps_socket(service):
    service.delay = 0.2
    with client.websocket_connect("/tts/websocket") as ws:
        ws.send_json(cartesia_message(context_id="old", transcript="A long reply."))
        ws.send_json({"context_id": "old", "cancel": True})
        service.delay = 0
        ws.send_json(
            cartesia_message(context_id="new", transcript="Hi.", **{"continue": False})
        )
        messages = receive_until(ws, lambda m: m["type"] == "done")
    assert {m["context_id"] for m in messages} == {"new"}


def test_cartesia_bad_request_reports_error(service):
    bad_format = {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000}
    with client.websocket_connect("/tts/websocket") as ws:
        ws.send_json(cartesia_message(context_id="bad", output_format=bad_format))
        error = ws.receive_json()
        ws.send_json(
            cartesia_message(context_id="good", transcript="Hi.", **{"continue": False})
        )
        messages = receive_until(ws, lambda m: m["type"] == "done")
    assert error["type"] == "error" and error["context_id"] == "bad"
    assert messages[-1]["context_id"] == "good"


def test_open_contexts_are_capped(service):
    service.delay = 1
    with client.websocket_connect(el_url()) as ws:
        for i in range(MAX_CONTEXTS + 1):
            ws.send_json({"text": "Hi.", "context_id": str(i)})
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 1008

    with client.websocket_connect("/tts/websocket") as ws:
        for i in range(MAX_CONTEXTS + 1):
            ws.send_json(cartesia_message(context_id=str(i), transcript="Hi."))
        error = ws.receive_json()
    assert error["type"] == "error" and error["context_id"] == str(MAX_CONTEXTS)


def test_context_text_is_capped(service, monkeypatch):
    monkeypatch.setattr(settings, "max_input_length", 10)
    with client.websocket_connect(el_url()) as ws:
        ws.send_json({"text": "Short. ", "context_id": "c1"})
        ws.send_json({"text": "Over the cap.", "context_id": "c1"})
        with pytest.raises(WebSocketDisconnect) as e:
            receive_until(ws, lambda m: m.get("isFinal"))
    assert e.value.code == 1008

    with client.websocket_connect("/tts/websocket") as ws:
        ws.send_json(cartesia_message(context_id="big", transcript="Over the cap."))
        error = ws.receive_json()
        ws.send_json(
            cartesia_message(context_id="ok", transcript="Hi.", **{"continue": False})
        )
        messages = receive_until(ws, lambda m: m["type"] == "done")
    assert error["type"] == "error" and error["context_id"] == "big"
    assert messages[-1]["context_id"] == "ok"
    assert service.spoken == ["Short. ", "Hi."]


def test_generation_failure_reports_error(service):
    service.fail = True
    with client.websocket_connect(el_url()) as ws:
        ws.send_json({"text": "Hi.", "context_id": "c1"})
        messages = receive_until(ws, lambda m: m.get("isFinal"))
        service.fail = False
        ws.send_json({"text": "Hi.", "context_id": "c2"})
        ws.send_json({"context_id": "c2", "close_context": True})
        recovered = receive_until(ws, lambda m: m.get("isFinal"))
    assert messages[0] == {"error": "Model failed", "contextId": "c1"}
    assert recovered[0]["audio"] and recovered[0]["contextId"] == "c2"

    service.fail = True
    with client.websocket_connect("/tts/websocket") as ws:
        ws.send_json(cartesia_message(transcript="Hi.", **{"continue": False}))
        error = ws.receive_json()
    assert error["type"] == "error" and error["context_id"] == "k1"
    assert "Model failed" in error["error"]


def test_resample_lengths():
    audio = np.ones(SAMPLES, np.int16)
    for rate in (8000, 16000, 22050, 44100, 48000):
        assert len(resample(audio, rate)) == SAMPLES * rate // 24000


def test_sentence_boundary_floods_are_fast():
    for payload in (".)" * 20_000, "." * 40_000, ".)" + ")" * 40_000, '!"' * 20_000):
        start = time.monotonic()
        list(SENTENCE_END.finditer(payload))
        assert time.monotonic() - start < 2.0
