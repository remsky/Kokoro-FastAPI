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
from api.src.routers.openai_compatible import resolve_voice_alias
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
        self.voice, self.speed = voice, kwargs.get("speed")
        self.kwargs = kwargs
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
    voices = AsyncMock(
        side_effect=lambda voice, _service, aliases=None: resolve_voice_alias(
            voice, aliases
        )
    )
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
        patch(
            "api.src.routers.openai_realtime.get_tts_service",
            AsyncMock(return_value=fake),
        ),
        patch("api.src.routers.openai_realtime.process_and_validate_voices", voices),
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
    for url in (el_url(), "/tts/websocket", "/v1/realtime"):
        with client.websocket_connect(url) as ws:
            ws.send_bytes(b"hi")
            with pytest.raises(WebSocketDisconnect) as e:
                receive_until(ws, lambda m: False)
        assert e.value.code == 1008


def test_kill_switch(service, monkeypatch):
    monkeypatch.setattr(settings, "enable_websocket_tts", False)
    for url in (el_url(), "/tts/websocket", "/v1/realtime"):
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
    for url in (el_url(), "/tts/websocket", "/v1/realtime"):
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


def realtime_response(text: str) -> dict:
    message = {"type": "message", "role": "user"}
    message["content"] = [{"type": "input_text", "text": text}]
    return {
        "type": "response.create",
        "response": {
            "conversation": "none",
            "output_modalities": ["audio"],
            "input": [message],
        },
    }


def is_response_done(message: dict) -> bool:
    return message["type"] == "response.done"


def test_realtime_tts_turn(service):
    session = {
        "type": "realtime",
        "output_modalities": ["audio"],
        "tool_choice": "none",
        "instructions": "Speak the user's message aloud exactly as written.",
        "audio": {
            "output": {
                "voice": "alloy",
                "format": {"type": "audio/pcm", "rate": 24000},
                "speed": 1.2,
            }
        },
    }
    with client.websocket_connect("/v1/realtime?model=kokoro") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "session.update", "session": session})
        updated = ws.receive_json()
        ws.send_json(realtime_response("Hello there. Bye now."))
        messages = receive_until(ws, is_response_done)

    assert updated["type"] == "session.updated"
    assert updated["session"]["audio"]["output"]["voice"] == "alloy"
    assert service.spoken == ["Hello there. ", "Bye now."]
    assert (service.voice, service.speed) == ("alloy", 1.2)
    types = [m["type"] for m in messages]
    assert types[:4] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_audio_transcript.delta",
    ]
    assert types[-4:] == [
        "response.output_audio_transcript.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    audio = [m for m in messages if m["type"] == "response.output_audio.delta"]
    assert len(audio) == 2
    assert all(len(base64.b64decode(m["delta"])) == SAMPLES * 2 for m in audio)
    transcript = next(
        m for m in messages if m["type"] == "response.output_audio_transcript.done"
    )
    assert transcript["transcript"] == "Hello there. Bye now."
    assert messages[-1]["response"]["status"] == "completed"
    assert len({m["event_id"] for m in messages}) == len(messages)


def test_realtime_voice_object(service):
    session = {"type": "realtime", "audio": {"output": {"voice": {"id": "af_bella"}}}}
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": session})
        updated = ws.receive_json()
        ws.send_json(realtime_response("Hi."))
        receive_until(ws, is_response_done)
    assert updated["type"] == "session.updated"
    assert service.voice == "af_bella"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "input_audio_buffer.append", "audio": ""},
        {"type": "response.create", "response": {}},
        {
            "type": "session.update",
            "session": {"audio": {"output": {"format": {"type": "audio/pcmu"}}}},
        },
        {
            "type": "session.update",
            "session": {"audio": {"output": {"format": {"rate": 16000}}}},
        },
        {
            "type": "session.update",
            "session": {"audio": {"output": {"speed": float("nan")}}},
        },
    ],
)
def test_realtime_rejects_with_error_event(service, event):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({**event, "event_id": "client_1"})
        error = ws.receive_json()
        ws.send_json(realtime_response("Hi."))
        messages = receive_until(ws, is_response_done)
    assert error["type"] == "error"
    assert error["error"]["event_id"] == "client_1"
    assert messages[-1]["response"]["status"] == "completed"
    assert service.spoken == ["Hi."]


@pytest.mark.parametrize("text", ["not json", "[]"])
def test_realtime_bad_json_is_an_error_event(service, text):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_text(text)
        error = ws.receive_json()
        ws.send_json(realtime_response("Hi."))
        messages = receive_until(ws, is_response_done)
    assert error["type"] == "error"
    assert error["error"]["event_id"] is None
    assert messages[-1]["response"]["status"] == "completed"


def test_realtime_cancel(service):
    service.delay = 0.2
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json(realtime_response("A long reply."))
        response_id = ws.receive_json()["response"]["id"]
        ws.send_json({"type": "response.cancel", "response_id": response_id})
        done = receive_until(ws, is_response_done)[-1]
    assert done["type"] == "response.done"
    assert done["response"]["id"] == response_id
    assert done["response"]["status"] == "cancelled"


def test_realtime_cancel_without_response_is_an_error(service):
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "response.cancel"})
        error = ws.receive_json()
    assert error["type"] == "error"


def test_realtime_generation_failure(service):
    service.fail = True
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json(realtime_response("Hi."))
        messages = receive_until(ws, is_response_done)
    response = messages[-1]["response"]
    assert response["status"] == "failed"
    assert response["status_details"]["error"]["message"] == "Model failed"


SPEECH_FIELDS = {
    "voice_aliases": {"narrator": {"voice": "af_bella", "rate": 0.5}},
    "lang_code": "b",
    "volume_multiplier": 2.0,
    "normalization_options": {"normalize": False},
}


def speak_with_fields(route: str, fields: dict) -> list:
    if route == "elevenlabs":
        url = "/v1/text-to-speech/narrator/multi-stream-input?output_format=pcm_24000"
        with client.websocket_connect(url) as ws:
            ws.send_json(
                {
                    "text": "Hi.",
                    "context_id": "c1",
                    "voice_settings": {"speed": 1.2},
                    **fields,
                }
            )
            ws.send_json({"context_id": "c1", "flush": True})
            return receive_until(ws, lambda m: "error" in m or m.get("audio"))
    if route == "cartesia":
        with client.websocket_connect("/tts/websocket") as ws:
            ws.send_json(
                cartesia_message(
                    voice={"mode": "id", "id": "narrator"},
                    generation_config={"speed": 1.2},
                    transcript="Hi.",
                    **{"continue": False},
                    **fields,
                )
            )
            return receive_until(ws, lambda m: m["type"] in ("done", "error"))
    session = {
        "type": "realtime",
        "audio": {"output": {"voice": "narrator", "speed": 1.2}},
        **fields,
    }
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": session})
        updated = ws.receive_json()
        if updated["type"] == "error":
            return [updated]
        ws.send_json(realtime_response("Hi."))
        return receive_until(ws, is_response_done)


@pytest.mark.parametrize("route", ["elevenlabs", "cartesia", "realtime"])
def test_speech_fields_reach_generation(service, route):
    speak_with_fields(route, SPEECH_FIELDS)
    assert service.voice == "af_bella"
    assert service.speed == pytest.approx(0.6)
    assert service.kwargs["lang_code"] == "b"
    assert service.kwargs["volume_multiplier"] == 2.0
    assert service.kwargs["normalization_options"].normalize is False


@pytest.mark.parametrize("route", ["elevenlabs", "cartesia", "realtime"])
def test_bad_speech_fields_are_refused(service, route):
    fields = {**SPEECH_FIELDS, "lang_code": "zz"}
    if route == "elevenlabs":
        with pytest.raises(WebSocketDisconnect) as e:
            speak_with_fields(route, fields)
        assert e.value.code == 1008
    else:
        assert speak_with_fields(route, fields)[-1]["type"] == "error"
    assert service.spoken == []


def test_resample_lengths():
    audio = np.ones(SAMPLES, np.int16)
    for rate in (8000, 16000, 22050, 44100, 48000):
        assert len(resample(audio, rate)) == SAMPLES * rate // 24000


def test_sentence_boundary_floods_are_fast():
    for payload in (".)" * 20_000, "." * 40_000, ".)" + ")" * 40_000, '!"' * 20_000):
        start = time.monotonic()
        list(SENTENCE_END.finditer(payload))
        assert time.monotonic() - start < 2.0
