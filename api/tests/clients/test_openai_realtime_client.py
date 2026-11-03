import base64

import pytest
from openai import AsyncOpenAI
from openai.types.realtime import ResponseAudioDeltaEvent, ResponseDoneEvent
from websockets.exceptions import InvalidStatus

from api.src.core.config import settings
from api.tests.test_websocket_tts import SAMPLES

SESSION = {
    "type": "realtime",
    "output_modalities": ["audio"],
    "tool_choice": "none",
    "instructions": "Speak the user's message aloud exactly as written.",
    "audio": {
        "output": {
            "voice": "alloy",
            "format": {"type": "audio/pcm", "rate": 24000},
            "speed": 1.0,
        }
    },
}


def response(text: str) -> dict:
    message = {"type": "message", "role": "user"}
    message["content"] = [{"type": "input_text", "text": text}]
    return {"conversation": "none", "output_modalities": ["audio"], "input": [message]}


async def test_wyoming_style_synthesis(service, server_url):
    client = AsyncOpenAI(api_key="x", base_url=server_url)
    events = []
    async with client.realtime.connect(model="kokoro") as conn:
        await conn.session.update(session=SESSION)
        await conn.response.create(response=response("Hello there. Bye now."))
        async for event in conn:
            events.append(event)
            if event.type == "response.done":
                break

    audio = [e for e in events if isinstance(e, ResponseAudioDeltaEvent)]
    assert b"".join(base64.b64decode(e.delta) for e in audio) == b"\x01\x00" * (
        SAMPLES * 2
    )
    transcript = next(
        e for e in events if e.type == "response.output_audio_transcript.done"
    )
    assert transcript.transcript == "Hello there. Bye now."
    assert isinstance(events[-1], ResponseDoneEvent)
    assert events[-1].response.status == "completed"
    assert service.spoken == ["Hello there. ", "Bye now."]


async def test_error_event_parses(service, server_url):
    client = AsyncOpenAI(api_key="x", base_url=server_url)
    async with client.realtime.connect(model="kokoro") as conn:
        await conn.session.update(
            session={
                "type": "realtime",
                "audio": {"output": {"format": {"type": "audio/pcmu"}}},
            }
        )
        event = await conn.recv()
        while event.type == "session.created":
            event = await conn.recv()
    assert event.type == "error"
    assert "audio/pcm" in event.error.message


async def test_api_key_is_sent_as_bearer(service, server_url, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", ["secret"])
    async with AsyncOpenAI(api_key="secret", base_url=server_url).realtime.connect(
        model="kokoro"
    ) as conn:
        assert (await conn.recv()).type == "session.created"
    with pytest.raises(InvalidStatus):
        async with AsyncOpenAI(api_key="wrong", base_url=server_url).realtime.connect(
            model="kokoro"
        ):
            pass
