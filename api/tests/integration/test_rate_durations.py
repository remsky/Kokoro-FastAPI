"""Rate semantics measured on real audio against a live server.

`speed` divides the model's predicted phoneme durations, so output length
scales as baseline/rate. Each case synthesizes the same sentence and checks
the duration ratio, which catches rate plumbing regressions no unit test
of tag parsing can.
"""

from __future__ import annotations

import io
import wave

import pytest

pytestmark = pytest.mark.integration

TEXT = "The quick brown fox jumps over the lazy dog and keeps on running through the open field."
REL = 0.15


def _wav_seconds(audio: bytes) -> float:
    """Streamed WAVs carry a placeholder frame count, so size the data chunk instead."""
    with wave.open(io.BytesIO(audio), "rb") as wf:
        bytes_per_second = wf.getframerate() * wf.getnchannels() * wf.getsampwidth()
    data_start = audio.find(b"data") + 8
    return (len(audio) - data_start) / bytes_per_second


def _synth_seconds(client, text: str = TEXT, voice: str = "af_bella", **extra) -> float:
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="wav",
        extra_body=extra or None,
    )
    seconds = _wav_seconds(response.content)
    assert seconds > 0.5, f"implausibly short audio ({seconds:.2f}s)"
    return seconds


@pytest.fixture(scope="module")
def baseline_seconds(openai_client) -> float:
    return _synth_seconds(openai_client)


def test_speed_param_scales_duration(openai_client, baseline_seconds):
    slow = _synth_seconds(openai_client, speed=0.5)
    assert slow / baseline_seconds == pytest.approx(2.0, rel=REL)


def test_rate_tag_scales_duration(openai_client, baseline_seconds):
    slow = _synth_seconds(openai_client, text=f"[rate:0.5] {TEXT}", allow_voice_tags=True)
    assert slow / baseline_seconds == pytest.approx(2.0, rel=REL)


def test_rate_tag_scales_the_alias_base_rate(openai_client, baseline_seconds):
    """Alias at 0.8 under [rate:1.1] speaks at 0.88, not an absolute 1.1.

    The old override semantics produced audio shorter than baseline
    (1/1.1). Multiplicative semantics produce longer audio (1/0.88), so
    the ratio's direction alone separates the two behaviors.
    """
    seconds = _synth_seconds(
        openai_client,
        text=f"[rate:1.1] {TEXT}",
        voice="slowpoke",
        voice_aliases={"slowpoke": {"voice": "af_bella", "rate": 0.8}},
        allow_voice_tags=True,
    )
    ratio = seconds / baseline_seconds
    assert ratio > 1.0, f"rate tag overrode the alias pace (ratio {ratio:.3f})"
    assert ratio == pytest.approx(1 / 0.88, rel=REL)


def test_ssml_prosody_rate_slows_duration(openai_client, server_url, baseline_seconds):
    """The full SSML path: /dev/ssml translation fed to /v1/audio/speech."""
    import httpx

    ssml = f'<speak><prosody rate="slow">{TEXT}</prosody></speak>'
    r = httpx.post(
        f"{server_url}/dev/ssml", json={"text": ssml, "voice": "af_bella"}, timeout=10
    )
    if r.status_code == 403:
        pytest.skip("SSML disabled on this server")
    r.raise_for_status()
    translated = r.json()["text"]
    assert "[rate:0.75]" in translated

    seconds = _synth_seconds(openai_client, text=translated, allow_voice_tags=True)
    assert seconds / baseline_seconds == pytest.approx(1 / 0.75, rel=REL)


def test_alias_rate_alone_slows_duration(openai_client, baseline_seconds):
    seconds = _synth_seconds(
        openai_client,
        voice="slowpoke",
        voice_aliases={"slowpoke": {"voice": "af_bella", "rate": 0.8}},
    )
    assert seconds / baseline_seconds == pytest.approx(1 / 0.8, rel=REL)
