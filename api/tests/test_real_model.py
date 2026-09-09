"""Route tests that run real inference on the CPU model.

Every other unit suite mocks the backend, this exercises loading the
weights, warmup, or the generation path end to end. Skipped unless the v1.0
weights are present:

    python docker/scripts/download_model.py --output api/src/models/v1_0
"""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from unittest.mock import patch

import pytest
import soundfile
import torch
from fastapi.testclient import TestClient

from api.src.core.config import settings
from api.src.inference.model_manager import ModelManager
from api.src.inference.voice_manager import VoiceManager
from api.src.main import app

API_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = "src/models"
VOICES_DIR = "src/voices/v1_0"
WEIGHTS = API_DIR / MODEL_DIR / "v1_0" / "kokoro-v1_0.pth"

TEXT = "The quick brown fox jumps over the lazy dog."

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(),
    reason=f"real model weights not found at {WEIGHTS}",
)


def _wav_seconds(audio: bytes) -> float:
    """Streamed WAVs carry a placeholder frame count, so size the data chunk instead."""
    with wave.open(io.BytesIO(audio), "rb") as wf:
        bytes_per_second = wf.getframerate() * wf.getnchannels() * wf.getsampwidth()
    data_start = audio.find(b"data") + 8
    return (len(audio) - data_start) / bytes_per_second


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """One client per module: loading and warming the model costs seconds."""
    from api.src.routers import openai_compatible

    with (
        patch.object(settings, "model_dir", MODEL_DIR),
        patch.object(settings, "voices_dir", VOICES_DIR),
        patch.object(settings, "use_gpu", False),
        patch.object(
            settings, "temp_file_dir", str(tmp_path_factory.mktemp("temp_files"))
        ),
        TestClient(app) as test_client,
    ):
        yield test_client

    openai_compatible._tts_service = None
    ModelManager._instance = None
    VoiceManager._instance = None


@pytest.fixture(scope="module")
def baseline_wav(client):
    """One af_bella render of TEXT, reused as the yardstick for other formats and voices."""
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": TEXT,
            "voice": "af_bella",
            "response_format": "wav",
            "stream": False,
        },
    )
    assert response.status_code == 200
    return response.content


def test_speech_returns_playable_audio(baseline_wav):
    assert baseline_wav.startswith(b"RIFF")
    assert _wav_seconds(baseline_wav) > 0.5


def test_speech_streams_mp3(client, baseline_wav):
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": TEXT,
            "voice": "af_bella",
            "response_format": "mp3",
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"

    audio, sample_rate = soundfile.read(io.BytesIO(response.content))
    assert sample_rate == 24000
    assert len(audio) / sample_rate == pytest.approx(
        _wav_seconds(baseline_wav), rel=0.1
    )


def test_speech_accepts_combined_voice(client, baseline_wav):
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": TEXT,
            "voice": "af_bella+am_adam",
            "response_format": "wav",
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert _wav_seconds(response.content) > 0.5
    assert response.content != baseline_wav


def test_speech_speed_scales_duration(client):
    def seconds(speed):
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": TEXT,
                "voice": "af_bella",
                "response_format": "wav",
                "stream": False,
                "speed": speed,
            },
        )
        assert response.status_code == 200
        return _wav_seconds(response.content)

    assert seconds(0.5) / seconds(1.0) == pytest.approx(2.0, rel=0.15)


def test_download_link_serves_the_generated_file(client):
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": TEXT,
            "voice": "af_bella",
            "response_format": "mp3",
            "stream": True,
            "return_download_link": True,
        },
    )

    assert response.status_code == 200
    download_path = response.headers["x-download-path"]

    downloaded = client.get(f"/v1{download_path}")
    assert downloaded.status_code == 200
    assert downloaded.content == response.content


def test_captioned_speech_returns_word_timestamps(client):
    response = client.post(
        "/dev/captioned_speech",
        json={
            "input": TEXT,
            "voice": "af_bella",
            "response_format": "wav",
            "stream": False,
            "return_timestamps": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert base64.b64decode(body["audio"]).startswith(b"RIFF")

    timestamps = body["timestamps"]
    assert [t["word"] for t in timestamps][:3] == ["The", "quick", "brown"]
    assert all(t["end_time"] >= t["start_time"] for t in timestamps)
    assert timestamps == sorted(timestamps, key=lambda t: t["start_time"])


def test_phonemes_round_trip_to_audio(client):
    phonemized = client.post("/dev/phonemize", json={"text": TEXT, "language": "a"})
    assert phonemized.status_code == 200
    phonemes = phonemized.json()["phonemes"]
    assert phonemes

    response = client.post(
        "/dev/generate_from_phonemes",
        json={"phonemes": phonemes, "voice": "af_bella"},
    )

    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert _wav_seconds(response.content) > 0.5


def test_dialogue_renders_every_turn(client):
    first = {"voice": "af_bella", "text": "Who left the gate open?"}
    second = {"voice": "am_adam", "text": "Not me, I was asleep."}

    def seconds(turns):
        response = client.post(
            "/dev/dialogue",
            json={"turns": turns, "response_format": "wav", "stream": False},
        )
        assert response.status_code == 200
        return _wav_seconds(response.content)

    one_turn = seconds([first])
    assert seconds([first, second]) > one_turn * 1.5


def test_voice_combine_returns_a_loadable_tensor(client):
    with patch.object(settings, "allow_local_voice_saving", True):
        response = client.post("/v1/audio/voices/combine", json=["af_bella", "am_adam"])

    assert response.status_code == 200
    tensor = torch.load(
        io.BytesIO(response.content), map_location="cpu", weights_only=False
    )
    assert (
        tensor.shape
        == torch.load(
            API_DIR / VOICES_DIR / "af_bella.pt", map_location="cpu", weights_only=False
        ).shape
    )
