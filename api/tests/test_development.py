import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from api.src.inference.base import AudioChunk
from api.src.main import app
from api.src.services import temp_manager
from api.src.services.tts_service import TTSService
from api.src.structures.schemas import WordTimestamp

client = TestClient(app)


@pytest.fixture
def mock_tts_service():
    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["test_voice"]
    with patch.object(TTSService, "create", AsyncMock(return_value=service)):
        yield service


def captioned(**body):
    return client.post(
        "/dev/captioned_speech",
        json={
            "model": "kokoro",
            "voice": "test_voice",
            "response_format": "mp3",
            **body,
        },
    )


def word(text, start):
    return WordTimestamp(word=text, start_time=start, end_time=start + 0.1)


def test_captioned_speech_nothing_speakable_is_400(mock_tts_service):
    """Blank input, or emoji-only with remove_emoji, 400s before the stream opens (issue #353)."""
    for body in [
        {"input": "   "},
        {"input": "😊", "normalization_options": {"remove_emoji": True}},
    ]:
        response = captioned(stream=True, **body)
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "validation_error"


def test_captioned_speech_stream_carries_timestamps_forward(mock_tts_service):
    """A chunk with timestamps but no audio hands them to the next audio chunk."""

    async def stream(*args, **kwargs):
        yield AudioChunk(np.zeros(1, np.int16), word_timestamps=[word("a", 0.0)])
        yield AudioChunk(
            np.zeros(1, np.int16), output=b"mp3", word_timestamps=[word("b", 0.2)]
        )
        yield AudioChunk(np.zeros(1, np.int16), output=b"mp3", word_timestamps=None)

    mock_tts_service.generate_audio_stream = stream
    response = captioned(input="a b", stream=True, return_timestamps=True)

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert [[t["word"] for t in line["timestamps"]] for line in lines] == [
        ["a", "b"],
        [],
    ]
    assert all(base64.b64decode(line["audio"]) == b"mp3" for line in lines)


def test_captioned_speech_download_link_writes_the_temp_file(
    mock_tts_service, tmp_path, monkeypatch
):
    monkeypatch.setattr(temp_manager.settings, "temp_file_dir", str(tmp_path))

    async def stream(*args, **kwargs):
        yield AudioChunk(
            np.zeros(1, np.int16), output=b"mp3", word_timestamps=[word("hi", 0.0)]
        )

    mock_tts_service.generate_audio_stream = stream
    response = captioned(input="hi", stream=True, return_download_link=True)

    assert response.status_code == 200
    name = response.headers["X-Download-Path"].rsplit("/", 1)[-1]
    assert (tmp_path / name).read_bytes() == b"mp3"
    assert json.loads(response.text)["timestamps"][0]["word"] == "hi"


def test_captioned_speech_non_stream_returns_audio_and_timestamps(mock_tts_service):
    mock_tts_service.generate_audio.return_value = AudioChunk(
        np.zeros(1, np.int16), output=b"mp3", word_timestamps=[word("hi", 0.0)]
    )
    response = captioned(input="hi", stream=False, return_timestamps=True)

    assert response.status_code == 200
    body = response.json()
    assert base64.b64decode(body["audio"]) == b"mp3"
    assert [t["word"] for t in body["timestamps"]] == ["hi"]


def test_generate_captioned_speech():
    """Test the generate_captioned_speech function with mocked responses"""
    # Mock the API responses
    mock_audio_response = MagicMock()
    mock_audio_response.status_code = 200

    mock_timestamps_response = MagicMock()
    mock_timestamps_response.status_code = 200
    mock_timestamps_response.content = json.dumps(
        {
            "audio": base64.b64encode(b"mock audio data").decode("utf-8"),
            "timestamps": [{"word": "test", "start_time": 0.0, "end_time": 1.0}],
        }
    )

    # Patch the HTTP requests
    with patch("requests.post", return_value=mock_timestamps_response):
        # Import here to avoid module-level import issues
        from examples.captioned_speech_example import generate_captioned_speech

        # Test the function
        audio, timestamps = generate_captioned_speech("test text")

        # Verify we got both audio and timestamps
        assert audio == b"mock audio data"
        assert timestamps == [{"word": "test", "start_time": 0.0, "end_time": 1.0}]
