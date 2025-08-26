import asyncio
import json
import os
from typing import AsyncGenerator, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.src.core.config import settings
from api.src.inference.base import AudioChunk
from api.src.main import app
from api.src.routers.openai_compatible import (
    get_tts_service,
    load_openai_mappings,
    stream_audio_chunks,
)
from api.src.services.streaming_audio_writer import StreamingAudioWriter
from api.src.services.tts_service import TTSService
from api.src.structures.schemas import OpenAISpeechRequest

client = TestClient(app)


@pytest.fixture
def test_voice():
    """Fixture providing a test voice name."""
    return "test_voice"


@pytest.fixture
def mock_openai_mappings():
    """Mock OpenAI mappings for testing."""
    with patch(
        "api.src.routers.openai_compatible._openai_mappings",
        {
            "models": {"tts-1": "kokoro-v1_0", "tts-1-hd": "kokoro-v1_0"},
            "voices": {"alloy": "am_adam", "nova": "bf_isabella"},
        },
    ):
        yield


@pytest.fixture
def mock_json_file(tmp_path):
    """Create a temporary mock JSON file."""
    content = {
        "models": {"test-model": "test-kokoro"},
        "voices": {"test-voice": "test-internal"},
    }
    json_file = tmp_path / "test_mappings.json"
    json_file.write_text(json.dumps(content))
    return json_file


def test_load_openai_mappings(mock_json_file):
    """Test loading OpenAI mappings from JSON file"""
    with patch("os.path.join", return_value=str(mock_json_file)):
        mappings = load_openai_mappings()
        assert "models" in mappings
        assert "voices" in mappings
        assert mappings["models"]["test-model"] == "test-kokoro"
        assert mappings["voices"]["test-voice"] == "test-internal"


def test_load_openai_mappings_file_not_found():
    """Test handling of missing mappings file"""
    with patch("os.path.join", return_value="/nonexistent/path"):
        mappings = load_openai_mappings()
        assert mappings == {"models": {}, "voices": {}}


def test_list_models(mock_openai_mappings):
    """Test listing available models endpoint"""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 3  # tts-1, tts-1-hd, and kokoro

    # Verify all expected models are present
    model_ids = [model["id"] for model in data["data"]]
    assert "tts-1" in model_ids
    assert "tts-1-hd" in model_ids
    assert "kokoro" in model_ids

    # Verify model format
    for model in data["data"]:
        assert model["object"] == "model"
        assert "created" in model
        assert model["owned_by"] == "kokoro"


def test_retrieve_model(mock_openai_mappings):
    """Test retrieving a specific model endpoint"""
    # Test successful model retrieval
    response = client.get("/v1/models/tts-1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "tts-1"
    assert data["object"] == "model"
    assert data["owned_by"] == "kokoro"
    assert "created" in data

    # Test non-existent model
    response = client.get("/v1/models/nonexistent-model")
    assert response.status_code == 404
    error = response.json()
    assert error["detail"]["error"] == "model_not_found"
    assert "not found" in error["detail"]["message"]
    assert error["detail"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_get_tts_service_initialization():
    """Test TTSService initialization"""
    with patch("api.src.routers.openai_compatible._tts_service", None):
        with patch("api.src.routers.openai_compatible._init_lock", None):
            with patch("api.src.services.tts_service.TTSService.create") as mock_create:
                mock_service = AsyncMock()
                mock_create.return_value = mock_service

                # Test concurrent access
                async def get_service():
                    return await get_tts_service()

                # Create multiple concurrent requests
                tasks = [get_service() for _ in range(5)]
                results = await asyncio.gather(*tasks)

                # Verify service was created only once
                mock_create.assert_called_once()
                assert all(r == mock_service for r in results)


@pytest.mark.asyncio
async def test_stream_audio_chunks_client_disconnect():
    """Test handling of client disconnect during streaming"""
    mock_request = MagicMock()
    mock_request.is_disconnected = AsyncMock(return_value=True)

    mock_service = AsyncMock()

    async def mock_stream(*args, **kwargs):
        for i in range(5):
            yield AudioChunk(np.ndarray([], np.int16), output=b"chunk")

    mock_service.generate_audio_stream = mock_stream
    mock_service.list_voices.return_value = ["test_voice"]

    request = OpenAISpeechRequest(
        model="kokoro",
        input="Test text",
        voice="test_voice",
        response_format="mp3",
        stream=True,
        speed=1.0,
    )

    writer = StreamingAudioWriter("mp3", 24000)

    chunks = []
    async for chunk in stream_audio_chunks(mock_service, request, mock_request, writer):
        chunks.append(chunk)

    writer.close()

    assert len(chunks) == 0  # Should stop immediately due to disconnect


def test_openai_voice_mapping(mock_tts_service, mock_openai_mappings):
    """Test OpenAI voice name mapping"""
    mock_tts_service.list_voices.return_value = ["am_adam", "bf_isabella"]

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hello world",
            "voice": "alloy",  # OpenAI voice name
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 200
    mock_tts_service.generate_audio.assert_called_once()
    assert mock_tts_service.generate_audio.call_args[1]["voice"] == "am_adam"


def test_openai_voice_mapping_streaming(
    mock_tts_service, mock_openai_mappings, mock_audio_bytes
):
    """Test OpenAI voice mapping in streaming mode"""
    mock_tts_service.list_voices.return_value = ["am_adam", "bf_isabella"]

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1-hd",
            "input": "Hello world",
            "voice": "nova",  # OpenAI voice name
            "response_format": "mp3",
            "stream": True,
        },
    )
    assert response.status_code == 200
    content = b""
    for chunk in response.iter_bytes():
        content += chunk
    assert content == mock_audio_bytes


def test_invalid_openai_model(mock_tts_service, mock_openai_mappings):
    """Test error handling for invalid OpenAI model"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "invalid-model",
            "input": "Hello world",
            "voice": "alloy",
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 400
    error_response = response.json()
    assert error_response["detail"]["error"] == "invalid_model"
    assert "Unsupported model" in error_response["detail"]["message"]


@pytest.fixture
def mock_audio_bytes():
    """Mock audio bytes for testing."""
    return b"mock audio data"


@pytest.fixture
def mock_tts_service(mock_audio_bytes):
    """Mock TTS service for testing."""
    with patch("api.src.routers.openai_compatible.get_tts_service") as mock_get:
        service = AsyncMock(spec=TTSService)
        service.generate_audio.return_value = AudioChunk(np.zeros(1000, np.int16))

        async def mock_stream(*args, **kwargs) -> AsyncGenerator[AudioChunk, None]:
            yield AudioChunk(np.ndarray([], np.int16), output=mock_audio_bytes)

        service.generate_audio_stream = mock_stream
        service.list_voices.return_value = ["test_voice", "voice1", "voice2"]
        service.combine_voices.return_value = "voice1_voice2"

        mock_get.return_value = service
        mock_get.side_effect = None
        yield service


@patch("api.src.services.audio.AudioService.convert_audio")
def test_openai_speech_endpoint(
    mock_convert, mock_tts_service, test_voice, mock_audio_bytes
):
    """Test the OpenAI-compatible speech endpoint with basic MP3 generation"""
    # Configure mocks
    mock_tts_service.generate_audio.return_value = AudioChunk(np.zeros(1000, np.int16))
    mock_convert.return_value = AudioChunk(
        np.zeros(1000, np.int16), output=mock_audio_bytes
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0
    assert response.content == mock_audio_bytes + mock_audio_bytes

    mock_tts_service.generate_audio.assert_called_once()
    assert mock_convert.call_count == 2


def test_openai_speech_streaming(mock_tts_service, test_voice, mock_audio_bytes):
    """Test the OpenAI-compatible speech endpoint with streaming"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert "Transfer-Encoding" in response.headers
    assert response.headers["Transfer-Encoding"] == "chunked"

    content = b""
    for chunk in response.iter_bytes():
        content += chunk
    assert content == mock_audio_bytes


def test_openai_speech_pcm_streaming(mock_tts_service, test_voice, mock_audio_bytes):
    """Test PCM streaming format"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "pcm",
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"

    content = b""
    for chunk in response.iter_bytes():
        content += chunk
    assert content == mock_audio_bytes


def test_openai_speech_invalid_voice(mock_tts_service):
    """Test error handling for invalid voice"""
    mock_tts_service.generate_audio.side_effect = ValueError(
        "Voice 'invalid_voice' not found"
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": "invalid_voice",
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 400
    error_response = response.json()
    assert error_response["detail"]["error"] == "validation_error"
    assert "Voice 'invalid_voice' not found" in error_response["detail"]["message"]
    assert error_response["detail"]["type"] == "invalid_request_error"


def test_openai_speech_empty_text(mock_tts_service, test_voice):
    """Test error handling for empty text"""

    async def mock_error_stream(*args, **kwargs):
        raise ValueError("Text is empty after preprocessing")

    mock_tts_service.generate_audio = mock_error_stream
    mock_tts_service.list_voices.return_value = ["test_voice"]

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 400
    error_response = response.json()
    assert error_response["detail"]["error"] == "validation_error"
    assert "Text is empty after preprocessing" in error_response["detail"]["message"]
    assert error_response["detail"]["type"] == "invalid_request_error"


def test_openai_speech_invalid_format(mock_tts_service, test_voice):
    """Test error handling for invalid format"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "invalid_format",
            "stream": False,
        },
    )
    assert response.status_code == 422  # Validation error from Pydantic


def test_list_voices(mock_tts_service):
    """Test listing available voices"""
    # Override the mock for this specific test
    mock_tts_service.list_voices.return_value = ["voice1", "voice2"]

    response = client.get("/v1/audio/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert len(data["voices"]) == 2
    assert "voice1" in data["voices"]
    assert "voice2" in data["voices"]


@patch("api.src.routers.openai_compatible.settings")
def test_combine_voices(mock_settings, mock_tts_service):
    """Test combining voices endpoint"""
    # Enable local voice saving for this test
    mock_settings.allow_local_voice_saving = True

    response = client.post("/v1/audio/voices/combine", json="voice1+voice2")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "voice1+voice2.pt" in response.headers["content-disposition"]


def test_server_error(mock_tts_service, test_voice):
    """Test handling of server errors"""

    async def mock_error_stream(*args, **kwargs):
        raise RuntimeError("Internal server error")

    mock_tts_service.generate_audio = mock_error_stream
    mock_tts_service.list_voices.return_value = ["test_voice"]

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 500
    error_response = response.json()
    assert error_response["detail"]["error"] == "processing_error"
    assert error_response["detail"]["type"] == "server_error"


def test_streaming_error(mock_tts_service, test_voice):
    """Test handling streaming errors"""
    # Mock process_voices to raise the error
    mock_tts_service.list_voices.side_effect = RuntimeError("Streaming failed")

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": True,
        },
    )

    assert response.status_code == 500
    error_data = response.json()
    assert error_data["detail"]["error"] == "processing_error"
    assert error_data["detail"]["type"] == "server_error"
    assert "Streaming failed" in error_data["detail"]["message"]


@pytest.mark.asyncio
async def test_streaming_initialization_error():
    """Test handling of streaming initialization errors"""
    mock_service = AsyncMock()

    async def mock_error_stream(*args, **kwargs):
        if False:  # This makes it a proper generator
            yield b""
        raise RuntimeError("Failed to initialize stream")

    mock_service.generate_audio_stream = mock_error_stream
    mock_service.list_voices.return_value = ["test_voice"]

    request = OpenAISpeechRequest(
        model="kokoro",
        input="Test text",
        voice="test_voice",
        response_format="mp3",
        stream=True,
        speed=1.0,
    )

    writer = StreamingAudioWriter("mp3", 24000)

    with pytest.raises(RuntimeError) as exc:
        async for _ in stream_audio_chunks(mock_service, request, MagicMock(), writer):
            pass

    writer.close()
    assert "Failed to initialize stream" in str(exc.value)
