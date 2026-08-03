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
    _resolve_download_name,
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
    # Verify all expected models are present
    model_ids = [model["id"] for model in data["data"]]
    assert "tts-1" in model_ids
    assert "tts-1-hd" in model_ids
    assert "kokoro" in model_ids
    assert "gpt-4o-mini-tts" in model_ids

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
    assert {"id": "voice1", "name": "voice1"} in data["voices"]
    assert {"id": "voice2", "name": "voice2"} in data["voices"]

    legacy = client.get("/v1/audio/voices?legacy=true")
    assert legacy.status_code == 200
    assert legacy.json()["voices"] == ["voice1", "voice2"]


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


@pytest.mark.parametrize(
    "requested,expected",
    [
        (None, "tmprloey00i.mp3"),
        ("", "tmprloey00i.mp3"),
        (
            "af_bella_2026-08-01T12-30-00-000Z.mp3",
            "af_bella_2026-08-01T12-30-00-000Z.mp3",
        ),
        ("af_bella+af_sky", "af_bella_af_sky.mp3"),
        ("report.wav", "report.mp3"),  # extension always comes from the stored file
        ("../../etc/passwd", "etc_passwd.mp3"),
        ("sub/dir/name", "sub_dir_name.mp3"),
        ('bad";name', "bad_name.mp3"),
        ("...", "tmprloey00i.mp3"),
        ("x" * 200, f"{'x' * 100}.mp3"),
    ],
)
def test_resolve_download_name(requested, expected):
    """Client-supplied save-as names are sanitized and keep the stored extension"""
    assert _resolve_download_name(requested, "tmprloey00i.mp3") == expected


@pytest.fixture
def temp_download_file(tmp_path):
    """A stored temp audio file plus its patched temp dir"""
    audio_file = tmp_path / "tmprloey00i.mp3"
    audio_file.write_bytes(b"fake mp3 bytes")
    with patch("api.src.routers.openai_compatible.settings") as mock_settings:
        mock_settings.temp_file_dir = str(tmp_path)
        yield audio_file


def test_download_uses_temp_name_by_default(temp_download_file):
    """Without ?name= the stored temp name is served"""
    response = client.get("/v1/download/tmprloey00i.mp3")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert "tmprloey00i.mp3" in response.headers["content-disposition"]


def test_download_honors_requested_name(temp_download_file):
    """?name= drives Content-Disposition so the save dialog shows a friendly name"""
    response = client.get(
        "/v1/download/tmprloey00i.mp3",
        params={"name": "af_bella_2026-08-01T12-30-00-000Z.mp3"},
    )

    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert "af_bella_2026-08-01T12-30-00-000Z.mp3" in disposition
    assert "tmprloey00i" not in disposition


def test_download_rejects_traversal_in_requested_name(temp_download_file):
    """A path-like ?name= can't escape into a directory or swap the extension"""
    response = client.get(
        "/v1/download/tmprloey00i.mp3", params={"name": "../../evil.sh"}
    )

    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert "evil.mp3" in disposition
    assert "/" not in disposition and ".." not in disposition


def test_download_missing_file_returns_404(temp_download_file):
    """Unknown temp names still 404"""
    response = client.get("/v1/download/nope.mp3")

    assert response.status_code == 404


def test_dialogue_endpoint(mock_tts_service, mock_audio_bytes):
    """Test the multi-speaker dialogue endpoint streams audio"""
    response = client.post(
        "/dev/dialogue",
        json={
            "model": "kokoro",
            "turns": [
                {"voice": "voice1", "text": "Hello there."},
                {"voice": "voice2", "text": "Hi back."},
            ],
            "response_format": "mp3",
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == mock_audio_bytes


def test_dialogue_endpoint_rejects_unknown_voice(mock_tts_service):
    """An unknown turn voice fails validation before generation starts"""
    response = client.post(
        "/dev/dialogue",
        json={
            "turns": [{"voice": "not_a_voice", "text": "Hello."}],
            "stream": False,
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "validation_error"


def test_dialogue_endpoint_requires_turns(mock_tts_service):
    """An empty turn list is a schema error"""
    response = client.post("/dev/dialogue", json={"turns": []})
    assert response.status_code == 422


def test_dialogue_request_to_tagged_input():
    """Turns render to the inline tag form the text pipeline consumes"""
    from api.src.structures.schemas import DialogueRequest

    request = DialogueRequest(
        turns=[
            {"voice": "af_bella", "text": "One."},
            {"voice": "am_michael", "text": "Two."},
        ],
        pause_between_turns=0.5,
    )
    assert request.to_tagged_input() == (
        "[voice:af_bella] One. [pause:0.5s] [voice:am_michael] Two."
    )


def test_dialogue_request_no_pause_between_turns():
    """Zero pause joins turns with a plain space"""
    from api.src.structures.schemas import DialogueRequest

    request = DialogueRequest(
        turns=[
            {"voice": "af_bella", "text": "One."},
            {"voice": "am_michael", "text": "Two."},
        ],
    )
    assert request.to_tagged_input() == "[voice:af_bella] One. [voice:am_michael] Two."


def test_dialogue_request_accepts_elevenlabs_field_names():
    """inputs/voice_id are accepted as aliases for turns/voice"""
    from api.src.structures.schemas import DialogueRequest

    request = DialogueRequest.model_validate(
        {
            "inputs": [
                {"voice_id": "af_bella", "text": "One."},
                {"voice_id": "am_michael", "text": "Two."},
            ]
        }
    )
    assert [turn.voice for turn in request.turns] == ["af_bella", "am_michael"]
    assert request.to_tagged_input() == "[voice:af_bella] One. [voice:am_michael] Two."


def test_dialogue_endpoint_accepts_elevenlabs_field_names(mock_tts_service):
    """The alias form reaches the endpoint, not just the model"""
    response = client.post(
        "/dev/dialogue",
        json={
            "inputs": [
                {"voice_id": "voice1", "text": "One."},
                {"voice_id": "voice2", "text": "Two."},
            ],
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_process_and_validate_voice_tags_maps_openai_names(
    mock_openai_mappings,
):
    """Inline tags get the same OpenAI voice mapping as the voice parameter"""
    from api.src.routers.openai_compatible import process_and_validate_voice_tags

    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["am_adam", "bf_isabella"]

    result = await process_and_validate_voice_tags(
        "[voice:alloy] Hello. [voice:nova] Hi.", service, allow_voice_tags=True
    )
    assert result == "[voice:am_adam] Hello. [voice:bf_isabella] Hi."


@pytest.mark.asyncio
async def test_process_and_validate_voice_tags_rejects_unknown():
    """An unknown inline voice raises rather than failing mid stream"""
    from api.src.routers.openai_compatible import process_and_validate_voice_tags

    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["af_heart"]

    with pytest.raises(ValueError, match="not found"):
        await process_and_validate_voice_tags(
            "[voice:nope] Hello.", service, allow_voice_tags=True
        )


@pytest.mark.asyncio
async def test_process_and_validate_voice_tags_passthrough():
    """Untagged text is returned untouched without hitting the voice list"""
    from api.src.routers.openai_compatible import process_and_validate_voice_tags

    service = AsyncMock(spec=TTSService)
    result = await process_and_validate_voice_tags("Plain text.", service)

    assert result == "Plain text."
    service.list_voices.assert_not_called()


def test_speech_endpoint_with_inline_voice_tags(mock_tts_service, mock_audio_bytes):
    """Inline voice tags are accepted on the standard speech endpoint"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "[voice:voice1] Hello. [voice:voice2] Hi.",
            "voice": "test_voice",
            "response_format": "mp3",
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.content == mock_audio_bytes


def test_speech_endpoint_rejects_unknown_inline_voice(mock_tts_service):
    """A bad inline voice is a 400, not a mid stream failure"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "[voice:not_a_voice] Hello.",
            "voice": "test_voice",
            "response_format": "mp3",
            "stream": False,
            "allow_voice_tags": True,
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "validation_error"


def test_speech_endpoint_ignores_voice_tags_by_default(mock_tts_service):
    """Bracketed text is spoken as written unless the request opts in"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "He said [voice:not_a_voice] and left.",
            "voice": "test_voice",
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 200
    kwargs = mock_tts_service.generate_audio.call_args.kwargs
    assert kwargs["text"] == "He said [voice:not_a_voice] and left."
    assert kwargs["allow_voice_tags"] is False


@pytest.mark.asyncio
async def test_voice_aliases_stand_in_for_a_mix_in_tags():
    """A short name keeps the mix out of the text without changing what is spoken"""
    from api.src.routers.openai_compatible import process_and_validate_voice_tags

    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["af_bella", "af_sky", "am_michael"]

    result = await process_and_validate_voice_tags(
        "[voice:narrator] Once. [voice:villain] Never.",
        service,
        allow_voice_tags=True,
        aliases={"narrator": "af_bella(2)+af_sky", "villain": "am_michael"},
    )
    assert result == "[voice:af_bella(2)+af_sky] Once. [voice:am_michael] Never."


@pytest.mark.asyncio
async def test_voice_alias_pointing_at_an_unknown_voice_still_fails():
    """Aliases are a naming layer, not a way around validation"""
    from api.src.routers.openai_compatible import process_and_validate_voice_tags

    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["af_heart"]

    with pytest.raises(ValueError, match="not found"):
        await process_and_validate_voice_tags(
            "[voice:narrator] Hello.",
            service,
            allow_voice_tags=True,
            aliases={"narrator": "af_nope"},
        )


@pytest.mark.asyncio
async def test_unaliased_names_are_left_to_normal_validation():
    """A tag with no alias behaves exactly as it did before aliases existed"""
    from api.src.routers.openai_compatible import process_and_validate_voice_tags

    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["af_heart", "am_michael"]

    result = await process_and_validate_voice_tags(
        "[voice:af_heart] One. [voice:narrator] Two.",
        service,
        allow_voice_tags=True,
        aliases={"narrator": "am_michael"},
    )
    assert result == "[voice:af_heart] One. [voice:am_michael] Two."


@pytest.mark.asyncio
async def test_voice_alias_applies_to_the_voice_parameter():
    """The default speaker can be named too, since it is just another cast member"""
    from api.src.routers.openai_compatible import process_and_validate_voices

    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["af_bella", "af_sky"]

    resolved = await process_and_validate_voices(
        "narrator", service, {"narrator": "af_bella(2)+af_sky"}
    )
    assert resolved == "af_bella(2)+af_sky"


def test_speech_endpoint_accepts_voice_aliases(mock_tts_service, mock_audio_bytes):
    """The alias map travels with the request, so the payload is self contained"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "[voice:narrator] Hello. [voice:villain] Never.",
            "voice": "narrator",
            "response_format": "mp3",
            "stream": False,
            "allow_voice_tags": True,
            "voice_aliases": {"narrator": "voice1", "villain": "voice2"},
        },
    )
    assert response.status_code == 200
    kwargs = mock_tts_service.generate_audio.call_args.kwargs
    assert kwargs["text"] == "[voice:voice1] Hello. [voice:voice2] Never."
    assert kwargs["voice"] == "voice1"


def test_speech_endpoint_rejects_an_alias_to_nowhere(mock_tts_service):
    """A mistyped alias target is a 400 like any other unknown voice"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "[voice:narrator] Hello.",
            "voice": "voice1",
            "response_format": "mp3",
            "stream": False,
            "allow_voice_tags": True,
            "voice_aliases": {"narrator": "not_a_voice"},
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "validation_error"


@pytest.mark.asyncio
async def test_process_and_validate_voice_tags_disabled_skips_validation():
    """With tags off the text is untouched and the voice list is never read"""
    from api.src.routers.openai_compatible import process_and_validate_voice_tags

    service = AsyncMock(spec=TTSService)
    result = await process_and_validate_voice_tags("[voice:nope] Hello.", service)

    assert result == "[voice:nope] Hello."
    service.list_voices.assert_not_called()


def test_dialogue_endpoint_opts_into_voice_tags(mock_tts_service):
    """/dev/dialogue builds its own tags, so it opts in on the caller's behalf"""
    response = client.post(
        "/dev/dialogue",
        json={
            "turns": [
                {"voice": "voice1", "text": "One."},
                {"voice": "voice2", "text": "Two."},
            ],
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert mock_tts_service.generate_audio.call_args.kwargs["allow_voice_tags"] is True
