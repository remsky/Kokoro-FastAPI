import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from api.src.services.tts_service import TTSService


@pytest.fixture
def mock_managers():
    """Mock model and voice managers."""

    async def _mock_managers():
        model_manager = AsyncMock()
        model_manager.get_backend.return_value = MagicMock()

        voice_manager = AsyncMock()
        voice_manager.get_voice_path.return_value = "/path/to/voice.pt"
        voice_manager.list_voices.return_value = ["voice1", "voice2"]

        with (
            patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
            patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
        ):
            mock_get_model.return_value = model_manager
            mock_get_voice.return_value = voice_manager
            return model_manager, voice_manager

    return _mock_managers()


@pytest.fixture
def tts_service(mock_managers):
    """Create TTSService instance with mocked dependencies."""

    async def _create_service():
        return await TTSService.create("test_output")

    return _create_service()


@pytest.mark.asyncio
async def test_service_creation():
    """Test service creation and initialization."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        assert service.output_dir == "test_output"
        assert service.model_manager is model_manager
        assert service._voice_manager is voice_manager


@pytest.mark.asyncio
async def test_get_voice_path_single():
    """Test getting path for single voice."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()
    voice_manager.get_voice_path.return_value = "/path/to/voice1.pt"

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        name, path = await service._get_voices_path("voice1")
        assert name == "voice1"
        assert path == "/path/to/voice1.pt"
        voice_manager.get_voice_path.assert_called_once_with("voice1")


@pytest.mark.asyncio
async def test_get_voice_path_combined():
    """Test getting path for combined voices."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()
    voice_manager.get_voice_path.return_value = "/path/to/voice.pt"

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
        patch("torch.load") as mock_load,
        patch("torch.save") as mock_save,
        patch("tempfile.gettempdir") as mock_temp,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager
        mock_temp.return_value = "/tmp"
        mock_load.return_value = torch.ones(10)

        service = await TTSService.create("test_output")
        name, path = await service._get_voices_path("voice1+voice2")
        assert name == "voice1+voice2"
        # Verify the path points to a temporary file with expected format
        assert Path(path).parent == Path("/tmp")
        assert "voice1+voice2" in Path(path).name
        assert path.endswith(".pt")
        mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_list_voices():
    """Test listing available voices."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()
    voice_manager.list_voices.return_value = ["voice1", "voice2"]

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        voices = await service.list_voices()
        assert voices == ["voice1", "voice2"]
        voice_manager.list_voices.assert_called_once()


@pytest.mark.asyncio
async def test_split_multi_voice_resolves_each_speaker_once():
    """Repeated speakers reuse their resolved path instead of reloading tensors."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        service._get_voices_path = AsyncMock(
            side_effect=lambda voice: (voice, f"/path/to/{voice}.pt")
        )

        text = "[voice:af_bella] One. [voice:bm_george] Two. [voice:af_bella] Three."
        speakers = []
        async for (
            voice_name,
            _path,
            _lang,
            _text,
            _tokens,
            _pause,
        ) in service._split_multi_voice(text, "af_heart", None, None):
            speakers.append(voice_name)

        assert speakers == ["af_bella", "bm_george", "af_bella"]
        assert service._get_voices_path.await_count == 2


@pytest.mark.asyncio
async def test_split_multi_voice_lang_code_per_speaker():
    """Each speaker gets the pipeline implied by its own voice prefix."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        service._get_voices_path = AsyncMock(
            side_effect=lambda voice: (voice, f"/path/to/{voice}.pt")
        )

        text = "[voice:af_bella] Hello. [voice:bm_george] Hello."
        langs = [
            lang
            async for _name, _path, lang, _text, _tokens, _pause in (
                service._split_multi_voice(text, "af_heart", None, None)
            )
        ]

        assert langs == ["a", "b"]


@pytest.mark.asyncio
async def test_split_multi_voice_explicit_lang_code_wins():
    """An explicit request lang_code overrides every speaker's prefix."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        service._get_voices_path = AsyncMock(
            side_effect=lambda voice: (voice, f"/path/to/{voice}.pt")
        )

        text = "[voice:af_bella] Hello. [voice:bm_george] Hello."
        langs = [
            lang
            async for _name, _path, lang, _text, _tokens, _pause in (
                service._split_multi_voice(text, "af_heart", "e", None)
            )
        ]

        assert langs == ["e", "e"]
