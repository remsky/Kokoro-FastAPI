from unittest.mock import AsyncMock, Mock

import pytest

from api.src.services.tts_service import TTSService


@pytest.fixture
async def mock_model_manager():
    """Mock model manager for UI tests"""
    manager = AsyncMock()
    manager.get_backend = Mock(return_value=Mock(device="cpu"))
    return manager


@pytest.fixture
async def mock_voice_manager():
    """Mock voice manager for UI tests"""
    manager = AsyncMock()
    manager.list_voices = AsyncMock(return_value=["af_heart", "bm_lewis", "af_sarah"])
    return manager


@pytest.fixture
async def mock_tts_service(mock_model_manager, mock_voice_manager):
    """Mock TTSService for UI tests"""
    service = AsyncMock()
    service.model_manager = mock_model_manager
    service._voice_manager = mock_voice_manager
    return service


@pytest.fixture(autouse=True)
async def setup_mocks(
    monkeypatch, mock_model_manager, mock_voice_manager, mock_tts_service
):
    """Setup global mocks for UI tests"""

    async def mock_get_model():
        return mock_model_manager

    async def mock_get_voice():
        return mock_voice_manager

    async def mock_create_service():
        return mock_tts_service

    monkeypatch.setattr("api.src.inference.model_manager.get_manager", mock_get_model)
    monkeypatch.setattr("api.src.inference.voice_manager.get_manager", mock_get_voice)
    monkeypatch.setattr(
        "api.src.services.tts_service.TTSService.create", mock_create_service
    )
