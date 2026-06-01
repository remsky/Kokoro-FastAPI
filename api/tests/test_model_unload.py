"""Tests for ModelManager.unload(), lazy reinit in generate(), and POST /dev/unload."""

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.src.inference.base import AudioChunk
from api.src.inference.model_manager import ModelManager
from api.src.main import app
from api.src.routers.development import get_tts_service
from api.src.services.tts_service import TTSService

client = TestClient(app)


@contextmanager
def override_tts_service(service):
    """Override the get_tts_service FastAPI dependency for the duration of the block."""
    async def _override():
        return service

    app.dependency_overrides[get_tts_service] = _override
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_tts_service, None)


# ---------------------------------------------------------------------------
# ModelManager unit tests
# ---------------------------------------------------------------------------


def test_manager_init_creates_lock():
    manager = ModelManager()
    assert isinstance(manager._lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_unload_clears_backend():
    manager = ModelManager()
    mock_backend = MagicMock()
    manager._backend = mock_backend

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        await manager.unload()

    mock_backend.unload.assert_called_once()
    assert manager._backend is None


@pytest.mark.asyncio
async def test_unload_when_already_none_is_noop():
    manager = ModelManager()
    assert manager._backend is None

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        await manager.unload()  # must not raise

    assert manager._backend is None


@pytest.mark.asyncio
async def test_unload_calls_cuda_empty_cache_when_available():
    manager = ModelManager()
    manager._backend = MagicMock()

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        await manager.unload()

    mock_torch.cuda.empty_cache.assert_called_once()


@pytest.mark.asyncio
async def test_unload_skips_cuda_empty_cache_when_unavailable():
    manager = ModelManager()
    manager._backend = MagicMock()

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        await manager.unload()

    mock_torch.cuda.empty_cache.assert_not_called()


@pytest.mark.asyncio
async def test_generate_lazy_reinit_when_backend_none():
    """generate() initializes backend lazily when _backend is None."""
    manager = ModelManager()
    assert manager._backend is None

    mock_backend = MagicMock()
    audio_chunk = AudioChunk(np.zeros(10, dtype=np.float32))

    async def fake_generate(*args, **kwargs):
        yield audio_chunk

    mock_backend.generate = fake_generate

    async def fake_initialize():
        manager._backend = mock_backend

    with (
        patch.object(manager, "initialize", side_effect=fake_initialize) as mock_init,
        patch.object(manager, "load_model", new_callable=AsyncMock) as mock_load,
    ):
        chunks = []
        async for chunk in manager.generate("hello", ("voice", "/path/voice.pt")):
            chunks.append(chunk)

    mock_init.assert_called_once()
    mock_load.assert_called_once_with(manager._config.pytorch_kokoro_v1_file)
    assert len(chunks) == 1
    assert chunks[0] is audio_chunk


@pytest.mark.asyncio
async def test_generate_skips_reinit_when_backend_set():
    """generate() does not call initialize/load_model when backend already exists."""
    manager = ModelManager()
    mock_backend = MagicMock()
    audio_chunk = AudioChunk(np.zeros(10, dtype=np.float32))

    async def fake_generate(*args, **kwargs):
        yield audio_chunk

    mock_backend.generate = fake_generate
    manager._backend = mock_backend

    with (
        patch.object(manager, "initialize", new_callable=AsyncMock) as mock_init,
        patch.object(manager, "load_model", new_callable=AsyncMock) as mock_load,
    ):
        chunks = []
        async for chunk in manager.generate("hello", ("voice", "/path/voice.pt")):
            chunks.append(chunk)

    mock_init.assert_not_called()
    mock_load.assert_not_called()
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# POST /dev/unload endpoint tests
# ---------------------------------------------------------------------------


def _mock_service(manager=None):
    """Build a TTSService-shaped mock with the given model_manager."""
    service = MagicMock(spec=TTSService)
    service.model_manager = manager
    return service


def test_unload_endpoint_returns_200():
    mock_manager = AsyncMock()
    mock_manager.unload = AsyncMock()
    service = _mock_service(manager=mock_manager)

    with override_tts_service(service):
        response = client.post("/dev/unload")

    assert response.status_code == 200
    assert response.json() == {"status": "unloaded"}
    mock_manager.unload.assert_called_once()


def test_unload_endpoint_idempotent():
    """Calling /dev/unload twice both succeed — unload is a no-op when already clear."""
    mock_manager = AsyncMock()
    mock_manager.unload = AsyncMock()
    service = _mock_service(manager=mock_manager)

    with override_tts_service(service):
        r1 = client.post("/dev/unload")
        r2 = client.post("/dev/unload")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert mock_manager.unload.call_count == 2


def test_unload_endpoint_503_when_manager_none():
    """Returns 503 when model_manager has not been initialised on the service."""
    service = _mock_service(manager=None)

    with override_tts_service(service):
        response = client.post("/dev/unload")

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "Model manager not initialized"


def test_unload_endpoint_500_on_exception():
    """Returns 500 when manager.unload() raises unexpectedly."""
    mock_manager = AsyncMock()
    mock_manager.unload = AsyncMock(side_effect=RuntimeError("GPU exploded"))
    service = _mock_service(manager=mock_manager)

    with override_tts_service(service):
        response = client.post("/dev/unload")

    assert response.status_code == 500
    assert "GPU exploded" in response.json()["detail"]["error"]
