"""Tests for model unload, auto-unload, lazy reload, and dev lifecycle endpoints."""

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.src.core.config import settings
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


@pytest.fixture(autouse=True)
def _enable_dev_unload(monkeypatch):
    """Enable the /dev/unload gate for the endpoint tests in this module."""
    monkeypatch.setattr(settings, "allow_dev_unload", True)
    monkeypatch.setattr(settings, "model_auto_unload_timeout_seconds", 0.0)
    monkeypatch.setattr(settings, "model_unload_strategy", "destroy")


# ---------------------------------------------------------------------------
# ModelManager unit tests
# ---------------------------------------------------------------------------


def test_manager_init_creates_lock():
    manager = ModelManager()
    assert isinstance(manager._lock, asyncio.Lock)
    assert manager._active_requests == 0


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
async def test_unload_move_to_cpu_keeps_backend(monkeypatch):
    monkeypatch.setattr(settings, "use_gpu", True)
    monkeypatch.setattr(settings, "model_unload_strategy", "move_to_cpu")

    manager = ModelManager()
    mock_backend = MagicMock()
    mock_backend.is_loaded = True
    mock_backend.is_cpu_cached = True
    manager._backend = mock_backend

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        await manager.unload()

    mock_backend.unload.assert_called_once_with(strategy="move_to_cpu")
    assert manager._backend is mock_backend


def test_move_to_cpu_strategy_warns_and_uses_destroy_without_gpu(monkeypatch):
    monkeypatch.setattr(settings, "use_gpu", False)
    monkeypatch.setattr(settings, "model_unload_strategy", "move_to_cpu")

    manager = ModelManager()

    with patch("api.src.inference.model_manager.logger.warning") as mock_warning:
        assert manager._unload_strategy() == "destroy"

    mock_warning.assert_called_once_with(
        "MODEL_UNLOAD_STRATEGY=move_to_cpu requires USE_GPU=true; using destroy"
    )


@pytest.mark.asyncio
async def test_unload_move_to_cpu_destroys_backend_without_gpu(monkeypatch):
    monkeypatch.setattr(settings, "use_gpu", False)
    monkeypatch.setattr(settings, "model_unload_strategy", "move_to_cpu")

    manager = ModelManager()
    mock_backend = MagicMock()
    mock_backend.is_loaded = True
    mock_backend.is_cpu_cached = True
    manager._backend = mock_backend

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        await manager.unload()

    mock_backend.unload.assert_called_once_with(strategy="destroy")
    assert manager._backend is None


@pytest.mark.asyncio
async def test_reload_restores_cpu_cached_backend(monkeypatch):
    monkeypatch.setattr(settings, "use_gpu", True)
    monkeypatch.setattr(settings, "model_unload_strategy", "move_to_cpu")

    manager = ModelManager()
    mock_backend = MagicMock()
    mock_backend.is_loaded = True
    mock_backend.is_cpu_cached = True
    manager._backend = mock_backend

    with (
        patch.object(manager, "initialize", new_callable=AsyncMock) as mock_init,
        patch.object(manager, "load_model", new_callable=AsyncMock) as mock_load,
    ):
        await manager.reload()

    mock_backend.restore_to_device.assert_called_once()
    mock_init.assert_not_called()
    mock_load.assert_not_called()


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
async def test_ensure_backend_serializes_concurrent_reloads():
    """Concurrent callers when _backend is None should trigger only one load cycle."""
    manager = ModelManager()
    assert manager._backend is None

    mock_backend = MagicMock()
    init_count = 0
    load_count = 0

    async def fake_initialize():
        nonlocal init_count
        init_count += 1
        await asyncio.sleep(0)  # yield so other tasks can attempt entry
        manager._backend = mock_backend

    async def fake_load(path):
        nonlocal load_count
        load_count += 1

    with (
        patch.object(manager, "initialize", side_effect=fake_initialize),
        patch.object(manager, "load_model", side_effect=fake_load),
    ):
        await asyncio.gather(*[manager.ensure_backend() for _ in range(5)])

    assert init_count == 1
    assert load_count == 1


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


@pytest.mark.asyncio
async def test_generate_schedules_idle_unload_when_enabled(monkeypatch):
    """Finished generation schedules model unload after the configured idle period."""
    monkeypatch.setattr(settings, "model_auto_unload_timeout_seconds", 0.01)

    manager = ModelManager()
    mock_backend = MagicMock()
    audio_chunk = AudioChunk(np.zeros(10, dtype=np.float32))

    async def fake_generate(*args, **kwargs):
        yield audio_chunk

    mock_backend.generate = fake_generate
    manager._backend = mock_backend

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        chunks = []
        async for chunk in manager.generate("hello", ("voice", "/path/voice.pt")):
            chunks.append(chunk)
        await asyncio.sleep(0.03)

    assert len(chunks) == 1
    mock_backend.unload.assert_called_once()
    assert manager._backend is None


@pytest.mark.asyncio
async def test_idle_auto_unload_log_includes_configured_timeout(monkeypatch):
    monkeypatch.setattr(settings, "model_auto_unload_timeout_seconds", 30.0)

    manager = ModelManager()
    mock_backend = MagicMock()
    manager._backend = mock_backend
    manager._last_used_at = 0.0

    with (
        patch("api.src.inference.model_manager.time.monotonic", return_value=31.0),
        patch("api.src.inference.model_manager.torch") as mock_torch,
        patch("api.src.inference.model_manager.logger.info") as mock_log_info,
    ):
        mock_torch.cuda.is_available.return_value = False
        await manager._idle_unload_after(0)

    mock_backend.unload.assert_called_once()
    mock_log_info.assert_any_call("Model auto-unloaded after idle timeout of 30s")


@pytest.mark.asyncio
async def test_load_model_schedules_idle_unload_when_enabled(monkeypatch):
    """Startup-style model loads also schedule unload without waiting for traffic."""
    monkeypatch.setattr(settings, "model_auto_unload_timeout_seconds", 0.01)

    manager = ModelManager()
    mock_backend = MagicMock()
    mock_backend.load_model = AsyncMock()
    manager._backend = mock_backend

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        await manager.load_model("/path/model.pt")
        await asyncio.sleep(0.03)

    mock_backend.load_model.assert_called_once_with("/path/model.pt")
    mock_backend.unload.assert_called_once()
    assert manager._backend is None


@pytest.mark.asyncio
async def test_load_model_does_not_schedule_idle_unload_during_active_request(
    monkeypatch,
):
    """Lazy loads during generation wait for request completion before scheduling."""
    monkeypatch.setattr(settings, "model_auto_unload_timeout_seconds", 0.01)

    manager = ModelManager()
    mock_backend = MagicMock()
    mock_backend.load_model = AsyncMock()
    manager._backend = mock_backend
    manager._active_requests = 1

    await manager.load_model("/path/model.pt")
    await asyncio.sleep(0.03)

    mock_backend.load_model.assert_called_once_with("/path/model.pt")
    mock_backend.unload.assert_not_called()
    assert manager._backend is mock_backend
    assert manager._idle_unload_task is None


@pytest.mark.asyncio
async def test_active_request_blocks_idle_unload(monkeypatch):
    """The idle timer does not unload while generation is still active."""
    monkeypatch.setattr(settings, "model_auto_unload_timeout_seconds", 0.01)

    manager = ModelManager()
    mock_backend = MagicMock()

    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(0.03)
        yield AudioChunk(np.zeros(10, dtype=np.float32))

    mock_backend.generate = slow_generate
    manager._backend = mock_backend

    with patch("api.src.inference.model_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        chunks = []
        async for chunk in manager.generate("hello", ("voice", "/path/voice.pt")):
            assert manager._active_requests == 1
            mock_backend.unload.assert_not_called()
            chunks.append(chunk)

    assert len(chunks) == 1
    assert manager._active_requests == 0
    assert manager._idle_unload_task is not None
    manager._cancel_idle_unload_timer()


def test_status_reports_model_lifecycle_state(monkeypatch):
    monkeypatch.setattr(settings, "model_auto_unload_timeout_seconds", 30.0)
    manager = ModelManager()
    manager._backend = MagicMock()
    manager._device = "cuda"
    manager._last_used_at = 10.0

    with patch("api.src.inference.model_manager.time.monotonic", return_value=15.0):
        status = manager.status()

    assert status["backend"] == "kokoro_v1"
    assert status["device"] == "cuda"
    assert status["loaded"] is True
    assert status["active_requests"] == 0
    assert status["unload_strategy"] == "destroy"
    assert status["cpu_cached"] is False
    assert status["auto_unload_enabled"] is True
    assert status["auto_unload_timeout_seconds"] == 30.0
    assert status["idle_seconds"] == 5.0
    assert status["seconds_until_auto_unload"] == 25.0


def test_status_reports_cpu_cached_state(monkeypatch):
    monkeypatch.setattr(settings, "use_gpu", True)
    monkeypatch.setattr(settings, "model_unload_strategy", "move_to_cpu")
    monkeypatch.setattr(settings, "model_auto_unload_timeout_seconds", 30.0)

    manager = ModelManager()
    mock_backend = MagicMock()
    mock_backend.is_loaded = True
    mock_backend.is_cpu_cached = True
    manager._backend = mock_backend
    manager._last_used_at = 10.0

    with patch("api.src.inference.model_manager.time.monotonic", return_value=15.0):
        status = manager.status()

    assert status["loaded"] is False
    assert status["cpu_cached"] is True
    assert status["unload_strategy"] == "move_to_cpu"
    assert status["seconds_until_auto_unload"] is None


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


def test_unload_endpoint_403_when_disabled(monkeypatch):
    """Returns 403 without touching the model when the gate is off."""
    monkeypatch.setattr(settings, "allow_dev_unload", False)
    mock_manager = AsyncMock()
    mock_manager.unload = AsyncMock()
    service = _mock_service(manager=mock_manager)

    with override_tts_service(service):
        response = client.post("/dev/unload")

    assert response.status_code == 403
    mock_manager.unload.assert_not_called()


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


def test_reload_endpoint_returns_status():
    mock_manager = MagicMock()
    mock_manager.reload = AsyncMock()
    mock_manager.status.return_value = {"loaded": True}
    service = _mock_service(manager=mock_manager)

    with override_tts_service(service):
        response = client.post("/dev/reload")

    assert response.status_code == 200
    assert response.json() == {"status": "loaded", "model": {"loaded": True}}
    mock_manager.reload.assert_called_once()


def test_model_status_endpoint_returns_status():
    mock_manager = MagicMock()
    mock_manager.status.return_value = {"loaded": False}
    service = _mock_service(manager=mock_manager)

    with override_tts_service(service):
        response = client.get("/dev/model")

    assert response.status_code == 200
    assert response.json() == {"loaded": False}
