from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
import torch

from api.src.inference.kokoro_v1 import KokoroV1


@pytest.fixture
def kokoro_backend():
    """Create a KokoroV1 instance for testing."""
    return KokoroV1()


def test_initial_state(kokoro_backend):
    """Test initial state of KokoroV1."""
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    assert kokoro_backend._pipelines == {}  # Now using dict of pipelines
    # Device should be set based on settings
    assert kokoro_backend.device in ["cuda", "cpu"]


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.memory_allocated", return_value=5e9)
def test_memory_management(mock_memory, mock_cuda, kokoro_backend):
    """Test GPU memory management functions."""
    # Patch backend so it thinks we have cuda
    with patch.object(kokoro_backend, "_device", "cuda"):
        # Test memory check
        with patch("api.src.inference.kokoro_v1.model_config") as mock_config:
            mock_config.pytorch_gpu.memory_threshold = 4
            assert kokoro_backend._check_memory() == True

            mock_config.pytorch_gpu.memory_threshold = 6
            assert kokoro_backend._check_memory() == False


@patch("torch.cuda.empty_cache")
@patch("torch.cuda.synchronize")
def test_clear_memory(mock_sync, mock_clear, kokoro_backend):
    """Test memory clearing."""
    with patch.object(kokoro_backend, "_device", "cuda"):
        kokoro_backend._clear_memory()
        mock_clear.assert_called_once()
        mock_sync.assert_called_once()


@pytest.mark.asyncio
async def test_load_model_validation(kokoro_backend):
    """Test model loading validation."""
    with pytest.raises(RuntimeError, match="Failed to load Kokoro model"):
        await kokoro_backend.load_model("nonexistent_model.pth")


def test_unload_with_pipelines(kokoro_backend):
    """Test model unloading with multiple pipelines."""
    # Mock loaded state with multiple pipelines
    kokoro_backend._model = MagicMock()
    pipeline_a = MagicMock()
    pipeline_e = MagicMock()
    kokoro_backend._pipelines = {"a": pipeline_a, "e": pipeline_e}
    assert kokoro_backend.is_loaded

    # Test unload
    kokoro_backend.unload()
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    assert kokoro_backend._pipelines == {}  # All pipelines should be cleared


@pytest.mark.asyncio
async def test_generate_validation(kokoro_backend):
    """Test generation validation."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate("test", "voice"):
            pass


@pytest.mark.asyncio
async def test_generate_from_tokens_validation(kokoro_backend):
    """Test token generation validation."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate_from_tokens("test tokens", "voice"):
            pass


def test_get_pipeline_creates_new(kokoro_backend):
    """Test that _get_pipeline creates new pipeline for new language code."""
    # Mock loaded state
    kokoro_backend._model = MagicMock()

    # Mock KPipeline
    mock_pipeline = MagicMock()
    with patch(
        "api.src.inference.kokoro_v1.KPipeline", return_value=mock_pipeline
    ) as mock_kpipeline:
        # Get pipeline for Spanish
        pipeline_e = kokoro_backend._get_pipeline("e")

        # Should create new pipeline with correct params
        mock_kpipeline.assert_called_once_with(
            lang_code="e", model=kokoro_backend._model, device=kokoro_backend._device
        )
        assert pipeline_e == mock_pipeline
        assert kokoro_backend._pipelines["e"] == mock_pipeline


def test_get_pipeline_reuses_existing(kokoro_backend):
    """Test that _get_pipeline reuses existing pipeline for same language code."""
    # Mock loaded state
    kokoro_backend._model = MagicMock()

    # Mock KPipeline
    mock_pipeline = MagicMock()
    with patch(
        "api.src.inference.kokoro_v1.KPipeline", return_value=mock_pipeline
    ) as mock_kpipeline:
        # Get pipeline twice for same language
        pipeline1 = kokoro_backend._get_pipeline("e")
        pipeline2 = kokoro_backend._get_pipeline("e")

        # Should only create pipeline once
        mock_kpipeline.assert_called_once()
        assert pipeline1 == pipeline2
        assert kokoro_backend._pipelines["e"] == mock_pipeline


@pytest.mark.asyncio
async def test_generate_uses_correct_pipeline(kokoro_backend):
    """Test that generate uses correct pipeline for language code."""
    # Mock loaded state
    kokoro_backend._model = MagicMock()

    # Mock voice path handling
    with (
        patch("api.src.core.paths.load_voice_tensor") as mock_load_voice,
        patch("api.src.core.paths.save_voice_tensor"),
        patch("tempfile.gettempdir") as mock_tempdir,
    ):
        mock_load_voice.return_value = torch.ones(1)
        mock_tempdir.return_value = "/tmp"

        # Mock KPipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([])  # Empty generator for testing
        with patch("api.src.inference.kokoro_v1.KPipeline", return_value=mock_pipeline):
            # Generate with Spanish voice and explicit lang_code
            async for _ in kokoro_backend.generate("test", "ef_voice", lang_code="e"):
                pass

            # Should create pipeline with Spanish lang_code
            assert "e" in kokoro_backend._pipelines
            # Use ANY to match the temp file path since it's dynamic
            mock_pipeline.assert_called_with(
                "test",
                voice=ANY,  # Don't check exact path since it's dynamic
                speed=1.0,
                model=kokoro_backend._model,
            )
            # Verify the voice path is a temp file path
            call_args = mock_pipeline.call_args
            assert isinstance(call_args[1]["voice"], str)
            assert call_args[1]["voice"].startswith("/tmp/temp_voice_")
