import tempfile
from pathlib import Path
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
    assert kokoro_backend.device in ["cuda", "cpu", "mps"]


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


def test_unload_with_pipelines(kokoro_backend):
    """Test model unloading with multiple pipelines."""
    # Mock loaded state with multiple pipelines
    kokoro_backend._model = MagicMock()
    pipeline_a = MagicMock()
    pipeline_e = MagicMock()
    kokoro_backend._pipelines = {"a": pipeline_a, "e": pipeline_e}
    kokoro_backend._voice_cache = {"af_heart.pt:cpu": MagicMock()}
    assert kokoro_backend.is_loaded

    # Test unload
    kokoro_backend.unload()
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    assert kokoro_backend._pipelines == {}  # All pipelines should be cleared
    assert kokoro_backend._voice_cache == {}  # Voice tensors should be released


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
    ):
        mock_load_voice.return_value = torch.ones(1)

        # Mock KPipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([])  # Empty generator for testing
        with patch("api.src.inference.kokoro_v1.KPipeline", return_value=mock_pipeline):
            # Generate with Spanish voice and explicit lang_code
            async for _ in kokoro_backend.generate("test", "ef_voice", lang_code="e"):
                pass

            # Should create pipeline with Spanish lang_code and call it.
            assert "e" in kokoro_backend._pipelines
            mock_pipeline.assert_called_with(
                "test",
                voice=ANY,
                speed=1.0,
                model=kokoro_backend._model,
            )
            # Voice was staged to a temp file with the expected basename in
            # the OS temp dir (cross-platform: don't string-match separators).
            voice_arg = Path(mock_pipeline.call_args[1]["voice"])
            assert voice_arg.name == "temp_voice_ef_voice"
            assert voice_arg.parent == Path(tempfile.gettempdir())


def test_espeak_word_timestamps_basic():
    """Word times derived from pred_dur for an espeak (non-English) result."""
    from api.src.inference.kokoro_v1 import _espeak_word_timestamps

    # phonemes "ab cd": BOS=8, a=4, b=4, space=8, c=4, d=4, EOS=0
    # scale is 2/80 s per unit -> BOS 0.2s, each char 0.1s, space 0.2s
    pred_dur = torch.tensor([8, 4, 4, 8, 4, 4, 0])
    ts = _espeak_word_timestamps("hola mundo", "ab cd", pred_dur)

    assert [t.word for t in ts] == ["hola", "mundo"]
    assert ts[0].start_time == pytest.approx(0.2)
    assert ts[0].end_time == pytest.approx(0.4)
    assert ts[1].start_time == pytest.approx(0.6)
    assert ts[1].end_time == pytest.approx(0.8)


def test_espeak_word_timestamps_expansion():
    """Words espeak expands (numbers) are merged back via per-word g2p."""
    from api.src.inference.kokoro_v1 import _espeak_word_timestamps

    # Three text words but four phoneme groups ("1863" speaks as two words).
    phonemes = "a bb cc d"
    pred_dur = torch.tensor([0, 4] + [8, 4, 4] + [8, 4, 4] + [8, 4] + [0])

    def g2p(word):
        return ("bb cc" if word == "1863" else "x", None)

    ts = _espeak_word_timestamps("en 1863 el", phonemes, pred_dur, g2p=g2p)

    assert [t.word for t in ts] == ["en", "1863", "el"]
    # "1863" spans both expanded groups.
    assert ts[1].start_time == pytest.approx(0.3)
    assert ts[1].end_time == pytest.approx(0.9)
    # Whole-string timing stays monotonic and inside the clip.
    assert ts[0].end_time <= ts[1].start_time <= ts[2].start_time


def test_espeak_word_timestamps_unreconcilable():
    """Group/word count mismatch that g2p can't explain yields None."""
    from api.src.inference.kokoro_v1 import _espeak_word_timestamps

    phonemes = "a b c"
    pred_dur = torch.tensor([0, 4, 8, 4, 8, 4, 0])

    # g2p says every word is a single group: 2 != 3 -> give up.
    ts = _espeak_word_timestamps("uno dos", phonemes, pred_dur, g2p=lambda w: (w, None))
    assert ts is None

    # No g2p available -> also None.
    assert _espeak_word_timestamps("uno dos", phonemes, pred_dur) is None


def test_espeak_word_timestamps_degenerate_inputs():
    """Missing durations or empty text return None instead of raising."""
    from api.src.inference.kokoro_v1 import _espeak_word_timestamps

    assert _espeak_word_timestamps("hola", "abc", None) is None
    assert _espeak_word_timestamps("hola", "", torch.tensor([0, 0])) is None
    assert _espeak_word_timestamps("", "abc", torch.tensor([0, 4, 4, 4, 0])) is None
    # pred_dur too short for the phoneme string.
    assert _espeak_word_timestamps("hola", "abcdef", torch.tensor([0, 4])) is None
