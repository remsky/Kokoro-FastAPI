import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from api.src.inference.base import AudioChunk
from api.src.services.tts_service import TTSService
from api.src.structures.schemas import WordTimestamp


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
        return await TTSService.create()

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

        service = await TTSService.create()
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

        service = await TTSService.create()
        name, path = await service.get_voices_path("voice1")
        assert name == "voice1"
        assert path == "/path/to/voice1.pt"
        voice_manager.get_voice_path.assert_called_once_with("voice1")


@pytest.mark.asyncio
async def test_get_voice_path_single_with_weight_normalized():
    """A weighted single voice loads by bare name when normalization is on."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()
    voice_manager.get_voice_path.return_value = "/path/to/voice1.pt"

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
        patch("api.src.services.tts_service.settings") as mock_settings,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager
        mock_settings.voice_weight_normalization = True

        service = await TTSService.create()
        name, path = await service.get_voices_path("voice1(2)")
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

        service = await TTSService.create()
        name, path = await service.get_voices_path("voice1+voice2")
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

        service = await TTSService.create()
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

        service = await TTSService.create()
        service.get_voices_path = AsyncMock(
            side_effect=lambda voice: (voice, f"/path/to/{voice}.pt")
        )

        text = "[voice:af_bella] One. [voice:bm_george] Two. [voice:af_bella] Three."
        speakers = []
        async for (
            voice_name,
            _path,
            _lang,
            _rate,
            _text,
            _tokens,
            _pause,
        ) in service._split_multi_voice(
            text, "af_heart", None, None, allow_voice_tags=True
        ):
            speakers.append(voice_name)

        assert speakers == ["af_bella", "bm_george", "af_bella"]
        assert service.get_voices_path.await_count == 2


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

        service = await TTSService.create()
        service.get_voices_path = AsyncMock(
            side_effect=lambda voice: (voice, f"/path/to/{voice}.pt")
        )

        text = "[voice:af_bella] Hello. [voice:bm_george] Hello."
        langs = [
            lang
            async for _name, _path, lang, _rate, _text, _tokens, _pause in (
                service._split_multi_voice(
                    text, "af_heart", None, None, allow_voice_tags=True
                )
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

        service = await TTSService.create()
        service.get_voices_path = AsyncMock(
            side_effect=lambda voice: (voice, f"/path/to/{voice}.pt")
        )

        text = "[voice:af_bella] Hello. [voice:bm_george] Hello."
        langs = [
            lang
            async for _name, _path, lang, _rate, _text, _tokens, _pause in (
                service._split_multi_voice(
                    text, "af_heart", "e", None, allow_voice_tags=True
                )
            )
        ]

        assert langs == ["e", "e"]


async def _stubbed_service():
    """A service whose inference yields one 0.1s chunk timestamped with the chunk's first word."""
    model_manager = AsyncMock()
    model_manager.get_backend = MagicMock()

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = AsyncMock()
        service = await TTSService.create()

    service.get_voices_path = AsyncMock(
        side_effect=lambda voice: (voice, f"/path/to/{voice}.pt")
    )

    def fake_process_chunk(text, *args, **kwargs):
        async def _gen():
            if not text:
                return
            yield AudioChunk(
                audio=np.zeros(2400, dtype=np.int16),
                word_timestamps=[
                    WordTimestamp(word=text.split()[0], start_time=0.0, end_time=0.05)
                ],
            )

        return _gen()

    service._process_chunk = fake_process_chunk
    return service


async def _stamped_words(service, allow_voice_tags):
    text = "[voice:af_bella] One. [voice:bm_george] Two."
    return [
        (t.word, t.voice, t.start_time)
        async for chunk in service.generate_audio_stream(
            text,
            "af_heart",
            MagicMock(),
            return_timestamps=True,
            allow_voice_tags=allow_voice_tags,
        )
        for t in chunk.word_timestamps
    ]


@pytest.mark.asyncio
async def test_timestamps_carry_speaker_when_tags_allowed():
    """Each word names the voice that said it, and offsets still run across the switch."""
    service = await _stubbed_service()

    stamped = await _stamped_words(service, allow_voice_tags=True)

    assert [(word, voice) for word, voice, _ in stamped] == [
        ("One.", "af_bella"),
        ("Two.", "bm_george"),
    ]
    assert [start for _, _, start in stamped] == [0.0, 0.1]


@pytest.mark.asyncio
async def test_timestamps_omit_speaker_by_default():
    """Without the opt in the field stays unset, so existing callers see the same shape."""
    service = await _stubbed_service()

    stamped = await _stamped_words(service, allow_voice_tags=False)

    assert stamped and all(voice is None for _, voice, _ in stamped)


@pytest.mark.asyncio
async def test_timings_collect_chunks_and_pauses():
    """Spoken chunks land as {text, start, end} from sample counts, pauses as empty-text gaps."""
    service = await _stubbed_service()

    timings = []
    async for _ in service.generate_audio_stream(
        "One. [pause:0.5s] Two.",
        "af_heart",
        MagicMock(),
        output_format=None,
        timings=timings,
    ):
        pass

    assert [(t["text"].strip(), t["start"], t["end"]) for t in timings] == [
        ("One.", 0.0, 0.1),
        ("", 0.1, 0.6),
        ("Two.", 0.6, 0.7),
    ]


@pytest.mark.asyncio
async def test_timings_carry_speaker_when_tags_allowed():
    """Chunk entries name their voice with the opt in, and never without it."""
    service = await _stubbed_service()

    tagged = []
    async for _ in service.generate_audio_stream(
        "[voice:af_bella] One. [voice:bm_george] Two.",
        "af_heart",
        MagicMock(),
        output_format=None,
        allow_voice_tags=True,
        timings=tagged,
    ):
        pass
    assert [t.get("voice") for t in tagged] == ["af_bella", "bm_george"]

    plain = []
    async for _ in service.generate_audio_stream(
        "One. Two.",
        "af_heart",
        MagicMock(),
        output_format=None,
        timings=plain,
    ):
        pass
    assert plain and all("voice" not in t for t in plain)


@pytest.mark.asyncio
async def test_rate_tags_multiply_request_speed():
    """Segment rates scale the request speed per chunk, clamped to the speed bounds."""
    service = await _stubbed_service()

    speeds = []
    original = service._process_chunk

    def capture(text, tokens, voice_name, voice_path, speed, *args, **kwargs):
        if text:  # skip the empty stream-finalizer chunk
            speeds.append(speed)
        return original(text, tokens, voice_name, voice_path, speed, *args, **kwargs)

    service._process_chunk = capture

    async for _ in service.generate_audio_stream(
        "One. [rate:1.5] Two. [rate:4.0] Three.",
        "af_heart",
        MagicMock(),
        speed=2.0,
        output_format=None,
        allow_voice_tags=True,
    ):
        pass

    assert speeds == [2.0, 3.0, 4.0]


@pytest.mark.asyncio
async def test_generate_audio_joins_encoded_chunks():
    """Non-streaming accumulates encoded bytes as they arrive, never raw PCM."""
    service = await _stubbed_service()

    async def fake_stream(*args, **kwargs):
        assert kwargs["output_format"] == "mp3"
        for part in (b"id3", b"frame1", b"", b"frame2"):
            yield AudioChunk(
                np.array([], dtype=np.int16),
                word_timestamps=[
                    WordTimestamp(word=part.decode(), start_time=0.0, end_time=0.05)
                ]
                if part
                else [],
                output=part,
            )

    service.generate_audio_stream = fake_stream

    result = await service.generate_audio(
        "hi", "af_heart", MagicMock(), output_format="mp3"
    )

    assert result.output == b"id3frame1frame2"
    assert [t.word for t in result.word_timestamps] == ["id3", "frame1", "frame2"]


@pytest.mark.asyncio
async def test_pause_budget_survives_voice_tag_segmentation():
    """Per-segment splitting must not reset the request's pause budget."""
    service = await _stubbed_service()

    with pytest.raises(ValueError, match="exceeds"):
        async for _ in service.generate_audio_stream(
            "[voice:af_bella] [pause:60s] [voice:bm_george] [pause:60s] " * 3,
            "af_heart",
            MagicMock(),
            output_format=None,
            allow_voice_tags=True,
        ):
            pass


@pytest.mark.asyncio
async def test_generate_audio_with_nothing_speakable_raises():
    """Tags-only input is a ValueError the routers map to a 400, not an IndexError"""
    service = await _stubbed_service()

    with pytest.raises(ValueError, match="no speakable text"):
        await service.generate_audio(
            "[voice:af_bella] [voice:bm_george]",
            "af_heart",
            MagicMock(),
            allow_voice_tags=True,
        )
