"""Tests for AudioService"""

from unittest.mock import patch

import numpy as np
import pytest

from api.src.inference.base import AudioChunk
from api.src.services.audio import AudioNormalizer, AudioService
from api.src.services.streaming_audio_writer import StreamingAudioWriter


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests"""
    with patch("api.src.services.audio.settings") as mock_settings:
        mock_settings.gap_trim_ms = 250
        yield mock_settings


@pytest.fixture
def sample_audio():
    """Generate a simple sine wave for testing"""
    sample_rate = 24000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    return np.sin(2 * np.pi * frequency * t).astype(np.float32), sample_rate


@pytest.mark.asyncio
async def test_convert_to_wav(sample_audio):
    """Test converting to WAV format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("wav", sample_rate=24000)
    # Write and finalize in one step for WAV
    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "wav", writer, is_last_chunk=False
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check WAV header
    assert audio_chunk.output.startswith(b"RIFF")
    assert b"WAVE" in audio_chunk.output[:12]


@pytest.mark.asyncio
async def test_convert_to_mp3(sample_audio):
    """Test converting to MP3 format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("mp3", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "mp3", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check MP3 header (ID3 or MPEG frame sync)
    assert audio_chunk.output.startswith(b"ID3") or audio_chunk.output.startswith(
        b"\xff\xfb"
    )


@pytest.mark.asyncio
async def test_convert_to_opus(sample_audio):
    """Test converting to Opus format"""

    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("opus", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "opus", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check OGG header
    assert audio_chunk.output.startswith(b"OggS")


@pytest.mark.asyncio
async def test_convert_to_flac(sample_audio):
    """Test converting to FLAC format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("flac", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "flac", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check FLAC header
    assert audio_chunk.output.startswith(b"fLaC")


@pytest.mark.asyncio
async def test_convert_to_aac(sample_audio):
    """Test converting to M4A format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("aac", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "aac", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check ADTS header (AAC)
    assert audio_chunk.output.startswith(b"\xff\xf0") or audio_chunk.output.startswith(
        b"\xff\xf1"
    )


@pytest.mark.asyncio
async def test_convert_to_pcm(sample_audio):
    """Test converting to PCM format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("pcm", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "pcm", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # PCM is raw bytes, so no header to check


@pytest.mark.asyncio
async def test_convert_to_invalid_format_raises_error(sample_audio):
    """Test that converting to an invalid format raises an error"""
    # audio_data, sample_rate = sample_audio
    with pytest.raises(ValueError, match="Unsupported format: invalid"):
        writer = StreamingAudioWriter("invalid", sample_rate=24000)


@pytest.mark.asyncio
async def test_normalization_wav(sample_audio):
    """Test that WAV output is properly normalized to int16 range"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("wav", sample_rate=24000)

    # Create audio data outside int16 range
    large_audio = audio_data * 1e5
    # Write and finalize in one step for WAV
    audio_chunk = await AudioService.convert_audio(
        AudioChunk(large_audio), "wav", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0


@pytest.mark.asyncio
async def test_normalization_pcm(sample_audio):
    """Test that PCM output is properly normalized to int16 range"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("pcm", sample_rate=24000)

    # Create audio data outside int16 range
    large_audio = audio_data * 1e5
    audio_chunk = await AudioService.convert_audio(
        AudioChunk(large_audio), "pcm", writer
    )
    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0


@pytest.mark.asyncio
async def test_invalid_audio_data():
    """Test handling of invalid audio data"""
    invalid_audio = np.array([])  # Empty array
    sample_rate = 24000

    writer = StreamingAudioWriter("wav", sample_rate=24000)

    with pytest.raises(ValueError):
        await AudioService.convert_audio(invalid_audio, sample_rate, "wav", writer)


@pytest.mark.asyncio
async def test_different_sample_rates(sample_audio):
    """Test converting audio with different sample rates"""
    audio_data, _ = sample_audio
    sample_rates = [8000, 16000, 44100, 48000]

    for rate in sample_rates:
        writer = StreamingAudioWriter("wav", sample_rate=rate)

        audio_chunk = await AudioService.convert_audio(
            AudioChunk(audio_data), "wav", writer
        )

        writer.close()

        assert isinstance(audio_chunk.output, bytes)
        assert isinstance(audio_chunk, AudioChunk)
        assert len(audio_chunk.output) > 0


@pytest.mark.asyncio
async def test_buffer_position_after_conversion(sample_audio):
    """Test that buffer position is reset after writing"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("wav", sample_rate=24000)

    # Write and finalize in one step for first conversion
    audio_chunk1 = await AudioService.convert_audio(
        AudioChunk(audio_data), "wav", writer, is_last_chunk=True
    )
    assert isinstance(audio_chunk1.output, bytes)
    assert isinstance(audio_chunk1, AudioChunk)
    # Convert again to ensure buffer was properly reset

    writer = StreamingAudioWriter("wav", sample_rate=24000)

    audio_chunk2 = await AudioService.convert_audio(
        AudioChunk(audio_data), "wav", writer, is_last_chunk=True
    )
    assert isinstance(audio_chunk2.output, bytes)
    assert isinstance(audio_chunk2, AudioChunk)
    assert len(audio_chunk1.output) == len(audio_chunk2.output)
