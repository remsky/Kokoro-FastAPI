"""Tests for AudioService"""

from unittest.mock import patch

import numpy as np
import pytest

from api.src.services.audio import AudioNormalizer, AudioService
from api.src.inference.base import AudioChunk

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
    # Write and finalize in one step for WAV
    result, audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), sample_rate, "wav", is_first_chunk=True, is_last_chunk=True
    )
    assert isinstance(result, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result) > 0
    # Check WAV header
    assert result.startswith(b"RIFF")
    assert b"WAVE" in result[:12]


@pytest.mark.asyncio
async def test_convert_to_mp3(sample_audio):
    """Test converting to MP3 format"""
    audio_data, sample_rate = sample_audio
    result, audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), sample_rate, "mp3"
    )
    assert isinstance(result, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result) > 0
    # Check MP3 header (ID3 or MPEG frame sync)
    assert result.startswith(b"ID3") or result.startswith(b"\xff\xfb")


@pytest.mark.asyncio
async def test_convert_to_opus(sample_audio):
    """Test converting to Opus format"""
    audio_data, sample_rate = sample_audio
    result, audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), sample_rate, "opus"
    )
    assert isinstance(result, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result) > 0
    # Check OGG header
    assert result.startswith(b"OggS")


@pytest.mark.asyncio
async def test_convert_to_flac(sample_audio):
    """Test converting to FLAC format"""
    audio_data, sample_rate = sample_audio
    result, audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), sample_rate, "flac"
    )
    assert isinstance(result, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result) > 0
    # Check FLAC header
    assert result.startswith(b"fLaC")


@pytest.mark.asyncio
async def test_convert_to_aac(sample_audio):
    """Test converting to AAC format"""
    audio_data, sample_rate = sample_audio
    result, audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), sample_rate, "aac"
    )
    assert isinstance(result, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result) > 0
    # Check ADTS header (AAC)
    assert result.startswith(b"\xff\xf0") or result.startswith(b"\xff\xf1")


@pytest.mark.asyncio
async def test_convert_to_pcm(sample_audio):
    """Test converting to PCM format"""
    audio_data, sample_rate = sample_audio
    result, audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), sample_rate, "pcm"
    )
    assert isinstance(result, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result) > 0
    # PCM is raw bytes, so no header to check


@pytest.mark.asyncio
async def test_convert_to_invalid_format_raises_error(sample_audio):
    """Test that converting to an invalid format raises an error"""
    audio_data, sample_rate = sample_audio
    with pytest.raises(ValueError, match="Format invalid not supported"):
        await AudioService.convert_audio(audio_data, sample_rate, "invalid")


@pytest.mark.asyncio
async def test_normalization_wav(sample_audio):
    """Test that WAV output is properly normalized to int16 range"""
    audio_data, sample_rate = sample_audio
    # Create audio data outside int16 range
    large_audio = audio_data * 1e5
    # Write and finalize in one step for WAV
    result, audio_chunk = await AudioService.convert_audio(
        AudioChunk(large_audio), sample_rate, "wav", is_first_chunk=True, is_last_chunk=True
    )
    assert isinstance(result, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_normalization_pcm(sample_audio):
    """Test that PCM output is properly normalized to int16 range"""
    audio_data, sample_rate = sample_audio
    # Create audio data outside int16 range
    large_audio = audio_data * 1e5
    result, audio_chunk = await AudioService.convert_audio(
        AudioChunk(large_audio), sample_rate, "pcm"
    )
    assert isinstance(result, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_invalid_audio_data():
    """Test handling of invalid audio data"""
    invalid_audio = np.array([])  # Empty array
    sample_rate = 24000
    with pytest.raises(ValueError):
        await AudioService.convert_audio(invalid_audio, sample_rate, "wav")


@pytest.mark.asyncio
async def test_different_sample_rates(sample_audio):
    """Test converting audio with different sample rates"""
    audio_data, _ = sample_audio
    sample_rates = [8000, 16000, 44100, 48000]

    for rate in sample_rates:
        result, audio_chunk = await AudioService.convert_audio(
            AudioChunk(audio_data), rate, "wav", is_first_chunk=True, is_last_chunk=True
        )
        assert isinstance(result, bytes)
        assert isinstance(audio_chunk, AudioChunk)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_buffer_position_after_conversion(sample_audio):
    """Test that buffer position is reset after writing"""
    audio_data, sample_rate = sample_audio
    # Write and finalize in one step for first conversion
    result1, audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), sample_rate, "wav", is_first_chunk=True, is_last_chunk=True
    )
    assert isinstance(result1, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    # Convert again to ensure buffer was properly reset
    result2, audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), sample_rate, "wav", is_first_chunk=True, is_last_chunk=True
    )
    assert isinstance(result2, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(result1) == len(result2)
