"""Tests for AudioService"""

from unittest.mock import patch

import numpy as np
import pytest

from api.src.services.audio import AudioService, AudioNormalizer


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


def test_convert_to_wav(sample_audio):
    """Test converting to WAV format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "wav")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_mp3(sample_audio):
    """Test converting to MP3 format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "mp3")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_opus(sample_audio):
    """Test converting to Opus format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "opus")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_flac(sample_audio):
    """Test converting to FLAC format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "flac")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_aac_raises_error(sample_audio):
    """Test that converting to AAC raises an error"""
    audio_data, sample_rate = sample_audio
    with pytest.raises(
        ValueError,
        match="Failed to convert audio to aac: Format aac not currently supported. Supported formats are: wav, mp3, opus, flac, pcm.",
    ):
        AudioService.convert_audio(audio_data, sample_rate, "aac")


def test_convert_to_pcm(sample_audio):
    """Test converting to PCM format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "pcm")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_invalid_format_raises_error(sample_audio):
    """Test that converting to an invalid format raises an error"""
    audio_data, sample_rate = sample_audio
    with pytest.raises(ValueError, match="Format invalid not supported"):
        AudioService.convert_audio(audio_data, sample_rate, "invalid")


def test_normalization_wav(sample_audio):
    """Test that WAV output is properly normalized to int16 range"""
    audio_data, sample_rate = sample_audio
    # Create audio data outside int16 range
    large_audio = audio_data * 1e5
    result = AudioService.convert_audio(large_audio, sample_rate, "wav")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_normalization_pcm(sample_audio):
    """Test that PCM output is properly normalized to int16 range"""
    audio_data, sample_rate = sample_audio
    # Create audio data outside int16 range
    large_audio = audio_data * 1e5
    result = AudioService.convert_audio(large_audio, sample_rate, "pcm")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_invalid_audio_data():
    """Test handling of invalid audio data"""
    invalid_audio = np.array([])  # Empty array
    sample_rate = 24000
    with pytest.raises(ValueError):
        AudioService.convert_audio(invalid_audio, sample_rate, "wav")


def test_different_sample_rates(sample_audio):
    """Test converting audio with different sample rates"""
    audio_data, _ = sample_audio
    sample_rates = [8000, 16000, 44100, 48000]

    for rate in sample_rates:
        result = AudioService.convert_audio(audio_data, rate, "wav")
        assert isinstance(result, bytes)
        assert len(result) > 0


def test_buffer_position_after_conversion(sample_audio):
    """Test that buffer position is reset after writing"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "wav")
    # Convert again to ensure buffer was properly reset
    result2 = AudioService.convert_audio(audio_data, sample_rate, "wav")
    assert len(result) == len(result2)
