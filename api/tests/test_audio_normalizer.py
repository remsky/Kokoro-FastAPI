"""Tests for AudioNormalizer"""

import numpy as np
import pytest
from api.src.services.audio import AudioNormalizer

@pytest.fixture
def normalizer():
    """Create an AudioNormalizer instance"""
    return AudioNormalizer()

@pytest.fixture
def silent_audio():
    """Generate silent audio data"""
    return np.zeros(24000, dtype=np.int16)  # 1 second of silence

@pytest.fixture
def speech_audio():
    """Generate audio data with speech-like content"""
    # Create 1 second of audio with speech in the middle
    audio = np.zeros(24000, dtype=np.int16)
    # Add speech-like content from 0.25s to 0.75s (leaving silence at start/end)
    speech_start = 6000  # 0.25s * 24000
    speech_end = 18000   # 0.75s * 24000
    # Generate non-zero values for speech section
    audio[speech_start:speech_end] = np.random.randint(-32768//2, 32767//2, speech_end-speech_start, dtype=np.int16)
    return audio

def test_find_first_last_non_silent_all_silent(normalizer, silent_audio):
    """Test silence detection with completely silent audio"""
    start, end = normalizer.find_first_last_non_silent(silent_audio,"",1)
    assert start == 0
    assert end == len(silent_audio)

def test_find_first_last_non_silent_with_speech(normalizer, speech_audio):
    """Test silence detection with audio containing speech"""
    start, end = normalizer.find_first_last_non_silent(speech_audio,"",1)
    
    # Should detect speech section with padding
    # Start should be before 0.25s (with 50ms padding)
    assert start < 6000
    # End should be after 0.75s (with dynamic padding)
    assert end > 18000
    # But shouldn't extend beyond audio length
    assert end <= len(speech_audio)

def test_normalize_streaming_chunks(normalizer):
    """Test normalizing streaming audio chunks"""
    # Create three 100ms chunks
    chunk_samples = 2400  # 100ms at 24kHz
    chunks = []
    
    # First chunk: silence then speech
    chunk1 = np.zeros(chunk_samples, dtype=np.float32)
    chunk1[1200:] = np.random.random(1200) * 2 - 1  # Speech in second half
    chunks.append(chunk1)
    
    # Second chunk: all speech
    chunk2 = (np.random.random(chunk_samples) * 2 - 1).astype(np.float32)
    chunks.append(chunk2)
    
    # Third chunk: speech then silence
    chunk3 = np.zeros(chunk_samples, dtype=np.float32)
    chunk3[:1200] = np.random.random(1200) * 2 - 1  # Speech in first half
    chunks.append(chunk3)
    
    # Process chunks
    results = []
    for i, chunk in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        normalized = normalizer.normalize(chunk, is_last_chunk=is_last)
        results.append(normalized)
        
    # Verify results
    # First chunk should trim silence from start but keep end for continuity
    assert len(results[0]) < len(chunk1)
    # Middle chunk should be similar length to input
    assert abs(len(results[1]) - len(chunk2)) < 100
    # Last chunk should trim silence from end
    assert len(results[2]) < len(chunk3)

def test_normalize_amplitude(normalizer):
    """Test audio amplitude normalization"""
    # Create audio with values outside int16 range
    audio = np.random.random(1000) * 1e5
    
    result = normalizer.normalize(audio)
    
    # Check result is within int16 range
    assert np.max(np.abs(result)) <= 32767
    assert result.dtype == np.int16

def test_padding_behavior(normalizer, speech_audio):
    """Test start and end padding behavior"""
    result = normalizer.normalize(speech_audio)
    
    # Find actual speech content in result (non-zero values)
    non_zero = np.nonzero(result)[0]
    first_speech = non_zero[0]
    last_speech = non_zero[-1]
    
    # Verify we have some padding before first speech
    # Should be close to 50ms (1200 samples at 24kHz)
    start_padding = first_speech
    assert 0 < start_padding <= 1200
    
    # Verify we have some padding after last speech
    # Should be close to dynamic_gap_trim_padding_ms - 50ms
    end_padding = len(result) - last_speech - 1
    expected_end_padding = int((410 - 50) * 24000 / 1000)  # ~8640 samples
    padding_tolerance = 100  # Allow some variation
    assert abs(end_padding - expected_end_padding) < padding_tolerance
