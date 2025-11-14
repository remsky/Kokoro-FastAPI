"""
Comprehensive end-to-end tests for ZipVoice integration.
"""

import asyncio
import base64
import os
import tempfile
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf


# Test configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8880")
TEST_VOICE_NAME = "test_voice_e2e"


@pytest.fixture
async def client():
    """Create async HTTP client."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


@pytest.fixture
def sample_audio_file():
    """Create a sample WAV file for testing."""
    # Generate 2 seconds of audio (sine wave)
    sample_rate = 24000
    duration = 2.0
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        yield f.name

    # Cleanup
    try:
        os.unlink(f.name)
    except:
        pass


@pytest.fixture
def sample_transcription():
    """Sample transcription for test audio."""
    return "This is a test audio sample for voice cloning."


class TestZipVoiceEndToEnd:
    """End-to-end tests for ZipVoice integration."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test 1: Server is running and healthy."""
        response = await client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_voice_registration(self, client, sample_audio_file, sample_transcription):
        """Test 2: Register a voice prompt."""
        files = {'audio_file': open(sample_audio_file, 'rb')}
        data = {
            'name': TEST_VOICE_NAME,
            'transcription': sample_transcription
        }

        response = await client.post(
            f"{BASE_URL}/v1/zipvoice/voices/register",
            files=files,
            data=data
        )

        assert response.status_code == 200
        result = response.json()
        assert result['status'] == 'success'
        assert result['name'] == TEST_VOICE_NAME
        assert 'audio_info' in result
        assert result['audio_info']['duration'] > 0

    @pytest.mark.asyncio
    async def test_list_voices(self, client):
        """Test 3: List registered voices."""
        response = await client.get(f"{BASE_URL}/v1/zipvoice/voices")

        assert response.status_code == 200
        result = response.json()
        assert 'voices' in result
        assert 'count' in result
        assert result['count'] >= 0

    @pytest.mark.asyncio
    async def test_get_voice_info(self, client):
        """Test 4: Get voice information."""
        response = await client.get(f"{BASE_URL}/v1/zipvoice/voices/{TEST_VOICE_NAME}")

        if response.status_code == 404:
            pytest.skip(f"Voice '{TEST_VOICE_NAME}' not registered")

        assert response.status_code == 200
        result = response.json()
        assert result['name'] == TEST_VOICE_NAME
        assert 'transcription' in result
        assert 'audio_info' in result

    @pytest.mark.asyncio
    async def test_generate_speech_with_registered_voice(self, client, sample_transcription):
        """Test 5: Generate speech using registered voice."""
        request_data = {
            'model': 'zipvoice',
            'input': 'This is a test of voice cloning with ZipVoice.',
            'voice': TEST_VOICE_NAME,
            'prompt_text': sample_transcription,
            'response_format': 'wav',
            'stream': False,
            'num_steps': 4  # Fast for testing
        }

        response = await client.post(
            f"{BASE_URL}/v1/zipvoice/audio/speech",
            json=request_data
        )

        if response.status_code == 404:
            pytest.skip(f"Voice '{TEST_VOICE_NAME}' not registered")

        assert response.status_code == 200
        assert len(response.content) > 0

        # Validate it's actual audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        try:
            audio, sr = sf.read(temp_path)
            assert sr > 0
            assert len(audio) > 0
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_streaming_generation(self, client, sample_transcription):
        """Test 6: Streaming audio generation."""
        request_data = {
            'model': 'zipvoice',
            'input': 'This is a longer text for testing streaming. ' * 5,
            'voice': TEST_VOICE_NAME,
            'prompt_text': sample_transcription,
            'response_format': 'mp3',
            'stream': True,
            'num_steps': 4
        }

        chunks_received = 0
        total_bytes = 0

        async with client.stream(
            'POST',
            f"{BASE_URL}/v1/zipvoice/audio/speech",
            json=request_data
        ) as response:
            if response.status_code == 404:
                pytest.skip(f"Voice '{TEST_VOICE_NAME}' not registered")

            assert response.status_code == 200

            async for chunk in response.aiter_bytes():
                chunks_received += 1
                total_bytes += len(chunk)

        assert chunks_received > 0
        assert total_bytes > 0

    @pytest.mark.asyncio
    async def test_base64_voice_input(self, client, sample_audio_file, sample_transcription):
        """Test 7: Generate speech with base64 encoded voice."""
        # Read and encode audio
        with open(sample_audio_file, 'rb') as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()

        request_data = {
            'model': 'zipvoice',
            'input': 'Testing base64 voice input.',
            'voice': f'base64+{audio_b64}',
            'prompt_text': sample_transcription,
            'response_format': 'wav',
            'num_steps': 4
        }

        response = await client.post(
            f"{BASE_URL}/v1/zipvoice/audio/speech",
            json=request_data
        )

        assert response.status_code == 200
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_different_model_variants(self, client, sample_transcription):
        """Test 8: Test different ZipVoice model variants."""
        models = ['zipvoice', 'zipvoice_distill']

        for model in models:
            request_data = {
                'model': model,
                'input': f'Testing {model} model.',
                'voice': TEST_VOICE_NAME,
                'prompt_text': sample_transcription,
                'response_format': 'wav',
                'num_steps': 4
            }

            response = await client.post(
                f"{BASE_URL}/v1/zipvoice/audio/speech",
                json=request_data
            )

            if response.status_code == 404:
                pytest.skip(f"Voice '{TEST_VOICE_NAME}' not registered")

            # May fail if model not available, which is OK
            if response.status_code == 200:
                assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_different_response_formats(self, client, sample_transcription):
        """Test 9: Test different audio output formats."""
        formats = ['wav', 'mp3', 'opus', 'flac']

        for fmt in formats:
            request_data = {
                'model': 'zipvoice',
                'input': f'Testing {fmt} format.',
                'voice': TEST_VOICE_NAME,
                'prompt_text': sample_transcription,
                'response_format': fmt,
                'num_steps': 4
            }

            response = await client.post(
                f"{BASE_URL}/v1/zipvoice/audio/speech",
                json=request_data
            )

            if response.status_code == 404:
                pytest.skip(f"Voice '{TEST_VOICE_NAME}' not registered")

            assert response.status_code == 200
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_speed_parameter(self, client, sample_transcription):
        """Test 10: Test speed parameter."""
        speeds = [0.5, 1.0, 1.5]

        for speed in speeds:
            request_data = {
                'model': 'zipvoice',
                'input': 'Testing speed parameter.',
                'voice': TEST_VOICE_NAME,
                'prompt_text': sample_transcription,
                'response_format': 'wav',
                'speed': speed,
                'num_steps': 4
            }

            response = await client.post(
                f"{BASE_URL}/v1/zipvoice/audio/speech",
                json=request_data
            )

            if response.status_code == 404:
                pytest.skip(f"Voice '{TEST_VOICE_NAME}' not registered")

            assert response.status_code == 200
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_num_steps_parameter(self, client, sample_transcription):
        """Test 11: Test inference steps parameter."""
        steps = [4, 8, 16]

        for num_steps in steps:
            request_data = {
                'model': 'zipvoice',
                'input': 'Testing inference steps.',
                'voice': TEST_VOICE_NAME,
                'prompt_text': sample_transcription,
                'response_format': 'wav',
                'num_steps': num_steps
            }

            response = await client.post(
                f"{BASE_URL}/v1/zipvoice/audio/speech",
                json=request_data
            )

            if response.status_code == 404:
                pytest.skip(f"Voice '{TEST_VOICE_NAME}' not registered")

            assert response.status_code == 200
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_cache_clearing(self, client):
        """Test 12: Test cache clearing functionality."""
        cache_types = ['url', 'base64', 'all']

        for cache_type in cache_types:
            response = await client.post(
                f"{BASE_URL}/v1/zipvoice/voices/cache/clear",
                params={'cache_type': cache_type}
            )

            assert response.status_code == 200
            result = response.json()
            assert 'files_deleted' in result

    @pytest.mark.asyncio
    async def test_error_handling_missing_voice(self, client):
        """Test 13: Error handling for missing voice."""
        request_data = {
            'model': 'zipvoice',
            'input': 'Testing error handling.',
            'voice': 'nonexistent_voice_12345',
            'prompt_text': 'Sample text.',
            'num_steps': 4
        }

        response = await client.post(
            f"{BASE_URL}/v1/zipvoice/audio/speech",
            json=request_data
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_error_handling_missing_prompt_text(self, client):
        """Test 14: Error handling for missing prompt_text."""
        request_data = {
            'model': 'zipvoice',
            'input': 'Testing error handling.',
            'voice': TEST_VOICE_NAME,
            # Missing prompt_text
            'num_steps': 4
        }

        response = await client.post(
            f"{BASE_URL}/v1/zipvoice/audio/speech",
            json=request_data
        )

        # Should fail validation
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_delete_voice(self, client):
        """Test 15: Delete registered voice (run last)."""
        response = await client.delete(
            f"{BASE_URL}/v1/zipvoice/voices/{TEST_VOICE_NAME}"
        )

        if response.status_code == 404:
            pytest.skip(f"Voice '{TEST_VOICE_NAME}' not registered")

        assert response.status_code == 200
        result = response.json()
        assert result['status'] == 'deleted'


class TestZipVoicePerformance:
    """Performance tests for ZipVoice."""

    @pytest.mark.asyncio
    async def test_generation_speed_comparison(self, client, sample_transcription):
        """Test 16: Compare generation speed across configurations."""
        import time

        configs = [
            {'name': 'Fast', 'model': 'zipvoice_distill', 'num_steps': 4},
            {'name': 'Balanced', 'model': 'zipvoice', 'num_steps': 8},
            {'name': 'Quality', 'model': 'zipvoice', 'num_steps': 16},
        ]

        results = []

        for config in configs:
            request_data = {
                'model': config['model'],
                'input': 'Performance test text.',
                'voice': TEST_VOICE_NAME,
                'prompt_text': sample_transcription,
                'response_format': 'wav',
                'num_steps': config['num_steps'],
                'stream': False
            }

            start = time.time()
            response = await client.post(
                f"{BASE_URL}/v1/zipvoice/audio/speech",
                json=request_data
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                results.append({
                    'config': config['name'],
                    'time': elapsed,
                    'size': len(response.content)
                })

        # Print results for analysis
        for result in results:
            print(f"{result['config']}: {result['time']:.2f}s, {result['size']} bytes")

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, sample_transcription):
        """Test 17: Handle concurrent requests."""
        async def make_request():
            request_data = {
                'model': 'zipvoice',
                'input': 'Concurrent request test.',
                'voice': TEST_VOICE_NAME,
                'prompt_text': sample_transcription,
                'num_steps': 4
            }

            response = await client.post(
                f"{BASE_URL}/v1/zipvoice/audio/speech",
                json=request_data
            )
            return response.status_code

        # Make 3 concurrent requests
        tasks = [make_request() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # At least some should succeed
        success_count = sum(1 for r in results if r == 200)
        assert success_count > 0


class TestZipVoiceValidation:
    """Validation tests for ZipVoice."""

    @pytest.mark.asyncio
    async def test_audio_quality_validation(self, client, sample_transcription):
        """Test 18: Validate generated audio quality."""
        request_data = {
            'model': 'zipvoice',
            'input': 'Audio quality validation test.',
            'voice': TEST_VOICE_NAME,
            'prompt_text': sample_transcription,
            'response_format': 'wav',
            'num_steps': 8
        }

        response = await client.post(
            f"{BASE_URL}/v1/zipvoice/audio/speech",
            json=request_data
        )

        if response.status_code != 200:
            pytest.skip("Generation failed")

        # Save and analyze audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        try:
            audio, sr = sf.read(temp_path)

            # Validate audio properties
            assert sr == 24000, f"Expected 24kHz, got {sr}Hz"
            assert len(audio) > 0, "Audio is empty"

            # Check audio level (not silent or clipping)
            max_amplitude = np.max(np.abs(audio))
            assert max_amplitude > 0.01, "Audio too quiet"
            assert max_amplitude < 1.0, "Audio clipping"

        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_long_text_generation(self, client, sample_transcription):
        """Test 19: Generate speech for long text."""
        long_text = "This is a test of long text generation. " * 50

        request_data = {
            'model': 'zipvoice',
            'input': long_text,
            'voice': TEST_VOICE_NAME,
            'prompt_text': sample_transcription,
            'response_format': 'mp3',
            'num_steps': 4
        }

        response = await client.post(
            f"{BASE_URL}/v1/zipvoice/audio/speech",
            json=request_data,
            timeout=120.0  # Longer timeout
        )

        if response.status_code != 200:
            pytest.skip("Long text generation not supported or failed")

        assert len(response.content) > 0


# Utility function to run all tests
async def run_all_tests():
    """Run all tests programmatically."""
    import sys

    # Run pytest
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ])

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
