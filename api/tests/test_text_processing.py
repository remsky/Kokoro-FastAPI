"""Tests for text processing endpoints"""
from unittest.mock import Mock, patch
import pytest
import pytest_asyncio
from httpx import AsyncClient
import numpy as np

from ..src.main import app
from .conftest import MockTTSModel

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_phonemize_endpoint(async_client):
    """Test phoneme generation endpoint"""
    with patch('api.src.routers.text_processing.phonemize') as mock_phonemize, \
         patch('api.src.routers.text_processing.tokenize') as mock_tokenize:
        
        # Setup mocks
        mock_phonemize.return_value = "həlˈoʊ"
        mock_tokenize.return_value = [1, 2, 3]
        
        # Test request
        response = await async_client.post("/text/phonemize", json={
            "text": "hello",
            "language": "a"
        })
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["phonemes"] == "həlˈoʊ"
        assert result["tokens"] == [0, 1, 2, 3, 0]  # Should add start/end tokens

@pytest.mark.asyncio
async def test_phonemize_empty_text(async_client):
    """Test phoneme generation with empty text"""
    response = await async_client.post("/text/phonemize", json={
        "text": "",
        "language": "a"
    })
    
    assert response.status_code == 500
    assert "error" in response.json()["detail"]

@pytest.mark.asyncio
async def test_generate_from_phonemes(async_client, mock_tts_service, mock_audio_service):
    """Test audio generation from phonemes"""
    with patch('api.src.routers.text_processing.TTSService', return_value=mock_tts_service):
        response = await async_client.post("/text/generate_from_phonemes", json={
            "phonemes": "həlˈoʊ",
            "voice": "af_bella",
            "speed": 1.0
        })
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["content-disposition"] == "attachment; filename=speech.wav"
        assert response.content == b"mock audio data"

@pytest.mark.asyncio
async def test_generate_from_phonemes_invalid_voice(async_client, mock_tts_service):
    """Test audio generation with invalid voice"""
    mock_tts_service._get_voice_path.return_value = None
    with patch('api.src.routers.text_processing.TTSService', return_value=mock_tts_service):
        response = await async_client.post("/text/generate_from_phonemes", json={
            "phonemes": "həlˈoʊ",
            "voice": "invalid_voice",
            "speed": 1.0
        })
        
        assert response.status_code == 400
        assert "Voice not found" in response.json()["detail"]["message"]

@pytest.mark.asyncio
async def test_generate_from_phonemes_invalid_speed(async_client, monkeypatch):
    """Test audio generation with invalid speed"""
    # Mock TTSModel initialization
    mock_model = Mock()
    mock_model.generate_from_tokens = Mock(return_value=np.zeros(48000))
    monkeypatch.setattr("api.src.services.tts_model.TTSModel._instance", mock_model)
    monkeypatch.setattr("api.src.services.tts_model.TTSModel.get_instance", Mock(return_value=mock_model))
    
    response = await async_client.post("/text/generate_from_phonemes", json={
        "phonemes": "həlˈoʊ",
        "voice": "af_bella",
        "speed": -1.0
    })
    
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_generate_from_phonemes_empty_phonemes(async_client, mock_tts_service):
    """Test audio generation with empty phonemes"""
    with patch('api.src.routers.text_processing.TTSService', return_value=mock_tts_service):
        response = await async_client.post("/text/generate_from_phonemes", json={
            "phonemes": "",
            "voice": "af_bella",
            "speed": 1.0
        })
        
        assert response.status_code == 400
        assert "Invalid request" in response.json()["detail"]["error"]
