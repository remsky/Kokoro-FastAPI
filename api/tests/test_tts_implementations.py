"""Tests for TTS model implementations"""
import os
import torch
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from api.src.services.tts_base import TTSBaseModel
from api.src.services.tts_cpu import TTSCPUModel
from api.src.services.tts_gpu import TTSGPUModel, length_to_mask

# Base Model Tests
def test_get_device_error():
    """Test get_device() raises error when not initialized"""
    TTSBaseModel._device = None
    with pytest.raises(RuntimeError, match="Model not initialized"):
        TTSBaseModel.get_device()

@pytest.mark.asyncio
@patch('torch.cuda.is_available')
@patch('os.path.exists')
@patch('os.path.join')
@patch('os.listdir')
@patch('torch.load')
@patch('torch.save')
async def test_setup_cuda_available(mock_save, mock_load, mock_listdir, mock_join, mock_exists, mock_cuda_available):
    """Test setup with CUDA available"""
    TTSBaseModel._device = None
    mock_cuda_available.return_value = True
    mock_exists.return_value = True
    mock_load.return_value = torch.zeros(1)
    mock_listdir.return_value = ["voice1.pt", "voice2.pt"]
    mock_join.return_value = "/mocked/path"
    
    # Create mock model
    mock_model = MagicMock()
    mock_model.bert = MagicMock()
    mock_model.process_text = MagicMock(return_value=("dummy", [1,2,3]))
    mock_model.generate_from_tokens = MagicMock(return_value=np.zeros(1000))
    
    # Mock initialize to return our mock model
    TTSBaseModel.initialize = MagicMock(return_value=mock_model)
    TTSBaseModel._instance = mock_model
    
    voice_count = await TTSBaseModel.setup()
    assert TTSBaseModel._device == "cuda"
    assert voice_count == 2

@pytest.mark.asyncio
@patch('torch.cuda.is_available')
@patch('os.path.exists')
@patch('os.path.join')
@patch('os.listdir')
@patch('torch.load')
@patch('torch.save')
async def test_setup_cuda_unavailable(mock_save, mock_load, mock_listdir, mock_join, mock_exists, mock_cuda_available):
    """Test setup with CUDA unavailable"""
    TTSBaseModel._device = None
    mock_cuda_available.return_value = False
    mock_exists.return_value = True
    mock_load.return_value = torch.zeros(1)
    mock_listdir.return_value = ["voice1.pt", "voice2.pt"]
    mock_join.return_value = "/mocked/path"
    
    # Create mock model
    mock_model = MagicMock()
    mock_model.bert = MagicMock()
    mock_model.process_text = MagicMock(return_value=("dummy", [1,2,3]))
    mock_model.generate_from_tokens = MagicMock(return_value=np.zeros(1000))
    
    # Mock initialize to return our mock model
    TTSBaseModel.initialize = MagicMock(return_value=mock_model)
    TTSBaseModel._instance = mock_model
    
    voice_count = await TTSBaseModel.setup()
    assert TTSBaseModel._device == "cpu"
    assert voice_count == 2

# CPU Model Tests
def test_cpu_initialize_missing_model():
    """Test CPU initialize with missing model"""
    TTSCPUModel._onnx_session = None  # Reset the session
    with patch('os.path.exists', return_value=False), \
         patch('onnxruntime.InferenceSession', return_value=None):
        result = TTSCPUModel.initialize("dummy_dir")
        assert result is None

def test_cpu_generate_uninitialized():
    """Test CPU generate methods with uninitialized model"""
    TTSCPUModel._onnx_session = None
    
    with pytest.raises(RuntimeError, match="ONNX model not initialized"):
        TTSCPUModel.generate_from_text("test", torch.zeros(1), "en", 1.0)
        
    with pytest.raises(RuntimeError, match="ONNX model not initialized"):
        TTSCPUModel.generate_from_tokens([1,2,3], torch.zeros(1), 1.0)

def test_cpu_process_text():
    """Test CPU process_text functionality"""
    with patch('api.src.services.tts_cpu.phonemize') as mock_phonemize, \
         patch('api.src.services.tts_cpu.tokenize') as mock_tokenize:
        
        mock_phonemize.return_value = "test phonemes"
        mock_tokenize.return_value = [1, 2, 3]
        
        phonemes, tokens = TTSCPUModel.process_text("test", "en")
        assert phonemes == "test phonemes"
        assert tokens == [0, 1, 2, 3, 0]  # Should add start/end tokens

# GPU Model Tests
@patch('torch.cuda.is_available')
def test_gpu_initialize_cuda_unavailable(mock_cuda_available):
    """Test GPU initialize with CUDA unavailable"""
    mock_cuda_available.return_value = False
    TTSGPUModel._instance = None
    
    result = TTSGPUModel.initialize("dummy_dir", "dummy_path")
    assert result is None

@patch('api.src.services.tts_gpu.length_to_mask')
def test_gpu_length_to_mask(mock_length_to_mask):
    """Test length_to_mask function"""
    # Setup mock return value
    expected_mask = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, False, False]
    ])
    mock_length_to_mask.return_value = expected_mask
    
    # Call function with test input
    lengths = torch.tensor([3, 5])
    mask = mock_length_to_mask(lengths)
    
    # Verify mock was called with correct input
    mock_length_to_mask.assert_called_once()
    assert torch.equal(mask, expected_mask)

def test_gpu_generate_uninitialized():
    """Test GPU generate methods with uninitialized model"""
    TTSGPUModel._instance = None
    
    with pytest.raises(RuntimeError, match="GPU model not initialized"):
        TTSGPUModel.generate_from_text("test", torch.zeros(1), "en", 1.0)
        
    with pytest.raises(RuntimeError, match="GPU model not initialized"):
        TTSGPUModel.generate_from_tokens([1,2,3], torch.zeros(1), 1.0)

def test_gpu_process_text():
    """Test GPU process_text functionality"""
    with patch('api.src.services.tts_gpu.phonemize') as mock_phonemize, \
         patch('api.src.services.tts_gpu.tokenize') as mock_tokenize:
        
        mock_phonemize.return_value = "test phonemes"
        mock_tokenize.return_value = [1, 2, 3]
        
        phonemes, tokens = TTSGPUModel.process_text("test", "en")
        assert phonemes == "test phonemes"
        assert tokens == [1, 2, 3]  # GPU implementation doesn't add start/end tokens
