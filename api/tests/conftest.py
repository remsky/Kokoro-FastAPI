import os
import shutil
import sys
from unittest.mock import MagicMock, Mock, patch

import aiofiles.threadpool
import numpy as np
import pytest


def cleanup_mock_dirs():
    """Clean up any MagicMock directories created during tests"""
    mock_dir = "MagicMock"
    if os.path.exists(mock_dir):
        shutil.rmtree(mock_dir)


@pytest.fixture(autouse=True)
def setup_aiofiles():
    """Setup aiofiles mock wrapper"""
    aiofiles.threadpool.wrap.register(MagicMock)(
        lambda *args, **kwargs: aiofiles.threadpool.AsyncBufferedIOBase(*args, **kwargs)
    )
    yield


@pytest.fixture(autouse=True)
def cleanup():
    """Automatically clean up before and after each test"""
    cleanup_mock_dirs()
    yield
    cleanup_mock_dirs()


# Mock modules before they're imported
sys.modules["transformers"] = Mock()
sys.modules["phonemizer"] = Mock()
sys.modules["models"] = Mock()
sys.modules["models.build_model"] = Mock()
sys.modules["kokoro"] = Mock()
sys.modules["kokoro.generate"] = Mock()
sys.modules["kokoro.phonemize"] = Mock()
sys.modules["kokoro.tokenize"] = Mock()

# Mock ONNX runtime
mock_onnx = Mock()
mock_onnx.InferenceSession = Mock()
mock_onnx.SessionOptions = Mock()
mock_onnx.GraphOptimizationLevel = Mock()
mock_onnx.ExecutionMode = Mock()
sys.modules["onnxruntime"] = mock_onnx

# Create mock settings module
mock_settings_module = Mock()
mock_settings = Mock()
mock_settings.model_dir = "/mock/model/dir"
mock_settings.onnx_model_path = "mock.onnx"
mock_settings_module.settings = mock_settings
sys.modules["api.src.core.config"] = mock_settings_module


class MockTTSModel:
    _instance = None
    _onnx_session = None
    VOICES_DIR = "/mock/voices/dir"

    def __init__(self):
        self._initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def initialize(cls, model_dir):
        cls._onnx_session = Mock()
        cls._onnx_session.run = Mock(return_value=[np.zeros(48000)])
        cls._instance._initialized = True
        return cls._onnx_session

    @classmethod
    def setup(cls):
        if not cls._instance._initialized:
            cls.initialize("/mock/model/dir")
        return cls._instance

    @classmethod
    def generate_from_tokens(cls, tokens, voicepack, speed):
        if not cls._instance._initialized:
            raise RuntimeError("Model not initialized. Call setup() first.")
        return np.zeros(48000)

    @classmethod
    def process_text(cls, text, language):
        return "mock phonemes", [1, 2, 3]

    @staticmethod
    def get_device():
        return "cpu"


@pytest.fixture
def mock_tts_service(monkeypatch):
    """Mock TTSService for testing"""
    mock_service = Mock()
    mock_service._get_voice_path.return_value = "/mock/path/voice.pt"
    mock_service._load_voice.return_value = np.zeros((1, 192))

    # Mock TTSModel.generate_from_tokens since we call it directly
    mock_generate = Mock(return_value=np.zeros(48000))
    monkeypatch.setattr(
        "api.src.routers.development.TTSModel.generate_from_tokens", mock_generate
    )

    return mock_service


@pytest.fixture
def mock_audio_service(monkeypatch):
    """Mock AudioService"""
    mock_service = Mock()
    mock_service.convert_audio.return_value = b"mock audio data"
    monkeypatch.setattr("api.src.routers.development.AudioService", mock_service)
    return mock_service
