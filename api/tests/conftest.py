import os
import sys
import shutil
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest
import aiofiles.threadpool


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


# Create mock torch module
mock_torch = Mock()
mock_torch.cuda = Mock()
mock_torch.cuda.is_available = Mock(return_value=False)


# Create a mock tensor class that supports basic operations
class MockTensor:
    def __init__(self, data):
        self.data = data
        if isinstance(data, (list, tuple)):
            self.shape = [len(data)]
        elif isinstance(data, MockTensor):
            self.shape = data.shape
        else:
            self.shape = getattr(data, "shape", [1])

    def __getitem__(self, idx):
        if isinstance(self.data, (list, tuple)):
            if isinstance(idx, slice):
                return MockTensor(self.data[idx])
            return self.data[idx]
        return self

    def max(self):
        if isinstance(self.data, (list, tuple)):
            max_val = max(self.data)
            return MockTensor(max_val)
        return 5  # Default for testing

    def item(self):
        if isinstance(self.data, (list, tuple)):
            return max(self.data)
        if isinstance(self.data, (int, float)):
            return self.data
        return 5  # Default for testing

    def cuda(self):
        """Support cuda conversion"""
        return self

    def any(self):
        if isinstance(self.data, (list, tuple)):
            return any(self.data)
        return False

    def all(self):
        if isinstance(self.data, (list, tuple)):
            return all(self.data)
        return True

    def unsqueeze(self, dim):
        return self

    def expand(self, *args):
        return self

    def type_as(self, other):
        return self


# Add tensor operations to mock torch
mock_torch.tensor = lambda x: MockTensor(x)
mock_torch.zeros = lambda *args: MockTensor(
    [0] * (args[0] if isinstance(args[0], int) else args[0][0])
)
mock_torch.arange = lambda x: MockTensor(list(range(x)))
mock_torch.gt = lambda x, y: MockTensor([False] * x.shape[0])

# Mock modules before they're imported
sys.modules["torch"] = mock_torch
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
