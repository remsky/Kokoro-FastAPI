import os
import threading
from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
import numpy as np
from loguru import logger

from ..core.config import settings

class TTSBaseModel(ABC):
    _instance = None
    _lock = threading.Lock()
    _device = None
    VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voices")

    @classmethod
    async def setup(cls):
        """Initialize model and setup voices"""
        with cls._lock:
            # Set device
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                try:
                    # Test CUDA device
                    test_tensor = torch.zeros(1).cuda()
                    logger.info("CUDA test successful")
                    model_path = os.path.join(settings.model_dir, settings.pytorch_model_path)
                    cls._device = "cuda"
                except Exception as e:
                    logger.error(f"CUDA test failed: {e}")
                    cls._device = "cpu"
            else:
                cls._device = "cpu"
                model_path = os.path.join(settings.model_dir, settings.onnx_model_path)
            logger.info(f"Initializing model on {cls._device}")

            # Initialize model first
            model = cls.initialize(settings.model_dir, model_path=model_path)
            if model is None:
                raise RuntimeError(f"Failed to initialize {cls._device.upper()} model")
            cls._instance = model

            # Setup voices directory
            os.makedirs(cls.VOICES_DIR, exist_ok=True)

            # Copy base voices to local directory
            base_voices_dir = os.path.join(settings.model_dir, settings.voices_dir)
            if os.path.exists(base_voices_dir):
                for file in os.listdir(base_voices_dir):
                    if file.endswith(".pt"):
                        voice_name = file[:-3]
                        voice_path = os.path.join(cls.VOICES_DIR, file)
                        if not os.path.exists(voice_path):
                            try:
                                logger.info(f"Copying base voice {voice_name} to voices directory")
                                base_path = os.path.join(base_voices_dir, file)
                                voicepack = torch.load(base_path, map_location=cls._device, weights_only=True)
                                torch.save(voicepack, voice_path)
                            except Exception as e:
                                logger.error(f"Error copying voice {voice_name}: {str(e)}")

            # Count voices in directory
            voice_count = len([f for f in os.listdir(cls.VOICES_DIR) if f.endswith(".pt")])

            # Now that model and voices are ready, do warmup
            try:
                with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "don_quixote.txt")) as f:
                    warmup_text = f.read()
            except Exception as e:
                logger.warning(f"Failed to load warmup text: {e}")
                warmup_text = "This is a warmup text that will be split into chunks for processing."

            # Use warmup service after model is fully initialized
            from .warmup import WarmupService
            warmup = WarmupService()
            
            # Load and warm up voices
            loaded_voices = warmup.load_voices()
            await warmup.warmup_voices(warmup_text, loaded_voices)
            
            logger.info("Model warm-up complete")

            # Count voices in directory
            voice_count = len([f for f in os.listdir(cls.VOICES_DIR) if f.endswith(".pt")])
            return voice_count

    @classmethod
    @abstractmethod
    def initialize(cls, model_dir: str, model_path: str = None):
        """Initialize the model"""
        pass

    @classmethod
    @abstractmethod
    def process_text(cls, text: str, language: str) -> Tuple[str, List[int]]:
        """Process text into phonemes and tokens
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            tuple[str, list[int]]: Phonemes and token IDs
        """
        pass

    @classmethod
    @abstractmethod
    def generate_from_text(cls, text: str, voicepack: torch.Tensor, language: str, speed: float) -> Tuple[np.ndarray, str]:
        """Generate audio from text
        
        Args:
            text: Input text
            voicepack: Voice tensor
            language: Language code
            speed: Speed factor
            
        Returns:
            tuple[np.ndarray, str]: Generated audio samples and phonemes
        """
        pass

    @classmethod
    @abstractmethod
    def generate_from_tokens(cls, tokens: List[int], voicepack: torch.Tensor, speed: float) -> np.ndarray:
        """Generate audio from tokens
        
        Args:
            tokens: Token IDs
            voicepack: Voice tensor
            speed: Speed factor
            
        Returns:
            np.ndarray: Generated audio samples
        """
        pass

    @classmethod
    def get_device(cls):
        """Get the current device"""
        if cls._device is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        return cls._device
