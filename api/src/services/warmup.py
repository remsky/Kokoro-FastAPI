import os
from typing import List, Tuple
import torch
from loguru import logger

from .tts_service import TTSService
from .tts_model import TTSModel
from ..core.config import settings


class WarmupService:
    """Service for warming up TTS models and voice caches"""
    
    def __init__(self):
        """Initialize warmup service and ensure model is ready"""
        # Initialize model if not already initialized
        if TTSModel._instance is None:
            TTSModel.initialize(settings.model_dir)
        self.tts_service = TTSService()
        
    def load_voices(self) -> List[Tuple[str, torch.Tensor]]:
        """Load and cache voices up to LRU limit"""
        # Get all voices sorted by filename length (shorter names first, usually base voices)
        voice_files = sorted(
            [f for f in os.listdir(TTSModel.VOICES_DIR) if f.endswith(".pt")],
            key=len
        )
        
        n_voices_cache=1
        loaded_voices = []
        for voice_file in voice_files[:n_voices_cache]:
            try:
                voice_path = os.path.join(TTSModel.VOICES_DIR, voice_file)
                # load using service, lru cache
                voicepack = self.tts_service._load_voice(voice_path)
                loaded_voices.append((voice_file[:-3], voicepack))  # Store name and tensor
                # voicepack = torch.load(voice_path, map_location=TTSModel.get_device(), weights_only=True)
                # logger.info(f"Loaded voice {voice_file[:-3]} into cache")
            except Exception as e:
                logger.error(f"Failed to load voice {voice_file}: {e}")
        logger.info(f"Pre-loaded {len(loaded_voices)} voices into cache")
        return loaded_voices
        
    async def warmup_voices(self, warmup_text: str, loaded_voices: List[Tuple[str, torch.Tensor]]):
        """Warm up voice inference and streaming"""
        n_warmups = 1
        for voice_name, _ in loaded_voices[:n_warmups]:
            try:
                logger.info(f"Running warmup inference on voice {voice_name}")
                async for _ in self.tts_service.generate_audio_stream(
                    warmup_text,
                    voice_name,
                    1.0,
                    "pcm"
                ):
                    pass  # Process all chunks to properly warm up
                logger.info(f"Completed warmup for voice {voice_name}")
            except Exception as e:
                logger.warning(f"Warmup failed for voice {voice_name}: {e}")
