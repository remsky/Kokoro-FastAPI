"""Voice management with controlled resource handling."""

from typing import Dict, List, Optional

import aiofiles
import torch
from loguru import logger

from ..core import paths
from ..core.config import settings


class VoiceManager:
    """Manages voice loading and caching with controlled resource usage."""

    # Singleton instance
    _instance = None

    def __init__(self):
        """Initialize voice manager."""
        # Strictly respect settings.use_gpu
        self._device = settings.get_device()
        self._voices: Dict[str, torch.Tensor] = {}

    async def get_voice_path(self, voice_name: str) -> str:
        """Get path to voice file.

        Args:
            voice_name: Name of voice

        Returns:
            Path to voice file

        Raises:
            RuntimeError: If voice not found
        """
        return await paths.get_voice_path(voice_name)

    async def load_voice(
        self, voice_name: str, device: Optional[str] = None
    ) -> torch.Tensor:
        """Load voice tensor.

        Args:
            voice_name: Name of voice to load
            device: Optional override for target device

        Returns:
            Voice tensor

        Raises:
            RuntimeError: If voice not found
        """
        try:
            voice_path = await self.get_voice_path(voice_name)
            target_device = device or self._device
            voice = await paths.load_voice_tensor(voice_path, target_device)
            self._voices[voice_name] = voice
            return voice
        except Exception as e:
            raise RuntimeError(f"Failed to load voice {voice_name}: {e}")

    async def combine_voices(
        self, voices: List[str], device: Optional[str] = None
    ) -> torch.Tensor:
        """Combine multiple voices.

        Args:
            voices: List of voice names to combine
            device: Optional override for target device

        Returns:
            Combined voice tensor

        Raises:
            RuntimeError: If any voice not found
        """
        if len(voices) < 2:
            raise ValueError("Need at least 2 voices to combine")

        target_device = device or self._device
        voice_tensors = []
        for name in voices:
            voice = await self.load_voice(name, target_device)
            voice_tensors.append(voice)

        combined = torch.mean(torch.stack(voice_tensors), dim=0)
        return combined

    async def list_voices(self) -> List[str]:
        """List available voice names.

        Returns:
            List of voice names
        """
        return await paths.list_voices()

    def cache_info(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        return {"loaded_voices": len(self._voices), "device": self._device}


async def get_manager() -> VoiceManager:
    """Get voice manager instance.

    Returns:
        VoiceManager instance
    """
    if VoiceManager._instance is None:
        VoiceManager._instance = VoiceManager()
    return VoiceManager._instance
