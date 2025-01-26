"""Voice pack management and caching."""

import os
from typing import Dict, List, Optional

import torch
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..structures.model_schemas import VoiceConfig


class VoiceManager:
    """Manages voice loading and operations."""

    def __init__(self, config: Optional[VoiceConfig] = None):
        """Initialize voice manager.
        
        Args:
            config: Optional voice configuration
        """
        self._config = config or VoiceConfig()
        self._voice_cache: Dict[str, torch.Tensor] = {}

    def get_voice_path(self, voice_name: str) -> Optional[str]:
        """Get path to voice file.
        
        Args:
            voice_name: Name of voice
            
        Returns:
            Path to voice file if exists, None otherwise
        """
        api_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        voice_path = os.path.join(api_dir, settings.voices_dir, f"{voice_name}.pt")
        return voice_path if os.path.exists(voice_path) else None

    async def load_voice(self, voice_name: str, device: str = "cpu") -> torch.Tensor:
        """Load voice tensor.
        
        Args:
            voice_name: Name of voice to load
            device: Device to load voice on
            
        Returns:
            Voice tensor
            
        Raises:
            RuntimeError: If voice loading fails
        """
        # Check if it's a combined voice request
        if "+" in voice_name:
            voices = [v.strip() for v in voice_name.split("+") if v.strip()]
            if len(voices) < 2:
                raise RuntimeError(f"Invalid combined voice name: {voice_name}")
                
            # Load and combine voices
            voice_tensors = []
            for voice in voices:
                try:
                    voice_tensor = await self.load_voice(voice, device)
                    voice_tensors.append(voice_tensor)
                except Exception as e:
                    raise RuntimeError(f"Failed to load base voice {voice}: {e}")
                    
            return torch.mean(torch.stack(voice_tensors), dim=0)

        # Handle single voice
        voice_path = self.get_voice_path(voice_name)
        if not voice_path:
            raise RuntimeError(f"Voice not found: {voice_name}")

        # Check cache
        cache_key = f"{voice_path}_{device}"
        if self._config.use_cache and cache_key in self._voice_cache:
            return self._voice_cache[cache_key]

        # Load voice tensor
        try:
            voice = await paths.load_voice_tensor(voice_path, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load voice {voice_name}: {e}")

        # Cache if enabled
        if self._config.use_cache:
            self._manage_cache()
            self._voice_cache[cache_key] = voice
            logger.debug(f"Cached voice: {voice_name} on {device}")

        return voice

    def _manage_cache(self) -> None:
        """Manage voice cache size using simple LRU."""
        if len(self._voice_cache) >= self._config.cache_size:
            # Remove least recently used voice
            oldest = next(iter(self._voice_cache))
            del self._voice_cache[oldest]
            torch.cuda.empty_cache()  # Clean up GPU memory if needed
            logger.debug(f"Removed LRU voice from cache: {oldest}")

    async def combine_voices(self, voices: List[str], device: str = "cpu") -> str:
        """Combine multiple voices into a new voice.
        
        Args:
            voices: List of voice names to combine
            device: Device to load voices on
            
        Returns:
            Name of combined voice
            
        Raises:
            ValueError: If fewer than 2 voices provided
            RuntimeError: If voice combination fails
        """
        if len(voices) < 2:
            raise ValueError("At least 2 voices are required for combination")

        # Create combined name using + as separator
        combined_name = "+".join(voices)

        # If saving is enabled, try to save the combination
        if settings.allow_local_voice_saving:
            try:
                # Load and combine voices
                combined_tensor = await self.load_voice(combined_name, device)
                
                # Save to disk
                api_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                voices_dir = os.path.join(api_dir, settings.voices_dir)
                os.makedirs(voices_dir, exist_ok=True)
                
                combined_path = os.path.join(voices_dir, f"{combined_name}.pt")
                try:
                    torch.save(combined_tensor, combined_path)
                    # Cache with path-based key
                    self._voice_cache[f"{combined_path}_{device}"] = combined_tensor
                except Exception as e:
                    raise RuntimeError(f"Failed to save combined voice: {e}")

            except Exception as e:
                logger.warning(f"Failed to save combined voice: {e}")
                # Continue without saving - will be combined on-the-fly when needed

        return combined_name

    async def list_voices(self) -> List[str]:
        """List available voices.
        
        Returns:
            List of voice names
        """
        voices = set()  # Use set to avoid duplicates
        try:
            # Get voices from disk
            api_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            voices_dir = os.path.join(api_dir, settings.voices_dir)
            os.makedirs(voices_dir, exist_ok=True)
            
            for entry in os.listdir(voices_dir):
                if entry.endswith(".pt"):
                    voices.add(entry[:-3])
                
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
        return sorted(list(voices))

    def validate_voice(self, voice_path: str) -> bool:
        """Validate voice file.
        
        Args:
            voice_path: Path to voice file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not os.path.exists(voice_path):
                return False
            voice = torch.load(voice_path, map_location="cpu")
            return isinstance(voice, torch.Tensor)
        except Exception:
            return False

    async def create_weighted_voice(
    self,
    formula: str, 
    normalize: bool = False,
    device: str = "cpu",
    ) -> str:
        """
        Parse the voice formula string (e.g. '0.3 * voiceA + 0.5 * voiceB')
        and return a combined torch.Tensor representing the weighted sum
        of the given voices.

        Args:
            formula: Weighted voice formula string.
            voice_manager: A class that has a `load_voice(voice_name, device)` -> Tensor
            device: 'cpu' or 'cuda' for the final tensor.
            normalize: If True, divide the final result by the sum of the weights
                    so the total "magnitude" remains consistent.

        Returns:
            A torch.Tensor containing the combined voice embedding.
        """
        pairs = self.parse_voice_formula(formula)  # [(weight, voiceName), (weight, voiceName), ...]
        
        # Validate the pairs
        for weight, voice_name in pairs:
            if weight <= 0:
                raise ValueError(f"Invalid weight {weight} for voice {voice_name}.")

        if not pairs:
            raise ValueError("No valid weighted voices found in formula.")

        # Keep track of total weight if we plan to normalize.
        total_weight = 0.0
        weighted_sum = None
        combined_name = ""

        for weight, voice_name in pairs:
            # 1) Load each base voice from your manager/service
            base_voice = await self.load_voice(voice_name, device=device)
            
            
            # 3) Combine the base voices using the weights
            if combined_name == "":
                combined_name = voice_name
            else:
                combined_name += f"+{voice_name}"

               
        
            
            # 2) Multiply by weight and accumulate
            if weighted_sum is None:
                # Clone so we don't modify the base voice in memory
                weighted_sum = base_voice.clone() * weight
            else:
                weighted_sum += (base_voice * weight)
            
            total_weight += weight

        if weighted_sum is None:
            raise ValueError("No voices were combined. Check the formula syntax.")

        # Optional normalization
        if normalize and total_weight != 0.0:
            weighted_sum /= total_weight
        
        if settings.allow_local_voice_saving:
            
            # Save to disk
            api_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            voices_dir = os.path.join(api_dir, settings.voices_dir)
            os.makedirs(voices_dir, exist_ok=True)
            combined_path = os.path.join(voices_dir, f"{formula}.pt")
            try:
                torch.save(weighted_sum, combined_path)
            except Exception as e:
                logger.warning(f"Failed to save combined voice: {e}")
                # Continue without saving - will be combined on-the-fly when needed

        return combined_name
        
      
    

    def parse_voice_formula(self,formula: str) -> List[tuple[float, str]]:
        """
        Parse the voice formula string (e.g. '0.3 * voiceA + 0.5 * voiceB')
        and return a list of (weight, voiceName) pairs.
        Args:
            formula: Weighted voice formula string.
        Returns:
            List of (weight, voiceName) pairs.
        """
        pairs = []
        parts = formula.split('+')
        for part in parts:
            weight, voice_name = part.strip().split('*')
            pairs.append((float(weight), voice_name.strip()))
        return pairs

    @property
    def cache_info(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache info
        """
        return {
            'size': len(self._voice_cache),
            'max_size': self._config.cache_size
        }


# Global singleton instance and lock
_manager_instance = None


async def get_manager(config: Optional[VoiceConfig] = None) -> VoiceManager:
    """Get global voice manager instance.
    
    Args:
        config: Optional voice configuration
        
    Returns:
        VoiceManager instance
    """
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = VoiceManager(config)
    return _manager_instance