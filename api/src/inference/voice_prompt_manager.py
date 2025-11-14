"""Voice Prompt Manager for ZipVoice zero-shot voice cloning."""

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx
import soundfile as sf
from loguru import logger

from api.src.core.config import settings


class VoicePromptManager:
    """Manages voice prompts for ZipVoice zero-shot voice cloning.

    Supports multiple input methods:
    - Pre-registered voice names (cached locally)
    - URL downloads (with caching)
    - Base64 encoded audio (decoded and cached)
    - Direct file paths
    """

    def __init__(self, cache_dir: str = None):
        """Initialize the voice prompt manager.

        Args:
            cache_dir: Directory to store cached voice prompts
        """
        self.cache_dir = Path(cache_dir or settings.zipvoice_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Registry file stores voice name -> {audio_path, transcription}
        self.registry_file = self.cache_dir / "voice_registry.json"
        self.registry: Dict[str, Dict[str, str]] = self._load_registry()

        # URL cache directory
        self.url_cache_dir = self.cache_dir / "url_cache"
        self.url_cache_dir.mkdir(parents=True, exist_ok=True)

        # Base64 cache directory
        self.base64_cache_dir = self.cache_dir / "base64_cache"
        self.base64_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VoicePromptManager initialized with cache dir: {self.cache_dir}")
        logger.info(f"Loaded {len(self.registry)} registered voices")

    def _load_registry(self) -> Dict[str, Dict[str, str]]:
        """Load voice registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load voice registry: {e}")
                return {}
        return {}

    def _save_registry(self) -> None:
        """Save voice registry to disk."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
            logger.debug(f"Saved voice registry with {len(self.registry)} entries")
        except Exception as e:
            logger.error(f"Failed to save voice registry: {e}")

    def register_voice(self, name: str, audio_path: str, transcription: str) -> None:
        """Register a voice prompt for reuse.

        Args:
            name: Voice identifier
            audio_path: Path to audio file
            transcription: Text transcription of the audio

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio validation fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate audio file
        if not self.validate_audio(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")

        # Copy to cache if not already there
        cache_path = self.cache_dir / f"{name}.wav"
        if str(Path(audio_path).resolve()) != str(cache_path.resolve()):
            # Convert to WAV if needed
            data, samplerate = sf.read(audio_path)
            sf.write(str(cache_path), data, samplerate)
            logger.info(f"Copied voice prompt to cache: {cache_path}")

        # Update registry
        self.registry[name] = {
            "audio_path": str(cache_path),
            "transcription": transcription
        }
        self._save_registry()
        logger.info(f"Registered voice: {name}")

    def get_voice_prompt(self, identifier: str) -> Tuple[str, str]:
        """Get voice prompt by name.

        Args:
            identifier: Voice name from registry

        Returns:
            Tuple of (audio_path, transcription)

        Raises:
            KeyError: If voice not found in registry
        """
        if identifier not in self.registry:
            raise KeyError(f"Voice not found in registry: {identifier}. "
                          f"Available voices: {list(self.registry.keys())}")

        voice_data = self.registry[identifier]
        audio_path = voice_data["audio_path"]
        transcription = voice_data["transcription"]

        if not os.path.exists(audio_path):
            logger.warning(f"Cached audio file missing: {audio_path}")
            raise FileNotFoundError(f"Cached audio file not found: {audio_path}")

        return audio_path, transcription

    async def download_from_url(self, url: str) -> str:
        """Download audio from URL and cache it.

        Args:
            url: URL to audio file

        Returns:
            Path to cached audio file

        Raises:
            ValueError: If URL download is disabled or URL is invalid
            httpx.HTTPError: If download fails
        """
        if not settings.zipvoice_allow_url_download:
            raise ValueError("URL downloads are disabled in settings")

        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        # Create cache filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.url_cache_dir / f"{url_hash}.wav"

        # Return cached if exists
        if cache_path.exists():
            logger.debug(f"Using cached audio from URL: {url}")
            return str(cache_path)

        # Download
        logger.info(f"Downloading audio from URL: {url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            # Check size
            content_length = int(response.headers.get('content-length', 0))
            max_size = int(settings.zipvoice_max_download_size_mb * 1024 * 1024)
            if content_length > max_size:
                raise ValueError(f"File too large: {content_length} bytes "
                               f"(max: {max_size} bytes)")

            # Save to temp file first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

        try:
            # Validate and convert to WAV
            data, samplerate = sf.read(tmp_path)
            sf.write(str(cache_path), data, samplerate)
            logger.info(f"Cached audio from URL: {cache_path}")
            return str(cache_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def decode_base64(self, data: str) -> str:
        """Decode base64 audio and save to cache.

        Args:
            data: Base64 encoded audio data (with or without data URI prefix)

        Returns:
            Path to cached audio file

        Raises:
            ValueError: If base64 is disabled or decoding fails
        """
        if not settings.zipvoice_allow_base64:
            raise ValueError("Base64 audio input is disabled in settings")

        # Strip data URI prefix if present (e.g., "data:audio/wav;base64,...")
        if data.startswith('data:'):
            # Find the comma that separates metadata from data
            comma_idx = data.find(',')
            if comma_idx != -1:
                data = data[comma_idx + 1:]

        # Create cache filename from data hash
        data_hash = hashlib.md5(data.encode()).hexdigest()
        cache_path = self.base64_cache_dir / f"{data_hash}.wav"

        # Return cached if exists
        if cache_path.exists():
            logger.debug(f"Using cached base64 audio: {cache_path}")
            return str(cache_path)

        # Decode
        try:
            audio_bytes = base64.b64decode(data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {e}")

        # Save to temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # Validate and convert to WAV
            data, samplerate = sf.read(tmp_path)
            sf.write(str(cache_path), data, samplerate)
            logger.info(f"Cached base64 audio: {cache_path}")
            return str(cache_path)
        except Exception as e:
            raise ValueError(f"Invalid audio data: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def validate_audio(self, path: str, max_duration: float = None) -> bool:
        """Validate audio file.

        Args:
            path: Path to audio file
            max_duration: Maximum allowed duration in seconds

        Returns:
            True if valid, False otherwise
        """
        try:
            data, samplerate = sf.read(path)

            # Check duration
            duration = len(data) / samplerate
            max_dur = max_duration or settings.zipvoice_max_prompt_duration
            if duration > max_dur:
                logger.warning(f"Audio duration {duration}s exceeds maximum {max_dur}s")
                return False

            logger.debug(f"Audio validation passed: {path} ({duration:.2f}s, {samplerate}Hz)")
            return True
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False

    def get_audio_info(self, path: str) -> Dict[str, any]:
        """Get audio file information.

        Args:
            path: Path to audio file

        Returns:
            Dictionary with audio metadata
        """
        try:
            data, samplerate = sf.read(path)
            duration = len(data) / samplerate
            channels = 1 if len(data.shape) == 1 else data.shape[1]

            return {
                "duration": duration,
                "samplerate": samplerate,
                "channels": channels,
                "samples": len(data),
                "format": Path(path).suffix
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {}

    def list_voices(self) -> Dict[str, Dict[str, str]]:
        """List all registered voices.

        Returns:
            Dictionary of voice_name -> {audio_path, transcription}
        """
        return self.registry.copy()

    def delete_voice(self, name: str) -> bool:
        """Delete a registered voice.

        Args:
            name: Voice identifier

        Returns:
            True if deleted, False if not found
        """
        if name not in self.registry:
            return False

        # Delete audio file if in cache
        voice_data = self.registry[name]
        audio_path = Path(voice_data["audio_path"])
        if audio_path.exists() and audio_path.parent == self.cache_dir:
            try:
                audio_path.unlink()
                logger.info(f"Deleted audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to delete audio file: {e}")

        # Remove from registry
        del self.registry[name]
        self._save_registry()
        logger.info(f"Deleted voice: {name}")
        return True

    def clear_cache(self, cache_type: str = "all") -> int:
        """Clear cached audio files.

        Args:
            cache_type: Type of cache to clear ("url", "base64", or "all")

        Returns:
            Number of files deleted
        """
        count = 0

        if cache_type in ["url", "all"]:
            for file in self.url_cache_dir.glob("*.wav"):
                try:
                    file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file}: {e}")

        if cache_type in ["base64", "all"]:
            for file in self.base64_cache_dir.glob("*.wav"):
                try:
                    file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file}: {e}")

        logger.info(f"Cleared {count} cached files ({cache_type})")
        return count


# Global instance (will be initialized in main.py)
voice_prompt_manager: Optional[VoicePromptManager] = None


def get_voice_prompt_manager() -> VoicePromptManager:
    """Get the global voice prompt manager instance."""
    global voice_prompt_manager
    if voice_prompt_manager is None:
        voice_prompt_manager = VoicePromptManager()
    return voice_prompt_manager
