"""
Automatic transcription service using Whisper for ZipVoice voice prompts.

This eliminates the need to manually provide prompt_text by auto-transcribing audio.
"""

import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from loguru import logger

from api.src.core.config import settings


class AutoTranscriptionService:
    """Automatic transcription service for voice prompts using Whisper."""

    def __init__(self, model_size: str = "base"):
        """Initialize auto-transcription service.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self._model = None
        self._whisper_available = False

        # Check if Whisper is available
        try:
            import whisper
            self._whisper_available = True
            logger.info(f"Whisper available, using {model_size} model")
        except ImportError:
            logger.warning("Whisper not available. Install with: pip install openai-whisper")

    def _load_model(self):
        """Lazy load Whisper model."""
        if self._model is None and self._whisper_available:
            try:
                import whisper
                logger.info(f"Loading Whisper {self.model_size} model...")
                self._model = whisper.load_model(self.model_size)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self._whisper_available = False

    async def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Optional[str]:
        """Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code (en, es, etc.) or None for auto-detect
            task: "transcribe" or "translate" (to English)

        Returns:
            Transcription text or None if failed
        """
        if not self._whisper_available:
            logger.warning("Whisper not available, cannot transcribe")
            return None

        # Load model if not already loaded
        self._load_model()

        if self._model is None:
            return None

        try:
            # Transcribe
            logger.info(f"Transcribing audio: {audio_path}")

            options = {"task": task}
            if language:
                options["language"] = language

            result = self._model.transcribe(audio_path, **options)

            transcription = result["text"].strip()
            detected_language = result.get("language", "unknown")

            logger.info(f"Transcription complete ({detected_language}): {transcription[:100]}...")

            return transcription

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    async def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None
    ) -> Optional[str]:
        """Transcribe audio from bytes.

        Args:
            audio_bytes: Audio data as bytes
            language: Language code or None for auto-detect

        Returns:
            Transcription text or None if failed
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            return await self.transcribe_audio(temp_path, language=language)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    async def transcribe_from_numpy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None
    ) -> Optional[str]:
        """Transcribe audio from numpy array.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            language: Language code or None for auto-detect

        Returns:
            Transcription text or None if failed
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            temp_path = f.name

        try:
            return await self.transcribe_audio(temp_path, language=language)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def is_available(self) -> bool:
        """Check if Whisper transcription is available."""
        return self._whisper_available

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        if not self._whisper_available:
            return []

        try:
            import whisper
            return list(whisper.tokenizer.LANGUAGES.keys())
        except:
            return []


# Global instance
_auto_transcription_service: Optional[AutoTranscriptionService] = None


def get_auto_transcription_service(model_size: str = "base") -> AutoTranscriptionService:
    """Get global auto-transcription service instance.

    Args:
        model_size: Whisper model size

    Returns:
        AutoTranscriptionService instance
    """
    global _auto_transcription_service

    if _auto_transcription_service is None:
        _auto_transcription_service = AutoTranscriptionService(model_size)

    return _auto_transcription_service
