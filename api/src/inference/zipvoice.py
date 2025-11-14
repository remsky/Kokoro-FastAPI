"""ZipVoice TTS backend with zero-shot voice cloning."""

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from api.src.core.config import settings
from api.src.inference.base import AudioChunk, BaseModelBackend


class ZipVoiceBackend(BaseModelBackend):
    """ZipVoice TTS backend with zero-shot voice cloning.

    Uses the ZipVoice model for fast, high-quality speech synthesis
    with zero-shot voice cloning capabilities.
    """

    def __init__(self, model_name: str = None):
        """Initialize ZipVoice backend.

        Args:
            model_name: Model variant to use (zipvoice, zipvoice_distill, etc.)
        """
        super().__init__()
        self._backend_type = "zipvoice"
        self.model_name = model_name or settings.zipvoice_model
        self._is_loaded = False

        # Detect device
        self._device = settings.get_device()
        logger.info(f"ZipVoice backend initializing on device: {self._device}")

        # Check if ZipVoice is available
        try:
            import zipvoice
            self.zipvoice_available = True
            logger.info(f"ZipVoice library available, using model: {self.model_name}")
        except ImportError:
            self.zipvoice_available = False
            logger.warning("ZipVoice library not found. Will use command-line interface.")

    async def load_model(self, path: str = None) -> None:
        """Load ZipVoice model.

        Args:
            path: Not used for ZipVoice (models auto-download)

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading ZipVoice model: {self.model_name}")

            # ZipVoice models are loaded on-demand during inference
            # We just verify the package is available
            if not self.zipvoice_available:
                logger.warning("ZipVoice package not available, using CLI fallback")

            self._is_loaded = True
            logger.info(f"ZipVoice backend ready (model: {self.model_name})")

        except Exception as e:
            logger.error(f"Failed to load ZipVoice model: {e}")
            raise RuntimeError(f"ZipVoice model loading failed: {e}")

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        **kwargs
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio from text using ZipVoice.

        Args:
            text: Input text to synthesize
            voice: For ZipVoice, this should be the prompt_wav path
            speed: Speed multiplier (applied via processing)
            **kwargs: Additional parameters:
                - prompt_text: Transcription of prompt_wav (required)
                - num_steps: Number of inference steps (default: 8)
                - remove_long_silence: Remove silences (default: True)
                - max_duration: Maximum duration constraint

        Yields:
            AudioChunk objects with generated audio

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        # Extract ZipVoice-specific parameters
        prompt_text = kwargs.get('prompt_text')
        num_steps = kwargs.get('num_steps', settings.zipvoice_num_steps)
        remove_long_silence = kwargs.get('remove_long_silence', settings.zipvoice_remove_long_silence)
        max_duration = kwargs.get('max_duration')

        # Get prompt_wav path
        if isinstance(voice, tuple):
            prompt_wav = voice[1]
        else:
            prompt_wav = voice

        # Validate inputs
        if not prompt_text:
            raise ValueError("prompt_text is required for ZipVoice generation")
        if not prompt_wav or not os.path.exists(str(prompt_wav)):
            raise ValueError(f"prompt_wav file not found: {prompt_wav}")

        logger.info(f"Generating audio with ZipVoice (model: {self.model_name})")
        logger.debug(f"Text: {text[:100]}...")
        logger.debug(f"Prompt: {prompt_wav}, Transcription: {prompt_text}")

        # Split text into chunks for streaming-like behavior
        chunks = self._split_text_into_chunks(text)
        logger.debug(f"Split into {len(chunks)} chunks")

        # Generate each chunk
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            logger.debug(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")

            # Generate audio for this chunk
            audio_data = await self._generate_chunk(
                text=chunk,
                prompt_wav=prompt_wav,
                prompt_text=prompt_text,
                num_steps=num_steps,
                remove_long_silence=remove_long_silence,
                max_duration=max_duration,
                speed=speed
            )

            # Convert to AudioChunk format
            if audio_data is not None:
                # Apply speed adjustment if needed
                if speed != 1.0:
                    audio_data = self._apply_speed(audio_data, speed)

                chunk_obj = AudioChunk(
                    audio=audio_data,
                    word_timestamps=[],  # ZipVoice doesn't provide timestamps
                    output=b""
                )
                yield chunk_obj

        logger.info("ZipVoice generation complete")

    async def _generate_chunk(
        self,
        text: str,
        prompt_wav: str,
        prompt_text: str,
        num_steps: int,
        remove_long_silence: bool,
        max_duration: Optional[float],
        speed: float
    ) -> Optional[np.ndarray]:
        """Generate a single chunk using ZipVoice.

        Returns:
            Audio data as numpy array (24kHz, mono, float32)
        """
        try:
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name

            # Build command for ZipVoice CLI
            cmd = [
                'python3', '-m', 'zipvoice.bin.infer_zipvoice',
                '--model-name', self.model_name,
                '--prompt-wav', prompt_wav,
                '--prompt-text', prompt_text,
                '--text', text,
                '--res-wav-path', output_path,
                '--num-steps', str(num_steps)
            ]

            if remove_long_silence:
                cmd.append('--remove-long-sil')

            if max_duration:
                cmd.extend(['--max-duration', str(max_duration)])

            if speed != 1.0:
                # ZipVoice has --speed parameter
                cmd.extend(['--speed', str(1.0 / speed)])  # Inverse for duration control

            # Run inference
            logger.debug(f"Running ZipVoice command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"ZipVoice inference failed: {stderr.decode()}")
                return None

            # Read generated audio
            if os.path.exists(output_path):
                audio_data, sample_rate = sf.read(output_path)

                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                # Ensure float32
                audio_data = audio_data.astype(np.float32)

                # Clean up temp file
                try:
                    os.unlink(output_path)
                except:
                    pass

                logger.debug(f"Generated {len(audio_data)/sample_rate:.2f}s of audio at {sample_rate}Hz")
                return audio_data
            else:
                logger.error(f"Output file not created: {output_path}")
                return None

        except Exception as e:
            logger.error(f"ZipVoice generation failed: {e}")
            return None

    def _split_text_into_chunks(self, text: str, max_chunk_length: int = 200) -> list:
        """Split text into chunks for pseudo-streaming.

        Args:
            text: Input text
            max_chunk_length: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentence_endings = r'[.!?。！？]\s+'
        sentences = re.split(sentence_endings, text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not sentence.strip():
                continue

            # If adding this sentence exceeds max length, save current chunk
            if current_chunk and len(current_chunk) + len(sentence) > max_chunk_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Apply speed adjustment to audio.

        Args:
            audio: Input audio array
            speed: Speed multiplier (>1 = faster, <1 = slower)

        Returns:
            Speed-adjusted audio
        """
        if speed == 1.0:
            return audio

        try:
            import librosa
            # Use librosa for time stretching
            return librosa.effects.time_stretch(audio, rate=speed)
        except ImportError:
            logger.warning("librosa not available, speed adjustment skipped")
            return audio

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def unload(self) -> None:
        """Unload model and free resources."""
        self._is_loaded = False
        logger.info("ZipVoice backend unloaded")

        # Clear CUDA cache if applicable
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class ZipVoiceDialogBackend(ZipVoiceBackend):
    """ZipVoice backend for dialogue generation (multi-speaker).

    Supports stereo output with speakers in separate channels.
    """

    def __init__(self, model_name: str = "zipvoice_dialog"):
        """Initialize ZipVoice dialogue backend."""
        super().__init__(model_name=model_name)
        self._backend_type = "zipvoice_dialog"
        logger.info("Initialized ZipVoice dialogue backend")

    async def generate_dialog(
        self,
        text: str,
        speaker1_prompt: Tuple[str, str],  # (wav_path, transcription)
        speaker2_prompt: Tuple[str, str],  # (wav_path, transcription)
        num_steps: int = None,
        stereo: bool = True
    ) -> np.ndarray:
        """Generate dialogue audio with multiple speakers.

        Args:
            text: Dialogue text with [S1] and [S2] markers
            speaker1_prompt: Voice prompt for speaker 1
            speaker2_prompt: Voice prompt for speaker 2
            num_steps: Inference steps
            stereo: Output stereo (speakers in separate channels)

        Returns:
            Generated audio array
        """
        num_steps = num_steps or settings.zipvoice_num_steps
        model = "zipvoice_dialog_stereo" if stereo else "zipvoice_dialog"

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name

            # Build command
            cmd = [
                'python3', '-m', 'zipvoice.bin.infer_zipvoice_dialog',
                '--model-name', model,
                '--spk1-prompt-wav', speaker1_prompt[0],
                '--spk1-prompt-text', speaker1_prompt[1],
                '--spk2-prompt-wav', speaker2_prompt[0],
                '--spk2-prompt-text', speaker2_prompt[1],
                '--text', text,
                '--res-wav-path', output_path,
                '--num-steps', str(num_steps)
            ]

            # Run inference
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"ZipVoice dialog inference failed: {stderr.decode()}")
                raise RuntimeError(f"Dialog generation failed: {stderr.decode()}")

            # Read result
            audio_data, sample_rate = sf.read(output_path)

            # Clean up
            try:
                os.unlink(output_path)
            except:
                pass

            logger.info(f"Generated dialogue: {len(audio_data)/sample_rate:.2f}s")
            return audio_data.astype(np.float32)

        except Exception as e:
            logger.error(f"Dialog generation failed: {e}")
            raise RuntimeError(f"Failed to generate dialogue: {e}")
