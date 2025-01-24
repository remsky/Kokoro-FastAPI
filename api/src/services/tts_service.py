"""TTS service using model and voice managers."""

import io
import time
from typing import List, Tuple, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from loguru import logger

from ..core.config import settings
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from .audio import AudioNormalizer, AudioService
from .text_processing import chunker, normalize_text, process_text


import asyncio

class TTSService:
    """Text-to-speech service."""

    # Limit concurrent chunk processing
    _chunk_semaphore = asyncio.Semaphore(4)

    def __init__(self, output_dir: str = None):
        """Initialize service.
        
        Args:
            output_dir: Optional output directory for saving audio
        """
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None

    @classmethod
    async def create(cls, output_dir: str = None) -> 'TTSService':
        """Create and initialize TTSService instance.
        
        Args:
            output_dir: Optional output directory for saving audio
            
        Returns:
            Initialized TTSService instance
        """
        service = cls(output_dir)
        # Initialize managers
        service.model_manager = await get_model_manager()
        service._voice_manager = await get_voice_manager()
        return service

    async def generate_audio(
        self, text: str, voice: str, speed: float = 1.0, stitch_long_output: bool = True
    ) -> Tuple[np.ndarray, float]:
        """Generate audio for text.
        
        Args:
            text: Input text
            voice: Voice name
            speed: Speed multiplier
            stitch_long_output: Whether to stitch together long outputs
            
        Returns:
            Audio samples and processing time
            
        Raises:
            ValueError: If text is empty after preprocessing or no chunks generated
            RuntimeError: If audio generation fails
        """
        start_time = time.time()
        voice_tensor = None

        try:
            # Normalize text
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Get backend and load voice
            backend = self.model_manager.get_backend()
            voice_tensor = await self._voice_manager.load_voice(voice, device=backend.device)

            # Get all chunks upfront
            chunks = list(chunker.split_text(text))
            if not chunks:
                raise ValueError("No text chunks to process")

            # Process chunk with concurrency control
            async def process_chunk(chunk: str) -> Optional[np.ndarray]:
                async with self._chunk_semaphore:
                    try:
                        tokens = process_text(chunk)
                        if not tokens:
                            return None
                        
                        # Generate audio
                        return await self.model_manager.generate(
                            tokens,
                            voice_tensor,
                            speed=speed
                        )
                    except Exception as e:
                        logger.error(f"Failed to process chunk: '{chunk}'. Error: {str(e)}")
                        return None

            # Process all chunks concurrently
            chunk_results = await asyncio.gather(*[
                process_chunk(chunk) for chunk in chunks
            ])

            # Filter out None results and combine
            audio_chunks = [chunk for chunk in chunk_results if chunk is not None]
            if not audio_chunks:
                raise ValueError("No audio chunks were generated successfully")

            # Combine chunks
            audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise
        finally:
            # Always clean up voice tensor
            if voice_tensor is not None:
                del voice_tensor
                torch.cuda.empty_cache()

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav",
    ):
        """Generate and stream audio chunks.
        
        Args:
            text: Input text
            voice: Voice name
            speed: Speed multiplier
            output_format: Output audio format
            
        Yields:
            Audio chunks as bytes
        """
        # Setup audio processing
        stream_normalizer = AudioNormalizer()
        voice_tensor = None
        
        try:
            # Normalize text
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Get backend and load voice
            backend = self.model_manager.get_backend()
            voice_tensor = await self._voice_manager.load_voice(voice, device=backend.device)

            # Get all chunks upfront
            chunks = list(chunker.split_text(text))
            if not chunks:
                raise ValueError("No text chunks to process")

            # Process chunk with concurrency control
            async def process_chunk(chunk: str, is_first: bool, is_last: bool) -> Optional[bytes]:
                async with self._chunk_semaphore:
                    try:
                        tokens = process_text(chunk)
                        if not tokens:
                            return None

                        # Generate audio and phenome duration
                        chunk_audio,durations = await self.model_manager.generate(
                            tokens,
                            voice_tensor,
                            speed=speed
                        )

                        if chunk_audio is not None:
                            # Convert to bytes
                            return AudioService.convert_audio(
                                chunk_audio,
                                24000,
                                output_format,
                                is_first_chunk=is_first,
                                normalizer=stream_normalizer,
                                is_last_chunk=is_last,
                                stream=True
                            )
                    except Exception as e:
                        logger.error(f"Failed to generate audio for chunk: '{chunk}'. Error: {str(e)}")
                        return None

            # Create tasks for all chunks
            tasks = [
                process_chunk(chunk, i==0, i==len(chunks)-1)
                for i, chunk in enumerate(chunks)
            ]

            # Process chunks concurrently and yield results in order
            for chunk_bytes in await asyncio.gather(*tasks):
                if chunk_bytes is not None:
                    yield chunk_bytes

        except Exception as e:
            logger.error(f"Error in audio generation stream: {str(e)}")
            raise
        finally:
            # Always clean up voice tensor
            if voice_tensor is not None:
                del voice_tensor
                torch.cuda.empty_cache()

    async def combine_voices(self, voices: List[str]) -> str:
        """Combine multiple voices.
        
        Args:
            voices: List of voice names
            
        Returns:
            Name of combined voice
        """
        return await self._voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices.
        
        Returns:
            List of voice names
        """
        return await self._voice_manager.list_voices()

    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio to WAV bytes.
        
        Args:
            audio: Audio samples
            
        Returns:
            WAV bytes
        """
        buffer = io.BytesIO()
        wavfile.write(buffer, 24000, audio)
        return buffer.getvalue()