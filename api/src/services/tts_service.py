"""TTS service using model and voice managers."""

import asyncio
import os
import tempfile
import time
from typing import AsyncGenerator, List, Optional, Tuple, Union

from ..inference.base import AudioChunk
import numpy as np
import torch
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from ..inference.kokoro_v1 import KokoroV1
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from .audio import AudioNormalizer, AudioService
from .text_processing import tokenize
from .text_processing.text_processor import process_text_chunk, smart_split
from ..structures.schemas import NormalizationOptions

class TTSService:
    """Text-to-speech service."""

    # Limit concurrent chunk processing
    _chunk_semaphore = asyncio.Semaphore(4)

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None

    @classmethod
    async def create(cls, output_dir: str = None) -> "TTSService":
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        service.model_manager = await get_model_manager()
        service._voice_manager = await get_voice_manager()
        return service

    async def _process_chunk(
        self,
        chunk_text: str,
        tokens: List[int],
        voice_name: str,
        voice_path: str,
        speed: float,
        output_format: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False,
        normalizer: Optional[AudioNormalizer] = None,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Process tokens into audio."""
        async with self._chunk_semaphore:
            try:
                # Handle stream finalization
                if is_last:
                    # Skip format conversion for raw audio mode
                    if not output_format:
                        yield AudioChunk(np.array([], dtype=np.int16),output=b'')
                        return
                    chunk_data = await AudioService.convert_audio(
                        AudioChunk(np.array([], dtype=np.float32)),  # Dummy data for type checking
                        24000,
                        output_format,
                        speed,
                        "",
                        is_first_chunk=False,
                        normalizer=normalizer,
                        is_last_chunk=True,
                    )
                    yield chunk_data
                    return

                # Skip empty chunks
                if not tokens and not chunk_text:
                    return

                # Get backend
                backend = self.model_manager.get_backend()

                # Generate audio using pre-warmed model
                if isinstance(backend, KokoroV1):
                    chunk_index=0
                    # For Kokoro V1, pass text and voice info with lang_code
                    async for chunk_data in self.model_manager.generate(
                        chunk_text,
                        (voice_name, voice_path),
                        speed=speed,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    ):
                        # For streaming, convert to bytes
                        if output_format:
                            try:
                                chunk_data = await AudioService.convert_audio(
                                    chunk_data,
                                    24000,
                                    output_format,
                                    speed,
                                    chunk_text,
                                    is_first_chunk=is_first and chunk_index == 0,
                                    is_last_chunk=is_last,
                                    normalizer=normalizer,
                                )
                                yield chunk_data
                            except Exception as e:
                                logger.error(f"Failed to convert audio: {str(e)}")
                        else:
                            chunk_data = AudioService.trim_audio(chunk_data,
                                                                    chunk_text,
                                                                    speed,
                                                                    is_last,
                                                                    normalizer)
                            yield chunk_data
                        chunk_index+=1
                else:

                    # For legacy backends, load voice tensor
                    voice_tensor = await self._voice_manager.load_voice(
                        voice_name, device=backend.device
                    )
                    chunk_data = await self.model_manager.generate(
                        tokens, voice_tensor, speed=speed, return_timestamps=return_timestamps
                    )

                    if chunk_data.audio is None:
                        logger.error("Model generated None for audio chunk")
                        return

                    if len(chunk_data.audio) == 0:
                        logger.error("Model generated empty audio chunk")
                        return

                    # For streaming, convert to bytes
                    if output_format:
                        try:
                            chunk_data = await AudioService.convert_audio(
                                chunk_data,
                                24000,
                                output_format,
                                speed,
                                chunk_text,
                                is_first_chunk=is_first,
                                normalizer=normalizer,
                                is_last_chunk=is_last,
                            )
                            yield chunk_data
                        except Exception as e:
                            logger.error(f"Failed to convert audio: {str(e)}")
                    else:
                        trimmed = AudioService.trim_audio(chunk_data,
                                                                    chunk_text,
                                                                    speed,
                                                                    is_last,
                                                                    normalizer)
                        yield trimmed
            except Exception as e:
                logger.error(f"Failed to process tokens: {str(e)}")

    async def _get_voice_path(self, voice: str) -> Tuple[str, str]:
        """Get voice path, handling combined voices.

        Args:
            voice: Voice name or combined voice names (e.g., 'af_jadzia+af_jessica')

        Returns:
            Tuple of (voice name to use, voice path to use)

        Raises:
            RuntimeError: If voice not found
        """
        try:
            # Check if it's a combined voice
            if "+" in voice:
                # Split on + but preserve any parentheses
                voice_parts = []
                weights = []
                for part in voice.split("+"):
                    part = part.strip()
                    if not part:
                        continue
                    # Extract voice name and weight if present
                    if "(" in part and ")" in part:
                        voice_name = part.split("(")[0].strip()
                        weight = float(part.split("(")[1].split(")")[0])
                    else:
                        voice_name = part
                        weight = 1.0
                    voice_parts.append(voice_name)
                    weights.append(weight)

                if len(voice_parts) < 2:
                    raise RuntimeError(f"Invalid combined voice name: {voice}")

                # Normalize weights to sum to 1
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Load and combine voices
                voice_tensors = []
                for v, w in zip(voice_parts, weights):
                    path = await self._voice_manager.get_voice_path(v)
                    if not path:
                        raise RuntimeError(f"Voice not found: {v}")
                    logger.debug(f"Loading voice tensor from: {path}")
                    voice_tensor = torch.load(path, map_location="cpu")
                    voice_tensors.append(voice_tensor * w)

                # Sum the weighted voice tensors
                logger.debug(
                    f"Combining {len(voice_tensors)} voice tensors with weights {weights}"
                )
                combined = torch.sum(torch.stack(voice_tensors), dim=0)

                # Save combined tensor
                temp_dir = tempfile.gettempdir()
                combined_path = os.path.join(temp_dir, f"{voice}.pt")
                logger.debug(f"Saving combined voice to: {combined_path}")
                torch.save(combined, combined_path)

                return voice, combined_path
            else:
                # Single voice
                if "(" in voice and ")" in voice:
                    voice = voice.split("(")[0].strip()
                path = await self._voice_manager.get_voice_path(voice)
                if not path:
                    raise RuntimeError(f"Voice not found: {voice}")
                logger.debug(f"Using single voice path: {path}")
                return voice, path
        except Exception as e:
            logger.error(f"Failed to get voice path: {e}")
            raise

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav",
        lang_code: Optional[str] = None,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate and stream audio chunks."""
        stream_normalizer = AudioNormalizer()
        chunk_index = 0
        current_offset=0.0
        try:
            # Get backend
            backend = self.model_manager.get_backend()

            # Get voice path, handling combined voices
            voice_name, voice_path = await self._get_voice_path(voice)
            logger.debug(f"Using voice path: {voice_path}")

            # Use provided lang_code or determine from voice name
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(
                f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in audio stream"
            )
            
            
            # Process text in chunks with smart splitting
            async for chunk_text, tokens in smart_split(text,lang_code=lang_code,normalization_options=normalization_options):
                try:
                    # Process audio for chunk
                    async for chunk_data in self._process_chunk(
                        chunk_text,  # Pass text for Kokoro V1
                        tokens,  # Pass tokens for legacy backends
                        voice_name,  # Pass voice name
                        voice_path,  # Pass voice path
                        speed,
                        output_format,
                        is_first=(chunk_index == 0),
                        is_last=False,  # We'll update the last chunk later
                        normalizer=stream_normalizer,
                        lang_code=pipeline_lang_code,  # Pass lang_code
                        return_timestamps=return_timestamps,
                    ):
                        if chunk_data.word_timestamps is not None:
                            for timestamp in chunk_data.word_timestamps:
                                timestamp.start_time+=current_offset
                                timestamp.end_time+=current_offset
                        
                        current_offset+=len(chunk_data.audio) / 24000
                        
                        if chunk_data.output is not None:
                            yield chunk_data
                            
                        else:
                            logger.warning(
                                f"No audio generated for chunk: '{chunk_text[:100]}...'"
                            )
                        chunk_index += 1
                except Exception as e:
                    logger.error(
                        f"Failed to process audio for chunk: '{chunk_text[:100]}...'. Error: {str(e)}"
                    )
                    continue

            # Only finalize if we successfully processed at least one chunk
            if chunk_index > 0:
                try:
                    # Empty tokens list to finalize audio
                    async for chunk_data in self._process_chunk(
                        "",  # Empty text
                        [],  # Empty tokens
                        voice_name,
                        voice_path,
                        speed,
                        output_format,
                        is_first=False,
                        is_last=True,  # Signal this is the last chunk
                        normalizer=stream_normalizer,
                        lang_code=pipeline_lang_code,  # Pass lang_code
                    ):
                        if chunk_data.output is not None:
                            yield chunk_data
                except Exception as e:
                    logger.error(f"Failed to finalize audio stream: {str(e)}")

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise e
        

    async def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        return_timestamps: bool = False,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        lang_code: Optional[str] = None,
    ) -> AudioChunk:
        """Generate complete audio for text using streaming internally."""
        audio_data_chunks=[]
  
        try:
            async for audio_stream_data in self.generate_audio_stream(text,voice,speed=speed,normalization_options=normalization_options,return_timestamps=return_timestamps,lang_code=lang_code,output_format=None):
                if len(audio_stream_data.audio) > 0:
                    audio_data_chunks.append(audio_stream_data)


            combined_audio_data=AudioChunk.combine(audio_data_chunks)
            return combined_audio_data
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise
        

    async def combine_voices(self, voices: List[str]) -> torch.Tensor:
        """Combine multiple voices.

        Returns:
            Combined voice tensor
        """
        return await self._voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices."""
        return await self._voice_manager.list_voices()

    async def generate_from_phonemes(
        self,
        phonemes: str,
        voice: str,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Generate audio directly from phonemes.

        Args:
            phonemes: Phonemes in Kokoro format
            voice: Voice name
            speed: Speed multiplier
            lang_code: Optional language code override

        Returns:
            Tuple of (audio array, processing time)
        """
        start_time = time.time()
        try:
            # Get backend and voice path
            backend = self.model_manager.get_backend()
            voice_name, voice_path = await self._get_voice_path(voice)

            if isinstance(backend, KokoroV1):
                # For Kokoro V1, use generate_from_tokens with raw phonemes
                result = None
                # Use provided lang_code or determine from voice name
                pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
                logger.info(
                    f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in phoneme pipeline"
                )

                try:
                    # Use backend's pipeline management
                    for r in backend._get_pipeline(
                        pipeline_lang_code
                    ).generate_from_tokens(
                        tokens=phonemes,  # Pass raw phonemes string
                        voice=voice_path,
                        speed=speed,
                    ):
                        if r.audio is not None:
                            result = r
                            break
                except Exception as e:
                    logger.error(f"Failed to generate from phonemes: {e}")
                    raise RuntimeError(f"Phoneme generation failed: {e}")

                if result is None or result.audio is None:
                    raise ValueError("No audio generated")

                processing_time = time.time() - start_time
                return result.audio.numpy(), processing_time
            else:
                raise ValueError(
                    "Phoneme generation only supported with Kokoro V1 backend"
                )

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise
