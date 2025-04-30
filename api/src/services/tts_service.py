"""TTS service using model and voice managers."""

import asyncio
import os
import re
import tempfile
import time
from typing import AsyncGenerator, List, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from ..inference.base import AudioChunk
from ..inference.kokoro_v1 import KokoroV1
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from ..structures.schemas import NormalizationOptions
from .audio import AudioNormalizer, AudioService
from .streaming_audio_writer import StreamingAudioWriter
from .text_processing import tokenize
from .text_processing.text_processor import process_text_chunk, smart_split


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
        writer: StreamingAudioWriter,
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
                        yield AudioChunk(np.array([], dtype=np.int16), output=b"")
                        return
                    chunk_data = await AudioService.convert_audio(
                        AudioChunk(
                            np.array([], dtype=np.float32)
                        ),  # Dummy data for type checking
                        output_format,
                        writer,
                        speed,
                        "",
                        normalizer=normalizer,
                        is_last_chunk=True,
                    )
                    yield chunk_data
                    return

                # Skip empty chunks (shouldn't happen if called correctly, but safety)
                if not tokens and not chunk_text:
                     logger.warning("Empty chunk passed to _process_chunk")
                     return

                # Get backend
                backend = self.model_manager.get_backend()

                # Generate audio using pre-warmed model
                if isinstance(backend, KokoroV1):
                    # TODO: In the future, we may need to restore custom phonemes here
                    # This would involve finding phoneme markers in chunk_text and restoring them
                    # Currently, we assume smart_split has already handled this
                    
                    internal_chunk_index = 0
                    async for chunk_data in self.model_manager.generate(
                        chunk_text,
                        (voice_name, voice_path),
                        speed=speed,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    ):
                        # For streaming, convert to bytes if format specified
                        if output_format:
                            try:
                                chunk_data = await AudioService.convert_audio(
                                    chunk_data,
                                    output_format,
                                    writer,
                                    speed,
                                    chunk_text,
                                    is_last_chunk=is_last, # Should always be False here, handled above
                                    normalizer=normalizer,
                                    trim_audio=True # Trim speech parts
                                )
                                yield chunk_data
                            except Exception as e:
                                logger.error(f"Failed to convert audio: {str(e)}")
                        else: # Raw audio mode
                            chunk_data = AudioService.trim_audio(
                                chunk_data, chunk_text.strip(), speed, False, normalizer # Trim speech parts
                            )
                            yield chunk_data
                        internal_chunk_index += 1
                    if internal_chunk_index == 0:
                         logger.warning(f"Model generation yielded no audio chunks for: '{chunk_text[:50]}...'")

                else:
                    # For legacy backends, load voice tensor
                    voice_tensor = await self._voice_manager.load_voice(
                        voice_name, device=backend.device
                    )
                    async for chunk_data in self.model_manager.generate( # Needs to be async generator
                        tokens, # Legacy uses tokens
                        (voice_name, voice_tensor), # Pass tuple as expected
                        speed=speed,
                        return_timestamps=return_timestamps,
                    ):

                        if chunk_data.audio is None or len(chunk_data.audio) == 0:
                            logger.error("Legacy model generated empty or None audio chunk")
                            continue # Skip this chunk

                        # For streaming, convert to bytes
                        if output_format:
                            try:
                                chunk_data = await AudioService.convert_audio(
                                    chunk_data,
                                    output_format,
                                    writer,
                                    speed,
                                    chunk_text.strip(), # Pass original text for trimming logic
                                    normalizer=normalizer,
                                    is_last_chunk=is_last, # Should be False here
                                    trim_audio=True # Trim speech parts
                                )
                                yield chunk_data
                            except Exception as e:
                                logger.error(f"Failed to convert legacy audio: {str(e)}")
                        else: # Raw audio mode
                            trimmed = AudioService.trim_audio(
                                chunk_data, chunk_text.strip(), speed, False, normalizer # Trim speech parts
                            )
                            yield trimmed
            except Exception as e:
                logger.exception(f"Failed to process chunk: '{chunk_text[:50]}...'. Error: {str(e)}")


    async def _load_voice_from_path(self, path: str, weight: float):
        # Check if the path is None and raise a ValueError if it is not
        if not path:
            raise ValueError(f"Voice not found at path: {path}")

        logger.debug(f"Loading voice tensor from path: {path}")
        # Ensure loading happens on CPU initially to avoid device mismatches
        tensor = torch.load(path, map_location="cpu")
        return tensor * weight

    async def _get_voices_path(self, voice: str) -> Tuple[str, str]:
        """Get voice path, handling combined voices.

        Args:
            voice: Voice name or combined voice names (e.g., 'af_jadzia+af_jessica')

        Returns:
            Tuple of (voice name to use, voice path to use)

        Raises:
            RuntimeError: If voice not found
        """
        try:
            # Split the voice on + and - and ensure that they get added to the list eg: hi+bob = ["hi","+","bob"]
            split_voice = re.split(r"([-+])", voice)

            # If it is only once voice there is no point in loading it up, doing nothing with it, then saving it
            if len(split_voice) == 1:
                # Since its a single voice the only time that the weight would matter is if voice_weight_normalization is off
                if (
                    "(" not in voice and ")" not in voice
                ) or settings.voice_weight_normalization == True:
                    path = await self._voice_manager.get_voice_path(voice)
                    if not path:
                        raise RuntimeError(f"Voice not found: {voice}")
                    logger.debug(f"Using single voice path: {path}")
                    return voice, path

            total_weight = 0

            for voice_index in range(0, len(split_voice), 2):
                voice_object = split_voice[voice_index]

                if "(" in voice_object and ")" in voice_object:
                    voice_name = voice_object.split("(")[0].strip()
                    voice_weight = float(voice_object.split("(")[1].split(")")[0])
                else:
                    voice_name = voice_object
                    voice_weight = 1

                total_weight += voice_weight
                split_voice[voice_index] = (voice_name, voice_weight)

            # If voice_weight_normalization is false prevent normalizing the weights by setting the total_weight to 1 so it divides each weight by 1
            if settings.voice_weight_normalization == False:
                total_weight = 1

            # Load the first voice as the starting point for voices to be combined onto
            path = await self._voice_manager.get_voice_path(split_voice[0][0])
            combined_tensor = await self._load_voice_from_path(
                path, split_voice[0][1] / total_weight
            )

            # Loop through each + or - in split_voice so they can be applied to combined voice
            for operation_index in range(1, len(split_voice) - 1, 2):
                # Get the voice path of the voice 1 index ahead of the operator
                path = await self._voice_manager.get_voice_path(
                    split_voice[operation_index + 1][0]
                )
                voice_tensor = await self._load_voice_from_path(
                    path, split_voice[operation_index + 1][1] / total_weight
                )

                # Either add or subtract the voice from the current combined voice
                if split_voice[operation_index] == "+":
                    combined_tensor += voice_tensor
                else:
                    combined_tensor -= voice_tensor

            # Save the new combined voice so it can be loaded latter
            temp_dir = tempfile.gettempdir()
            combined_path = os.path.join(temp_dir, f"{voice}.pt")
            logger.debug(f"Saving combined voice to: {combined_path}")
            torch.save(combined_tensor, combined_path)
            return voice, combined_path
        except Exception as e:
            logger.error(f"Failed to get voice path: {e}")
            raise


    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        output_format: str = "wav",
        lang_code: Optional[str] = None,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate and stream audio chunks, handling text, pauses, and newlines."""
        stream_normalizer = AudioNormalizer()
        chunk_index = 0
        current_offset = 0.0 # Track audio time offset for timestamps
        try:
            # Get backend
            backend = self.model_manager.get_backend()

            # Get voice path, handling combined voices
            voice_name, voice_path = await self._get_voices_path(voice)
            logger.debug(f"Using voice path: {voice_path}")

            # Use provided lang_code or determine from voice name
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(
                f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in audio stream"
            )

            # Process text in chunks (handling pauses and newlines within smart_split)
            async for chunk_text, tokens, pause_duration_s in smart_split(
                text,
                lang_code=pipeline_lang_code,
                normalization_options=normalization_options,
            ):
                if pause_duration_s is not None and pause_duration_s > 0:
                    # --- Handle Pause Chunk ---
                    try:
                        logger.debug(f"Generating {pause_duration_s}s silence chunk")
                        silence_samples = int(pause_duration_s * settings.sample_rate)
                        # Create silence appropriate for AudioService (float32)
                        silence_audio = np.zeros(silence_samples, dtype=np.float32)
                        pause_chunk = AudioChunk(audio=silence_audio, word_timestamps=[]) # Empty timestamps for silence

                        # Format and yield the silence chunk
                        if output_format:
                            formatted_pause_chunk = await AudioService.convert_audio(
                                pause_chunk, output_format, writer, speed=1.0, chunk_text="",
                                is_last_chunk=False, trim_audio=False, normalizer=stream_normalizer,
                            )
                            if formatted_pause_chunk.output:
                                yield formatted_pause_chunk
                        else: # Raw audio mode
                            # Normalize to int16 for raw output consistency
                            pause_chunk.audio = stream_normalizer.normalize(pause_chunk.audio)
                            if len(pause_chunk.audio) > 0:
                                yield pause_chunk

                        # Update offset based on silence duration
                        current_offset += pause_duration_s
                        chunk_index += 1 # Count pause as a yielded chunk

                    except Exception as e:
                        logger.error(f"Failed to process pause chunk: {str(e)}")
                        continue

                elif tokens or chunk_text.strip(): # Process if there are tokens OR non-whitespace text
                    # --- Handle Text Chunk ---
                    original_text_with_markers = chunk_text # Keep original including markers/newlines
                    text_chunk_for_model = chunk_text.strip() # Clean text for the model
                    has_trailing_newline = chunk_text.endswith('\n')

                    try:
                        # Process audio for the text chunk
                        async for chunk_data in self._process_chunk(
                            text_chunk_for_model,  # Pass cleaned text for model processing
                            tokens,
                            voice_name,
                            voice_path,
                            speed,
                            writer,
                            output_format,
                            is_first=(chunk_index == 0), # Check if this is the very first *audio* chunk
                            is_last=False, # is_last is handled separately after the loop
                            normalizer=stream_normalizer,
                            lang_code=pipeline_lang_code,
                            return_timestamps=return_timestamps,
                        ):
                            # Adjust timestamps relative to the stream start
                            if chunk_data.word_timestamps:
                                for timestamp in chunk_data.word_timestamps:
                                    timestamp.start_time += current_offset
                                    timestamp.end_time += current_offset

                            # Update offset based on the *actual duration* of the generated audio chunk
                            chunk_duration = 0
                            if chunk_data.audio is not None and len(chunk_data.audio) > 0:
                                chunk_duration = len(chunk_data.audio) / settings.sample_rate
                                current_offset += chunk_duration

                            # Yield the processed chunk (either formatted or raw)
                            if output_format and chunk_data.output:
                                yield chunk_data
                            elif not output_format and chunk_data.audio is not None and len(chunk_data.audio) > 0:
                                yield chunk_data
                            else:
                                logger.warning(
                                    f"No audio generated or output for text chunk: '{text_chunk_for_model[:50]}...'"
                                )

                        # --- Add pause after newline (if applicable) ---
                        if has_trailing_newline:
                            newline_pause_s = 0.5
                            try:
                                logger.debug(f"Adding {newline_pause_s}s pause after newline.")
                                silence_samples = int(newline_pause_s * settings.sample_rate)
                                silence_audio = np.zeros(silence_samples, dtype=np.float32)
                                # Create a *new* AudioChunk instance for the newline pause
                                newline_pause_chunk = AudioChunk(audio=silence_audio, word_timestamps=[])

                                if output_format:
                                    formatted_pause_chunk = await AudioService.convert_audio(
                                        newline_pause_chunk, output_format, writer, speed=1.0, chunk_text="", # Use newline_pause_chunk
                                        is_last_chunk=False, trim_audio=False, normalizer=stream_normalizer,
                                    )
                                    if formatted_pause_chunk.output:
                                        yield formatted_pause_chunk
                                else:
                                    # Normalize the *new* chunk before yielding
                                    newline_pause_chunk.audio = stream_normalizer.normalize(newline_pause_chunk.audio)
                                    if len(newline_pause_chunk.audio) > 0:
                                        yield newline_pause_chunk # Yield the normalized newline pause chunk

                                current_offset += newline_pause_s # Add newline pause to offset

                            except Exception as pause_e:
                                 logger.error(f"Failed to process newline pause chunk: {str(pause_e)}")
                        # ------------------------------------------------

                        chunk_index += 1 # Increment chunk index after processing text and potential newline pause

                    except Exception as e:
                        logger.exception( # Use exception to include traceback
                            f"Failed processing audio for chunk: '{text_chunk_for_model[:50]}...'. Error: {str(e)}"
                        )
                        continue

            # --- End of main loop ---

            # Finalize the stream (sends any remaining buffered data)
            # Only finalize if we successfully processed at least one chunk (text or pause)
            if chunk_index > 0:
                try:
                    async for final_chunk_data in self._process_chunk(
                        "", [], voice_name, voice_path, speed, writer, output_format,
                        is_first=False, is_last=True, normalizer=stream_normalizer, lang_code=pipeline_lang_code
                    ):
                         # Yield final formatted chunk or raw empty chunk
                        if output_format and final_chunk_data.output:
                             yield final_chunk_data
                        elif not output_format: # Raw mode: Finalize yields empty chunk signal
                             yield final_chunk_data # Yields empty AudioChunk
                except Exception as e:
                    logger.error(f"Failed to finalize audio stream: {str(e)}")

        except Exception as e:
            logger.exception(f"Error during audio stream generation: {str(e)}") # Use exception for traceback
            # Ensure writer is closed on error - moved to caller (e.g., route handler)
            try:
                writer.close()
            except Exception as close_e:
                logger.error(f"Error closing writer during exception handling: {close_e}")
            raise e # Re-raise the original exception


    async def generate_audio(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        return_timestamps: bool = False,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        lang_code: Optional[str] = None,
    ) -> AudioChunk:
        """Generate complete audio for text using streaming internally."""
        audio_data_chunks = []

        try:
            async for audio_stream_data in self.generate_audio_stream(
                text,
                voice,
                writer,
                speed=speed,
                normalization_options=normalization_options,
                return_timestamps=return_timestamps,
                lang_code=lang_code,
                output_format=None,
            ):
                if len(audio_stream_data.audio) > 0:
                    audio_data_chunks.append(audio_stream_data)

            combined_audio_data = AudioChunk.combine(audio_data_chunks)
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
            voice_name, voice_path = await self._get_voices_path(voice)

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