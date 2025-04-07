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
                        trim_audio=False, # Don't trim final silence
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
                # Note: chunk_text is the *original* text including custom phoneme markers and newlines
                # The model needs the text *with phonemes restored*
                text_for_model = chunk_text # Start with original
                # Restore custom phonemes if backend needs it (like KokoroV1)
                if isinstance(backend, KokoroV1):
                     # Find phoneme markers in this specific chunk_text and restore
                     # (This assumes smart_split yielded text with markers) - let's refine smart_split yield
                     # For now, assume chunk_text is ready for the model (phonemes restored by smart_split)
                     pass


                if isinstance(backend, KokoroV1):
                    internal_chunk_index = 0
                    async for chunk_data in self.model_manager.generate(
                        text_for_model.strip(), # Pass cleaned text to model
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
                                    chunk_text.strip(), # Pass original text for trimming logic
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
                         logger.warning(f"Model generation yielded no audio chunks for: '{text_for_model[:50]}...'")

                else:
                    # --- Legacy backend path (using tokens) ---
                    # This path might not work correctly with custom phonemes restored in text_for_model
                    logger.warning("Using legacy backend path with tokens - custom phonemes might not be handled.")
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
            voice: Voice name or combined voice names (e.g., 'af_jadzia(0.7)+af_jessica(0.3)')

        Returns:
            Tuple of (voice name to use, voice path to use)

        Raises:
            RuntimeError: If voice not found
        """
        try:
            # Regex to handle names, weights, and operators: af_name(weight)[+-]af_other(weight)...
            pattern = re.compile(r"([a-zA-Z0-9_]+)(?:\((\d+(?:\.\d+)?)\))?([+-]?)")
            matches = pattern.findall(voice.replace(" ", "")) # Remove spaces

            if not matches:
                raise ValueError(f"Could not parse voice string: {voice}")

            # If only one voice and no explicit weight or operators, handle directly
            if len(matches) == 1 and not matches[0][1] and not matches[0][2]:
                voice_name = matches[0][0]
                path = await self._voice_manager.get_voice_path(voice_name)
                if not path:
                    raise RuntimeError(f"Voice not found: {voice_name}")
                logger.debug(f"Using single voice path: {path}")
                return voice_name, path

            # Process combinations
            voice_parts = []
            total_weight = 0
            for name, weight_str, operator in matches:
                 weight = float(weight_str) if weight_str else 1.0
                 voice_parts.append({"name": name, "weight": weight, "op": operator})
                 # Use weight directly for total, normalization happens later if enabled
                 total_weight += weight # Summing base weights before potential normalization

            # Check base voices exist
            available_voices = await self._voice_manager.list_voices()
            for part in voice_parts:
                if part["name"] not in available_voices:
                     raise ValueError(f"Base voice '{part['name']}' not found in combined string '{voice}'. Available: {available_voices}")


            # Determine normalization factor
            norm_factor = total_weight if settings.voice_weight_normalization and total_weight > 0 else 1.0
            if settings.voice_weight_normalization:
                 logger.debug(f"Normalizing combined voice weights by factor: {norm_factor:.2f}")
            else:
                 logger.debug("Voice weight normalization disabled, using raw weights.")


            # Load and combine tensors
            first_part = voice_parts[0]
            base_path = await self._voice_manager.get_voice_path(first_part["name"])
            combined_tensor = await self._load_voice_from_path(base_path, first_part["weight"] / norm_factor)

            current_op = "+" # Implicitly start with addition for the first voice

            for i in range(len(voice_parts) - 1):
                 current_part = voice_parts[i]
                 next_part = voice_parts[i+1]

                 # Determine the operation based on the *current* part's operator
                 op_symbol = current_part["op"] if current_part["op"] else "+" # Default to '+' if no operator

                 path = await self._voice_manager.get_voice_path(next_part["name"])
                 voice_tensor = await self._load_voice_from_path(path, next_part["weight"] / norm_factor)

                 if op_symbol == "+":
                    combined_tensor += voice_tensor
                    logger.debug(f"Adding voice {next_part['name']} (weight {next_part['weight']/norm_factor:.2f})")
                 elif op_symbol == "-":
                    combined_tensor -= voice_tensor
                    logger.debug(f"Subtracting voice {next_part['name']} (weight {next_part['weight']/norm_factor:.2f})")


            # Save the new combined voice so it can be loaded later
            # Use a safe filename based on the original input string
            safe_filename = re.sub(r'[^\w+-]', '_', voice) + ".pt"
            temp_dir = tempfile.gettempdir()
            combined_path = os.path.join(temp_dir, safe_filename)
            logger.debug(f"Saving combined voice '{voice}' to temporary path: {combined_path}")
            # Save the tensor to the device specified by settings for model loading consistency
            target_device = settings.get_device()
            torch.save(combined_tensor.to(target_device), combined_path)
            return voice, combined_path # Return original name and temp path

        except Exception as e:
            logger.error(f"Failed to get or combine voice path for '{voice}': {e}")
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
            # voice_name will be the potentially complex combined name string
            voice_name, voice_path = await self._get_voices_path(voice)
            logger.debug(f"Using voice path for '{voice_name}': {voice_path}")

            # Determine language code
            # Use provided lang_code, fallback to settings override, then first letter of first base voice
            first_base_voice_match = re.match(r"([a-zA-Z0-9_]+)", voice)
            first_base_voice = first_base_voice_match.group(1) if first_base_voice_match else "a" # Default 'a'
            pipeline_lang_code = lang_code if lang_code else (settings.default_voice_code if settings.default_voice_code else first_base_voice[:1].lower())
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
                            pause_chunk.audio = stream_normalizer.normalize(pause_chunk.audio)
                            if len(pause_chunk.audio) > 0:
                                yield pause_chunk

                        # Update offset based on silence duration
                        current_offset += pause_duration_s
                        chunk_index += 1

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
                            is_first=(chunk_index == 0),
                            is_last=False,
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
                            # Check if audio data exists before calculating duration
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
                        if output_format and final_chunk_data.output:
                             yield final_chunk_data
                        elif not output_format and final_chunk_data.audio is not None and len(final_chunk_data.audio) > 0:
                             yield final_chunk_data # Should yield empty chunk in raw mode upon finalize
                except Exception as e:
                    logger.error(f"Failed to finalize audio stream: {str(e)}")

        except Exception as e:
            logger.exception(f"Error during audio stream generation: {str(e)}") # Use exception for traceback
            # Ensure writer is closed on error
            try:
                writer.close()
            except Exception as close_e:
                logger.error(f"Error closing writer during exception handling: {close_e}")
            raise e # Re-raise the original exception


    async def generate_audio(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter, # Writer needed even for non-streaming internally
        speed: float = 1.0,
        return_timestamps: bool = False,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        lang_code: Optional[str] = None,
    ) -> AudioChunk:
        """Generate complete audio for text using streaming internally."""
        audio_data_chunks = []
        output_format = None # Signal raw audio mode for internal streaming
        combined_chunk = None
        try:
            async for audio_stream_data in self.generate_audio_stream(
                text,
                voice,
                writer, # Pass writer, although it won't be used for formatting here
                speed=speed,
                normalization_options=normalization_options,
                return_timestamps=return_timestamps,
                lang_code=lang_code,
                output_format=output_format, # Explicitly None for raw audio
            ):
                # Ensure we only append chunks with actual audio data
                # Raw silence chunks generated for pauses will have audio data (zeros)
                if audio_stream_data.audio is not None and len(audio_stream_data.audio) > 0:
                    audio_data_chunks.append(audio_stream_data)

            if not audio_data_chunks:
                 logger.warning("No valid audio chunks generated.")
                 combined_chunk = AudioChunk(audio=np.array([], dtype=np.int16), word_timestamps=[])
            else:
                combined_chunk = AudioChunk.combine(audio_data_chunks)

            return combined_chunk
        except Exception as e:
            logger.error(f"Error in combined audio generation: {str(e)}")
            raise # Re-raise after logging
        finally:
            # Explicitly close the writer if it was passed, though it shouldn't hold resources in raw mode
             try:
                 writer.close()
             except Exception:
                 pass # Ignore errors during cleanup



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
            # Use _get_voices_path to handle potential combined voice names passed here too
            voice_name, voice_path = await self._get_voices_path(voice)

            if isinstance(backend, KokoroV1):
                # For Kokoro V1, use generate_from_tokens with raw phonemes
                result_audio = None
                # Determine language code
                first_base_voice_match = re.match(r"([a-zA-Z0-9_]+)", voice_name)
                first_base_voice = first_base_voice_match.group(1) if first_base_voice_match else "a"
                pipeline_lang_code = lang_code if lang_code else (settings.default_voice_code if settings.default_voice_code else first_base_voice[:1].lower())

                logger.info(
                    f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in phoneme generation"
                )

                # Use backend's pipeline management and iterate through potential chunks
                full_audio_list = []
                async for r in backend.generate_from_tokens( # generate_from_tokens is now async
                    tokens=phonemes,  # Pass raw phonemes string
                    voice=(voice_name, voice_path), # Pass tuple
                    speed=speed,
                    lang_code=pipeline_lang_code,
                ):
                    if r is not None and len(r) > 0:
                         # r is directly the numpy array chunk
                        full_audio_list.append(r)


                if not full_audio_list:
                    raise ValueError("No audio generated from phonemes")

                # Combine chunks if necessary
                result_audio = np.concatenate(full_audio_list) if len(full_audio_list) > 1 else full_audio_list[0]

                processing_time = time.time() - start_time
                 # Normalize the final audio before returning
                normalizer = AudioNormalizer()
                normalized_audio = normalizer.normalize(result_audio)
                return normalized_audio, processing_time
            else:
                raise ValueError(
                    "Phoneme generation only supported with Kokoro V1 backend"
                )

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise