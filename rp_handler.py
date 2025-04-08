import runpod
import json
import asyncio
from loguru import logger
import time
import re
import numpy as np
import base64
# Import necessary functions and classes from the API source
from api.src.services.tts_service import TTSService
from api.src.services.streaming_audio_writer import StreamingAudioWriter
from api.src.routers.openai_compatible import (
    get_tts_service as api_get_tts_service, # Keep original function for potential reuse/consistency
    process_and_validate_voices,
    load_openai_mappings,
)
from api.src.inference.base import AudioChunk
from api.src.structures.schemas import NormalizationOptions # Import normalization options
from api.src.core.config import settings # Import settings for lang_code logic

# Load OpenAI mappings
openai_mappings = load_openai_mappings()

# Global service instance
tts_service = None
_init_lock = None

async def get_tts_service():
    """Get or initialize the TTS service instance, ensuring the backend is warmed up."""
    global tts_service, _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()

    if tts_service is None:
        async with _init_lock:
            # Double-check pattern
            if tts_service is None:
                temp_service = None # Use a temporary variable
                try:
                    logger.info("Initializing TTS service for RunPod handler...")
                    # Create the service instance first
                    # Note: TTSService.create() gets managers but doesn't init the backend
                    temp_service = await TTSService.create()

                    if temp_service.model_manager is None or temp_service._voice_manager is None:
                        raise RuntimeError("TTS service managers not initialized by create()")

                    logger.info("Warming up TTS model...")
                    # Explicitly initialize and warm up the model manager's backend
                    device, model_name, voice_count = await temp_service.model_manager.initialize_with_warmup(
                        temp_service._voice_manager
                    )
                    logger.info(f"Model '{model_name}' warmed up on {device} with {voice_count} voices.")

                    # Assign to global variable only after successful warmup
                    tts_service = temp_service
                    logger.info("TTS service initialized and warmed up successfully for RunPod handler.")

                except Exception as e:
                    logger.error(f"Failed to initialize and warm up TTS service: {e}", exc_info=True)
                    # Ensure tts_service remains None if initialization failed
                    tts_service = None
                    raise RuntimeError(f"Failed to initialize TTS service: {str(e)}")
    # Check if the backend is actually loaded after potential initialization
    if tts_service and tts_service.model_manager.get_backend() is None:
         logger.error("TTS Service exists but backend is still None after initialization attempt.")
         raise RuntimeError("Backend failed to initialize properly.")

    return tts_service


async def handler(event):
    """RunPod async handler, mirroring openai_compatible.py create_speech logic."""
    start_handler_time = time.time()
    writer = None # Initialize writer to None
    try:
        input_data = event.get("input", {})

        # Extract parameters with defaults from schemas/openai_compatible
        text = input_data.get("input")
        model = input_data.get("model", "kokoro")
        voice_input = input_data.get("voice", "af_heart") # Use a default from schemas
        response_format = input_data.get("response_format", "mp3")
        speed = float(input_data.get("speed", 1.0))
        lang_code_input = input_data.get("lang_code") # Can be None
        # Add normalization options extraction
        norm_options_input = input_data.get("normalization_options", {})
        normalization_options = NormalizationOptions(**norm_options_input)

        # --- Parameter Validation ---
        if not text:
            yield {"status": "error", "error": "input_missing", "message": "Input text is required"}
            return

        if model not in openai_mappings["models"]:
            yield {"status": "error", "error": "invalid_model", "message": f"Unsupported model: {model}"}
            return

        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        if response_format not in valid_formats:
            yield {"status": "error", "error": "invalid_format", "message": f"Unsupported format: {response_format}"}
            return

        if not (0.25 <= speed <= 4.0):
            yield {"status": "error", "error": "invalid_speed", "message": "Speed must be between 0.25 and 4.0"}
            return

        content_type_map = {
            "mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
            "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm",
        }
        content_type = content_type_map[response_format]

        # --- Initialization & Processing ---
        yield {"status": "initializing", "progress": 10, "message": "Initializing TTS service..."}
        service = await get_tts_service() # Ensures service is initialized AND warmed up

        yield {"status": "processing", "progress": 25, "message": "Processing voice..."}
        # Use the imported voice processing function
        voice_name = await process_and_validate_voices(voice_input, service)

        # Determine effective language code (logic from tts_service.py)
        first_base_voice_match = re.match(r"([a-zA-Z0-9_]+)", voice_name)
        first_base_voice = first_base_voice_match.group(1) if first_base_voice_match else "a"
        effective_lang_code = lang_code_input if lang_code_input else (settings.default_voice_code if settings.default_voice_code else first_base_voice[:1].lower())

        # Create the writer needed by generate_audio_stream
        writer = StreamingAudioWriter(response_format, sample_rate=settings.sample_rate)

        # --- Streaming Audio Generation ---
        yield {"status": "generating", "progress": 40, "message": "Generating audio stream..."}
        yield {
            "status": "stream_start",
            "content_type": content_type,
            "format": response_format,
            "sample_rate": settings.sample_rate,
            "lang_code_used": effective_lang_code,
            "voice_used": voice_name
        }

        start_generation_time = time.time()
        chunk_count = 0
        total_size = 0
        has_yielded_audio = False

        # Stream audio chunks
        async for audio_chunk in service.generate_audio_stream(
            text=text,
            voice=voice_name,
            writer=writer, # Pass the writer
            speed=speed,
            output_format=response_format,
            lang_code=effective_lang_code,
            normalization_options=normalization_options, # Pass normalization options
            return_timestamps=False # Timestamps not typically needed for basic RunPod handler
        ):
            # audio_chunk is now an AudioChunk object containing raw audio and/or output bytes
            if audio_chunk.output and isinstance(audio_chunk.output, bytes) and len(audio_chunk.output) > 0:
                has_yielded_audio = True
                chunk_count += 1
                chunk_size = len(audio_chunk.output)
                total_size += chunk_size

                # *** FIX: Base64 encode the audio bytes ***
                encoded_audio_data = base64.b64encode(audio_chunk.output).decode('utf-8')
                
                yield {
                    "status": "audio_chunk",
                    "chunk_index": chunk_count,
                    "audio_data": encoded_audio_data, # Yield the formatted bytes
                    "chunk_size": chunk_size
                }
                # Optional progress update
                # progress = min(40 + int(chunk_count * 50 / max(len(text) // 50, 1)), 95)
                # yield {"status": "progress", "progress": progress, "message": f"Streamed chunk {chunk_count}"}
            elif not audio_chunk.output and audio_chunk.audio is not None and len(audio_chunk.audio) == 0 and chunk_count == 0 and not has_yielded_audio:
                 # This might be the final empty chunk signaling end, or an initial empty chunk.
                 # If no audio was ever yielded, log a warning later.
                 pass
            # else: log potential empty chunks if needed

        generation_time = time.time() - start_generation_time
        handler_time = time.time() - start_handler_time

        if not has_yielded_audio and len(text.strip()) > 0 : # Check if input text was actually present
             logger.warning(f"No audio data was yielded for the request. Input text: '{text[:100]}...'")
             yield {
                 "status": "error",
                 "error": "generation_failed",
                 "message": "Audio generation completed but produced no output data. Please check logs."
             }
             return # Stop processing if no audio was generated

        # --- Completion ---
        yield {
            "status": "complete",
            "progress": 100,
            "message": "Audio generation complete.",
            "format": response_format,
            "content_type": content_type,
            "total_chunks": chunk_count,
            "total_size_bytes": total_size,
            "generation_time_seconds": round(generation_time, 3),
            "total_handler_time_seconds": round(handler_time, 3)
        }

    except ValueError as e:
        logger.warning(f"Validation error in RunPod handler: {str(e)}")
        yield {"status": "error", "error": "validation_error", "message": str(e)}
    except RuntimeError as e:
        logger.error(f"Runtime error in RunPod handler: {str(e)}")
        yield {"status": "error", "error": "processing_error", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in RunPod handler: {e}", exc_info=True)
        yield {"status": "error", "error": "server_error", "message": str(e)}
    finally:
        # Ensure the writer is closed if it was created
        if writer:
            try:
                writer.close()
            except Exception as close_e:
                logger.error(f"Error closing audio writer: {close_e}")


if __name__ == '__main__':
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({
        'handler': handler,
        'return_aggregate_stream': True # Keep streaming enabled
    })