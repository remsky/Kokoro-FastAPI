import runpod
import json
import asyncio
from loguru import logger
import time
import re
import numpy as np

# Import necessary functions and classes from the API source
from api.src.services.tts_service import TTSService
from api.src.services.streaming_audio_writer import StreamingAudioWriter
from api.src.routers.openai_compatible import (
    get_tts_service as api_get_tts_service,
    process_and_validate_voices,
    load_openai_mappings,
)
from api.src.inference.base import AudioChunk
from api.src.structures.schemas import NormalizationOptions # Import normalization options
from api.src.core.config import settings # Import settings for lang_code logic

# Load OpenAI mappings (remains the same)
openai_mappings = load_openai_mappings()

# Global service instance (remains the same)
tts_service = None
_init_lock = None

async def get_tts_service():
    """Get or initialize the TTS service instance."""
    global tts_service, _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()

    if tts_service is None:
        async with _init_lock:
            if tts_service is None:
                try:
                    logger.info("Initializing TTS service for RunPod handler...")
                    # Use the imported function which handles initialization and warmup
                    tts_service = await api_get_tts_service()
                    # Verify it's ready
                    if tts_service.model_manager is None or tts_service._voice_manager is None:
                         raise RuntimeError("TTS service backend not initialized after get_tts_service call")
                    await tts_service.list_voices() # Test call
                    logger.info("TTS service initialized successfully for RunPod handler.")
                except Exception as e:
                    logger.error(f"Failed to initialize TTS service: {e}", exc_info=True)
                    tts_service = None # Reset on failure
                    raise RuntimeError(f"Failed to initialize TTS service: {str(e)}")
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
        service = await get_tts_service() # Ensures service is initialized

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

                yield {
                    "status": "audio_chunk",
                    "chunk_index": chunk_count,
                    "audio_data": audio_chunk.output, # Yield the formatted bytes
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

        if not has_yielded_audio:
             logger.warning(f"No audio data was yielded for the request. Input text: '{text[:100]}...'")
             yield {
                 "status": "error",
                 "error": "generation_failed",
                 "message": "Audio generation completed but produced no output data."
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