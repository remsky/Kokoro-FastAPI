import json
import os
from typing import AsyncGenerator, Dict, List, Union

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from ..services.audio import AudioService
from ..services.tts_service import TTSService
from ..structures.schemas import OpenAISpeechRequest
from ..core.config import settings

# Load OpenAI mappings
def load_openai_mappings() -> Dict:
    """Load OpenAI voice and model mappings from JSON"""
    api_dir = os.path.dirname(os.path.dirname(__file__))
    mapping_path = os.path.join(api_dir, "core", "openai_mappings.json")
    try:
        with open(mapping_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI mappings: {e}")
        return {"models": {}, "voices": {}}

# Global mappings
_openai_mappings = load_openai_mappings()


router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

# Global TTSService instance with lock
_tts_service = None
_init_lock = None


async def get_tts_service() -> TTSService:
    """Get global TTSService instance"""
    global _tts_service, _init_lock
    
    # Create lock if needed
    if _init_lock is None:
        import asyncio
        _init_lock = asyncio.Lock()
    
    # Initialize service if needed
    if _tts_service is None:
        async with _init_lock:
            # Double check pattern
            if _tts_service is None:
                _tts_service = await TTSService.create()
                logger.info("Created global TTSService instance")
    
    return _tts_service


def get_model_name(model: str) -> str:
    """Get internal model name from OpenAI model name"""
    base_name = _openai_mappings["models"].get(model)
    if not base_name:
        raise ValueError(f"Unsupported model: {model}")
    # Add extension based on runtime config
    extension = ".onnx" if settings.use_onnx else ".pth"
    return base_name + extension

async def process_voices(
    voice_input: Union[str, List[str]], tts_service: TTSService
) -> str:
    """Process voice input into a combined voice, handling both string and list formats"""
    # Convert input to list of voices
    if isinstance(voice_input, str):
        # Check if it's an OpenAI voice name
        mapped_voice = _openai_mappings["voices"].get(voice_input)
        if mapped_voice:
            voice_input = mapped_voice
        voices = [v.strip() for v in voice_input.split("+") if v.strip()]
    else:
        # For list input, map each voice if it's an OpenAI voice name
        voices = [_openai_mappings["voices"].get(v, v) for v in voice_input]
        voices = [v.strip() for v in voices if v.strip()]

    if not voices:
        raise ValueError("No voices provided")

    # If single voice, validate and return it
    if len(voices) == 1:
        available_voices = await tts_service.list_voices()
        if voices[0] not in available_voices:
            raise ValueError(
                f"Voice '{voices[0]}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )
        return voices[0]

    # For multiple voices, validate base voices exist
    available_voices = await tts_service.list_voices()
    for voice in voices:
        if voice not in available_voices:
            raise ValueError(
                f"Base voice '{voice}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )

    # Combine voices
    return await tts_service.combine_voices(voices=voices)


async def stream_audio_chunks(
    tts_service: TTSService, 
    request: OpenAISpeechRequest,
    client_request: Request
) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks as they're generated with client disconnect handling"""
    # Check if 'request.voice' is a weighted formula (contains '*')
    if '*' in request.voice:
         # Weighted formula path
        voice_to_use = await tts_service._voice_manager.create_weighted_voice(
                formula=request.voice,
                normalize=True
            )
      
    else:
        # Normal single or multi-voice path
        voice_to_use = await process_voices(request.voice, tts_service)
    
    try:
        async for chunk in tts_service.generate_audio_stream(
            text=request.input,
            voice=voice_to_use,
            speed=request.speed,
            output_format=request.response_format,
        ):
            # Check if client is still connected
            is_disconnected = client_request.is_disconnected
            if callable(is_disconnected):
                is_disconnected = await is_disconnected()
            if is_disconnected:
                logger.info("Client disconnected, stopping audio generation")
                break
            yield chunk
    except Exception as e:
        logger.error(f"Error in audio streaming: {str(e)}")
        # Let the exception propagate to trigger cleanup
        raise


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    """OpenAI-compatible endpoint for text-to-speech"""
    # Validate model before processing request
    if request.model not in _openai_mappings["models"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}",
                "type": "invalid_request_error"
            }
        )
    
    try:
        model_name = get_model_name(request.model)

        # Get global service instance
        tts_service = await get_tts_service()
        
         # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        # Check if streaming is requested (default for OpenAI client)
        if request.stream:
            # Create generator but don't start it yet
            generator = stream_audio_chunks(tts_service, request, client_request)
            
            # Test the generator by attempting to get first chunk
            try:
                first_chunk = await anext(generator)
            except StopAsyncIteration:
                first_chunk = b""  # Empty audio case
            except Exception as e:
                # Re-raise any errors to be caught by the outer try-except
                raise RuntimeError(f"Failed to initialize audio stream: {str(e)}") from e
            
            # If we got here, streaming can begin
            async def safe_stream():
                yield first_chunk
                try:
                    async for chunk in generator:
                        yield chunk
                except Exception as e:
                    # Log the error but don't yield anything - the connection will close
                    logger.error(f"Error during streaming: {str(e)}")
                    raise
            
            # Stream audio chunks as they're generated
            return StreamingResponse(
                safe_stream(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",  # Disable proxy buffering
                    "Cache-Control": "no-cache",  # Prevent caching
                    "Transfer-Encoding": "chunked",  # Enable chunked transfer encoding
                },
            )
        else:
                # Check if 'request.voice' is a weighted formula (contains '*')
            if '*' in request.voice:
                # Weighted formula path
                print("Weighted formula path")
                voice_to_use = await tts_service._voice_manager.create_weighted_voice(
                    formula=request.voice,
                    normalize=True
                )
                print(voice_to_use)
            else:
                # Normal single or multi-voice path
                print("Normal single or multi-voice path")
                # Otherwise, handle normal single or multi-voice logic
                voice_to_use = await process_voices(request.voice, tts_service)
                # Generate complete audio using public interface
                audio, _ = await tts_service.generate_audio(
                    text=request.input,
                    voice=voice_to_use,
                    speed=request.speed,
                    stitch_long_output=True
                )

            # Convert to requested format
            content = await AudioService.convert_audio(
                audio, 24000, request.response_format, is_first_chunk=True, stream=False
            )

            return Response(
                content=content,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",  # Prevent caching
                },
            )

    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except RuntimeError as e:
        # Handle runtime/processing errors
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error"
            }
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error"
            }
        )


@router.get("/audio/voices")
async def list_voices():
    """List all available voices for text-to-speech"""
    try:
        tts_service = await get_tts_service()
        voices = await tts_service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve voice list",
                "type": "server_error"
            }
        )


@router.post("/audio/voices/combine")
async def combine_voices(request: Union[str, List[str]]):
    """Combine multiple voices into a new voice.

    Args:
        request: Either a string with voices separated by + (e.g. "voice1+voice2")
                or a list of voice names to combine

    Returns:
        Dict with combined voice name and list of all available voices

    Raises:
        HTTPException:
            - 400: Invalid request (wrong number of voices, voice not found)
            - 500: Server error (file system issues, combination failed)
    """
    try:
        tts_service = await get_tts_service()
        combined_voice = await process_voices(request, tts_service)
        voices = await tts_service.list_voices()
        return {"voices": voices, "voice": combined_voice}

    except ValueError as e:
        logger.warning(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except RuntimeError as e:
        logger.error(f"Voice combination processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": "Failed to process voice combination request",
                "type": "server_error"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in voice combination: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "An unexpected error occurred",
                "type": "server_error"
            }
        )
