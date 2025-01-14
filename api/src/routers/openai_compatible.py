from typing import List, Union, AsyncGenerator

from loguru import logger
from fastapi import Header, Depends, Response, APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..services.audio import AudioService
from ..structures.schemas import OpenAISpeechRequest
from ..services.tts_service import TTSService

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance with database session"""
    return TTSService()  # Initialize TTSService with default settings


async def process_voices(
    voice_input: Union[str, List[str]], tts_service: TTSService
) -> str:
    """Process voice input into a combined voice, handling both string and list formats"""
    # Convert input to list of voices
    if isinstance(voice_input, str):
        voices = [v.strip() for v in voice_input.split("+") if v.strip()]
    else:
        voices = voice_input

    if not voices:
        raise ValueError("No voices provided")

    # Check if all voices exist
    available_voices = await tts_service.list_voices()
    for voice in voices:
        if voice not in available_voices:
            raise ValueError(
                f"Voice '{voice}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )

    # If single voice, return it directly
    if len(voices) == 1:
        return voices[0]

    # Otherwise combine voices
    return await tts_service.combine_voices(voices=voices)


async def stream_audio_chunks(
    tts_service: TTSService, request: OpenAISpeechRequest
) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks as they're generated"""
    voice_to_use = await process_voices(request.voice, tts_service)
    async for chunk in tts_service.generate_audio_stream(
        text=request.input,
        voice=voice_to_use,
        speed=request.speed,
        output_format=request.response_format,
    ):
        yield chunk


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    tts_service: TTSService = Depends(get_tts_service),
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    """OpenAI-compatible endpoint for text-to-speech"""
    try:
        # Process voice combination and validate
        voice_to_use = await process_voices(request.voice, tts_service)

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
            # Stream audio chunks as they're generated
            return StreamingResponse(
                stream_audio_chunks(tts_service, request),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",  # Disable proxy buffering
                    "Cache-Control": "no-cache",  # Prevent caching
                    "Transfer-Encoding": "chunked",  # Enable chunked transfer encoding
                },
            )
        else:
            # Generate complete audio
            audio, _ = tts_service._generate_audio(
                text=request.input,
                voice=voice_to_use,
                speed=request.speed,
                stitch_long_output=True,
            )

            # Convert to requested format
            content = AudioService.convert_audio(
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
        logger.error(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400, detail={"error": "Invalid request", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )


@router.get("/audio/voices")
async def list_voices(tts_service: TTSService = Depends(get_tts_service)):
    """List all available voices for text-to-speech"""
    try:
        voices = await tts_service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/voices/combine")
async def combine_voices(
    request: Union[str, List[str]], tts_service: TTSService = Depends(get_tts_service)
):
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
        combined_voice = await process_voices(request, tts_service)
        voices = await tts_service.list_voices()
        return {"voices": voices, "voice": combined_voice}

    except ValueError as e:
        logger.error(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(
            status_code=400, detail={"error": "Invalid request", "message": str(e)}
        )

    except Exception as e:
        logger.error(f"Server error during voice combination: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": "Server error"}
        )
