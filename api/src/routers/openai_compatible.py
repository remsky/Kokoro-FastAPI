"""OpenAI-compatible router for text-to-speech"""

import io
import json
import os
import re
import tempfile
from typing import AsyncGenerator, Dict, List, Tuple, Union
from urllib import response

import aiofiles
import numpy as np
import torch
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

from ..core.config import settings
from ..inference.base import AudioChunk
from ..services.audio import AudioService
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.tts_service import TTSService
from ..structures import OpenAISpeechRequest
from ..structures.schemas import CaptionedSpeechRequest


# Load OpenAI mappings
def load_openai_mappings() -> Dict:
    """Load OpenAI voice and model mappings from JSON"""
    api_dir = os.path.dirname(os.path.dirname(__file__))
    mapping_path = os.path.join(api_dir, "core", "openai_mappings.json")
    try:
        with open(mapping_path, "r") as f:
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
    return base_name + ".pth"


async def process_and_validate_voices(
    voice_input: Union[str, List[str]], tts_service: TTSService
) -> str:
    """Process voice input, handling both string and list formats

    Returns:
        Voice name to use (with weights if specified)
    """
    voices = []
    # Convert input to list of voices
    if isinstance(voice_input, str):
        voice_input = voice_input.replace(" ", "").strip()

        if voice_input[-1] in "+-" or voice_input[0] in "+-":
            raise ValueError(f"Voice combination contains empty combine items")

        if re.search(r"[+-]{2,}", voice_input) is not None:
            raise ValueError(f"Voice combination contains empty combine items")
        voices = re.split(r"([-+])", voice_input)
    else:
        voices = [[item, "+"] for item in voice_input][:-1]

    available_voices = await tts_service.list_voices()

    for voice_index in range(0, len(voices), 2):
        mapped_voice = voices[voice_index].split("(")
        mapped_voice = list(map(str.strip, mapped_voice))

        if len(mapped_voice) > 2:
            raise ValueError(
                f"Voice '{voices[voice_index]}' contains too many weight items"
            )

        if mapped_voice.count(")") > 1:
            raise ValueError(
                f"Voice '{voices[voice_index]}' contains too many weight items"
            )

        mapped_voice[0] = _openai_mappings["voices"].get(
            mapped_voice[0], mapped_voice[0]
        )

        if mapped_voice[0] not in available_voices:
            raise ValueError(
                f"Voice '{mapped_voice[0]}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )

        voices[voice_index] = "(".join(mapped_voice)

    return "".join(voices)


async def stream_audio_chunks(
    tts_service: TTSService,
    request: Union[OpenAISpeechRequest, CaptionedSpeechRequest],
    client_request: Request,
    writer: StreamingAudioWriter,
) -> AsyncGenerator[AudioChunk, None]:
    """Stream audio chunks as they're generated with client disconnect handling"""
    voice_name = await process_and_validate_voices(request.voice, tts_service)
    unique_properties = {"return_timestamps": False}
    if hasattr(request, "return_timestamps"):
        unique_properties["return_timestamps"] = request.return_timestamps

    try:
        async for chunk_data in tts_service.generate_audio_stream(
            text=request.input,
            voice=voice_name,
            writer=writer,
            speed=request.speed,
            output_format=request.response_format,
            lang_code=request.lang_code,
            volume_multiplier=request.volume_multiplier,
            normalization_options=request.normalization_options,
            return_timestamps=unique_properties["return_timestamps"],
        ):
            # Check if client is still connected
            is_disconnected = client_request.is_disconnected
            if callable(is_disconnected):
                is_disconnected = await is_disconnected()
            if is_disconnected:
                logger.info("Client disconnected, stopping audio generation")
                break

            yield chunk_data
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
                "type": "invalid_request_error",
            },
        )

    try:
        # model_name = get_model_name(request.model)
        tts_service = await get_tts_service()
        voice_name = await process_and_validate_voices(request.voice, tts_service)

        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        writer = StreamingAudioWriter(request.response_format, sample_rate=24000)

        # Check if streaming is requested (default for OpenAI client)
        if request.stream:
            # Create generator but don't start it yet
            generator = stream_audio_chunks(
                tts_service, request, client_request, writer
            )

            # If download link requested, wrap generator with temp file writer
            if request.return_download_link:
                from ..services.temp_manager import TempFileWriter

                # Use download_format if specified, otherwise use response_format
                output_format = request.download_format or request.response_format
                temp_writer = TempFileWriter(output_format)
                await temp_writer.__aenter__()  # Initialize temp file

                # Get download path immediately after temp file creation
                download_path = temp_writer.download_path

                # Create response headers with download path
                headers = {
                    "Content-Disposition": f"attachment; filename=speech.{output_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                    "X-Download-Path": download_path,
                }

                # Add header to indicate if temp file writing is available
                if temp_writer._write_error:
                    headers["X-Download-Status"] = "unavailable"

                # Create async generator for streaming
                async def dual_output():
                    try:
                        # Write chunks to temp file and stream
                        async for chunk_data in generator:
                            if chunk_data.output:  # Skip empty chunks
                                await temp_writer.write(chunk_data.output)
                                # if return_json:
                                #    yield chunk, chunk_data
                                # else:
                                yield chunk_data.output

                        # Finalize the temp file
                        await temp_writer.finalize()
                    except Exception as e:
                        logger.error(f"Error in dual output streaming: {e}")
                        await temp_writer.__aexit__(type(e), e, e.__traceback__)
                        raise
                    finally:
                        # Ensure temp writer is closed
                        if not temp_writer._finalized:
                            await temp_writer.__aexit__(None, None, None)
                        writer.close()

                # Stream with temp file writing
                return StreamingResponse(
                    dual_output(), media_type=content_type, headers=headers
                )

            async def single_output():
                try:
                    # Stream chunks
                    async for chunk_data in generator:
                        if chunk_data.output:  # Skip empty chunks
                            yield chunk_data.output
                except Exception as e:
                    logger.error(f"Error in single output streaming: {e}")
                    writer.close()
                    raise

            # Standard streaming without download link
            return StreamingResponse(
                single_output(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            headers = {
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Cache-Control": "no-cache",  # Prevent caching
            }

            # Generate complete audio using public interface
            audio_data = await tts_service.generate_audio(
                text=request.input,
                voice=voice_name,
                writer=writer,
                speed=request.speed,
                volume_multiplier=request.volume_multiplier,
                normalization_options=request.normalization_options,
                lang_code=request.lang_code,
            )

            audio_data = await AudioService.convert_audio(
                audio_data,
                request.response_format,
                writer,
                is_last_chunk=False,
                trim_audio=False,
            )

            # Convert to requested format with proper finalization
            final = await AudioService.convert_audio(
                AudioChunk(np.array([], dtype=np.int16)),
                request.response_format,
                writer,
                is_last_chunk=True,
            )
            output = audio_data.output + final.output

            if request.return_download_link:
                from ..services.temp_manager import TempFileWriter

                # Use download_format if specified, otherwise use response_format
                output_format = request.download_format or request.response_format
                temp_writer = TempFileWriter(output_format)
                await temp_writer.__aenter__()  # Initialize temp file

                # Get download path immediately after temp file creation
                download_path = temp_writer.download_path
                headers["X-Download-Path"] = download_path

                try:
                    # Write chunks to temp file
                    logger.info("Writing chunks to tempory file for download")
                    await temp_writer.write(output)
                    # Finalize the temp file
                    await temp_writer.finalize()

                except Exception as e:
                    logger.error(f"Error in dual output: {e}")
                    await temp_writer.__aexit__(type(e), e, e.__traceback__)
                    raise
                finally:
                    # Ensure temp writer is closed
                    if not temp_writer._finalized:
                        await temp_writer.__aexit__(None, None, None)
                    writer.close()

            return Response(
                content=output,
                media_type=content_type,
                headers=headers,
            )

    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Invalid request: {str(e)}")

        try:
            writer.close()
        except:
            pass

        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except RuntimeError as e:
        # Handle runtime/processing errors
        logger.error(f"Processing error: {str(e)}")

        try:
            writer.close()
        except:
            pass

        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in speech generation: {str(e)}")

        try:
            writer.close()
        except:
            pass

        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/download/{filename}")
async def download_audio_file(filename: str):
    """Download a generated audio file from temp storage"""
    try:
        from ..core.paths import _find_file, get_content_type

        # Search for file in temp directory
        file_path = await _find_file(
            filename=filename, search_paths=[settings.temp_file_dir]
        )

        # Get content type from path helper
        content_type = await get_content_type(file_path)

        return FileResponse(
            file_path,
            media_type=content_type,
            filename=filename,
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

    except Exception as e:
        logger.error(f"Error serving download file {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to serve audio file",
                "type": "server_error",
            },
        )


@router.get("/models")
async def list_models():
    """List all available models"""
    try:
        # Create standard model list
        models = [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            },
            {
                "id": "kokoro",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            },
        ]

        return {"object": "list", "data": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model list",
                "type": "server_error",
            },
        )


@router.get("/models/{model}")
async def retrieve_model(model: str):
    """Retrieve a specific model"""
    try:
        # Define available models
        models = {
            "tts-1": {
                "id": "tts-1",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            },
            "tts-1-hd": {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            },
            "kokoro": {
                "id": "kokoro",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            },
        }

        # Check if requested model exists
        if model not in models:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "model_not_found",
                    "message": f"Model '{model}' not found",
                    "type": "invalid_request_error",
                },
            )

        # Return the specific model
        return models[model]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model {model}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model information",
                "type": "server_error",
            },
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
                "type": "server_error",
            },
        )


@router.post("/audio/voices/combine")
async def combine_voices(request: Union[str, List[str]]):
    """Combine multiple voices into a new voice and return the .pt file.

    Args:
        request: Either a string with voices separated by + (e.g. "voice1+voice2")
                or a list of voice names to combine

    Returns:
        FileResponse with the combined voice .pt file

    Raises:
        HTTPException:
            - 400: Invalid request (wrong number of voices, voice not found)
            - 500: Server error (file system issues, combination failed)
    """
    # Check if local voice saving is allowed
    if not settings.allow_local_voice_saving:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "permission_denied",
                "message": "Local voice saving is disabled",
                "type": "permission_error",
            },
        )

    try:
        # Convert input to list of voices
        if isinstance(request, str):
            # Check if it's an OpenAI voice name
            mapped_voice = _openai_mappings["voices"].get(request)
            if mapped_voice:
                request = mapped_voice
            voices = [v.strip() for v in request.split("+") if v.strip()]
        else:
            # For list input, map each voice if it's an OpenAI voice name
            voices = [_openai_mappings["voices"].get(v, v) for v in request]
            voices = [v.strip() for v in voices if v.strip()]

        if not voices:
            raise ValueError("No voices provided")

        # For multiple voices, validate base voices exist
        tts_service = await get_tts_service()
        available_voices = await tts_service.list_voices()
        for voice in voices:
            if voice not in available_voices:
                raise ValueError(
                    f"Base voice '{voice}' not found. Available voices: {', '.join(sorted(available_voices))}"
                )

        # Combine voices
        combined_tensor = await tts_service.combine_voices(voices=voices)
        combined_name = "+".join(voices)

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        voice_path = os.path.join(temp_dir, f"{combined_name}.pt")
        buffer = io.BytesIO()
        torch.save(combined_tensor, buffer)
        async with aiofiles.open(voice_path, "wb") as f:
            await f.write(buffer.getvalue())

        return FileResponse(
            voice_path,
            media_type="application/octet-stream",
            filename=f"{combined_name}.pt",
            headers={
                "Content-Disposition": f"attachment; filename={combined_name}.pt",
                "Cache-Control": "no-cache",
            },
        )

    except ValueError as e:
        logger.warning(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except RuntimeError as e:
        logger.error(f"Voice combination processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": "Failed to process voice combination request",
                "type": "server_error",
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error in voice combination: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "An unexpected error occurred",
                "type": "server_error",
            },
        )
