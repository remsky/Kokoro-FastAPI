import base64
import json
import os
import re
from pathlib import Path
from typing import AsyncGenerator, List, Tuple, Union

import numpy as np
import torch
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from ..inference.base import AudioChunk
from ..services.audio import AudioNormalizer, AudioService
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.temp_manager import TempFileWriter
from ..services.text_processing import smart_split
from ..services.tts_service import TTSService
from ..structures import CaptionedSpeechRequest, CaptionedSpeechResponse, WordTimestamp
from ..structures.custom_responses import JSONStreamingResponse
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
)
from .openai_compatible import process_and_validate_voices, stream_audio_chunks

router = APIRouter(tags=["text processing"])


async def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance"""
    return (
        await TTSService.create()
    )  # Create service with properly initialized managers


@router.post("/dev/phonemize", response_model=PhonemeResponse)
async def phonemize_text(request: PhonemeRequest) -> PhonemeResponse:
    """Convert text to phonemes using Kokoro's quiet mode.

    Args:
        request: Request containing text and language

    Returns:
        Phonemes and token IDs
    """
    try:
        if not request.text:
            raise ValueError("Text cannot be empty")

        # Initialize Kokoro pipeline in quiet mode (no model)
        pipeline = KPipeline(lang_code=request.language, model=False)

        # Get first result from pipeline (we only need one since we're not chunking)
        for result in pipeline(request.text):
            # result.graphemes = original text
            # result.phonemes = phonemized text
            # result.tokens = token objects (if available)
            return PhonemeResponse(phonemes=result.phonemes, tokens=[])

        raise ValueError("Failed to generate phonemes")
    except ValueError as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )


@router.post("/dev/generate_from_phonemes")
async def generate_from_phonemes(
    request: GenerateFromPhonemesRequest,
    client_request: Request,
    tts_service: TTSService = Depends(get_tts_service),
) -> StreamingResponse:
    """Generate audio directly from phonemes using Kokoro's phoneme format"""
    try:
        # Basic validation
        if not isinstance(request.phonemes, str):
            raise ValueError("Phonemes must be a string")
        if not request.phonemes:
            raise ValueError("Phonemes cannot be empty")

        # Create streaming audio writer and normalizer
        writer = StreamingAudioWriter(format="wav", sample_rate=24000, channels=1)
        normalizer = AudioNormalizer()

        async def generate_chunks():
            try:
                # Generate audio from phonemes
                chunk_audio, _ = await tts_service.generate_from_phonemes(
                    phonemes=request.phonemes,  # Pass complete phoneme string
                    voice=request.voice,
                    speed=1.0,
                )

                if chunk_audio is not None:
                    # Normalize audio before writing
                    normalized_audio = normalizer.normalize(chunk_audio)
                    # Write chunk and yield bytes
                    chunk_bytes = writer.write_chunk(normalized_audio)
                    if chunk_bytes:
                        yield chunk_bytes

                    # Finalize and yield remaining bytes
                    final_bytes = writer.write_chunk(finalize=True)
                    if final_bytes:
                        yield final_bytes
                        writer.close()
                else:
                    raise ValueError("Failed to generate audio data")

            except Exception as e:
                logger.error(f"Error in audio generation: {str(e)}")
                # Clean up writer on error
                writer.close()
                # Re-raise the original exception
                raise

        return StreamingResponse(
            generate_chunks(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )

    except ValueError as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.post("/dev/captioned_speech")
async def create_captioned_speech(
    request: CaptionedSpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
    tts_service: TTSService = Depends(get_tts_service),
):
    """Generate audio with word-level timestamps using streaming approach"""

    try:
        # model_name = get_model_name(request.model)
        tts_service = await get_tts_service()
        voice_name = await process_and_validate_voices(request.voice, tts_service)

        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "m4a": "audio/mp4",
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

                temp_writer = TempFileWriter(request.response_format)
                await temp_writer.__aenter__()  # Initialize temp file

                # Get download path immediately after temp file creation
                download_path = temp_writer.download_path

                # Create response headers with download path
                headers = {
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                    "X-Download-Path": download_path,
                }

                # Create async generator for streaming
                async def dual_output():
                    try:
                        # Write chunks to temp file and stream
                        async for chunk_data in generator:
                            # The timestamp acumulator is only used when word level time stamps are generated but no audio is returned.
                            timestamp_acumulator = []

                            if chunk_data.output:  # Skip empty chunks
                                await temp_writer.write(chunk_data.output)
                                base64_chunk = base64.b64encode(
                                    chunk_data.output
                                ).decode("utf-8")

                                # Add any chunks that may be in the acumulator into the return word_timestamps
                                if chunk_data.word_timestamps is not None:
                                    chunk_data.word_timestamps = (
                                        timestamp_acumulator + chunk_data.word_timestamps
                                    )
                                    timestamp_acumulator = []
                                else:
                                    chunk_data.word_timestamps = []

                                yield CaptionedSpeechResponse(
                                    audio=base64_chunk,
                                    audio_format=content_type,
                                    timestamps=chunk_data.word_timestamps,
                                )
                            else:
                                if (
                                    chunk_data.word_timestamps is not None
                                    and len(chunk_data.word_timestamps) > 0
                                ):
                                    timestamp_acumulator += chunk_data.word_timestamps

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
                return JSONStreamingResponse(
                    dual_output(), media_type="application/json", headers=headers
                )

            async def single_output():
                try:
                    # The timestamp acumulator is only used when word level time stamps are generated but no audio is returned.
                    timestamp_acumulator = []

                    # Stream chunks
                    async for chunk_data in generator:
                        if chunk_data.output:  # Skip empty chunks
                            # Encode the chunk bytes into base 64
                            base64_chunk = base64.b64encode(chunk_data.output).decode(
                                "utf-8"
                            )

                            # Add any chunks that may be in the acumulator into the return word_timestamps
                            if chunk_data.word_timestamps is not None:
                                chunk_data.word_timestamps = (
                                    timestamp_acumulator + chunk_data.word_timestamps
                                )
                            else:
                                chunk_data.word_timestamps = []
                            timestamp_acumulator = []

                            yield CaptionedSpeechResponse(
                                audio=base64_chunk,
                                audio_format=content_type,
                                timestamps=chunk_data.word_timestamps,
                            )
                        else:
                            if (
                                chunk_data.word_timestamps is not None
                                and len(chunk_data.word_timestamps) > 0
                            ):
                                timestamp_acumulator += chunk_data.word_timestamps

                except Exception as e:
                    logger.error(f"Error in single output streaming: {e}")
                    writer.close()
                    raise

            # Standard streaming without download link
            return JSONStreamingResponse(
                single_output(),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            # Generate complete audio using public interface
            audio_data = await tts_service.generate_audio(
                text=request.input,
                voice=voice_name,
                writer=writer,
                speed=request.speed,
                return_timestamps=request.return_timestamps,
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

            base64_output = base64.b64encode(output).decode("utf-8")

            content = CaptionedSpeechResponse(
                audio=base64_output,
                audio_format=content_type,
                timestamps=audio_data.word_timestamps,
            ).model_dump()

            writer.close()

            return JSONResponse(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",  # Prevent caching
                },
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
        logger.error(f"Unexpected error in captioned speech generation: {str(e)}")

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
