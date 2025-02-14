import re
from typing import List, Union, AsyncGenerator, Tuple

import numpy as np
import torch
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from kokoro import KPipeline
from loguru import logger

from ..inference.base import AudioChunk
from ..core.config import settings
from ..services.audio import AudioNormalizer, AudioService
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.text_processing import smart_split
from ..services.tts_service import TTSService
from ..services.temp_manager import TempFileWriter
from ..structures import CaptionedSpeechRequest, CaptionedSpeechResponse, WordTimestamp
from ..structures.custom_responses import JSONStreamingResponse
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
)
from .openai_compatible import process_voices, stream_audio_chunks
import json
import os
import base64
from pathlib import Path


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
                    normalized_audio = await normalizer.normalize(chunk_audio)
                    # Write chunk and yield bytes
                    chunk_bytes = writer.write_chunk(normalized_audio)
                    if chunk_bytes:
                        yield chunk_bytes

                    # Finalize and yield remaining bytes
                    final_bytes = writer.write_chunk(finalize=True)
                    if final_bytes:
                        yield final_bytes
                else:
                    raise ValueError("Failed to generate audio data")

            except Exception as e:
                logger.error(f"Error in audio generation: {str(e)}")
                # Clean up writer on error
                writer.write_chunk(finalize=True)
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
        voice_name = await process_voices(request.voice, tts_service)

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
                        async for chunk,chunk_data in generator:
                            if chunk:  # Skip empty chunks
                                await temp_writer.write(chunk)
                                base64_chunk= base64.b64encode(chunk).decode("utf-8")
                            
                                yield CaptionedSpeechResponse(audio=base64_chunk,audio_format=content_type,timestamps=chunk_data.word_timestamps)

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

                # Stream with temp file writing
                return JSONStreamingResponse(
                    dual_output(), media_type="application/json", headers=headers
                )

            async def single_output():
                try:
                    # Stream chunks
                    async for chunk,chunk_data in generator:
                        if chunk:  # Skip empty chunks
                            # Encode the chunk bytes into base 64
                            base64_chunk= base64.b64encode(chunk).decode("utf-8")
                            
                            yield CaptionedSpeechResponse(audio=base64_chunk,audio_format=content_type,timestamps=chunk_data.word_timestamps)
                except Exception as e:
                    logger.error(f"Error in single output streaming: {e}")
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
            _, audio_data = await tts_service.generate_audio(
                text=request.input,
                voice=voice_name,
                speed=request.speed,
                return_timestamps=request.return_timestamps,
                normalization_options=request.normalization_options,
                lang_code=request.lang_code,
            )
            
            content, audio_data = await AudioService.convert_audio(
                audio_data,
                24000,
                request.response_format,
                is_first_chunk=True,
                is_last_chunk=False,
            )
            
            # Convert to requested format with proper finalization
            final, _ = await AudioService.convert_audio(
                AudioChunk(np.array([], dtype=np.int16)),
                24000,
                request.response_format,
                is_first_chunk=False,
                is_last_chunk=True,
            )
            output=content+final
            
            base64_output= base64.b64encode(output).decode("utf-8")
            
            content=CaptionedSpeechResponse(audio=base64_output,audio_format=content_type,timestamps=audio_data.word_timestamps).model_dump()
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
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
    
    """
    try:
        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        # Create streaming audio writer and normalizer
        writer = StreamingAudioWriter(
            format=request.response_format, sample_rate=24000, channels=1
        )
        normalizer = AudioNormalizer()

        # Get voice path
        voice_name, voice_path = await tts_service._get_voice_path(request.voice)

        # Use provided lang_code or determine from voice name
        pipeline_lang_code = request.lang_code if request.lang_code else request.voice[0].lower()
        logger.info(
            f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in text chunking"
        )

        # Get backend and pipeline
        backend = tts_service.model_manager.get_backend()
        pipeline = backend._get_pipeline(pipeline_lang_code)

        # Create temp file writer for timestamps
        temp_writer = TempFileWriter("json")
        await temp_writer.__aenter__()  # Initialize temp file
        # Get just the filename without the path
        timestamps_filename = Path(temp_writer.download_path).name

        # Initialize variables for timestamps
        word_timestamps = []
        current_offset = 0.0

        async def generate_chunks():
            nonlocal current_offset, word_timestamps
            try:
                # Process text in chunks with smart splitting
                async for chunk_text, tokens in smart_split(request.input):
                    # Process chunk with pipeline
                    for result in pipeline(chunk_text, voice=voice_path, speed=request.speed):
                        if result.audio is not None:
                            # Process timestamps for this chunk
                            if hasattr(result, "tokens") and result.tokens and result.pred_dur is not None:
                                try:
                                    # Join timestamps for this chunk's tokens
                                    KPipeline.join_timestamps(result.tokens, result.pred_dur)

                                    # Add timestamps with offset
                                    for token in result.tokens:
                                        if not all(
                                            hasattr(token, attr)
                                            for attr in ["text", "start_ts", "end_ts"]
                                        ):
                                            continue
                                        if not token.text or not token.text.strip():
                                            continue

                                        # Apply offset to timestamps
                                        start_time = float(token.start_ts) + current_offset
                                        end_time = float(token.end_ts) + current_offset

                                        word_timestamps.append(
                                            {
                                                "word": str(token.text).strip(),
                                                "start_time": start_time,
                                                "end_time": end_time,
                                            }
                                        )

                                    # Update offset for next chunk
                                    chunk_duration = float(result.pred_dur.sum()) / 80  # Convert frames to seconds
                                    current_offset = max(current_offset + chunk_duration, end_time)

                                except Exception as e:
                                    logger.error(f"Failed to process timestamps for chunk: {e}")

                            # Process audio
                            audio_chunk = result.audio.numpy()
                            normalized_audio = await normalizer.normalize(audio_chunk)
                            chunk_bytes = writer.write_chunk(normalized_audio)
                            if chunk_bytes:
                                yield chunk_bytes

                # Write timestamps to temp file
                timestamps_json = json.dumps(word_timestamps)
                await temp_writer.write(timestamps_json.encode())
                await temp_writer.finalize()

                # Finalize audio
                final_bytes = writer.write_chunk(finalize=True)
                if final_bytes:
                    yield final_bytes

            except Exception as e:
                logger.error(f"Error in audio generation: {str(e)}")
                # Clean up writer on error
                writer.write_chunk(finalize=True)
                await temp_writer.__aexit__(type(e), e, e.__traceback__)
                # Re-raise the original exception
                raise

        return StreamingResponse(
            generate_chunks(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
                "X-Timestamps-Path": timestamps_filename,
            },
        )

    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except RuntimeError as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
    """