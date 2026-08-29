import base64
import os
import re
from pathlib import Path
from typing import AsyncGenerator, List, Tuple, Union

import soundfile as sf
import torch
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from kokoro import KPipeline
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..inference import voice_tune
from ..services.audio import AudioNormalizer
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.temp_manager import TempFileWriter
from ..services.text_processing import smart_split
from ..services.text_processing.text_processor import check_pause_budget
from ..services.tts_service import TTSService
from ..structures import (
    CaptionedSpeechRequest,
    CaptionedSpeechResponse,
    DialogueRequest,
    OpenAISpeechRequest,
    VoiceTuneRequest,
    VoiceTuneResponse,
    WordTimestamp,
)
from ..structures.custom_responses import JSONStreamingResponse
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
)
from .openai_compatible import (
    apply_alias_rate,
    create_speech,
    process_and_validate_voice_tags,
    process_and_validate_voices,
    require_voice_tags_enabled,
    stream_audio_chunks,
)
from .ssml import apply_ssml

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
        pipeline = KPipeline(
            lang_code=request.language, repo_id=settings.model_repo_id, model=False
        )

        # Get first result from pipeline (we only need one since we're not chunking)
        for result in pipeline(request.text):
            # result.graphemes = original text
            # result.phonemes = phonemized text
            # result.tokens = token objects (if available)
            return PhonemeResponse(phonemes=result.phonemes, tokens=[])

        raise ValueError("Failed to generate phonemes")
    except ValueError as e:
        logger.warning(f"Invalid phoneme request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
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


@router.post("/dev/dialogue")
async def create_dialogue(
    request: DialogueRequest,
    client_request: Request,
):
    """Generate multi-speaker audio from an ordered list of turns.

    Thin wrapper over /v1/audio/speech: turns are rendered to the existing
    inline [voice:...] and [pause:Xs] tags, so streaming, formats, and
    download links behave identically.
    """
    require_voice_tags_enabled()

    speech_request = OpenAISpeechRequest(
        model=request.model,
        input=request.to_tagged_input(),
        voice=request.turns[0].voice,
        response_format=request.response_format,
        download_format=request.download_format,
        speed=request.speed,
        stream=request.stream,
        return_download_link=request.return_download_link,
        return_timing=request.return_timing,
        lang_code=request.lang_code,
        volume_multiplier=request.volume_multiplier,
        normalization_options=request.normalization_options,
        allow_voice_tags=True,  # always on here
        voice_aliases=request.voice_aliases,
    )
    return await create_speech(
        request=speech_request,
        client_request=client_request,
        x_raw_response=None,
    )


@router.post("/dev/captioned_speech")
async def create_captioned_speech(
    request: CaptionedSpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
    tts_service: TTSService = Depends(get_tts_service),
):
    """Generate audio with word-level timestamps using streaming approach"""

    if request.allow_voice_tags:
        require_voice_tags_enabled()

    if request.ssml:
        await apply_ssml(request)

    try:
        # model_name = get_model_name(request.model)
        tts_service = await get_tts_service()
        voice_name = await process_and_validate_voices(
            request.voice, tts_service, request.voice_aliases
        )
        # resolved here, not in the generator, so a bad tag 400s before the stream opens
        request.input = await process_and_validate_voice_tags(
            request.input, tts_service, request.allow_voice_tags, request.voice_aliases
        )
        apply_alias_rate(request)
        # checked post-SSML and pre-stream, so an over-budget request 400s before headers
        check_pause_budget(request.input)

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
                tts_service, request, client_request, writer, voice_name
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
                                        timestamp_acumulator
                                        + chunk_data.word_timestamps
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
                allow_voice_tags=request.allow_voice_tags,
                output_format=request.response_format,
            )
            output = audio_data.output

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


@router.post("/dev/unload")
async def unload_model(
    tts_service: TTSService = Depends(get_tts_service),
):
    """Release the model from GPU VRAM without stopping the container.

    The model reloads automatically on the next inference request.
    Useful for homelab deployments where GPU memory is shared across services.
    """
    if not settings.allow_dev_unload:
        raise HTTPException(
            status_code=403,
            detail={"error": "The /dev/unload endpoint is disabled"},
        )
    try:
        if tts_service.model_manager is None:
            raise HTTPException(
                status_code=503, detail={"error": "Model manager not initialized"}
            )
        await tts_service.model_manager.unload()
        return JSONResponse({"status": "unloaded"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post("/dev/voices/tune", response_model=VoiceTuneResponse)
async def tune_voice(
    request: VoiceTuneRequest,
    tts_service: TTSService = Depends(get_tts_service),
) -> VoiceTuneResponse:
    """Enroll a reference clip as a voice pack.

    Saves `<name>.pt` into the tune voices dir, so the voice then works by name in
    every speech route. Needs TUNE_ADAPTER; re-enrolling a name replaces it, stock
    names are refused.
    """
    if not settings.tune_adapter:
        raise HTTPException(
            status_code=403,
            detail={"error": "Voice tuning is disabled, set TUNE_ADAPTER"},
        )
    try:
        if tts_service.model_manager is None:
            raise HTTPException(
                status_code=503, detail={"error": "Model manager not initialized"}
            )
        if os.path.exists(os.path.join(paths.get_voices_dir(), f"{request.name}.pt")):
            raise HTTPException(
                status_code=409,
                detail={"error": f"{request.name} is a stock voice, pick another name"},
            )
        wav, sr = voice_tune.decode_audio(base64.b64decode(request.audio))
        await tts_service.model_manager.ensure_backend()
        backend = tts_service.model_manager.get_backend()
        pack = backend.enroll_voice(wav, sr, request.strength)
        voice_path = os.path.join(paths.get_tune_voices_dir(), f"{request.name}.pt")
        await paths.save_voice_tensor(pack, voice_path)
        backend.evict_voice(voice_path)
        return VoiceTuneResponse(
            voice=request.name,
            adapter=paths.tune_alias(),
            speed=round(pack[0, 0, -1].item(), 3),
            f0_mean_st=round(pack[0, 0, -2].item(), 2),
        )
    except HTTPException:
        raise
    except (ValueError, sf.LibsndfileError) as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except Exception as e:
        logger.error(f"Error tuning voice: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
