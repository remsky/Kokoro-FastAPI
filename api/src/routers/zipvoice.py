"""ZipVoice TTS router with voice cloning support."""

import base64
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from ..core.config import settings
from ..inference.model_manager import ModelManager, get_manager
from ..inference.voice_prompt_manager import get_voice_prompt_manager, VoicePromptManager
from ..services.audio import AudioService
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..structures.schemas import (
    VoiceInfoResponse,
    VoiceListResponse,
    VoiceRegistrationRequest,
    ZipVoiceSpeechRequest,
)

router = APIRouter(
    prefix="/zipvoice",
    tags=["ZipVoice TTS"],
    responses={404: {"description": "Not found"}},
)


@router.post("/audio/speech")
async def create_speech_zipvoice(
    request: ZipVoiceSpeechRequest,
    prompt_wav_file: Optional[UploadFile] = File(None),
    model_manager: ModelManager = Depends(get_manager),
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Generate speech using ZipVoice with zero-shot voice cloning.

    Supports multiple methods to provide the voice prompt:
    - Upload file via multipart/form-data (voice="file+<name>")
    - URL download (voice="url+<url>")
    - Base64 encoded (voice="base64+<data>")
    - Pre-registered voice (voice="<name>")
    """
    try:
        # Determine how to get the prompt_wav
        prompt_wav_path = None
        prompt_text = request.prompt_text

        if request.voice.startswith("file+"):
            # Handle file upload
            if not prompt_wav_file:
                raise HTTPException(
                    status_code=400,
                    detail="prompt_wav file required when using 'file+' voice format"
                )

            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                content = await prompt_wav_file.read()
                tmp_file.write(content)
                prompt_wav_path = tmp_file.name

            # Validate audio
            if not voice_prompt_mgr.validate_audio(prompt_wav_path):
                os.unlink(prompt_wav_path)
                raise HTTPException(
                    status_code=400,
                    detail="Invalid audio file or duration exceeds maximum"
                )

        elif request.voice.startswith("url+"):
            # Download from URL
            url = request.voice[4:]  # Strip "url+" prefix
            try:
                prompt_wav_path = await voice_prompt_mgr.download_from_url(url)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download audio from URL: {str(e)}"
                )

        elif request.voice.startswith("base64+"):
            # Decode base64
            b64_data = request.voice[7:]  # Strip "base64+" prefix
            try:
                prompt_wav_path = voice_prompt_mgr.decode_base64(b64_data)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode base64 audio: {str(e)}"
                )

        else:
            # Lookup pre-registered voice
            try:
                prompt_wav_path, cached_prompt_text = voice_prompt_mgr.get_voice_prompt(request.voice)
                # Use cached transcription if not provided
                if not prompt_text:
                    prompt_text = cached_prompt_text
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice '{request.voice}' not found in registry. "
                           f"Available voices: {list(voice_prompt_mgr.list_voices().keys())}"
                )

        # Ensure we have prompt_text
        if not prompt_text:
            raise HTTPException(
                status_code=400,
                detail="prompt_text is required for ZipVoice generation"
            )

        # Get ZipVoice backend
        try:
            backend = model_manager.get_backend('zipvoice')
        except KeyError:
            raise HTTPException(
                status_code=503,
                detail="ZipVoice backend not available. Enable it in settings."
            )

        # Generate audio
        logger.info(f"Generating speech with ZipVoice: {request.input[:50]}...")

        # Prepare generation parameters
        gen_params = {
            'prompt_text': prompt_text,
            'num_steps': request.num_steps,
            'remove_long_silence': request.remove_long_silence,
            'max_duration': request.max_duration,
        }

        # Streaming response
        if request.stream:
            async def generate_stream():
                writer = StreamingAudioWriter(
                    format=request.response_format,
                    sample_rate=24000
                )

                try:
                    async for chunk in model_manager.generate(
                        text=request.input,
                        voice=prompt_wav_path,
                        speed=request.speed,
                        backend_type='zipvoice',
                        **gen_params
                    ):
                        # Convert to requested format
                        audio_bytes = writer.write_chunk(chunk.audio)
                        if audio_bytes:
                            yield audio_bytes

                    # Finalize
                    final_bytes = writer.finalize()
                    if final_bytes:
                        yield final_bytes

                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    raise
                finally:
                    # Clean up temp file if we created one
                    if request.voice.startswith("file+") and prompt_wav_path:
                        try:
                            os.unlink(prompt_wav_path)
                        except:
                            pass

            return StreamingResponse(
                generate_stream(),
                media_type=f"audio/{request.response_format}",
                headers={"Content-Type": f"audio/{request.response_format}"}
            )

        else:
            # Non-streaming response
            import numpy as np

            all_audio = []
            try:
                async for chunk in model_manager.generate(
                    text=request.input,
                    voice=prompt_wav_path,
                    speed=request.speed,
                    backend_type='zipvoice',
                    **gen_params
                ):
                    all_audio.append(chunk.audio)

                # Concatenate all chunks
                if all_audio:
                    complete_audio = np.concatenate(all_audio)

                    # Convert to requested format
                    writer = StreamingAudioWriter(
                        format=request.response_format,
                        sample_rate=24000
                    )
                    audio_bytes = writer.write_chunk(complete_audio)
                    audio_bytes += writer.finalize()

                    return Response(
                        content=audio_bytes,
                        media_type=f"audio/{request.response_format}"
                    )
                else:
                    raise HTTPException(status_code=500, detail="No audio generated")

            finally:
                # Clean up temp file if we created one
                if request.voice.startswith("file+") and prompt_wav_path:
                    try:
                        os.unlink(prompt_wav_path)
                    except:
                        pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ZipVoice generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voices/register")
async def register_voice(
    name: str = Form(...),
    transcription: str = Form(...),
    audio_file: UploadFile = File(...),
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Register a reusable voice prompt for ZipVoice.

    Upload an audio file with its transcription to create a named voice
    that can be reused in future requests.
    """
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Register the voice
        try:
            voice_prompt_mgr.register_voice(name, tmp_path, transcription)
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=str(e))

        # Get audio info
        audio_info = voice_prompt_mgr.get_audio_info(
            voice_prompt_mgr.registry[name]['audio_path']
        )

        return {
            "status": "success",
            "name": name,
            "transcription": transcription,
            "audio_info": audio_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """List all registered ZipVoice voice prompts."""
    voices = voice_prompt_mgr.list_voices()
    return VoiceListResponse(voices=voices, count=len(voices))


@router.get("/voices/{voice_name}", response_model=VoiceInfoResponse)
async def get_voice_info(
    voice_name: str,
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Get information about a specific registered voice."""
    try:
        audio_path, transcription = voice_prompt_mgr.get_voice_prompt(voice_name)
        audio_info = voice_prompt_mgr.get_audio_info(audio_path)

        return VoiceInfoResponse(
            name=voice_name,
            transcription=transcription,
            audio_info=audio_info
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found"
        )


@router.delete("/voices/{voice_name}")
async def delete_voice(
    voice_name: str,
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Delete a registered voice prompt."""
    success = voice_prompt_mgr.delete_voice(voice_name)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found"
        )
    return {"status": "deleted", "name": voice_name}


@router.post("/voices/cache/clear")
async def clear_voice_cache(
    cache_type: str = "all",  # "url", "base64", or "all"
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Clear cached voice prompts (URL downloads and base64 decodes)."""
    if cache_type not in ["url", "base64", "all"]:
        raise HTTPException(
            status_code=400,
            detail="cache_type must be 'url', 'base64', or 'all'"
        )

    count = voice_prompt_mgr.clear_cache(cache_type)
    return {"status": "cleared", "cache_type": cache_type, "files_deleted": count}
