"""
Enhanced ZipVoice TTS router with smart features.

Includes:
- Auto-transcription with Whisper
- Quality detection and warnings
- Smart parameter tuning
- ONNX/TensorRT optimizations
"""

import base64
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Header, UploadFile, Query
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from ..core.config import settings
from ..inference.model_manager import ModelManager, get_manager
from ..inference.voice_prompt_manager import get_voice_prompt_manager, VoicePromptManager
from ..services.auto_transcription import get_auto_transcription_service
from ..services.quality_detection import get_quality_detection_service
from ..services.smart_tuning import get_smart_tuning_service
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..structures.schemas import (
    VoiceInfoResponse,
    VoiceListResponse,
    ZipVoiceSpeechRequest,
)

router = APIRouter(
    prefix="/zipvoice",
    tags=["ZipVoice TTS (Enhanced)"],
    responses={404: {"description": "Not found"}},
)


@router.post("/audio/speech")
async def create_speech_zipvoice(
    request: ZipVoiceSpeechRequest,
    prompt_wav_file: Optional[UploadFile] = File(None),
    auto_transcribe: Optional[bool] = Query(None, description="Auto-transcribe prompt_wav (overrides setting)"),
    auto_tune: Optional[bool] = Query(None, description="Auto-tune parameters (overrides setting)"),
    priority: Optional[str] = Query("balanced", description="Optimization priority: speed, balanced, quality"),
    model_manager: ModelManager = Depends(get_manager),
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Generate speech using ZipVoice with smart features.

    **Smart Features:**
    - Auto-transcription: Automatically transcribe prompt_wav if prompt_text not provided
    - Quality detection: Warn about low-quality prompts
    - Smart tuning: Auto-optimize parameters based on input text
    - ONNX/TensorRT: Use optimized inference if enabled

    **Voice Input Methods:**
    - `file+<name>`: Upload via multipart/form-data
    - `url+<url>`: Download from URL
    - `base64+<data>`: Base64 encoded audio
    - `<name>`: Pre-registered voice
    """
    try:
        # Initialize services
        auto_transcribe_enabled = auto_transcribe if auto_transcribe is not None else settings.enable_auto_transcription
        auto_tune_enabled = auto_tune if auto_tune is not None else settings.enable_smart_tuning
        quality_detection_enabled = settings.enable_quality_detection

        transcription_service = get_auto_transcription_service() if auto_transcribe_enabled else None
        quality_service = get_quality_detection_service() if quality_detection_enabled else None
        tuning_service = get_smart_tuning_service() if auto_tune_enabled else None

        # Get prompt_wav path
        prompt_wav_path = None
        prompt_text = request.prompt_text
        temp_file_created = False

        if request.voice.startswith("file+"):
            # Handle file upload
            if not prompt_wav_file:
                raise HTTPException(
                    status_code=400,
                    detail="prompt_wav file required when using 'file+' voice format"
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                content = await prompt_wav_file.read()
                tmp_file.write(content)
                prompt_wav_path = tmp_file.name
                temp_file_created = True

            # Validate audio
            if not voice_prompt_mgr.validate_audio(prompt_wav_path):
                os.unlink(prompt_wav_path)
                raise HTTPException(
                    status_code=400,
                    detail="Invalid audio file or duration exceeds maximum"
                )

        elif request.voice.startswith("url+"):
            url = request.voice[4:]
            try:
                prompt_wav_path = await voice_prompt_mgr.download_from_url(url)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download: {str(e)}")

        elif request.voice.startswith("base64+"):
            b64_data = request.voice[7:]
            try:
                prompt_wav_path = voice_prompt_mgr.decode_base64(b64_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode base64: {str(e)}")

        else:
            # Pre-registered voice
            try:
                prompt_wav_path, cached_prompt_text = voice_prompt_mgr.get_voice_prompt(request.voice)
                if not prompt_text:
                    prompt_text = cached_prompt_text
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice '{request.voice}' not found"
                )

        # **SMART FEATURE 1: Auto-transcription**
        if not prompt_text and transcription_service and transcription_service.is_available():
            logger.info("Auto-transcribing prompt_wav with Whisper...")
            prompt_text = await transcription_service.transcribe_audio(prompt_wav_path)

            if not prompt_text:
                logger.warning("Auto-transcription failed")
                raise HTTPException(
                    status_code=400,
                    detail="prompt_text required (auto-transcription failed)"
                )

            logger.info(f"Auto-transcribed: {prompt_text[:100]}...")

        if not prompt_text:
            raise HTTPException(
                status_code=400,
                detail="prompt_text is required (set enable_auto_transcription=true to auto-generate)"
            )

        # **SMART FEATURE 2: Quality Detection**
        quality_warnings = []
        if quality_service:
            logger.debug("Analyzing prompt audio quality...")
            is_valid, issues = quality_service.validate_prompt_quality(prompt_wav_path)

            if not is_valid:
                quality_warnings = issues
                logger.warning(f"Quality issues detected: {issues}")

                # If quality is very poor, reject
                analysis = quality_service.analyze_audio(prompt_wav_path)
                if analysis['quality_score'] < 0.3:
                    if temp_file_created:
                        os.unlink(prompt_wav_path)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Audio quality too low (score: {analysis['quality_score']:.2f}). Issues: {', '.join(issues)}"
                    )

        # **SMART FEATURE 3: Smart Parameter Tuning**
        if tuning_service:
            logger.debug("Smart tuning parameters...")
            recommendations = tuning_service.recommend_parameters(request.input, priority=priority)

            # Apply recommendations if not explicitly set
            if request.num_steps is None:
                request.num_steps = recommendations['num_steps']
                logger.info(f"Auto-tuned num_steps: {request.num_steps}")

            if request.model == "zipvoice" and priority == "speed":
                request.model = recommendations['model']
                logger.info(f"Auto-tuned model: {request.model}")

            # Estimate generation time
            estimated_time = tuning_service.estimate_generation_time(
                request.input,
                request.num_steps,
                request.model
            )
            logger.info(f"Estimated generation time: {estimated_time:.1f}s")

        # Get backend
        try:
            backend = model_manager.get_backend('zipvoice')
        except KeyError:
            raise HTTPException(
                status_code=503,
                detail="ZipVoice backend not available"
            )

        # Prepare generation parameters
        gen_params = {
            'prompt_text': prompt_text,
            'num_steps': request.num_steps or settings.zipvoice_num_steps,
            'remove_long_silence': request.remove_long_silence if request.remove_long_silence is not None else settings.zipvoice_remove_long_silence,
            'max_duration': request.max_duration,
        }

        logger.info(f"Generating with: model={request.model}, steps={gen_params['num_steps']}")

        # **Generate Audio**
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
                        audio_bytes = writer.write_chunk(chunk.audio)
                        if audio_bytes:
                            yield audio_bytes

                    final_bytes = writer.finalize()
                    if final_bytes:
                        yield final_bytes

                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    raise
                finally:
                    if temp_file_created:
                        try:
                            os.unlink(prompt_wav_path)
                        except:
                            pass

            headers = {}
            if quality_warnings:
                headers['X-Quality-Warnings'] = '; '.join(quality_warnings)

            return StreamingResponse(
                generate_stream(),
                media_type=f"audio/{request.response_format}",
                headers=headers
            )

        else:
            # Non-streaming
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

                if all_audio:
                    complete_audio = np.concatenate(all_audio)

                    writer = StreamingAudioWriter(
                        format=request.response_format,
                        sample_rate=24000
                    )
                    audio_bytes = writer.write_chunk(complete_audio)
                    audio_bytes += writer.finalize()

                    headers = {}
                    if quality_warnings:
                        headers['X-Quality-Warnings'] = '; '.join(quality_warnings)

                    return Response(
                        content=audio_bytes,
                        media_type=f"audio/{request.response_format}",
                        headers=headers
                    )
                else:
                    raise HTTPException(status_code=500, detail="No audio generated")

            finally:
                if temp_file_created:
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
    transcription: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    auto_transcribe: Optional[bool] = Form(None),
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Register a reusable voice prompt with smart features.

    **Smart Features:**
    - Auto-transcription: If transcription not provided, use Whisper
    - Quality analysis: Return quality score and recommendations
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        # Auto-transcribe if needed
        should_auto_transcribe = auto_transcribe if auto_transcribe is not None else settings.auto_transcribe_on_upload

        if not transcription and should_auto_transcribe:
            logger.info("Auto-transcribing uploaded voice...")
            transcription_service = get_auto_transcription_service()

            if transcription_service.is_available():
                transcription = await transcription_service.transcribe_audio(temp_path)

                if not transcription:
                    os.unlink(temp_path)
                    raise HTTPException(
                        status_code=400,
                        detail="Failed to auto-transcribe. Please provide transcription manually."
                    )

                logger.info(f"Auto-transcribed: {transcription[:100]}...")

        if not transcription:
            os.unlink(temp_path)
            raise HTTPException(
                status_code=400,
                detail="transcription required (set auto_transcribe=true to auto-generate)"
            )

        # Quality analysis
        quality_info = {}
        if settings.enable_quality_detection:
            quality_service = get_quality_detection_service()
            analysis = quality_service.analyze_audio(temp_path)
            quality_info = {
                'quality_score': analysis['quality_score'],
                'recommendations': analysis.get('recommendations', []),
                'warnings': analysis.get('warnings', [])
            }

            logger.info(f"Quality score: {analysis['quality_score']:.2f}")

        # Register voice
        try:
            voice_prompt_mgr.register_voice(name, temp_path, transcription)
        except Exception as e:
            os.unlink(temp_path)
            raise HTTPException(status_code=400, detail=str(e))

        # Get audio info
        audio_info = voice_prompt_mgr.get_audio_info(
            voice_prompt_mgr.registry[name]['audio_path']
        )

        return {
            "status": "success",
            "name": name,
            "transcription": transcription,
            "auto_transcribed": should_auto_transcribe and auto_transcribe,
            "audio_info": audio_info,
            "quality": quality_info
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


@router.get("/voices/{voice_name}/quality")
async def analyze_voice_quality(
    voice_name: str,
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Analyze quality of a registered voice.

    Returns detailed quality metrics and recommendations.
    """
    try:
        audio_path, _ = voice_prompt_mgr.get_voice_prompt(voice_name)

        if not settings.enable_quality_detection:
            raise HTTPException(
                status_code=503,
                detail="Quality detection disabled"
            )

        quality_service = get_quality_detection_service()
        analysis = quality_service.analyze_audio(audio_path)

        return {
            "voice_name": voice_name,
            "analysis": analysis
        }

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
    cache_type: str = "all",
    voice_prompt_mgr: VoicePromptManager = Depends(get_voice_prompt_manager),
):
    """Clear cached voice prompts."""
    if cache_type not in ["url", "base64", "all"]:
        raise HTTPException(
            status_code=400,
            detail="cache_type must be 'url', 'base64', or 'all'"
        )

    count = voice_prompt_mgr.clear_cache(cache_type)
    return {"status": "cleared", "cache_type": cache_type, "files_deleted": count}


@router.post("/tune")
async def get_tuning_recommendations(
    text: str = Form(...),
    priority: str = Form("balanced")
):
    """Get smart parameter tuning recommendations.

    Returns optimized parameters for the given text and priority.
    """
    if not settings.enable_smart_tuning:
        raise HTTPException(
            status_code=503,
            detail="Smart tuning disabled"
        )

    tuning_service = get_smart_tuning_service()

    # Get recommendations
    recommendations = tuning_service.recommend_parameters(text, priority=priority)

    # Get text analysis
    analysis = tuning_service.analyze_text(text)

    # Estimate time
    estimated_time = tuning_service.estimate_generation_time(
        text,
        recommendations['num_steps'],
        recommendations['model']
    )

    return {
        "text_analysis": analysis,
        "recommendations": recommendations,
        "estimated_time_seconds": estimated_time
    }
