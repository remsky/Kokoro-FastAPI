import asyncio
import json
import os
import re
import shutil
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from pydantic import ValidationError
from starlette.background import BackgroundTask

from ..core import paths
from ..core.config import settings
from ..inference import inno_tuner
from ..inference.voice_manager import get_manager as get_voice_manager
from ..services.tts_service import TTSService
from ..structures import OpenAISpeechRequest
from .openai_compatible import create_speech, get_tts_service

router = APIRouter(tags=["voice tuning"])

SAVE_NAME = re.compile(r"^[ab][a-z]?_[a-z0-9]+(_[a-z0-9]+)*$")
SAVE_SUFFIX = "_tuned"
SAVE_NAME_RULE = (
    "save_voice must start with a (US) or b (UK English), an optional second letter, then _ "
    "and lowercase letters, digits, single underscores, e.g. ax_me"
)


def _bad_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": "validation_error",
            "message": message,
            "type": "invalid_request_error",
        },
    )


def _copy_new(src: str, dest: str) -> None:
    with open(src, "rb") as f, open(dest, "xb") as out:
        shutil.copyfileobj(f, out)


async def _discard(tts_service: TTSService, voice_name: str, pack_path: str) -> None:
    (await get_voice_manager()).forget_transient(voice_name)
    try:
        tts_service.model_manager.get_backend().forget_voice(pack_path)
    except Exception:
        pass
    try:
        os.remove(pack_path)
    except OSError:
        pass


async def _discard_after(
    body: AsyncIterator, tts_service: TTSService, voice_name: str, pack_path: str
):
    try:
        async for chunk in body:
            yield chunk
    finally:
        await _discard(tts_service, voice_name, pack_path)


@router.post("/dev/tune")
async def tune_speech(
    client_request: Request,
    audio: UploadFile = File(..., description="Reference clip, 3 to 30 s, one speaker"),
    request: Optional[str] = Form(
        None, description="JSON body as for /v1/audio/speech, minus voice"
    ),
    prosody_head: bool = Form(True),
    fmax: Optional[float] = Form(None, ge=60, le=1000),
    return_voice_pack: bool = Form(False),
    save_voice: str = Form(""),
    tts_service: TTSService = Depends(get_tts_service),
):
    """Tune a voice from the reference clip, then speak `request` with it.

    return_voice_pack=true returns the [510, 1, 256] .pt instead. save_voice=<name>
    keeps it as VOICES_DIR/<name>_tuned.pt and needs ALLOW_LOCAL_VOICE_SAVING.
    Otherwise the pack lives in the temp dir for the length of the response only.
    """
    if not settings.enable_inno_tuner:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "permission_denied",
                "message": "Inno voice tuner is disabled",
                "type": "permission_error",
            },
        )
    if save_voice and not settings.allow_local_voice_saving:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "permission_denied",
                "message": "Local voice saving is disabled",
                "type": "permission_error",
            },
        )
    if not inno_tuner.available():
        raise HTTPException(
            status_code=503,
            detail={"error": "unavailable", "message": "Inno voice tuner not loaded"},
        )
    if not (request or return_voice_pack or save_voice):
        raise _bad_request("request, return_voice_pack, or save_voice is required")

    save_name = save_voice.strip().lower()
    if save_name:
        if not SAVE_NAME.match(save_name):
            raise _bad_request(SAVE_NAME_RULE)
        if not save_name.endswith(SAVE_SUFFIX):
            save_name += SAVE_SUFFIX

    speech = None
    if request:
        try:
            speech = OpenAISpeechRequest(**{**json.loads(request), "voice": "a_tune"})
        except (TypeError, ValueError, ValidationError) as e:
            raise _bad_request(f"request: {e}")
        speech.allow_voice_tags = False
        speech.ssml = False

    data = await audio.read(inno_tuner.MAX_UPLOAD_BYTES + 1)
    if len(data) > inno_tuner.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "too_large",
                "message": f"reference exceeds {inno_tuner.MAX_UPLOAD_BYTES >> 20} MB",
            },
        )
    if not inno_tuner.reserve():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "busy",
                "message": "Inno voice tuner is busy, retry shortly",
            },
        )
    try:
        pack_path = await asyncio.shield(
            asyncio.to_thread(inno_tuner.tune, data, prosody_head, fmax)
        )
    except ValueError as e:
        raise _bad_request(str(e))

    voice_name = os.path.splitext(os.path.basename(pack_path))[0]
    (await get_voice_manager()).register_transient(voice_name, pack_path)
    cleanup = BackgroundTask(_discard, tts_service, voice_name, pack_path)

    if save_name:
        dest = os.path.join(paths.voices_dir(), f"{save_name}.pt")
        try:
            await asyncio.to_thread(_copy_new, pack_path, dest)
        except FileExistsError:
            await _discard(tts_service, voice_name, pack_path)
            raise HTTPException(
                status_code=409,
                detail={"error": "conflict", "message": f"{save_name} already exists"},
            )
        logger.info(f"Saved tuned voice {save_name}")

    if return_voice_pack:
        return FileResponse(
            pack_path,
            media_type="application/octet-stream",
            filename=f"{save_name or 'a_tune'}.pt",
            headers={"Cache-Control": "no-cache"},
            background=cleanup,
        )
    if speech is None:
        await _discard(tts_service, voice_name, pack_path)
        return JSONResponse({"voice": save_name})

    speech.voice = voice_name
    try:
        response = await create_speech(
            request=speech,
            client_request=client_request,
            x_raw_response=None,
        )
    except Exception:
        await _discard(tts_service, voice_name, pack_path)
        raise
    if isinstance(response, StreamingResponse):
        response.body_iterator = _discard_after(
            response.body_iterator, tts_service, voice_name, pack_path
        )
    else:
        response.background = cleanup
    return response
