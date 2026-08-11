from xml.etree.ElementTree import ParseError

from fastapi import APIRouter, Depends, HTTPException

from ..core.config import settings
from ..services.text_processing.ssml import (
    BREAK_STRENGTH_S,
    PROSODY_RATE,
    SSML_ELEMENTS,
    translate_ssml,
)
from ..structures.schemas import RATE_MAX, RATE_MIN
from ..structures.text_schemas import SsmlCapabilities, SsmlRequest, SsmlResponse

async def _require_ssml_enabled() -> None:
    """403 when the server-level SSML kill switch is off"""
    if not settings.enable_ssml:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "permission_denied",
                "message": "SSML translation is disabled on this server",
                "type": "permission_error",
            },
        )


router = APIRouter(tags=["ssml"], dependencies=[Depends(_require_ssml_enabled)])


def _rejected(message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": "validation_error",
            "message": message,
            "type": "invalid_request_error",
        },
    )


@router.get("/dev/ssml", response_model=SsmlCapabilities)
async def ssml_capabilities() -> SsmlCapabilities:
    """The SSML subset this build translates, and what the rest of it does instead."""
    return SsmlCapabilities(
        elements={k: v for k, v in SSML_ELEMENTS.items() if v},
        ignored=[k for k, v in SSML_ELEMENTS.items() if not v],
        break_strengths=BREAK_STRENGTH_S,
        prosody_rates=PROSODY_RATE,
        rate_range=[RATE_MIN, RATE_MAX],
    )


@router.post("/dev/ssml", response_model=SsmlResponse)
async def translate_ssml_text(request: SsmlRequest) -> SsmlResponse:
    """Translate SSML into the native inline control tokens.

    The result is plain text for the speech endpoints; pass it with
    allow_voice_tags=true when a voice was given so [voice:]/[rate:] spans
    apply and get validated there. Non-SSML input passes through unchanged.
    """
    try:
        translated = translate_ssml(
            request.text,
            default_voice=request.voice,
            allow_voice_tags=bool(request.voice),
        )
    except ParseError as e:
        raise _rejected(f"Malformed SSML: {e}")
    # backstop for an ssml_max_depth set high enough to reach the interpreter's own limit
    except RecursionError:
        raise _rejected("SSML is nested too deeply to translate")
    return SsmlResponse(text=translated)
