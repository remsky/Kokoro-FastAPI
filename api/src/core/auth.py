"""API_KEYS check for the API routers."""

import secrets

from fastapi import HTTPException
from starlette.requests import HTTPConnection

from .config import settings


async def require_api_key(conn: HTTPConnection) -> None:
    """401 unless API_KEYS is empty or the request carries one of them.

    Bearer token (OpenAI clients), xi-api-key (ElevenLabs), X-API-Key (Cartesia).
    """
    if not settings.api_keys:
        return
    scheme, _, bearer = conn.headers.get("authorization", "").partition(" ")
    offered = [
        bearer if scheme.lower() == "bearer" else "",
        conn.headers.get("xi-api-key", ""),
        conn.headers.get("x-api-key", ""),
    ]
    if not any(
        secrets.compare_digest(token.encode(), key.encode())
        for token in offered
        for key in settings.api_keys
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "A valid API key is required",
                "type": "authentication_error",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
