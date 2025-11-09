"""Web player router with async file serving."""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from loguru import logger

from ..core.config import settings
from ..core.paths import get_content_type, get_web_file_path, read_bytes

router = APIRouter(
    tags=["Web Player"],
    responses={404: {"description": "Not found"}},
)


@router.get("/config")
async def get_web_config():
    """Get web player configuration including UVICORN_ROOT_PATH."""
    if not settings.enable_web_player:
        raise HTTPException(status_code=404, detail="Web player is disabled")
    
    root_path = os.environ.get("UVICORN_ROOT_PATH", "")
    
    return {
        "root_path": root_path,
        "version": settings.api_version,
    }


@router.get("/{filename:path}")
async def serve_web_file(filename: str):
    """Serve web player static files asynchronously."""
    if not settings.enable_web_player:
        raise HTTPException(status_code=404, detail="Web player is disabled")

    try:
        # Default to index.html for root path
        if filename == "" or filename == "/":
            filename = "index.html"

        # Get file path
        file_path = await get_web_file_path(filename)

        # Read file content
        content = await read_bytes(file_path)

        # Get content type
        content_type = await get_content_type(file_path)

        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Cache-Control": "no-cache",  # Prevent caching during development
            },
        )

    except RuntimeError as e:
        logger.warning(f"Web file not found: {filename}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error serving web file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
