"""
FastAPI OpenAI Compatible API
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings, unrecognized_env_file_keys
from .routers.debug import router as debug_router
from .routers.development import router as dev_router
from .routers.openai_compatible import router as openai_router
from .routers.ssml import router as ssml_router
from .routers.tune import router as tune_router
from .routers.web_player import router as web_router


def setup_logger():
    """Configure loguru logger with custom formatting"""
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    level = os.getenv("API_LOG_LEVEL", "DEBUG").upper()
    if level not in valid_levels:
        level = "DEBUG"
    print(f"Global API loguru logger level: {level}")
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
                "{level: <8} | "
                "<fg #4169E1>{module}:{line}</fg #4169E1> | "
                "{message}",
                "colorize": True,
                "level": level,
            },
        ],
    }
    logger.remove()
    logger.configure(**config)
    logger.level("ERROR", color="<red>")


# Configure logger
setup_logger()

# unknown keys are ignored rather than fatal, so name them or a typo'd setting fails silently
for _key in unrecognized_env_file_keys():
    logger.warning(f"ignoring unrecognized env file key: {_key}")


class _WebPlayerStartupHandler(logging.Handler):
    """Log the web player URL from uvicorn's bound address."""

    def emit(self, record: logging.LogRecord) -> None:
        if not str(record.msg).startswith("Uvicorn running on %s://"):
            return
        scheme, host, port = record.args
        logger.info(f"Beta Web Player: {scheme}://{host}:{port}/web/")
        logger.info(f"              or {scheme}://localhost:{port}/web/")


if settings.enable_web_player:
    logging.getLogger("uvicorn.error").addHandler(_WebPlayerStartupHandler())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization"""
    from .inference.model_manager import get_manager
    from .inference.voice_manager import get_manager as get_voice_manager
    from .services.temp_manager import cleanup_temp_files

    # Clean old temp files on startup
    await cleanup_temp_files()

    logger.info("Loading TTS model and voice packs...")

    try:
        # Initialize managers
        model_manager = await get_manager()
        voice_manager = await get_voice_manager()

        # Initialize model with warmup and get status
        device, model, voicepack_count = await model_manager.initialize_with_warmup(
            voice_manager
        )

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    if settings.enable_inno_tuner:
        from .inference import inno_tuner

        try:
            await asyncio.to_thread(inno_tuner.load)
        except Exception as e:
            logger.error(f"Inno voice tuner not loaded, /dev/tune will answer 503: {e}")

    boundary = "░" * 2 * 12
    startup_msg = f"""

{boundary}

    ╔═╗┌─┐┌─┐┌┬┐
    ╠╣ ├─┤└─┐ │ 
    ╚  ┴ ┴└─┘ ┴
    ╦╔═┌─┐┬┌─┌─┐
    ╠╩╗│ │├┴┐│ │
    ╩ ╩└─┘┴ ┴└─┘

{boundary}
                """
    startup_msg += f"\nModel warmed up on {device}: {model}"
    if device == "mps":
        startup_msg += "\nUsing Apple Metal Performance Shaders (MPS)"
    elif device == "cuda":
        startup_msg += f"\nCUDA: {torch.cuda.is_available()}"
    else:
        startup_msg += "\nRunning on CPU"
    startup_msg += f"\n{voicepack_count} voice packs loaded"
    if settings.enable_inno_tuner:
        startup_msg += "\nInno voice tuner: " + (
            "ready at /dev/tune" if inno_tuner.available() else "not available"
        )

    if settings.enable_web_player:
        startup_msg += "\n\nWeb Player: enabled (URL logged below)"
    else:
        startup_msg += "\n\nWeb Player: disabled"

    startup_msg += f"\n{boundary}\n"
    logger.info(startup_msg)

    yield


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    openapi_url="/openapi.json",  # Explicitly enable OpenAPI schema
)


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """413 a request whose body passes MAX_REQUEST_BODY_MB, counted as it arrives."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = settings.max_request_body_mb * 1024 * 1024
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            return await self._reject(send, limit)

        received = 0
        over = False
        started = False

        async def counted_receive():
            nonlocal received, over
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit and not started:
                    over = True
                    raise _BodyTooLarge()
            return message

        # FastAPI turns a failed body read into its own 400, so swap in the 413
        async def guarded_send(message):
            nonlocal started
            if over:
                if message["type"] == "http.response.start" and not started:
                    started = True
                    await self._reject(send, limit)
                return
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send, limit)

    @staticmethod
    async def _reject(send, limit):
        body = json.dumps(
            {
                "detail": {
                    "error": "too_large",
                    "message": f"request body exceeds {limit >> 20} MB",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


app.add_middleware(BodySizeLimitMiddleware)

# Add CORS middleware if enabled
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(openai_router, prefix="/v1")
app.include_router(dev_router)  # Development endpoints
app.include_router(ssml_router)  # SSML translation and capabilities
app.include_router(tune_router)  # /dev/tune, 403 unless enabled
app.include_router(debug_router)  # Debug endpoints (403 unless enabled)
if settings.enable_web_player:
    app.include_router(web_router, prefix="/web")  # Web player static files


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
