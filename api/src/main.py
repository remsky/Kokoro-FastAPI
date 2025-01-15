"""
FastAPI OpenAI Compatible API
"""
import sys
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import uvicorn
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .core.config import settings
from .services.tts_model import TTSModel
from .routers.development import router as dev_router
from .services.tts_service import TTSService
from .routers.openai_compatible import router as openai_router
from ..services.text_processing.phonemizer import app as phonemizer_app

def setup_logger():
    """Configure loguru logger with custom formatting"""
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
                "{level: <8} | "
                "{message}",
                "colorize": True,
                "level": "INFO",
            },
        ],
    }
    logger.remove()
    logger.configure(**config)
    logger.level("ERROR", color="<red>")

# Configure logger
setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization"""
    # Initialize startup time tracking
    app.state.startup_time = datetime.now()
    app.state.warmup_period = timedelta(seconds=60)  # Adjust based on your needs
    # Initialize espeak_error as False by default
    app.state.espeak_error = False

    logger.info("Loading TTS model and voice packs...")
    # Initialize the main model with warm-up
    voicepack_count = await TTSModel.setup()
    
    boundary = "░" * 24
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
    startup_msg += f"\nModel warmed up on {TTSModel.get_device()}"
    startup_msg += f"\n{voicepack_count} voice packs loaded\n"
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

# Share app instance with phonemizer
phonemizer_app.app = app

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(openai_router, prefix="/v1")
app.include_router(dev_router)  # New development endpoints

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check that accounts for startup warmup and espeak errors"""
    # Check if we're still in warmup period
    if not hasattr(app.state, "startup_time"):
        # If startup_time doesn't exist yet, we're still initializing
        return {"status": "healthy", "message": "initializing"}
    
    if datetime.now() - app.state.startup_time < app.state.warmup_period:
        return {"status": "healthy", "message": "warming up"}
    
    # After warmup, check espeak error flag
    if hasattr(app.state, "espeak_error") and app.state.espeak_error:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "reason": "espeak error detected during operation"
            }
        )
    
    return {"status": "healthy"}

@app.get("/v1/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
