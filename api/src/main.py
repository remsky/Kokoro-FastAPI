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

async def check_espeak():
    """Check if espeak is working by running a test command"""
    try:
        # Try to run espeak --version
        result = subprocess.run(['espeak', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"eSpeak check failed: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization"""
    # Initialize startup time tracking
    app.state.startup_time = datetime.now()
    app.state.warmup_period = timedelta(seconds=60)  # Adjust based on your needs

    logger.info("Loading TTS model and voice packs...")
    # Initialize the main model with warm-up
    voicepack_count = await TTSModel.setup()
    
    # Verify espeak is working
    espeak_working = await check_espeak()
    if not espeak_working:
        logger.error("eSpeak not working during initialization")
        
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
    startup_msg += f"\neSpeak Status: {'Working' if espeak_working else 'Not Working'}\n"
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
    """Health check that actively verifies espeak functionality"""
    # Check if we're still in warmup period
    if not hasattr(app.state, "startup_time"):
        # If startup_time doesn't exist yet, we're still initializing
        return {"status": "healthy", "message": "initializing"}
    
    if datetime.now() - app.state.startup_time < app.state.warmup_period:
        return {"status": "healthy", "message": "warming up"}
    
    # Actively check if espeak is working
    espeak_working = await check_espeak()
    if not espeak_working:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "reason": "espeak not working"
            }
        )
    
    return {"status": "healthy"}

@app.get("/v1/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
