"""Model configuration for Kokoro V1.

This module provides model-specific configuration settings that complement the application-level
settings in config.py. While config.py handles general application settings (API, paths, etc.),
this module focuses on memory management and model file paths.
"""

from pydantic import BaseModel, Field
from core.config import settings


class KokoroV1Config(BaseModel):
    """Kokoro V1 configuration."""

    languages: list[str] = ["en"]

    class Config:
        frozen = True


class PyTorchConfig(BaseModel):
    """PyTorch backend configuration."""

    memory_threshold: float = Field(0.8, description="Memory threshold for cleanup")
    retry_on_oom: bool = Field(True, description="Whether to retry on OOM errors")

    class Config:
        frozen = True


class ModelConfig(BaseModel):
    """Kokoro V1 model configuration."""

    # General settings
    cache_voices: bool = Field(True, description="Whether to cache voice tensors")
    voice_cache_size: int = Field(2, description="Maximum number of cached voices")

    # Model filename
    pytorch_kokoro_v1_file: str = Field(
        settings.kokoro_v1_file, description="PyTorch Kokoro V1 model filename"
    )

    # Backend config
    pytorch_gpu: PyTorchConfig = Field(default_factory=PyTorchConfig)

    class Config:
        frozen = True


# Global instance
model_config = ModelConfig()
