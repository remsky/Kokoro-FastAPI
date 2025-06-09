"""Kokoro V1 model management."""

from typing import Optional

from loguru import logger

from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None  # Explicitly type as KokoroV1
        self._device: Optional[str] = None

    @staticmethod
    def _determine_device() -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> str:
        """Initialize Kokoro V1 backend."""
        import time
        start = time.perf_counter()

        try:
            self._device = self._determine_device()
            logger.info(f"Initializing Kokoro V1 on {self._device}")
            self._backend = KokoroV1()

            model_path = self._config.pytorch_kokoro_v1_file
            await self.load_model(model_path)

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Initialized in {ms}ms")
            
            return self._device
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro V1: {e}")

    def get_backend(self) -> BaseModelBackend:
        """Get initialized backend.

        Returns:
            Initialized backend instance

        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        return self._backend

    async def load_model(self, path: str) -> None:
        """Load model using initialized backend.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If loading fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            await self._backend.load_model(path)
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend.

        Raises:
            RuntimeError: If generation fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            async for chunk in self._backend.generate(*args, **kwargs):
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def unload_all(self) -> None:
        """Unload model and free resources."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        """Get current backend type."""
        return "kokoro_v1"


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
