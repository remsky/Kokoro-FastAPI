"""Kokoro V1 model management."""

import asyncio
import time
from typing import Optional

import torch
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None  # Explicitly type as KokoroV1
        self._device: Optional[str] = None
        self._lock = asyncio.Lock()
        self._active_requests = 0
        self._last_used_at: Optional[float] = None
        self._idle_unload_task: Optional[asyncio.Task] = None

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize Kokoro V1 backend."""
        try:
            self._device = self._determine_device()
            logger.info(f"Initializing Kokoro V1 on {self._device}")
            self._backend = KokoroV1()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro V1: {e}")

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize and warm up model.

        Args:
            voice_manager: Voice manager instance for warmup

        Returns:
            Tuple of (device, backend type, voice count)

        Raises:
            RuntimeError: If initialization fails
        """
        import time

        start = time.perf_counter()

        try:
            # Initialize backend
            await self.initialize()

            # Load model
            model_path = self._config.pytorch_kokoro_v1_file
            await self.load_model(model_path)

            # Use paths module to get voice path
            try:
                voices = await paths.list_voices()
                voice_path = await paths.get_voice_path(settings.default_voice)

                # Warm up with short text
                warmup_text = "Warmup text for initialization."
                # Use default voice name for warmup
                voice_name = settings.default_voice
                logger.debug(f"Using default voice '{voice_name}' for warmup")
                async for _ in self.generate(warmup_text, (voice_name, voice_path)):
                    pass
            except Exception as e:
                raise RuntimeError(f"Failed to get default voice: {e}")

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Warmup completed in {ms}ms")

            return self._device, "kokoro_v1", len(voices)
        except FileNotFoundError as e:
            logger.error("""
Model files not found! You need to download the Kokoro V1 model:

1. Download model using the script:
   python docker/scripts/download_model.py --output api/src/models/v1_0

2. Or set environment variable in docker-compose:
   DOWNLOAD_MODEL=true
""")
            exit(0)
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    async def ensure_backend(self) -> None:
        """Reload the backend if it was unloaded."""
        if self._backend:
            return
        async with self._lock:
            if not self._backend:
                self._cancel_idle_unload_timer()
                await self.initialize()
                await self.load_model(self._config.pytorch_kokoro_v1_file)

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
            self._last_used_at = time.monotonic()
            self._schedule_idle_unload_timer_locked()
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _auto_unload_timeout(self) -> float:
        return max(0.0, float(settings.model_auto_unload_timeout_seconds))

    def _format_seconds(self, seconds: float) -> str:
        return f"{seconds:g}s"

    def _auto_unload_enabled(self) -> bool:
        return settings.model_auto_unload_enabled and self._auto_unload_timeout() > 0

    def _cancel_idle_unload_timer(self) -> None:
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if (
            self._idle_unload_task
            and not self._idle_unload_task.done()
            and self._idle_unload_task is not current_task
        ):
            self._idle_unload_task.cancel()
        self._idle_unload_task = None

    def _unload_backend_locked(self) -> bool:
        if self._backend is None:
            return False
        self._backend.unload()
        self._backend = None
        self._last_used_at = time.monotonic()
        return True

    def _schedule_idle_unload_timer_locked(self) -> None:
        self._cancel_idle_unload_timer()
        if (
            not self._auto_unload_enabled()
            or self._backend is None
            or self._active_requests > 0
        ):
            return

        timeout = self._auto_unload_timeout()
        self._idle_unload_task = asyncio.create_task(self._idle_unload_after(timeout))

    async def _idle_unload_after(self, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
            async with self._lock:
                if (
                    not self._auto_unload_enabled()
                    or self._backend is None
                    or self._active_requests > 0
                    or self._last_used_at is None
                ):
                    return

                idle_for = time.monotonic() - self._last_used_at
                if idle_for < self._auto_unload_timeout():
                    self._schedule_idle_unload_timer_locked()
                    return

                unloaded = self._unload_backend_locked()

            if unloaded:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(
                    "Model auto-unloaded after idle timeout of "
                    f"{self._format_seconds(self._auto_unload_timeout())}"
                )
        except asyncio.CancelledError:
            pass

    async def _begin_request(self) -> None:
        async with self._lock:
            self._active_requests += 1
            self._cancel_idle_unload_timer()

    async def _end_request(self) -> None:
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._last_used_at = time.monotonic()
            self._schedule_idle_unload_timer_locked()

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend.

        Raises:
            RuntimeError: If generation fails
        """
        await self._begin_request()
        try:
            await self.ensure_backend()
            assert self._backend is not None, "ensure_backend left no backend"
            async for chunk in self._backend.generate(*args, **kwargs):
                if settings.default_volume_multiplier != 1.0:
                    chunk.audio *= settings.default_volume_multiplier
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
        finally:
            await self._end_request()

    def unload_all(self) -> None:
        """Unload model and free resources."""
        self._cancel_idle_unload_timer()
        if self._backend:
            self._backend.unload()
            self._backend = None

    async def unload(self) -> None:
        """Release model from GPU memory. Reloads automatically on next request."""
        async with self._lock:
            self._cancel_idle_unload_timer()
            self._unload_backend_locked()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded from GPU memory")

    async def reload(self) -> None:
        """Reload the model immediately."""
        async with self._lock:
            self._cancel_idle_unload_timer()
            self._unload_backend_locked()
            await self.initialize()
            await self.load_model(self._config.pytorch_kokoro_v1_file)
            self._schedule_idle_unload_timer_locked()
        logger.info("Model reloaded")

    def status(self) -> dict:
        """Return model lifecycle state for API responses."""
        timeout = self._auto_unload_timeout()
        idle_for = None
        unload_in = None
        if self._last_used_at is not None:
            idle_for = max(0.0, time.monotonic() - self._last_used_at)
            if self._auto_unload_enabled() and self._backend is not None:
                unload_in = max(0.0, timeout - idle_for)
        return {
            "backend": self.current_backend,
            "device": self._device,
            "loaded": self._backend is not None,
            "active_requests": self._active_requests,
            "auto_unload_enabled": settings.model_auto_unload_enabled,
            "auto_unload_timeout_seconds": timeout,
            "idle_seconds": idle_for,
            "seconds_until_auto_unload": unload_in,
        }

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
