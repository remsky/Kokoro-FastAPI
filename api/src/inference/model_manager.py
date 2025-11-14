"""Multi-backend TTS model management."""

from typing import Dict, Optional

from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1


class ModelManager:
    """Manages multiple TTS backends (Kokoro, ZipVoice, etc.)."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backends: Dict[str, BaseModelBackend] = {}  # backend_type -> instance
        self._device: Optional[str] = None
        self._default_backend: str = settings.default_backend

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize enabled TTS backends."""
        try:
            self._device = self._determine_device()
            logger.info(f"Initializing TTS backends on {self._device}")

            # Initialize Kokoro if enabled
            if settings.enable_kokoro:
                logger.info("Initializing Kokoro backend...")
                self._backends['kokoro'] = KokoroV1()
                logger.info("Kokoro backend initialized")

            # Initialize ZipVoice if enabled
            if settings.enable_zipvoice:
                try:
                    from .zipvoice import ZipVoiceBackend
                    logger.info("Initializing ZipVoice backend...")
                    self._backends['zipvoice'] = ZipVoiceBackend()
                    logger.info("ZipVoice backend initialized")
                except ImportError as e:
                    logger.warning(f"ZipVoice backend not available: {e}")
                    logger.info("Install with: pip install zipvoice (or set enable_zipvoice=False)")

            if not self._backends:
                raise RuntimeError("No TTS backends enabled or available")

            logger.info(f"Initialized {len(self._backends)} backend(s): {list(self._backends.keys())}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize backends: {e}")

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
            # Initialize backends
            await self.initialize()

            voice_count = 0

            # Warmup Kokoro if enabled
            if 'kokoro' in self._backends:
                # Load model
                model_path = self._config.pytorch_kokoro_v1_file
                await self.load_model(model_path, backend_type='kokoro')

                # Use paths module to get voice path
                try:
                    voices = await paths.list_voices()
                    voice_count = len(voices)
                    voice_path = await paths.get_voice_path(settings.default_voice)

                    # Warm up with short text
                    warmup_text = "Warmup text for initialization."
                    # Use default voice name for warmup
                    voice_name = settings.default_voice
                    logger.debug(f"Using default voice '{voice_name}' for Kokoro warmup")
                    async for _ in self.generate(warmup_text, (voice_name, voice_path), backend_type='kokoro'):
                        pass
                    logger.info("Kokoro warmup completed")
                except Exception as e:
                    raise RuntimeError(f"Failed to warm up Kokoro: {e}")

            # Warmup ZipVoice if enabled
            if 'zipvoice' in self._backends:
                try:
                    # Load ZipVoice backend (models auto-download)
                    await self.load_model('', backend_type='zipvoice')
                    logger.info("ZipVoice backend ready")
                except Exception as e:
                    logger.warning(f"ZipVoice warmup skipped: {e}")

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"All backends warmed up in {ms}ms")

            return self._device, self._default_backend, voice_count
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

    def get_backend(self, backend_type: str = None) -> BaseModelBackend:
        """Get initialized backend by type.

        Args:
            backend_type: Backend identifier ('kokoro', 'zipvoice', etc.)
                         If None, returns default backend

        Returns:
            Initialized backend instance

        Raises:
            RuntimeError: If backend not initialized
            KeyError: If backend type not found
        """
        backend_type = backend_type or self._default_backend

        if backend_type not in self._backends:
            available = list(self._backends.keys())
            raise KeyError(f"Backend '{backend_type}' not available. "
                          f"Available backends: {available}")

        return self._backends[backend_type]

    async def load_model(self, path: str, backend_type: str = None) -> None:
        """Load model using specified backend.

        Args:
            path: Path to model file
            backend_type: Backend to load model on (default: default_backend)

        Raises:
            RuntimeError: If loading fails
        """
        backend = self.get_backend(backend_type)

        try:
            await backend.load_model(path)
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model on {backend_type}: {e}")

    async def generate(self, *args, backend_type: str = None, **kwargs):
        """Generate audio using specified backend.

        Args:
            backend_type: Backend to use (default: default_backend)
            *args, **kwargs: Passed to backend's generate method

        Raises:
            RuntimeError: If generation fails
        """
        backend = self.get_backend(backend_type)

        try:
            async for chunk in backend.generate(*args, **kwargs):
                # Apply volume multiplier if set
                volume_multiplier = kwargs.get('volume_multiplier', settings.default_volume_multiplier)
                if volume_multiplier != 1.0:
                    chunk.audio *= volume_multiplier
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Generation failed on {backend_type or self._default_backend}: {e}")

    def unload_all(self) -> None:
        """Unload all models and free resources."""
        for backend_type, backend in self._backends.items():
            try:
                backend.unload()
                logger.info(f"Unloaded {backend_type} backend")
            except Exception as e:
                logger.error(f"Failed to unload {backend_type}: {e}")
        self._backends.clear()

    @property
    def current_backend(self) -> str:
        """Get default backend type."""
        return self._default_backend

    @property
    def available_backends(self) -> list:
        """Get list of available backend types."""
        return list(self._backends.keys())

    def set_default_backend(self, backend_type: str) -> None:
        """Set the default backend.

        Args:
            backend_type: Backend to set as default

        Raises:
            KeyError: If backend not available
        """
        if backend_type not in self._backends:
            raise KeyError(f"Backend '{backend_type}' not available")
        self._default_backend = backend_type
        logger.info(f"Default backend set to: {backend_type}")


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
