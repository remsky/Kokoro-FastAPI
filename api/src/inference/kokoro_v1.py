"""Clean Kokoro implementation with controlled resource management."""

import os
from typing import AsyncGenerator, Dict, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KModel, KPipeline
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import model_config
from ..structures.schemas import WordTimestamp
from .base import AudioChunk, BaseModelBackend

_ESPEAK_TS_SCALE = (
    2.0 / 80.0
)  # pred_dur unit -> seconds (matches KPipeline.join_timestamps)

# English g2p is version-independent, so the helper pipeline always uses the v1_0 repo
_EN_G2P_REPO_ID = "hexgrad/Kokoro-82M"


def _espeak_word_timestamps(graphemes, phonemes, pred_dur, g2p=None):
    """Derive word timestamps for espeak-based (non-English) pipelines.

    KPipeline attaches timed tokens only for English (misaki G2P) voices, but
    the model predicts a duration for every phoneme in every language (the
    audio is rendered from exactly those durations). pred_dur[0] is BOS and
    pred_dur[1 + i] covers phonemes[i], so summing per-character durations and
    splitting at the phoneme string's spaces yields exact word times.

    Words espeak expands into several spoken words (numbers, some
    abbreviations) are reconciled by re-phonemizing per word via `g2p`; if the
    counts still disagree, returns None so the caller emits no timestamps and
    clients can fall back to their own handling.
    """
    try:
        if pred_dur is None or not phonemes or len(pred_dur) < len(phonemes) + 1:
            return None
        words = graphemes.split()
        if not words:
            return None
        durations = [dur * _ESPEAK_TS_SCALE for dur in pred_dur.tolist()]
        groups = []  # [start, end) seconds per space-separated phoneme group
        now = durations[0]
        start = None
        for i, ch in enumerate(phonemes):
            if ch.isspace():
                if start is not None:
                    groups.append((start, now))
                    start = None
            elif start is None:
                start = now
            now += durations[1 + i]
        if start is not None:
            groups.append((start, now))

        if len(groups) != len(words):
            if g2p is None:
                return None
            counts = []
            for word in words:
                phones = g2p(word)
                if isinstance(phones, tuple):
                    phones = phones[0]
                counts.append(max(len((phones or "").split()), 1))
            if sum(counts) != len(groups):
                return None
            merged, group_idx = [], 0
            for count in counts:
                merged.append((groups[group_idx][0], groups[group_idx + count - 1][1]))
                group_idx += count
            groups = merged

        return [
            WordTimestamp(word=word, start_time=round(start, 3), end_time=round(end, 3))
            for word, (start, end) in zip(words, groups)
        ]
    except Exception as e:
        logger.warning(f"espeak timestamp mapping failed: {e}")
        return None


class KokoroV1(BaseModelBackend):
    """Kokoro backend with controlled resource management."""

    def __init__(self):
        """Initialize backend with environment-based configuration."""
        super().__init__()
        # Strictly respect settings.use_gpu
        self._device = settings.get_device()
        self._model: Optional[KModel] = None
        self._pipelines: Dict[str, KPipeline] = {}  # Store pipelines by lang_code
        self._en_g2p: Optional[KPipeline] = (
            None  # model-free English g2p for zh mixed text
        )
        self._voice_cache: Dict[str, torch.Tensor] = {}  # Cache voice tensors by path

    async def _get_voice_tensor(self, voice_path: str) -> torch.Tensor:
        """Load voice tensor with in-memory caching to avoid repeated file I/O.

        Args:
            voice_path: Path to the .pt voice file

        Returns:
            Voice tensor on the target device
        """
        cache_key = f"{voice_path}:{self._device}"
        if cache_key not in self._voice_cache:
            self._voice_cache[cache_key] = await paths.load_voice_tensor(
                voice_path, device=self._device
            )
            logger.debug(f"Cached voice tensor from {voice_path}")
        return self._voice_cache[cache_key]

    async def load_model(self, path: str) -> None:
        """Load pre-baked model.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

            if not os.path.exists(config_path):
                raise RuntimeError(f"Config file not found: {config_path}")

            logger.info(f"Loading Kokoro model on {self._device}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Model path: {model_path}")

            # Load model and let KModel handle device mapping
            self._model = KModel(
                repo_id=settings.model_repo_id, config=config_path, model=model_path
            ).eval()
            # For MPS, manually move ISTFT layers to CPU while keeping rest on MPS
            if self._device == "mps":
                logger.info(
                    "Moving model to MPS device with CPU fallback for unsupported operations"
                )
                self._model = self._model.to(torch.device("mps"))
            elif self._device == "cuda":
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()

        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        """Get or create pipeline for language code.

        Args:
            lang_code: Language code to use

        Returns:
            KPipeline instance for the language
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        if lang_code not in self._pipelines:
            logger.info(f"Creating new pipeline for language code: {lang_code}")
            kwargs = {}
            if lang_code == "z":
                # v1.1-zh g2p drops embedded English spans unless it can phonemize them
                kwargs["en_callable"] = self._en_phonemes
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id=settings.model_repo_id,
                model=self._model,
                device=self._device,
                **kwargs,
            )
        return self._pipelines[lang_code]

    def _en_phonemes(self, text: str) -> str:
        """Phonemize English spans embedded in non-English text (zh en_callable)."""
        if self._en_g2p is None:
            self._en_g2p = KPipeline(
                lang_code="a", model=False, repo_id=_EN_G2P_REPO_ID
            )
        # long spans split across results, so join rather than take the first
        return "".join(r.phonemes for r in self._en_g2p(text) if r.phonemes)

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio from phoneme tokens.

        Args:
            tokens: Input phoneme tokens to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Memory management for GPU
            if self._device == "cuda":
                if self._check_memory():
                    self._clear_memory()

            # Handle voice input
            voice_path: str
            voice_name: str
            if isinstance(voice, tuple):
                voice_name, voice_data = voice
                if isinstance(voice_data, str):
                    voice_path = voice_data
                else:
                    # Save tensor to temporary file
                    import tempfile

                    temp_dir = tempfile.gettempdir()
                    voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                    # Save tensor with CPU mapping for portability
                    torch.save(voice_data.cpu(), voice_path)
            else:
                voice_path = voice
                voice_name = os.path.splitext(os.path.basename(voice_path))[0]

            # Load voice tensor with caching to avoid repeated file I/O
            voice_tensor = await self._get_voice_tensor(voice_path)
            # Save to temp file only if needed (pipeline requires a file path)
            import tempfile

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(
                temp_dir, f"temp_voice_{os.path.basename(voice_path)}"
            )
            if not os.path.exists(temp_path):
                await paths.save_voice_tensor(voice_tensor, temp_path)
            voice_path = temp_path

            # Use provided lang_code, settings voice code override, or first letter of voice name
            if lang_code:  # api is given priority
                pipeline_lang_code = lang_code
            elif settings.default_voice_code:  # settings is next priority
                pipeline_lang_code = settings.default_voice_code
            else:  # voice name is default/fallback
                pipeline_lang_code = voice_name[0].lower()

            pipeline = self._get_pipeline(pipeline_lang_code)

            logger.debug(
                f"Generating audio from tokens with lang_code '{pipeline_lang_code}': '{tokens[:100]}{'...' if len(tokens) > 100 else ''}'"
            )
            for result in pipeline.generate_from_tokens(
                tokens=tokens, voice=voice_path, speed=speed, model=self._model
            ):
                if result.audio is not None:
                    logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                    yield result.audio.numpy()
                else:
                    logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate_from_tokens(
                    tokens, voice, speed, lang_code
                ):
                    yield chunk
            raise

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio using model.

        Args:
            text: Input text to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        try:
            # Memory management for GPU
            if self._device == "cuda":
                if self._check_memory():
                    self._clear_memory()

            # Handle voice input
            voice_path: str
            voice_name: str
            if isinstance(voice, tuple):
                voice_name, voice_data = voice
                if isinstance(voice_data, str):
                    voice_path = voice_data
                else:
                    # Save tensor to temporary file
                    import tempfile

                    temp_dir = tempfile.gettempdir()
                    voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                    # Save tensor with CPU mapping for portability
                    torch.save(voice_data.cpu(), voice_path)
            else:
                voice_path = voice
                voice_name = os.path.splitext(os.path.basename(voice_path))[0]

            # Load voice tensor with caching to avoid repeated file I/O
            voice_tensor = await self._get_voice_tensor(voice_path)
            # Save to temp file only if needed (pipeline requires a file path)
            import tempfile

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(
                temp_dir, f"temp_voice_{os.path.basename(voice_path)}"
            )
            if not os.path.exists(temp_path):
                await paths.save_voice_tensor(voice_tensor, temp_path)
            voice_path = temp_path

            # Use provided lang_code, settings voice code override, or first letter of voice name
            pipeline_lang_code = (
                lang_code
                if lang_code
                else (
                    settings.default_voice_code
                    if settings.default_voice_code
                    else voice_name[0].lower()
                )
            )
            pipeline = self._get_pipeline(pipeline_lang_code)

            logger.debug(
                f"Generating audio for text with lang_code '{pipeline_lang_code}': '{text[:100]}{'...' if len(text) > 100 else ''}'"
            )
            for result in pipeline(
                text, voice=voice_path, speed=speed, model=self._model
            ):
                if result.audio is not None:
                    logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                    word_timestamps = None
                    if (
                        return_timestamps
                        and hasattr(result, "tokens")
                        and result.tokens
                    ):
                        word_timestamps = []
                        current_offset = 0.0
                        logger.debug(
                            f"Processing chunk timestamps with {len(result.tokens)} tokens"
                        )
                        if result.pred_dur is not None:
                            try:
                                # Add timestamps with offset
                                for token in result.tokens:
                                    if not all(
                                        hasattr(token, attr)
                                        for attr in [
                                            "text",
                                            "start_ts",
                                            "end_ts",
                                        ]
                                    ):
                                        continue
                                    if not token.text or not token.text.strip():
                                        continue

                                    start_time = float(token.start_ts) + current_offset
                                    end_time = float(token.end_ts) + current_offset
                                    word_timestamps.append(
                                        WordTimestamp(
                                            word=str(token.text).strip(),
                                            start_time=start_time,
                                            end_time=end_time,
                                        )
                                    )
                                    logger.debug(
                                        f"Added timestamp for word '{token.text}': {start_time:.3f}s - {end_time:.3f}s"
                                    )

                            except Exception as e:
                                logger.error(
                                    f"Failed to process timestamps for chunk: {e}"
                                )
                    elif (
                        return_timestamps
                        and result.phonemes
                        and type(getattr(pipeline, "g2p", None)).__name__ == "EspeakG2P"
                    ):
                        # espeak pipelines (es/fr/it/hi/pt) yield no timed
                        # tokens; derive word times from the model's own
                        # phoneme durations instead. Espeak-only: other
                        # non-English G2Ps (ja/zh) have no space-separated
                        # word groups to map.
                        pred_dur = getattr(result, "pred_dur", None)
                        if (
                            pred_dur is None
                            and getattr(result, "output", None) is not None
                        ):
                            pred_dur = getattr(result.output, "pred_dur", None)
                        word_timestamps = _espeak_word_timestamps(
                            result.graphemes,
                            result.phonemes,
                            pred_dur,
                            g2p=getattr(pipeline, "g2p", None),
                        )

                    yield AudioChunk(
                        result.audio.numpy(), word_timestamps=word_timestamps
                    )
                else:
                    logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate(text, voice, speed, lang_code):
                    yield chunk
            raise

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        if self._device == "cuda":
            memory_gb = torch.cuda.memory_allocated() / 1e9
            return memory_gb > model_config.pytorch_gpu.memory_threshold
        # MPS doesn't provide memory management APIs
        return False

    def _clear_memory(self) -> None:
        """Clear device memory."""
        if self._device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self._device == "mps":
            # Empty cache if available (future-proofing)
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        for pipeline in self._pipelines.values():
            del pipeline
        self._pipelines.clear()
        self._en_g2p = None
        self._voice_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device
