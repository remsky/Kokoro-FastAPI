"""Inno clone tuner: reference clip in, stock-shaped Kokoro voice pack out.

Wraps the inno-kokoro package. docker/scripts/download_model.py bakes the pinned
weights next to the Kokoro model; load() runs at startup when ENABLE_INNO_TUNER is
set and any failure leaves available() False, so /dev/tune answers 503.
"""

import io
import os
import tempfile
import threading
from typing import Optional

import soundfile as sf
import torch
from loguru import logger

from ..core import paths
from ..core.config import settings

MAX_UPLOAD_BYTES = 10 << 20
MAX_REF_SECONDS = 30
MAX_SAMPLE_RATE = 96000
MAX_CHANNELS = 2

_tuner = None
_lock = threading.Lock()


def weights_path() -> str:
    return os.path.join(paths.models_dir(), "v1_0", "inno_tuner", "model.safetensors")


def load() -> None:
    global _tuner
    from inno_kokoro.enroll import Tuner

    _tuner = Tuner(weights_path(), device=settings.get_device())
    logger.info(f"Inno voice tuner v{_tuner.version} loaded on {_tuner.device}")


def available() -> bool:
    return _tuner is not None


def tune(data: bytes, head: bool = True, fmax: Optional[float] = None) -> str:
    """Decode a clip, enroll it, write the pack to the temp dir, return its path.

    Raises ValueError on a clip under 3 s, over 2 channels, or over 96 kHz, and
    soundfile.LibsndfileError on bytes that do not decode. Only the first 30 s are
    decoded. Blocking, call it off the event loop.
    """
    if not available():
        raise RuntimeError("inno voice tuner not available")
    from inno_kokoro.enroll import enroll

    with sf.SoundFile(io.BytesIO(data)) as clip:
        sr = clip.samplerate
        if clip.channels > MAX_CHANNELS or sr > MAX_SAMPLE_RATE:
            raise ValueError(
                f"reference must be mono or stereo at {MAX_SAMPLE_RATE // 1000} kHz or less"
            )
        wav = torch.from_numpy(clip.read(MAX_REF_SECONDS * sr, dtype="float32"))
    if wav.ndim > 1:
        wav = wav.mean(-1)
    with _lock:
        pack, _ = enroll(wav, sr, _tuner, fmax=fmax, head=head)
    fd, path = tempfile.mkstemp(prefix="a_tune_", suffix=".pt")
    with os.fdopen(fd, "wb") as f:
        torch.save(pack, f)
    return path
