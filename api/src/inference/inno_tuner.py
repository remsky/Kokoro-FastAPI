"""Inno clone tuner: reference clip in, stock-shaped Kokoro voice pack out.

Wraps the inno-kokoro package. docker/scripts/download_model.py bakes the pinned
weights next to the Kokoro model; load() runs at startup when ENABLE_INNO_TUNER is
set and any failure leaves available() False, so /dev/tune answers 503.
"""

import os
import subprocess
import sys
import tempfile
import threading
from typing import Optional

import numpy as np
import torch
from loguru import logger

from ..core import paths
from ..core.config import settings
from .decode_clip import REFUSED

MAX_UPLOAD_BYTES = 10 << 20
DECODER = os.path.join(os.path.dirname(__file__), "decode_clip.py")
DECODE_TIMEOUT = 5

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


def reserve() -> bool:
    """Claim the tuner for one tune() call, False if held. tune() releases it."""
    return _lock.acquire(blocking=False)


def decode(data: bytes) -> tuple[int, torch.Tensor]:
    """Run decode_clip.py in a child process. ValueError with the child's reason when
    it refuses the clip; a crash or timeout is logged and reported as undecodable."""
    try:
        proc = subprocess.run(
            [sys.executable, DECODER],
            input=data,
            capture_output=True,
            timeout=DECODE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"Reference decode timed out after {DECODE_TIMEOUT}s")
        raise ValueError("reference audio could not be decoded")
    reason = proc.stderr.decode(errors="replace").strip()
    if proc.returncode == REFUSED:
        raise ValueError(reason)
    if proc.returncode != 0:
        logger.warning(f"Reference decode exited {proc.returncode}: {reason}")
        raise ValueError("reference audio could not be decoded")
    sr = int.from_bytes(proc.stdout[:4], "little")
    return sr, torch.from_numpy(np.frombuffer(proc.stdout[4:], dtype=np.float32).copy())


def tune(data: bytes, head: bool = True, fmax: Optional[float] = None) -> str:
    """Decode, enroll, write the pack to the temp dir, return its path. Blocking,
    call off the event loop after reserve(); releases the claim on return.
    ValueError on a refused clip or one under 3 s."""
    try:
        if not available():
            raise RuntimeError("inno voice tuner not available")
        from inno_kokoro.enroll import enroll

        sr, wav = decode(data)
        pack, _ = enroll(wav, sr, _tuner, fmax=fmax, head=head)
        fd, path = tempfile.mkstemp(prefix="a_tune_", suffix=".pt")
        with os.fdopen(fd, "wb") as f:
            torch.save(pack, f)
        return path
    finally:
        _lock.release()
