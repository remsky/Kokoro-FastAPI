"""FastAPI glue for the tune adapter.

tune_adapter/ is a byte-identical copy of the kokoro-inno-clone-adapter package; keep it that way
and put anything FastAPI-specific here.
"""

import hashlib
import io
from typing import Tuple

import soundfile as sf
import torch
from safetensors import safe_open

from .tune_adapter import (  # noqa: F401  re-exported for kokoro_v1
    enroll as _enroll,
    install,
)
from .tune_adapter.enroll import REF_MAX_S, enroll  # noqa: F401


def decode_audio(data: bytes) -> Tuple[torch.Tensor, int]:
    """Audio file bytes (wav, flac, ogg, mp3) -> mono float32 [T], sample rate. Reads at most REF_MAX_S seconds."""
    with sf.SoundFile(io.BytesIO(data)) as f:
        wav = torch.from_numpy(f.read(f.samplerate * REF_MAX_S, dtype="float32"))
        return (wav.mean(-1) if wav.ndim > 1 else wav), f.samplerate


def adapter_ids(path: str) -> Tuple[str, str]:
    """(alias, id) of a weights file: its metadata name, and that name plus `_<sha256[:8]>` of the bytes.

    The alias is what a request names (`<alias>/<voice>`); the id names the voice dir, since packs only
    work with the weights that enrolled them and a retrain has to get a fresh one.
    """
    with safe_open(path, "pt") as f:
        alias = (f.metadata() or {}).get("name", "adapter")
    with open(path, "rb") as f:
        return alias, f"{alias}_{hashlib.sha256(f.read()).hexdigest()[:8]}"


def release_encoder() -> None:
    """Drop the cached speaker encoder, so /dev/unload frees it too."""
    _enroll._encoder = None
