"""Decodes a reference clip in a child process so a decoder crash cannot take the
server down. Upload on stdin; sample rate as 4 little-endian bytes then float32 mono
samples on stdout. Exits REFUSED with the reason on stderr for a refused or
undecodable clip.
"""

import io
import sys

import numpy as np
import soundfile as sf

MAX_REF_SECONDS = 30
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 96000
MAX_CHANNELS = 2
REFUSED = 64


def decode(data: bytes) -> tuple[int, np.ndarray]:
    """First 30 s, mono, finite, clamped to [-1, 1]. ValueError on a clip over 2
    channels or outside 8 to 96 kHz."""
    with sf.SoundFile(io.BytesIO(data)) as clip:
        sr = clip.samplerate
        if clip.channels > MAX_CHANNELS or not MIN_SAMPLE_RATE <= sr <= MAX_SAMPLE_RATE:
            raise ValueError(
                f"reference must be mono or stereo at {MIN_SAMPLE_RATE // 1000} to "
                f"{MAX_SAMPLE_RATE // 1000} kHz"
            )
        wav = clip.read(MAX_REF_SECONDS * sr, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(-1)
    return sr, np.nan_to_num(wav).clip(-1, 1)


if __name__ == "__main__":
    try:
        sr, wav = decode(sys.stdin.buffer.read())
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(REFUSED)
    except sf.LibsndfileError:
        print("reference audio could not be decoded", file=sys.stderr)
        sys.exit(REFUSED)
    sys.stdout.buffer.write(sr.to_bytes(4, "little") + wav.tobytes())
