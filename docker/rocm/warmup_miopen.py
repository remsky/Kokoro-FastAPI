"""MIOpen kernel warmup for Kokoro on ROCm.

Kokoro's tensor shape varies with phoneme count, and MIOpen compiles a
unique kernel per shape. Without a populated find DB each new length pays
a 5-60s search. This sweeps lengths 1..340 to warm the cache; ~2 hours on
Strix Halo, runs once per ROCm/PyTorch upgrade, survives reboots.

Opt-in and unverified outside @realugbun's Strix Halo iGPU (recipe from
#454); reports from other hardware welcome.

Run inside the container with FIND_MODE overridden so MIOpen performs the
search (image default is FIND_MODE=2, which reuses the on-disk cache):

    docker exec -it <container> bash -lc '\
        MIOPEN_FIND_MODE=3 MIOPEN_FIND_ENFORCE=3 \
        python /app/docker/rocm/warmup_miopen.py'
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from kokoro import KModel

APP_ROOT = Path(os.environ.get("KOKORO_APP_ROOT", "/app"))
MODEL_DIR = APP_ROOT / "api/src/models/v1_0"
VOICE_PATH = APP_ROOT / "api/src/voices/v1_0/af_heart.pt"
MAX_PHONEMES = int(os.environ.get("WARMUP_MAX_PHONEMES", "340"))


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA/HIP device not available; aborting.", file=sys.stderr)
        return 1

    find_mode = os.environ.get("MIOPEN_FIND_MODE", "<unset>")
    if find_mode == "2":
        print(
            "WARNING: MIOPEN_FIND_MODE=2 is set. The warmup needs MODE=3 to "
            "actually run the kernel search. Re-run with "
            "MIOPEN_FIND_MODE=3 MIOPEN_FIND_ENFORCE=3.",
            file=sys.stderr,
        )

    print(f"[{time.strftime('%H:%M:%S')}] Loading model on GPU...", flush=True)
    model = KModel(
        config=str(MODEL_DIR / "config.json"),
        model=str(MODEL_DIR / "kokoro-v1_0.pth"),
    ).eval().cuda()
    voice = torch.load(VOICE_PATH, weights_only=True)

    print(f"[{time.strftime('%H:%M:%S')}] Warming lengths 1..{MAX_PHONEMES}...", flush=True)
    total = 0.0
    for n in range(1, MAX_PHONEMES + 1):
        ps = ("a " * ((n + 1) // 2))[:n]
        ref_s = voice[min(len(ps), 509)]
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            model(ps, ref_s, speed=1)
        except Exception as e:
            print(f"ERROR at n={n}: {e}", file=sys.stderr)
            continue
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        total += elapsed
        if n % 10 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] {n:3d}/{MAX_PHONEMES} | "
                f"this={elapsed:5.1f}s | total={total / 60:5.1f}m",
                flush=True,
            )

    print(f"\nWarmup complete in {total / 60:.1f} minutes")
    print("Restart the container (or unset the FIND_MODE override) so the "
          "default MIOPEN_FIND_MODE=2 picks up the populated cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
