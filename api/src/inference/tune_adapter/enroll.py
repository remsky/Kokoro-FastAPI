"""Reference clip -> Kokoro voice pack.

STRENGTH=1.0 python -m tune_adapter.enroll ref.wav name [voices_dir]  ->  voices_dir/name.pt ([ROWS, 1, STYLE + r + 2]) + name_test.wav
STRENGTH is the distance from the adapter's mean voice: the default 1 is the prediction as is; above 1
exaggerates the voice at a quality cost.
Then: KPipeline(...)(text, voice="voices/name.pt") on a model with install() applied (see residual.py).
"""

import os
import sys

import soundfile as sf
import torch
import torch.nn.functional as F

from .prosody import SR_ENC, rate, resample, stats, tilt
from .residual import ROWS, install, use

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(
    HERE, "..", "model.safetensors"
)  # repo root; pass your own path to load()
ENCODER = "microsoft/unispeech-sat-base-plus-sv"
SR_KOKORO = 24000
REF_MAX_S = 30  # only the first 30 s of the reference are embedded
SPEED_MIN, SPEED_MAX = 0.7, 1.4  # clamp on the enrolled speed
# Three sentences: the rate measurement is text-dependent and a single short sentence reads fast.
CAL_TEXT = (
    "The quick brown fox jumps over the lazy dog, and then it runs away. "
    "Nobody expected the weather to change so quickly this afternoon. "
    "Please remember to bring your umbrella if you go out later."
)
TEST_TEXT = "Hello, this is a quick test of the cloned voice."
_encoder = None


def load(device=None, adapter=ADAPTER):
    from kokoro import KModel, KPipeline

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = KModel().to(device).eval()
    pipe = KPipeline(lang_code="a", model=model, device=device)
    return model, pipe, install(model, adapter)


def read(path):
    """wav path -> (mono float tensor, sr)."""
    w, sr = sf.read(path)
    w = torch.tensor(w, dtype=torch.float32)
    if w.ndim > 1:
        w = w.mean(-1)
    return w, sr


def _speaker_embedding(wav, sr, device):
    global _encoder
    if _encoder is None:
        from transformers import UniSpeechSatForXVector

        _encoder = UniSpeechSatForXVector.from_pretrained(ENCODER).to(device).eval()
    w = resample(wav.float().cpu(), sr, SR_ENC)[: SR_ENC * REF_MAX_S]
    if len(w) < SR_ENC:
        raise ValueError("reference clip must be at least one second long")
    return F.normalize(_encoder(w.to(device)[None]).embeddings, dim=-1)  # [1, ENC]


@torch.no_grad()
def enroll(wav, sr, pipe, heads, strength=1.0):
    """wav: [T] float mono at sr. Returns the [ROWS, 1, STYLE + r + 2] pack on CPU (every row identical)."""
    device = heads["style"][0].weight.device
    use(
        heads.slot
    )  # the calibration render below has to go through the adapter that built this pack
    e = _speaker_embedding(wav, sr, device)

    # strength scales the distance from the mean voice; 1 is the prediction as is.
    style = heads.mu["style"] + strength * (heads["style"](e) - heads.mu["style"])
    r = heads.mu["r"] + strength * (heads["r"](e) - heads.mu["r"])
    # Spectral tilt of the same slice the embedding saw, z-scored as in training, along the adapter's learned direction.
    mean, sd = heads.tilt_norm
    ref_tilt = tilt(
        resample(wav.float().cpu(), sr, SR_ENC)[: SR_ENC * REF_MAX_S], SR_ENC
    )
    style = style + heads["tilt"](
        torch.tensor([[(ref_tilt - mean) / sd]], device=device)
    )

    f0_mean, _, _ = stats(wav, sr)
    row = torch.cat([style, r, torch.tensor([[f0_mean, 1.0]], device=device)], -1)
    pack = (
        row[None].repeat(ROWS, 1, 1).cpu()
    )  # KPipeline wants a CPU tensor; it moves it to the model

    # One calibration render at speed 1 measures Kokoro's own pace for this voice.
    cal = next(pipe(CAL_TEXT, voice=pack)).audio
    speed = rate(wav, sr) / rate(
        cal, SR_KOKORO
    )  # reference slower than the render -> speed < 1
    pack[..., -1] = min(max(speed, SPEED_MIN), SPEED_MAX)
    return pack


if __name__ == "__main__":
    ref, name = sys.argv[1:3]
    out = sys.argv[3] if len(sys.argv) > 3 else "voices"
    strength = float(os.environ.get("STRENGTH", 1.0))

    model, pipe, heads = load()
    pack = enroll(*read(ref), pipe, heads, strength)

    os.makedirs(out, exist_ok=True)
    pack_path = os.path.join(out, f"{name}.pt")
    torch.save(pack, pack_path)
    wav = next(pipe(TEST_TEXT, voice=pack_path)).audio
    sf.write(os.path.join(out, f"{name}_test.wav"), wav.cpu().numpy(), SR_KOKORO)
    print(
        f"{pack_path}  speed {pack[0, 0, -1]:.2f}  f0 mean {pack[0, 0, -2]:.1f} st  tilt {tilt(ref):.1f} dB/oct"
    )
