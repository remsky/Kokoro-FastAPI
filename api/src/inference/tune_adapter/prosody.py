"""Prosody measurements used at enrollment: log-F0 mean/std in semitones, voiced fraction, speaking rate, spectral tilt.

CLI diagnostic:  python -m tune_adapter.prosody <ref.wav> <clone.wav> ...
Self-check:      python -m tune_adapter.prosody
"""

import math
import sys

import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

SR_ENC = 16000  # pitch tracking runs at the speaker-encoder rate
FRAME_S = 0.02  # pitch frame length
HOP = int(SR_ENC * FRAME_S)  # samples per pitch frame (320)
F0_MAX = 400.0  # Hz, upper bound for voiced speech
F0_MIN = 60.0  # Hz, first-pass lower bound
MEDIAN_FRAMES = 30  # median-smoothing window over the tracked lag, 600 ms
ST_REF_HZ = 100.0  # semitones are measured relative to 100 Hz
SR_TILT = 24000  # tilt is measured at Kokoro's rate with a fixed window, so references and renders read alike
TILT_WIN, TILT_HOP = 1024, 256
TILT_BAND = (300.0, 4000.0)  # Hz, the slope is fitted over this band


def st(hz):
    """Hz -> semitones relative to ST_REF_HZ."""
    return 12 * torch.log2(hz / ST_REF_HZ)


def hz(semitones):
    """Semitones relative to ST_REF_HZ -> Hz."""
    return ST_REF_HZ * 2 ** (semitones / 12)


def resample(w, sr, target):
    """[T] float tensor at sr -> [T'] at target, polyphase."""
    if sr == target:
        return w
    g = math.gcd(sr, target)
    return torch.from_numpy(resample_poly(w.numpy(), target // g, sr // g)).float()


def _wav(path, sr):
    """Accept a wav path (sr=None) or a [T] tensor with its sample rate. Returns (mono float tensor, sr)."""
    if sr is None:
        w, sr = sf.read(path)
        if w.ndim > 1:
            w = w.mean(1)
        return torch.tensor(w, dtype=torch.float32), sr
    return path.detach().float().cpu(), sr


def _frame_rms(w, hop):
    """RMS energy per hop-sized frame, dropping the partial last frame."""
    n = len(w) // hop * hop
    return w[:n].reshape(-1, hop).pow(2).mean(1).sqrt()


def _track_f0(w, lo, hi, win=2 * HOP):
    """Normalized cross-correlation pitch tracker at SR_ENC, one estimate per HOP. Returns (f0 Hz, rms) per frame."""
    lags = range(int(SR_ENC / hi), int(SR_ENC / lo) + 1)
    length = win + lags[-1]
    frames = F.pad(w, (0, length)).unfold(0, length, HOP)
    x = frames[:, :win]
    xx = x.pow(2).sum(1)
    corr = []
    for lag in lags:
        y = frames[:, lag : lag + win]
        corr.append((x * y).sum(1) / (xx * y.pow(2).sum(1)).sqrt().clamp_min(1e-8))
    corr = torch.stack(corr, 1)
    # shortest lag among near-ties, else the 2x period wins on float noise
    near_best = corr >= corr.max(1, keepdim=True).values - 1e-3
    best_lag = (torch.tensor(lags[0]) + near_best.float().argmax(1))[
        : len(w) // HOP
    ].float()
    pad = MEDIAN_FRAMES // 2
    best_lag = F.pad(
        best_lag[None, None], (pad, MEDIAN_FRAMES - 1 - pad), mode="replicate"
    )[0, 0]
    best_lag = best_lag.unfold(0, MEDIAN_FRAMES, 1).median(1).values
    return SR_ENC / best_lag, _frame_rms(w, HOP)


def rate(path, sr=None, limit=60):
    """Speaking-rate proxy: smoothed-energy peaks (roughly syllables) per second of speech, over the first `limit` s."""
    w, sr = _wav(path, sr)
    hop = sr // 100  # 10 ms frames
    e = _frame_rms(w[: limit * sr], hop)
    e = F.avg_pool1d(e[None, None], 3, 1, 1)[0, 0]  # 3-frame smoothing
    speech = e > 0.05 * e.max()  # energy gate for "is speech"

    is_peak = (e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]) & speech[1:-1]
    peaks = is_peak.nonzero().flatten()
    if len(peaks) == 0:
        return 0.0
    # Peaks closer than 80 ms (8 frames) belong to the same syllable.
    syllables = 1 + (peaks[1:] - peaks[:-1] >= 8).sum().item()
    speech_seconds = speech.float().mean().item() * len(e) / 100
    return syllables / (speech_seconds + 1e-9)


def tilt(path, sr=None):
    """Spectral tilt in dB per octave: least-squares slope of the long-term power spectrum over TILT_BAND.
    Pink noise reads -3, speech about -8; less negative is brighter, more vocal effort."""
    w, sr = _wav(path, sr)
    w = resample(w, sr, SR_TILT)
    spec = (
        torch.stft(
            w,
            TILT_WIN,
            TILT_HOP,
            window=torch.hann_window(TILT_WIN),
            return_complex=True,
        )
        .abs()
        .pow(2)
        .mean(1)
    )
    f = torch.linspace(0, SR_TILT / 2, len(spec))
    band = (f > TILT_BAND[0]) & (f < TILT_BAND[1])
    x = torch.log2(f[band])
    y = 10 * torch.log10(spec[band] + 1e-12)
    x = x - x.mean()
    return ((x * (y - y.mean())).sum() / (x * x).sum()).item()


def stats(path, sr=None):
    """Log-F0 statistics of the voiced frames. Returns (mean st, std st, voiced fraction); zeros if nothing is voiced."""
    w, sr = _wav(path, sr)
    w = resample(w, sr, SR_ENC)

    # Pass 2 raises the floor to 0.65x the upper-quartile F0, killing octave-down picks on high voices.
    lo = F0_MIN
    for _ in range(2):
        f0, e = _track_f0(w, lo, F0_MAX)
        voiced = (e > 0.1 * e.max()) & (f0 > lo) & (f0 < F0_MAX)
        voiced_st = st(f0[voiced])
        if not len(voiced_st):
            return 0.0, 0.0, 0.0
        lo = min(0.65 * hz(voiced_st.quantile(0.75).item()), 180)

    return voiced_st.mean().item(), voiced_st.std().item(), voiced.float().mean().item()


def _self_check():
    # A 220 Hz harmonic tone sits at 13.7 st and must not be tracked an octave down.
    t = torch.arange(SR_ENC * 2) / SR_ENC
    tone = sum(torch.sin(2 * torch.pi * 220 * k * t) / k for k in range(1, 6))
    mean, std, _ = stats(tone, SR_ENC)
    assert abs(mean - 13.7) < 0.5 and std < 0.5, (mean, std)

    # Noise amplitude-modulated at 5 Hz reads as 5 syllables per second.
    torch.manual_seed(0)
    t = torch.arange(SR_ENC * 4) / SR_ENC
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * torch.cos(2 * torch.pi * 5 * t))
    r = rate(envelope * torch.randn(len(t)), SR_ENC)
    assert abs(r - 5) < 0.7, r
    # White noise is flat, pink noise (1/f power) falls 3 dB per octave.
    white = torch.randn(SR_ENC * 4)
    pink = torch.fft.irfft(
        torch.fft.rfft(white) / torch.arange(1, SR_ENC * 2 + 2).sqrt()
    )
    tw, tp = tilt(white, SR_ENC), tilt(pink, SR_ENC)
    assert abs(tw) < 0.3 and abs(tp + 3) < 0.3, (tw, tp)
    print(
        f"ok mean={mean:.1f} std={std:.2f} rate={r:.2f} tilt white={tw:.2f} pink={tp:.2f}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        sys.exit()
    print(
        f"{'file':40s} {'F0 mean(st)':>11s} {'F0 std':>7s} {'voiced%':>8s} {'peaks/s':>8s} {'tilt':>6s}"
    )
    for p in sys.argv[1:]:
        mean, std, voiced = stats(p)
        print(
            f"{p[-40:]:40s} {mean:11.1f} {std:7.1f} {100 * voiced:8.0f} {rate(p):8.1f} {tilt(p):6.1f}"
        )
