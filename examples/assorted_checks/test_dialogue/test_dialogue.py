"""Multi-speaker checks against a running Kokoro server.

Covers the inline [voice:...] token on /v1/audio/speech and the structured
/dev/dialogue endpoint: speaker switching, per-speaker language pipelines,
pauses between turns, and rejection of unknown voices.

Writes every rendered clip plus report.json to output/ so the audio can be
listened to afterwards.

Run from the repo root:
    uv run --project examples python examples/assorted_checks/test_dialogue/test_dialogue.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests

BASE_URL = os.getenv("KOKORO_BASE_URL", "http://localhost:8880").rstrip("/")
TIMEOUT = float(os.getenv("KOKORO_TIMEOUT", "600"))
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44

OUTPUT_DIR = Path(__file__).parent / "output"


def post(path: str, body: dict) -> requests.Response:
    return requests.post(f"{BASE_URL}{path}", json=body, timeout=TIMEOUT)


def speech(text: str, voice: str = "af_heart", **extra) -> requests.Response:
    body = {
        "model": "kokoro",
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "stream": False,
    }
    body.update(extra)
    return post("/v1/audio/speech", body)


def dialogue(turns: list[tuple[str, str]], **extra) -> requests.Response:
    body = {
        "model": "kokoro",
        "turns": [{"voice": voice, "text": text} for voice, text in turns],
        "response_format": "wav",
        "stream": False,
    }
    body.update(extra)
    return post("/dev/dialogue", body)


def audio_seconds(payload: bytes) -> float:
    """Duration from byte length, since the wav header carries a placeholder frame count."""
    return max(len(payload) - WAV_HEADER_BYTES, 0) / (SAMPLE_RATE * BYTES_PER_SAMPLE)


def samples(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload[WAV_HEADER_BYTES:], dtype=np.int16).astype(np.float64)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference over the overlapping span, scaled to full scale."""
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return float(np.abs(a[:n] - b[:n]).mean() / 32768)


def solo(response: requests.Response) -> bytes:
    """Raise on a failed reference render, so a bad body never reaches samples()."""
    response.raise_for_status()
    return response.content


def save(name: str, payload: bytes) -> str:
    path = OUTPUT_DIR / f"{name}.wav"
    path.write_bytes(payload)
    return path.name


def check_two_speakers() -> tuple[bool, str]:
    """Two turns, two voices, one request."""
    response = dialogue(
        [
            ("af_bella", "So the reconciliation job caught it before anyone noticed?"),
            ("am_michael", "It did. Fifteen minutes after the drift started."),
        ]
    )
    response.raise_for_status()
    save("two_speakers", response.content)
    seconds = audio_seconds(response.content)
    return seconds > 1.0, f"{seconds:.2f}s of audio"


def check_inline_tags() -> tuple[bool, str]:
    """The same thing through the plain OpenAI endpoint using inline tokens."""
    response = speech(
        "Here is the narrator. [voice:af_bella] And here is Bella. "
        "[voice:am_michael] And Michael closes it out.",
        voice="af_heart",
    )
    response.raise_for_status()
    save("inline_tags", response.content)
    seconds = audio_seconds(response.content)
    return seconds > 1.0, f"{seconds:.2f}s of audio"


def check_combined_voice_tag() -> tuple[bool, str]:
    """The weighted combine syntax survives inside a tag."""
    response = speech(
        "Plain narrator. [voice:af_bella(2)+af_sky] Now a blended speaker.",
        voice="af_heart",
    )
    response.raise_for_status()
    save("combined_voice_tag", response.content)
    seconds = audio_seconds(response.content)
    return seconds > 1.0, f"{seconds:.2f}s of audio"


def check_multilingual() -> tuple[bool, str]:
    """No lang_code sent, so each speaker's prefix picks its own pipeline."""
    response = dialogue(
        [
            ("af_bella", "Good morning, shall we begin?"),
            ("jf_alpha", "おはようございます、はじめましょう。"),
            ("bm_george", "Quite right, let us get on with it."),
            ("zf_xiaobei", "好的，我们开始吧。"),
        ]
    )
    response.raise_for_status()
    save("multilingual", response.content)
    seconds = audio_seconds(response.content)
    return seconds > 2.0, f"a/j/b/z speakers, {seconds:.2f}s of audio"


def check_pause_between_turns() -> tuple[bool, str]:
    """pause_between_turns should lengthen the render by roughly the requested gap."""
    turns = [
        ("af_bella", "First line of the exchange."),
        ("am_michael", "Second line of the exchange."),
        ("af_bella", "And a third to close."),
    ]
    tight = dialogue(turns, pause_between_turns=0.0)
    tight.raise_for_status()
    spaced = dialogue(turns, pause_between_turns=1.0)
    spaced.raise_for_status()
    save("pause_none", tight.content)
    save("pause_1s", spaced.content)

    added = audio_seconds(spaced.content) - audio_seconds(tight.content)
    # two gaps between three turns, allow slack for chunk boundary rounding
    return 1.0 < added < 3.0, f"{added:.2f}s added by two one second pauses"


def check_voice_actually_switches() -> tuple[bool, str]:
    """Compare a dialogue's tail against each speaker rendering the same line solo."""
    line_a = "Hello there friend, good to see you."
    line_b = "Goodbye for now, talk again soon."

    bella_solo = samples(solo(dialogue([("af_bella", line_b)])))
    george_solo = samples(solo(dialogue([("bm_george", line_b)])))
    mixed = dialogue(
        [("af_bella", line_a), ("bm_george", line_b)], pause_between_turns=0.0
    )
    mixed.raise_for_status()
    save("switch_probe", mixed.content)

    tail = samples(mixed.content)[-len(george_solo) :]
    to_george = distance(tail, george_solo)
    to_bella = distance(tail, bella_solo)
    ratio = to_bella / max(to_george, 1e-9)
    return (
        to_george < to_bella,
        f"tail is {ratio:.1f}x closer to bm_george than af_bella",
    )


def check_unknown_voice_rejected() -> tuple[bool, str]:
    """A bad speaker should fail up front, not part way through the stream."""
    tagged = speech("Narrator. [voice:not_a_real_voice] Nope.", voice="af_heart")
    structured = dialogue([("af_bella", "Fine."), ("not_a_real_voice", "Nope.")])
    ok = tagged.status_code == 400 and structured.status_code == 400
    return ok, f"inline {tagged.status_code}, structured {structured.status_code}"


CHECKS: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("two_speakers", check_two_speakers),
    ("inline_tags", check_inline_tags),
    ("combined_voice_tag", check_combined_voice_tag),
    ("multilingual", check_multilingual),
    ("pause_between_turns", check_pause_between_turns),
    ("voice_actually_switches", check_voice_actually_switches),
    ("unknown_voice_rejected", check_unknown_voice_rejected),
]


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"server: {BASE_URL}")
    print(f"output: {OUTPUT_DIR}\n")

    results = []
    for name, check in CHECKS:
        start = time.perf_counter()
        error: Optional[str] = None
        try:
            ok, detail = check()
        except Exception as exc:
            ok, detail, error = False, str(exc), type(exc).__name__
        elapsed = time.perf_counter() - start
        results.append(
            {
                "name": name,
                "passed": ok,
                "detail": detail,
                "error": error,
                "seconds": round(elapsed, 2),
            }
        )
        print(f"{'PASS' if ok else 'FAIL'}  {name:<24} {elapsed:6.2f}s  {detail}")

    passed = sum(1 for r in results if r["passed"])
    report = {
        "base_url": BASE_URL,
        "passed": passed,
        "total": len(results),
        "checks": results,
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(
        f"\n{passed}/{len(results)} checks passed, report written to {OUTPUT_DIR / 'report.json'}"
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
