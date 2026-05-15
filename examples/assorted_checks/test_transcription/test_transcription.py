"""TTS roundtrip validation.

Synthesizes short phrases via a running Kokoro server, transcribes the audio
locally with faster-whisper, and reports word error rate against the expected
text. Intended as a manual sanity check, not a pytest suite.

Usage (from repo root):
    uv sync --project examples --extra transcription
    uv run --project examples python examples/assorted_checks/test_transcription/test_transcription.py

Env overrides:
    KOKORO_BASE_URL  default http://localhost:8880/v1
    WHISPER_MODEL    default base.en
    WER_THRESHOLD    default 0.2
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import openai
from faster_whisper import WhisperModel
from jiwer import wer
from jiwer.transforms import (
    Compose,
    RemoveMultipleSpaces,
    RemovePunctuation,
    ReduceToListOfListOfWords,
    Strip,
    SubstituteWords,
    ToLowerCase,
)


# Whisper often writes small numbers as digits ("5") even when the prompt was
# spelled out ("five"). Normalize both directions before computing WER so the
# metric reflects pronunciation accuracy, not formatting.
_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve",
}

_NORMALIZE = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    SubstituteWords(_DIGIT_WORDS),
    RemoveMultipleSpaces(),
    Strip(),
    ReduceToListOfListOfWords(),
])


def _normalized_wer(reference: str, hypothesis: str) -> float:
    return wer(
        reference,
        hypothesis,
        reference_transform=_NORMALIZE,
        hypothesis_transform=_NORMALIZE,
    )


BASE_URL = os.environ.get("KOKORO_BASE_URL", "http://localhost:8880/v1")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WER_THRESHOLD = float(os.environ.get("WER_THRESHOLD", "0.2"))

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"


def rel(path: Path) -> str:
    """Path relative to the script dir, posix-style."""
    return path.resolve().relative_to(SCRIPT_DIR.resolve()).as_posix()

CASES: list[tuple[str, str]] = [
    ("af_heart", "The quick brown fox jumps over the lazy dog."),
    ("af_bella", "She sells seashells by the seashore."),
    ("am_adam", "Pack my box with five dozen liquor jugs."),
    ("af_nicole", "The five boxing wizards jump quickly across the field."),
]


@dataclass
class Result:
    voice: str
    expected: str
    transcript: str
    wer: float
    passed: bool
    audio_path: str
    synth_seconds: float
    transcribe_seconds: float


def synthesize(client: openai.OpenAI, voice: str, text: str, out_path: Path) -> float:
    start = time.perf_counter()
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="wav",
    )
    out_path.write_bytes(response.content)
    return time.perf_counter() - start


def transcribe(model: WhisperModel, audio_path: Path) -> tuple[str, float]:
    start = time.perf_counter()
    segments, _info = model.transcribe(str(audio_path), beam_size=1)
    text = " ".join(seg.text for seg in segments).strip()
    return text, time.perf_counter() - start


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Server:    {BASE_URL}")
    print(f"Whisper:   {WHISPER_MODEL} (CPU, int8)")
    print(f"Threshold: WER < {WER_THRESHOLD}")
    print()

    client = openai.OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=60)

    print("Loading Whisper model (first run downloads ~150MB)...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Loaded.\n")

    results: list[Result] = []
    for voice, expected in CASES:
        audio_path = OUTPUT_DIR / f"{voice}.wav"
        print(f"[{voice}] {expected!r}")
        try:
            synth_s = synthesize(client, voice, expected, audio_path)
        except Exception as exc:
            print(f"  synthesis failed: {exc}\n")
            continue

        transcript, trans_s = transcribe(model, audio_path)
        score = _normalized_wer(expected, transcript)
        passed = score < WER_THRESHOLD

        print(f"  heard:  {transcript!r}")
        print(f"  WER:    {score:.3f}  ({'PASS' if passed else 'FAIL'})")
        print(f"  timing: synth {synth_s:.2f}s, transcribe {trans_s:.2f}s\n")

        results.append(
            Result(
                voice=voice,
                expected=expected,
                transcript=transcript,
                wer=score,
                passed=passed,
                audio_path=rel(audio_path),
                synth_seconds=synth_s,
                transcribe_seconds=trans_s,
            )
        )

    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    print(f"Summary: {passed_count}/{total} passed. Report: {rel(report_path)}")

    return 0 if results and passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
