"""Multilingual TTS roundtrip validation.

Same idea as test_transcription.py, but with multilingual Whisper and
per-language reference phrases. CER for ja/zh (no word boundaries), WER for
everything else. Intended as a manual sanity check across Kokoro's non-English
voices — especially Japanese, which fails quietly often enough to want a
dedicated harness.

Usage (from repo root):
    uv sync --project examples --extra transcription
    uv run --project examples python examples/assorted_checks/test_transcription/test_transcription_multilingual.py

Env overrides:
    KOKORO_BASE_URL   default http://localhost:8880/v1
    KOKORO_DEVICE     default gpu (label only; picks report_{device}.json + meta)
    WHISPER_MODEL     default small (multilingual; ~470MB int8)
    WHISPER_DEVICE    default cpu
    WHISPER_COMPUTE   default int8 on cpu, float16 on cuda

Note on silent failures: when Kokoro emits silence/garbage, Whisper sometimes
hallucinates a canned phrase under a forced language hint (e.g. Japanese →
"ありがとうございました"). The score will still be poor; just don't be surprised
when the transcript reads as something the audio doesn't say.
"""

from __future__ import annotations

import json
import os
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Windows stdout defaults to cp1252 — re-encode so Devanagari / CJK
# reference strings and transcripts print without crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import openai
from faster_whisper import WhisperModel
from jiwer import cer, wer
from jiwer.transforms import (
    Compose,
    ReduceToListOfListOfWords,
    RemoveMultipleSpaces,
    RemovePunctuation,
    Strip,
    ToLowerCase,
)


BASE_URL = os.environ.get("KOKORO_BASE_URL", "http://localhost:8880/v1")
KOKORO_DEVICE = os.environ.get("KOKORO_DEVICE", "gpu").lower()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get(
    "WHISPER_COMPUTE", "int8" if WHISPER_DEVICE == "cpu" else "float16"
)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output_multilingual"


# (voice, whisper ISO lang code, expected text, pass threshold)
# CJK uses CER, the rest WER. Thresholds are first-pass and deliberately
# loose — tighten after we see what `small` actually delivers.
CASES: list[tuple[str, str, str, float]] = [
    ("af_heart",    "en", "The quick brown fox jumps over the lazy dog.", 0.20),
    ("bf_emma",     "en", "The rain in Spain falls mainly on the plain.", 0.25),
    ("ef_dora",     "es", "El sol brilla en el cielo azul.",              0.30),
    ("ff_siwis",    "fr", "Le soleil brille dans le ciel bleu.",           0.20),
    ("if_sara",     "it", "Il gatto dorme sul tappeto rosso.",             0.30),
    ("pf_dora",     "pt", "O gato dorme no tapete vermelho.",              0.30),
    ("hf_alpha",    "hi", "आज मौसम बहुत अच्छा है।",                       0.20),
    ("jf_alpha",    "ja", "今日はとても良い天気です。",                      0.30),
    ("zf_xiaobei",  "zh", "今天天气非常好。",                                0.25),
]

# Hindi uses combining diacritics that make single-vowel mistakes count as
# whole-word substitutions under WER; CER is the right metric for any script
# where character-level diffs dominate, not just CJK.
CER_LANGS = {"hi", "ja", "zh"}


_WORD_NORMALIZE = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
    ReduceToListOfListOfWords(),
])


def _strip_for_cer(s: str) -> str:
    return "".join(
        c for c in s
        if not c.isspace() and not unicodedata.category(c).startswith("P")
    )


def _score(lang: str, reference: str, hypothesis: str) -> float:
    if lang in CER_LANGS:
        return cer(_strip_for_cer(reference), _strip_for_cer(hypothesis))
    return wer(
        reference,
        hypothesis,
        reference_transform=_WORD_NORMALIZE,
        hypothesis_transform=_WORD_NORMALIZE,
    )


@dataclass
class Result:
    voice: str
    lang: str
    metric: str
    expected: str
    transcript: str
    score: float
    threshold: float
    passed: bool
    detected_lang: str
    detected_prob: float
    audio_path: str
    audio_bytes: int
    synth_seconds: float
    transcribe_seconds: float
    error: str = ""


def rel(path: Path) -> str:
    return path.resolve().relative_to(SCRIPT_DIR.resolve()).as_posix()


def run_meta() -> dict:
    return {
        "kokoro_device": KOKORO_DEVICE,
        "kokoro_base_url": BASE_URL,
        "whisper_model": WHISPER_MODEL,
        "whisper_device": WHISPER_DEVICE,
        "whisper_compute": WHISPER_COMPUTE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


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


def transcribe(
    model: WhisperModel, audio_path: Path, language: str
) -> tuple[str, str, float, float]:
    start = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path), beam_size=1, language=language
    )
    text = " ".join(seg.text for seg in segments).strip()
    return text, info.language, info.language_probability, time.perf_counter() - start


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Server:    {BASE_URL}")
    print(f"Kokoro:    {KOKORO_DEVICE} (label)")
    print(f"Whisper:   {WHISPER_MODEL} ({WHISPER_DEVICE}, {WHISPER_COMPUTE})")
    print()

    client = openai.OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=120)

    print(f"Loading Whisper model {WHISPER_MODEL!r} (first run downloads weights)...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print("Loaded.\n")

    results: list[Result] = []
    for voice, lang, expected, threshold in CASES:
        audio_path = OUTPUT_DIR / f"{voice}_{lang}.wav"
        metric = "CER" if lang in CER_LANGS else "WER"
        print(f"[{voice} / {lang}] {expected!r}")

        def _fail(error: str, synth_s: float = 0.0, trans_s: float = 0.0) -> None:
            audio_bytes = audio_path.stat().st_size if audio_path.exists() else 0
            print(f"  FAIL ({error})  audio={audio_bytes}B\n")
            results.append(
                Result(
                    voice=voice, lang=lang, metric=metric, expected=expected,
                    transcript="", score=1.0, threshold=threshold, passed=False,
                    detected_lang="", detected_prob=0.0,
                    audio_path=rel(audio_path), audio_bytes=audio_bytes,
                    synth_seconds=synth_s, transcribe_seconds=trans_s,
                    error=error,
                )
            )

        try:
            synth_s = synthesize(client, voice, expected, audio_path)
        except Exception as exc:
            _fail(f"synth_error: {exc}")
            continue

        audio_bytes = audio_path.stat().st_size
        # Catches Kokoro's silent-fail mode (notably Japanese): the HTTP call
        # returns 200 with an empty/near-empty body, so we never see an
        # exception — only a zero-byte WAV that downstream tooling chokes on.
        if audio_bytes < 100:
            _fail(f"empty_audio (returned {audio_bytes}B WAV)", synth_s=synth_s)
            continue

        try:
            transcript, det_lang, det_prob, trans_s = transcribe(model, audio_path, lang)
        except Exception as exc:
            _fail(f"transcribe_error: {exc}", synth_s=synth_s)
            continue

        score = _score(lang, expected, transcript)
        passed = score < threshold

        print(f"  heard:    {transcript!r}")
        print(f"  detected: lang={det_lang} (p={det_prob:.2f})")
        print(f"  {metric}:      {score:.3f}  threshold {threshold}  ({'PASS' if passed else 'FAIL'})")
        print(f"  timing:   synth {synth_s:.2f}s, transcribe {trans_s:.2f}s\n")

        results.append(
            Result(
                voice=voice, lang=lang, metric=metric, expected=expected,
                transcript=transcript, score=score, threshold=threshold, passed=passed,
                detected_lang=det_lang, detected_prob=det_prob,
                audio_path=rel(audio_path), audio_bytes=audio_bytes,
                synth_seconds=synth_s, transcribe_seconds=trans_s,
            )
        )

    report_path = OUTPUT_DIR / f"report_{KOKORO_DEVICE}.json"
    report_path.write_text(
        json.dumps(
            {"meta": run_meta(), "results": [asdict(r) for r in results]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    print(f"Summary: {passed_count}/{total} passed. Report: {rel(report_path)}")

    return 0 if results and passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
