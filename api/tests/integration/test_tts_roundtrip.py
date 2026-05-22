"""End-to-end smoke test: synthesize each language and transcribe back.

For each (voice, language) case we:
  1. Call the OpenAI-compatible /v1/audio/speech endpoint against a live server
  2. Require the response to be non-empty (catches Kokoro's silent-fail mode)
  3. Run faster-whisper over the audio and require the score under a
     deliberately generous threshold

This is a smoke test, not a quality bar. WER/CER thresholds are loose on
purpose. The signal we want is "did the pipeline break for language X."
Tighter thresholds belong in a separate manual evaluation harness.
"""

from __future__ import annotations

import io
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass

import pytest


# Windows stdout defaults to cp1252; printing Devanagari/CJK reference strings
# or transcripts raises UnicodeEncodeError and surfaces as a fake test failure.
# Reconfigure once at module import so any -s output is safe.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from jiwer import cer, wer
from jiwer.transforms import (
    Compose,
    ReduceToListOfListOfWords,
    RemoveMultipleSpaces,
    RemovePunctuation,
    Strip,
    ToLowerCase,
)


pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Case:
    voice: str
    lang: str
    text: str


# One short sentence per language. Keep these stable. Whisper handles them
# well in CPU/int8 with `small`, so a regression below the threshold means
# something actually broke in Kokoro, not Whisper drift.
CASES: list[Case] = [
    Case("af_heart",    "en", "The quick brown fox jumps over the lazy dog."),
    Case("bf_emma",     "en", "The rain in Spain falls mainly on the plain."),
    Case("ef_dora",     "es", "El sol brilla en el cielo azul."),
    Case("ff_siwis",    "fr", "Le soleil brille dans le ciel bleu."),
    Case("if_sara",     "it", "Il gatto dorme sul tappeto rosso."),
    Case("pf_dora",     "pt", "O gato dorme no tapete vermelho."),
    Case("hf_alpha",    "hi", "आज मौसम बहुत अच्छा है।"),
    Case("jf_alpha",    "ja", "今日はとても良い天気です。"),
    Case("zf_xiaobei",  "zh", "今天天气非常好。"),
]

# Hindi uses combining diacritics; CJK has no word boundaries. CER is the
# fair metric for any of these. Everything else gets WER.
CER_LANGS = {"hi", "ja", "zh"}

WER_THRESHOLD = 0.25
CER_THRESHOLD = 0.25
MIN_AUDIO_BYTES = 1000


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


def _wav_seconds(audio_bytes: bytes) -> float:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate else 0.0


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda c: f"{c.voice}-{c.lang}",
)
def test_tts_roundtrip(case: Case, openai_client, whisper_model, tmp_path):
    threshold = CER_THRESHOLD if case.lang in CER_LANGS else WER_THRESHOLD
    metric = "CER" if case.lang in CER_LANGS else "WER"

    t0 = time.perf_counter()
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice=case.voice,
        input=case.text,
        response_format="wav",
    )
    audio = response.content
    synth_s = time.perf_counter() - t0

    assert audio, f"empty response body for {case.voice}/{case.lang}"
    assert len(audio) >= MIN_AUDIO_BYTES, (
        f"suspiciously small WAV for {case.voice}/{case.lang}: "
        f"{len(audio)} bytes (server likely silent-failed)"
    )

    audio_path = tmp_path / f"{case.voice}_{case.lang}.wav"
    audio_path.write_bytes(audio)
    duration_s = _wav_seconds(audio)
    assert duration_s > 0.2, (
        f"WAV header parsed but duration is only {duration_s:.3f}s "
        f"for {case.voice}/{case.lang}"
    )

    t0 = time.perf_counter()
    segments, _info = whisper_model.transcribe(
        str(audio_path), beam_size=1, language=case.lang
    )
    transcript = " ".join(seg.text for seg in segments).strip()
    transcribe_s = time.perf_counter() - t0

    score = _score(case.lang, case.text, transcript)
    print(
        f"[{case.voice}/{case.lang}] "
        f"synth={synth_s:.2f}s transcribe={transcribe_s:.2f}s "
        f"{metric}={score:.3f} (<{threshold}) "
        f"heard={transcript!r}"
    )
    assert score < threshold, (
        f"{metric} {score:.3f} exceeded threshold {threshold} "
        f"for {case.voice}/{case.lang}. "
        f"Expected: {case.text!r}. Heard: {transcript!r}."
    )
