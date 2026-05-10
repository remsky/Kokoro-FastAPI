"""Long-form roundtrip baseline.

Synthesizes a long input file with one voice via a running Kokoro server,
transcribes the result with faster-whisper, and reports timing + WER. Goal:
establish a baseline for "can the server produce 30-60 min of clean audio
in one go, and how long does it take" — not to assert pass/fail.

Usage:
    uv sync --extra transcription
    uv run python assorted_checks/test_transcription/test_long_form.py

Env overrides:
    KOKORO_BASE_URL  default http://localhost:8880/v1
    LONGFORM_VOICE   default af_heart
    LONGFORM_INPUT   default input/journey_all.txt.gz (relative to this script; .gz auto-decoded)
    LONGFORM_CHARS   default unset (full file); int caps cleaned input length
    WHISPER_MODEL    default base.en
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import openai


BASE_URL = os.environ.get("KOKORO_BASE_URL", "http://localhost:8880/v1")
VOICE = os.environ.get("LONGFORM_VOICE", "af_heart")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

SCRIPT_DIR = Path(__file__).parent
INPUT_PATH = Path(os.environ.get("LONGFORM_INPUT", SCRIPT_DIR / "input" / "journey_all.txt.gz"))
OUTPUT_DIR = SCRIPT_DIR / "output_long_form"


def rel(path: Path) -> str:
    """Path relative to the script dir, posix-style — keeps reports portable."""
    return path.resolve().relative_to(SCRIPT_DIR.resolve()).as_posix()


def _build_normalizer():
    # Imported lazily so --synth-only runs don't require faster-whisper/jiwer.
    from jiwer.transforms import (
        Compose,
        ReduceToListOfListOfWords,
        RemoveMultipleSpaces,
        RemovePunctuation,
        Strip,
        SubstituteWords,
        ToLowerCase,
    )

    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve",
    }
    return Compose([
        ToLowerCase(),
        RemovePunctuation(),
        SubstituteWords(digit_words),
        RemoveMultipleSpaces(),
        Strip(),
        ReduceToListOfListOfWords(),
    ])


def clean_input(raw: str) -> str:
    # Strip HTML/XML-style tags ("<i>i.e.</i>" etc.) and collapse whitespace.
    no_tags = re.sub(r"<[^>]+>", "", raw)
    return re.sub(r"\s+", " ", no_tags).strip()


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def synthesize_streaming(client: openai.OpenAI, text: str, out_path: Path) -> float:
    start = time.perf_counter()
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=VOICE,
        input=text,
        response_format="wav",
    ) as response:
        with open(out_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)
    fix_streaming_wav_header(out_path)
    return time.perf_counter() - start


def fix_streaming_wav_header(wav_path: Path) -> None:
    # Streaming WAV responses stamp placeholder sizes (RIFF=0xFFFFFFFF, data
    # chunk size near 0) because the server doesn't know the total length up
    # front. Rewrite both fields in place so downstream tools see real sizes.
    import struct

    size = wav_path.stat().st_size
    with open(wav_path, "r+b") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return  # not a RIFF/WAVE file, leave it alone

        # Walk subchunks until we find 'data', record its size offset.
        data_size_offset = None
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack("<I", chunk_header[4:8])[0]
            if chunk_id == b"data":
                data_size_offset = f.tell() - 4
                break
            # Skip body (chunk_size from header, but it's the placeholder for
            # streaming; jump to end-of-file for the data chunk if we're not
            # there yet — for non-data chunks, trust the header).
            f.seek(chunk_size, 1)

        if data_size_offset is None:
            return

        actual_data_size = size - (data_size_offset + 4)
        f.seek(4)
        f.write(struct.pack("<I", size - 8))
        f.seek(data_size_offset)
        f.write(struct.pack("<I", actual_data_size))


def audio_duration_seconds(wav_path: Path) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def transcribe(model, audio_path: Path) -> tuple[str, float]:
    start = time.perf_counter()
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=1,
        vad_filter=True,
    )
    text = " ".join(seg.text for seg in segments).strip()
    return text, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synth-only",
        action="store_true",
        help="Generate audio and stop. Skip Whisper load and transcription.",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Skip synthesis and transcribe the existing wav from a previous run.",
    )
    parser.add_argument(
        "--chars",
        type=int,
        default=None,
        help="Cap cleaned input at N chars (cuts at the last whitespace before N). "
             "Default: full file. Also reads LONGFORM_CHARS.",
    )
    args = parser.parse_args()
    if args.synth_only and args.transcribe_only:
        parser.error("--synth-only and --transcribe-only are mutually exclusive")

    char_cap = args.chars
    if char_cap is None and os.environ.get("LONGFORM_CHARS"):
        char_cap = int(os.environ["LONGFORM_CHARS"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if INPUT_PATH.suffix == ".gz":
        with gzip.open(INPUT_PATH, "rt", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = INPUT_PATH.read_text(encoding="utf-8")
    text = clean_input(raw)
    full_chars = len(text)
    if char_cap is not None and char_cap < full_chars:
        cut = text.rfind(" ", 0, char_cap)
        text = text[: cut if cut > 0 else char_cap]
    word_count = len(text.split())
    char_count = len(text)

    print(f"Server:      {BASE_URL}")
    print(f"Voice:       {VOICE}")
    if char_cap is not None and char_count < full_chars:
        print(f"Input:       {INPUT_PATH.name} ({word_count} words, {char_count} chars; capped at {char_cap} of {full_chars})")
    else:
        print(f"Input:       {INPUT_PATH.name} ({word_count} words, {char_count} chars)")
    print(f"Whisper:     {WHISPER_MODEL} (CPU, int8, VAD on)")
    print()

    audio_path = OUTPUT_DIR / f"long_form_{VOICE}.wav"

    if args.transcribe_only:
        if not audio_path.exists():
            print(f"--transcribe-only set but no audio at {audio_path}")
            return 2
        synth_s = 0.0
        audio_s = audio_duration_seconds(audio_path)
        synth_rtf = 0.0
        file_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"Reusing existing audio ({fmt_time(audio_s)}, {file_mb:.1f} MB)")
        print()
    else:
        client = openai.OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=None)

        print("Synthesizing...")
        try:
            synth_s = synthesize_streaming(client, text, audio_path)
        except Exception as exc:
            print(f"  synthesis failed after {time.perf_counter():.1f}s: {exc}")
            return 2

        audio_s = audio_duration_seconds(audio_path)
        synth_rtf = audio_s / synth_s if synth_s > 0 else 0.0
        file_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"  synth time:    {fmt_time(synth_s)}")
        print(f"  audio length:  {fmt_time(audio_s)}")
        print(f"  synth speedup: {synth_rtf:.2f}x realtime")
        print(f"  file size:     {file_mb:.1f} MB ({rel(audio_path)})")
        print()

    report = {
        "voice": VOICE,
        "input_file": rel(INPUT_PATH),
        "input_words": word_count,
        "input_chars": char_count,
        "input_chars_cap": char_cap,
        "audio_file": rel(audio_path),
        "audio_seconds": round(audio_s, 2),
        "synth_seconds": round(synth_s, 2),
        "synth_realtime_factor": round(synth_rtf, 2),
        "audio_file_mb": round(file_mb, 2),
    }

    if args.synth_only:
        report_path = OUTPUT_DIR / "long_form_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print("Summary (synth-only)")
        print(f"  generated {fmt_time(audio_s)} of audio in {fmt_time(synth_s)} ({synth_rtf:.1f}x rt)")
        print(f"  audio:  {rel(audio_path)}")
        print(f"  report: {rel(report_path)}")
        return 0

    from faster_whisper import WhisperModel
    from jiwer import wer

    normalize = _build_normalizer()

    print("Loading Whisper model...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print("Transcribing (this can take a while on CPU)...")
    transcript, trans_s = transcribe(model, audio_path)
    trans_rtf = audio_s / trans_s if trans_s > 0 else 0.0
    print(f"  transcribe time: {fmt_time(trans_s)}")
    print(f"  transcribe speedup: {trans_rtf:.2f}x realtime")
    print(f"  transcript words: {len(transcript.split())}")
    print()

    score = wer(
        text,
        transcript,
        reference_transform=normalize,
        hypothesis_transform=normalize,
    )
    print(f"WER (normalized): {score:.4f}")
    print()

    transcript_path = OUTPUT_DIR / f"long_form_{VOICE}.transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")

    report.update({
        "whisper_model": WHISPER_MODEL,
        "transcribe_seconds": round(trans_s, 2),
        "transcribe_realtime_factor": round(trans_rtf, 2),
        "transcript_words": len(transcript.split()),
        "wer": round(score, 4),
        "transcript_file": rel(transcript_path),
    })
    report_path = OUTPUT_DIR / "long_form_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("Summary")
    print(f"  generated {fmt_time(audio_s)} of audio in {fmt_time(synth_s)} ({synth_rtf:.1f}x rt)")
    print(f"  transcribed in {fmt_time(trans_s)} ({trans_rtf:.1f}x rt)")
    print(f"  WER {score:.4f} vs cleaned input")
    print(f"  report: {rel(report_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
