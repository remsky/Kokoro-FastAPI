"""Compare multi-speaker generation cost against a single-voice baseline.

Renders the same ~3.5 minute corpus four ways and reports throughput relative
to one plain single-voice request. The number that matters is 4 speakers
against 2: if switching cost anything, the wider cast would be slower.

Run from the repo root:
    uv run --project examples python examples/assorted_checks/test_dialogue/bench_dialogue.py
"""

import json
import os
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("KOKORO_BASE_URL", "http://localhost:8880").rstrip("/")
TIMEOUT = float(os.getenv("KOKORO_TIMEOUT", "1800"))
RUNS = int(os.getenv("DIALOGUE_BENCH_RUNS", "1"))
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44

OUTPUT_DIR = Path(__file__).parent / "output"

SENTENCES = [
    "The morning shift handed over a quiet board with only two open alerts.",
    "One of them had been flapping since midnight without any clear root cause.",
    "We pulled the traces and found a retry storm hiding behind a healthy status code.",
    "The upstream service was returning success while dropping half of the payload.",
    "That explained why the dashboards looked fine but the totals never reconciled.",
    "Nobody had thought to compare the record counts on either side of the queue.",
    "Once we did, the gap was obvious and it grew by a few thousand rows an hour.",
    "The fix was small but the investigation took most of the afternoon to complete.",
    "We added a reconciliation job that runs every fifteen minutes and alerts on drift.",
    "It caught a second issue within a day, which paid for the effort immediately.",
    "The second issue was a clock skew problem on one of the older worker nodes.",
    "Its timestamps drifted far enough that the dedupe window silently stopped working.",
    "We rebuilt the node and pinned the time source to the same pool as everything else.",
    "After that the drift metric flattened out and stayed flat for the rest of the week.",
    "The team wrote it up and added both checks to the standard onboarding runbook.",
    "It is easy to trust a green dashboard when the underlying counts are never compared.",
    "That lesson shows up again and again in postmortems across completely different systems.",
    "The cheapest observability you can add is often a simple count on both ends of a pipe.",
    "Everything else is refinement once you know the two numbers are supposed to match.",
    "We shipped the change on a Thursday and watched it through the weekend without incident.",
    "The following sprint we extended the same pattern to the billing export pipeline.",
    "That pipeline had never had an independent check on the number of records emitted.",
    "It turned out to be correct, which was reassuring but slightly anticlimactic for everyone.",
    "Still, the check now exists and it will catch the problem the first time it happens.",
    "A good portion of reliability work looks like that, quiet and uneventful by design.",
    "The alerts that never fire are doing just as much work as the ones that do.",
    "We tried to capture that idea in the review so it would not get lost over time.",
    "New engineers tend to optimize for the dramatic incidents they read about in writeups.",
    "The unglamorous checks are what keep those incidents from happening in the first place.",
    "By the end of the quarter the reconciliation pattern covered eleven separate pipelines.",
    "Two of those found real discrepancies within the first month of being enabled.",
    "Both discrepancies traced back to the same misconfigured retry policy in a shared library.",
    "Fixing the library resolved them together, which was a satisfying way to close the loop.",
    "We deprecated the old policy and added a lint rule to catch it in future code reviews.",
    "The lint rule fired twice in the next two weeks, so it was clearly worth adding.",
    "None of this was especially clever, it was just consistent application of a simple idea.",
    "Count the things going in, count the things coming out, and complain when they differ.",
    "Everything downstream of that principle is tuning thresholds and reducing false positives.",
    "The team still refers back to that first flapping alert as the origin of the whole effort.",
    "It remains a useful reminder that a noisy alert is sometimes telling you the truth.",
]

VOICES_2 = ["af_bella", "am_michael"]
VOICES_4 = ["af_bella", "am_michael", "af_heart", "bm_george"]


def post(path: str, body: dict) -> bytes:
    response = requests.post(f"{BASE_URL}{path}", json=body, timeout=TIMEOUT)
    response.raise_for_status()
    return response.content


def audio_seconds(payload: bytes) -> float:
    """Duration from byte length, since the wav header carries a placeholder frame count."""
    return max(len(payload) - WAV_HEADER_BYTES, 0) / (SAMPLE_RATE * BYTES_PER_SAMPLE)


def run_single(text: str, voice: str = "af_heart") -> tuple[float, float]:
    start = time.perf_counter()
    audio = post(
        "/v1/audio/speech",
        {
            "model": "kokoro",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "stream": False,
        },
    )
    return time.perf_counter() - start, audio_seconds(audio)


def run_dialogue(voices: list[str]) -> tuple[float, float]:
    turns = [
        {"voice": voices[i % len(voices)], "text": sentence}
        for i, sentence in enumerate(SENTENCES)
    ]
    start = time.perf_counter()
    audio = post(
        "/dev/dialogue",
        {
            "model": "kokoro",
            "turns": turns,
            "pause_between_turns": 0.0,
            "response_format": "wav",
            "stream": False,
        },
    )
    return time.perf_counter() - start, audio_seconds(audio)


def run_separate_calls(voices: list[str]) -> tuple[float, float]:
    """The workaround this replaces: one request per turn, stitched client side."""
    start = time.perf_counter()
    total_audio = 0.0
    for i, sentence in enumerate(SENTENCES):
        audio = post(
            "/v1/audio/speech",
            {
                "model": "kokoro",
                "input": sentence,
                "voice": voices[i % len(voices)],
                "response_format": "wav",
                "stream": False,
            },
        )
        total_audio += audio_seconds(audio)
    return time.perf_counter() - start, total_audio


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    full_text = " ".join(SENTENCES)
    print(f"server: {BASE_URL}")
    print(f"corpus: {len(SENTENCES)} sentences, {len(full_text.split())} words")
    print(f"runs:   {RUNS}\n")

    print("warming model...")
    run_single("Warm up the model and the default pipeline.")

    cases = [
        ("single voice, one call (baseline)", lambda: run_single(full_text)),
        ("2 speakers, /dev/dialogue", lambda: run_dialogue(VOICES_2)),
        ("4 speakers, /dev/dialogue", lambda: run_dialogue(VOICES_4)),
        ("2 speakers, one call per turn", lambda: run_separate_calls(VOICES_2)),
    ]

    runs = []
    for run_index in range(RUNS):
        print(f"\nrun {run_index + 1}/{RUNS}")
        raw: list[tuple[str, float, float, float]] = []
        for label, case in cases:
            wall, audio = case()
            rtf = audio / wall if wall else 0.0
            raw.append((label, wall, audio, rtf))
            print(
                f"  {label:<34} wall={wall:7.2f}s  audio={audio:7.2f}s  {rtf:5.2f}x realtime"
            )

        baseline_rtf = raw[0][3]
        measured = [
            {
                "label": label,
                "wall": wall,
                "audio": audio,
                "rtf": rtf,
                "pct_of_baseline": rtf / baseline_rtf * 100 if baseline_rtf else 0.0,
            }
            for label, wall, audio, rtf in raw
        ]
        print()
        for entry in measured[1:]:
            print(
                f"  {entry['label']:<34} {entry['pct_of_baseline']:5.1f}% of baseline throughput"
            )
        runs.append(measured)

    report_path = OUTPUT_DIR / "bench_report.json"
    report_path.write_text(
        json.dumps(
            {"base_url": BASE_URL, "words": len(full_text.split()), "runs": runs},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport written to {report_path}")


if __name__ == "__main__":
    main()
