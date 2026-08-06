"""Dialogue benchmark: measures multi-speaker overhead against single-voice baseline.

Run: uv run --project examples python examples/assorted_checks/test_dialogue/bench_dialogue.py
"""

import json
import os
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("KOKORO_BASE_URL", "http://localhost:8880").rstrip("/")
TIMEOUT = float(os.getenv("KOKORO_TIMEOUT", "1800"))
RUNS = int(os.getenv("DIALOGUE_BENCH_RUNS", "3"))
DEVICE = os.getenv("KOKORO_DEVICE", "gpu")
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44
OUTPUT_DIR = Path(__file__).parent / "output"

# ── sweep shape
FIXED_SENTENCES = 40
GROUP_SIZES = [40, 20, 10, 4, 2, 1]
LENGTHS = [5, 10, 20, 40, 80]
LENGTH_GROUPS = [5, 1]

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

VOICES_8 = [
    "af_bella", "am_michael", "af_heart", "bm_george",
    "af_nicole", "am_adam", "bf_emma", "bm_lewis",
]
VOICE_SETS = {2: VOICES_8[:2], 4: VOICES_8[:4], 8: VOICES_8}
BASELINE_VOICE = "af_heart"


def corpus(count: int) -> list[str]:
    return [SENTENCES[i % len(SENTENCES)] for i in range(count)]

def post(path: str, body: dict) -> bytes:
    response = requests.post(f"{BASE_URL}{path}", json=body, timeout=TIMEOUT)
    response.raise_for_status()
    return response.content

def audio_seconds(payload: bytes) -> float:
    return max(len(payload) - WAV_HEADER_BYTES, 0) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

def timed(call) -> tuple[float, float]:
    start = time.perf_counter()
    audio = call()
    return time.perf_counter() - start, audio_seconds(audio)

def run_single(sentences: list[str], voice: str = BASELINE_VOICE) -> tuple[float, float]:
    return timed(lambda: post("/v1/audio/speech", {
        "model": "kokoro", "input": " ".join(sentences),
        "voice": voice, "response_format": "wav", "stream": False,
    }))

def turns_for(sentences: list[str], voices: list[str], group: int) -> list[dict]:
    return [
        {"voice": voices[(i // group) % len(voices)], "text": sentence}
        for i, sentence in enumerate(sentences)
    ]

def swap_count(turns: list[dict]) -> int:
    return sum(1 for a, b in zip(turns, turns[1:]) if a["voice"] != b["voice"])

def distinct_voices(turns: list[dict]) -> int:
    return len({turn["voice"] for turn in turns})

def merge_turns(turns: list[dict]) -> list[dict]:
    # collapse consecutive same-voice turns so per-turn method gets a fair comparison
    merged: list[dict] = []
    for turn in turns:
        if merged and merged[-1]["voice"] == turn["voice"]:
            merged[-1]["text"] += " " + turn["text"]
        else:
            merged.append(dict(turn))
    return merged

def run_dialogue(turns: list[dict]) -> tuple[float, float]:
    return timed(lambda: post("/dev/dialogue", {
        "model": "kokoro", "turns": turns,
        "pause_between_turns": 0.0, "response_format": "wav", "stream": False,
    }))

def run_separate_calls(turns: list[dict]) -> tuple[float, float]:
    start = time.perf_counter()
    total_audio = 0.0
    for turn in merge_turns(turns):
        audio = post("/v1/audio/speech", {
            "model": "kokoro", "input": turn["text"],
            "voice": turn["voice"], "response_format": "wav", "stream": False,
        })
        total_audio += audio_seconds(audio)
    return time.perf_counter() - start, total_audio

def warm():
    for voice in dict.fromkeys([BASELINE_VOICE] + VOICES_8):
        run_single(["Warm up the model and this voice pipeline."], voice)
    run_dialogue(turns_for(corpus(4), VOICE_SETS[2], 1))

def measure(run: list[dict], series: str, x: int, wall: float, audio: float, base: float) -> None:
    rtf = audio / wall if wall else 0.0
    run.append({
        "series": series, "x": x, "wall": wall, "audio": audio,
        "rtf": rtf, "pct_of_baseline": rtf / base * 100 if base else 0.0,
    })
    print(f"  {series:<10} x={x:<4} wall={wall:7.2f}s  {rtf:6.2f}x rt  {run[-1]['pct_of_baseline']:6.1f}% of baseline")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"server: {BASE_URL}  device label: {DEVICE}  runs: {RUNS}")
    print("warming every voice...")
    warm()

    fixed = corpus(FIXED_SENTENCES)
    runs = []
    for run_index in range(RUNS):
        print(f"\nrun {run_index + 1}/{RUNS}")
        entries: list[dict] = []

        wall, audio = run_single(fixed)
        fixed_rtf = audio / wall if wall else 0.0
        print(f"  baseline    n={FIXED_SENTENCES}  wall={wall:7.2f}s  {fixed_rtf:6.2f}x rt")
        entries.append({"series": "baseline", "x": FIXED_SENTENCES, "wall": wall,
                        "audio": audio, "rtf": fixed_rtf, "pct_of_baseline": 100.0})

        # ── turn length sweep
        for group in GROUP_SIZES:
            for count, voices in VOICE_SETS.items():
                wall, audio = run_dialogue(turns_for(fixed, voices, group))
                measure(entries, f"dialogue{count}", group, wall, audio, fixed_rtf)
            wall, audio = run_separate_calls(turns_for(fixed, VOICE_SETS[2], group))
            measure(entries, "per_turn", group, wall, audio, fixed_rtf)

        # ── length sweep
        for count in LENGTHS:
            sentences = corpus(count)
            wall, audio = run_single(sentences)
            base = audio / wall if wall else 0.0
            measure(entries, "length_single", count, wall, audio, base)
            for group in LENGTH_GROUPS:
                wall, audio = run_dialogue(turns_for(sentences, VOICE_SETS[2], group))
                measure(entries, f"length{group}", count, wall, audio, base)

        runs.append(entries)

    report_path = OUTPUT_DIR / "bench_report.json"
    report_path.write_text(json.dumps({
        "base_url": BASE_URL, "device": DEVICE,
        "fixed_sentences": FIXED_SENTENCES,
        "fixed_words": len(" ".join(fixed).split()),
        "voices": {str(k): v for k, v in VOICE_SETS.items()},
        "shape": {
            str(group): {
                "turns": len(merge_turns(turns_for(fixed, VOICE_SETS[2], group))),
                "swaps": swap_count(turns_for(fixed, VOICE_SETS[2], group)),
                "voices_used": {str(count): distinct_voices(turns_for(fixed, voices, group))
                                for count, voices in VOICE_SETS.items()},
            } for group in GROUP_SIZES
        },
        "runs": runs,
    }, indent=2), encoding="utf-8")
    print(f"\nreport written to {report_path}")


if __name__ == "__main__":
    main()
