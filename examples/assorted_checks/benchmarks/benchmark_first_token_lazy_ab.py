#!/usr/bin/env python3
"""Time to first audio chunk against input size, eager vs lazy get_sentence_info.

Capture one tree at a time, then plot both:

    uv run python assorted_checks/benchmarks/benchmark_first_token_lazy_ab.py --label base
    uv run python assorted_checks/benchmarks/benchmark_first_token_lazy_ab.py --label lazy
    uv run python assorted_checks/benchmarks/benchmark_first_token_lazy_ab.py --plot

The stream is dropped once the first chunk lands; the router checks
`is_disconnected` per chunk, so the server stops rather than synthesizing hours
of audio nobody reads. That stop lands one chunk late, so SETTLE seconds pass
between requests; without it every sample carries the tail of the one before.
"""

import argparse
import json
import os
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import requests

from lib.shared_plotting import STYLE_CONFIG, setup_plot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "output_data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "output_plots")
CORPUS = os.path.join(SCRIPT_DIR, "the_time_machine_hg_wells.txt")

# chars, the unit MAX_INPUT_LENGTH uses; 1M is that default limit
CHAR_SIZES = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
REPS = 5
WARMUP_CHARS = 1_000
SETTLE = 3.0
URL = "http://localhost:8880/v1/audio/speech"


def load_corpus(min_chars: int) -> str:
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    while len(text) < min_chars:
        text += "\n\n" + text
    return text


def time_to_first_chunk(text: str, timeout: int = 600) -> float:
    start = time.perf_counter()
    r = requests.post(
        URL,
        json={
            "model": "kokoro",
            "input": text,
            "voice": "af_heart",
            "response_format": "pcm",
            "stream": True,
        },
        stream=True,
        timeout=timeout,
    )
    r.raise_for_status()
    try:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                return time.perf_counter() - start
    finally:
        r.close()
    raise RuntimeError("no audio chunk received")


def wait_for_health(timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get("http://localhost:8880/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise RuntimeError("server never became healthy")


def capture(label: str) -> None:
    corpus = load_corpus(max(CHAR_SIZES))
    wait_for_health()
    print(f"warming up ({WARMUP_CHARS} chars)")
    for _ in range(5):
        time_to_first_chunk(corpus[:WARMUP_CHARS])
        time.sleep(SETTLE)

    runs = []
    for chars in CHAR_SIZES:
        text = corpus[:chars]
        samples = []
        for _ in range(REPS):
            samples.append(time_to_first_chunk(text))
            time.sleep(SETTLE)
        runs.append({"chars": chars, "samples": samples})
        print(
            f"{chars:>9} chars  "
            f"mean={statistics.mean(samples):.3f}s  "
            f"sd={statistics.stdev(samples):.3f}s"
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"lazy_ab_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"label": label, "reps": REPS, "runs": runs}, f, indent=2)
    print(f"wrote {path}")


def _series(label: str):
    with open(os.path.join(DATA_DIR, f"lazy_ab_{label}.json"), encoding="utf-8") as f:
        data = json.load(f)
    x = np.array([r["chars"] for r in data["runs"]], dtype=float)
    means = np.array([statistics.mean(r["samples"]) for r in data["runs"]])
    sds = np.array([statistics.stdev(r["samples"]) for r in data["runs"]])
    return x, means, sds


def plot() -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 3))

    for label, name, color, style in (
        ("base", "v0.8.0 (eager)", STYLE_CONFIG["primary_color"], "--"),
        ("lazy", "v0.8.1 (lazy)", STYLE_CONFIG["secondary_color"], "-"),
    ):
        x, means, sds = _series(label)
        ax.errorbar(
            x,
            means,
            yerr=sds,
            label=name,
            color=color,
            marker="o",
            markersize=6,
            capsize=4,
            linewidth=2,
            linestyle=style,
            elinewidth=1.5,
            alpha=0.9,
        )

    ax.set_xscale("log")
    ax.set_xticks(CHAR_SIZES)
    ax.set_xticklabels([f"{s // 1000}k" if s < 1_000_000 else "1M" for s in CHAR_SIZES])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    setup_plot(
        fig,
        ax,
        "Time to first audio",
        xlabel="Input size (characters)",
        ylabel="Seconds",
    )
    ax.legend(
        facecolor=STYLE_CONFIG["background_color"],
        edgecolor=STYLE_CONFIG["text_color"],
        fontsize=STYLE_CONFIG["font_sizes"]["text"],
    )
    ax.set_ylim(bottom=0)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "lazy_ab_first_token.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", choices=["base", "lazy"])
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    if args.plot:
        plot()
    elif args.label:
        capture(args.label)
    else:
        p.error("pass --label base|lazy or --plot")


if __name__ == "__main__":
    main()
