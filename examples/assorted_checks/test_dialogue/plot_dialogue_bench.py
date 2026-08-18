"""Plot dialogue benchmark results from bench_dialogue.py.

Run: uv run --project examples --extra benchmarks python examples/assorted_checks/test_dialogue/plot_dialogue_bench.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from lib.shared_plotting import STYLE_CONFIG, setup_plot  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"
REPORT_PATH = OUTPUT_DIR / "bench_report.json"
TURN_PLOT_PATH = OUTPUT_DIR / "dialogue_turn_length.png"
TEXT_PLOT_PATH = OUTPUT_DIR / "dialogue_text_length.png"

BASELINE = STYLE_CONFIG["text_color"]
KNEE = "#8a8aa0"
FIGSIZE = (12, 6)
Y_LIMITS = (55, 108)
TOKENS_PER_SENTENCE = 83

SERIES = [
    ("dialogue2", "2 voices", "#ff2a6d", "o"),
    ("dialogue4", "4 voices", "#ffb703", "^"),
    ("dialogue8", "8 voices", "#05d9e8", "s"),
    ("per_turn", "one call per turn (previous method)", "#9aa0b5", "D"),
]
LENGTH_SERIES = [
    ("length5", "a voice change every 5 sentences", "#05d9e8", "s"),
    ("length1", "a voice change every sentence", "#ff2a6d", "o"),
]
LENGTH_SKIP = {5}


def collect(report: dict, series: str, field: str = "pct_of_baseline") -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for run in report["runs"]:
        for entry in run:
            if entry["series"] == series:
                grouped.setdefault(entry["x"], []).append(entry[field])
    return grouped

def means(report: dict, series: str, order: list[int]) -> list[float]:
    return [float(np.mean(collect(report, series)[x])) for x in order]

def sweep_order(report: dict, series: str, descending: bool) -> list[int]:
    return sorted(collect(report, series), reverse=descending)

def token_label(x: int) -> str:
    tokens = x * TOKENS_PER_SENTENCE
    return f"~{tokens / 1000:.1f}k" if tokens >= 1000 else f"~{tokens}"

def sentence_label(x: int) -> str:
    return f"{x} sentence" if x == 1 else f"{x} sentences"

def clock(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"

def new_figure() -> tuple:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(STYLE_CONFIG["background_color"])
    return fig, ax

def draw_baseline(ax, label_x: float) -> None:
    ax.axhline(100, color=BASELINE, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(label_x, 100.8, "single-voice request of the same text",
            ha="right", va="bottom", color=BASELINE,
            fontsize=STYLE_CONFIG["font_sizes"]["text"], alpha=0.8)

def draw_caption(fig, text: str) -> None:
    fig.text(0.5, 0.015, text, ha="center", color=STYLE_CONFIG["text_color"],
             fontsize=STYLE_CONFIG["font_sizes"]["text"], alpha=0.7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

def draw_chunk_floor(ax, order: list[int], min_tokens: int) -> None:
    under = [i for i, x in enumerate(order) if x * TOKENS_PER_SENTENCE < min_tokens]
    if not under:
        return
    ax.axvspan(under[0] - 0.5, len(order) - 0.5, color=KNEE, alpha=0.12)
    ax.text(len(order) - 0.28, sum(Y_LIMITS) / 2,
            f"under the {min_tokens} token chunk minimum",
            ha="center", va="center", rotation=90, color=KNEE,
            fontsize=STYLE_CONFIG["font_sizes"]["text"], fontweight="bold")


def plot_turn_length(report: dict) -> None:
    order = sweep_order(report, "dialogue2", descending=True)
    positions = list(range(len(order)))
    fig, ax = new_figure()

    draw_baseline(ax, label_x=len(order) - 0.6)
    for key, label, color, marker in SERIES:
        ax.plot(positions, means(report, key, order), marker=marker, color=color,
                label=label, linewidth=2.5, markersize=8, alpha=0.95)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{token_label(x)}\n({sentence_label(x)})" for x in order])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(*Y_LIMITS)
    ax.legend(loc="lower left", facecolor=STYLE_CONFIG["background_color"],
              edgecolor=KNEE, framealpha=0.85, fontsize=STYLE_CONFIG["font_sizes"]["text"])
    draw_chunk_floor(ax, order, min_tokens=175)

    setup_plot(fig, ax, "Multi-Speaker Throughput vs Tokens per Turn",
               xlabel="phoneme tokens per speaker turn", ylabel="% of single-voice throughput")
    draw_caption(fig,
        f"{report['fixed_words']} words, {len(report['runs'])} runs, "
        f"{report.get('device', 'gpu')}. Same text every point, only the turn boundaries move.")
    fig.savefig(TURN_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {TURN_PLOT_PATH}")


def plot_text_length(report: dict) -> None:
    order = [x for x in sweep_order(report, "length1", descending=False) if x not in LENGTH_SKIP]
    positions = list(range(len(order)))
    audio = collect(report, "length_single", "audio")
    fig, ax = new_figure()

    draw_baseline(ax, label_x=len(order) - 0.55)
    for key, label, color, marker in LENGTH_SERIES:
        grouped = collect(report, key)
        for position, x in zip(positions, order):
            ax.scatter([position] * len(grouped[x]), grouped[x], color=color, alpha=0.25, s=40, zorder=2)
        ax.plot(positions, [float(np.mean(grouped[x])) for x in order], marker=marker,
                color=color, label=label, linewidth=2.5, markersize=8, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([clock(float(np.mean(audio[x]))) for x in order])
    ax.set_xlim(-0.4, len(order) - 0.6)
    ax.set_ylim(*Y_LIMITS)
    ax.legend(loc="lower left", facecolor=STYLE_CONFIG["background_color"],
              edgecolor=KNEE, framealpha=0.85, fontsize=STYLE_CONFIG["font_sizes"]["text"])

    setup_plot(fig, ax, "Multi-Speaker Throughput vs Generation Length",
               xlabel="audio generated", ylabel="% of single-voice throughput")
    draw_caption(fig,
        f"2 voices, {len(report['runs'])} runs, {report.get('device', 'gpu')}. "
        "Each length is normalized to a single-voice request of that same text.")
    fig.savefig(TEXT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {TEXT_PLOT_PATH}")


def widen_y_limits(report: dict) -> None:
    global Y_LIMITS
    drawn = [float(np.mean(v)) for key, *_ in SERIES for v in collect(report, key).values()]
    drawn += [v for key, *_ in LENGTH_SERIES for values in collect(report, key).values() for v in values]
    Y_LIMITS = (min(Y_LIMITS[0], min(drawn) - 3), max(Y_LIMITS[1], max(drawn) + 3))


def main() -> int:
    if not REPORT_PATH.exists():
        print(f"no report at {REPORT_PATH}, run bench_dialogue.py first")
        return 1
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if not report.get("runs"):
        print("report has no runs")
        return 1
    widen_y_limits(report)
    plot_turn_length(report)
    plot_text_length(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
