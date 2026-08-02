"""Plot the dialogue benchmark against the single-voice baseline.

Reads output/bench_report.json written by bench_dialogue.py and renders the
shared benchmark theme. The point of the top panel is that the dialogue cases
sit inside the baseline's own run to run spread, so speaker count does not
cost throughput.

Run from the repo root, after bench_dialogue.py:
    uv run --project examples --extra benchmarks python examples/assorted_checks/test_dialogue/plot_dialogue_bench.py
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
PLOT_PATH = OUTPUT_DIR / "dialogue_throughput.png"

# short labels, the report ones are too wide for an axis
SHORT_LABELS = {
    "single voice, one call (baseline)": "single voice\n(baseline)",
    "2 speakers, /dev/dialogue": "2 speakers\n/dev/dialogue",
    "4 speakers, /dev/dialogue": "4 speakers\n/dev/dialogue",
    "2 speakers, one call per turn": "2 speakers\none call per turn",
}


def load_cases(report: dict) -> dict[str, list[dict]]:
    """Group per-run entries by case label, preserving the order they ran in."""
    cases: dict[str, list[dict]] = {}
    for run in report["runs"]:
        for entry in run:
            cases.setdefault(entry["label"], []).append(entry)
    return cases


def draw_mean_line(ax, x, y_center, half_height, label):
    """Vertical mean marker with the gradient fade used elsewhere in the suite."""
    gradient = np.linspace(0.2, 0.9, 60)
    edges = np.linspace(y_center - half_height, y_center + half_height, len(gradient))
    for i in range(len(gradient) - 1):
        ax.plot(
            [x, x],
            [edges[i], edges[i + 1]],
            "-",
            color=STYLE_CONFIG["secondary_color"],
            linewidth=3,
            alpha=gradient[i],
        )
    ax.text(
        x,
        y_center + half_height + 0.16,
        label,
        ha="center",
        va="bottom",
        color=STYLE_CONFIG["text_color"],
        fontsize=STYLE_CONFIG["font_sizes"]["text"],
        fontweight="bold",
        bbox=dict(
            facecolor=STYLE_CONFIG["background_color"],
            edgecolor=STYLE_CONFIG["secondary_color"],
            alpha=0.85,
            pad=3,
            linewidth=1,
        ),
    )


def plot_relative(ax, cases: dict[str, list[dict]], baseline_label: str) -> None:
    """Throughput as a percentage of that run's baseline, against baseline noise."""
    baseline_rtfs = [entry["rtf"] for entry in cases[baseline_label]]
    baseline_mean = float(np.mean(baseline_rtfs))
    band_low = min(baseline_rtfs) / baseline_mean * 100
    band_high = max(baseline_rtfs) / baseline_mean * 100

    labels = [label for label in cases if label != baseline_label]
    positions = list(range(len(labels)))
    all_values = [
        entry["pct_of_baseline"] for label in labels for entry in cases[label]
    ]

    ax.axvspan(band_low, band_high, color=STYLE_CONFIG["secondary_color"], alpha=0.1)
    ax.axvline(
        100,
        color=STYLE_CONFIG["secondary_color"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )

    # annotate inline rather than with a legend, which collides with the dots
    ax.text(
        (band_low + band_high) / 2,
        -0.78,
        f"baseline run to run spread  {band_low:.0f}-{band_high:.0f}%",
        ha="center",
        va="center",
        color=STYLE_CONFIG["secondary_color"],
        fontsize=STYLE_CONFIG["font_sizes"]["text"],
        fontweight="bold",
        alpha=0.9,
    )

    for position, label in zip(positions, labels):
        values = [entry["pct_of_baseline"] for entry in cases[label]]
        ax.plot(
            values,
            [position] * len(values),
            "o",
            color=STYLE_CONFIG["primary_color"],
            markersize=10,
            alpha=0.55,
            markeredgewidth=0,
        )
        draw_mean_line(
            ax, float(np.mean(values)), position, 0.2, f"{np.mean(values):.1f}%"
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([SHORT_LABELS.get(label, label) for label in labels])
    ax.set_ylim(-1.0, len(labels) - 0.35)
    ax.set_xlim(min(all_values + [band_low]) - 2, max(all_values + [band_high]) + 2)
    ax.invert_yaxis()

    setup_plot(
        ax.figure,
        ax,
        "Multi-Speaker Throughput vs Single-Voice Baseline",
        xlabel="% of baseline throughput (higher is better)",
    )


def plot_per_run(ax, cases: dict[str, list[dict]], baseline_label: str) -> None:
    """Wall clock per run, which is why the top panel normalizes per run."""
    labels = list(cases)
    runs = range(1, len(cases[baseline_label]) + 1)
    styles = [
        (STYLE_CONFIG["secondary_color"], "o", "-", 1.0),
        (STYLE_CONFIG["primary_color"], "s", "-", 0.9),
        (STYLE_CONFIG["primary_color"], "^", "--", 0.65),
        (STYLE_CONFIG["primary_color"], "D", ":", 0.45),
    ]

    for label, (color, marker, linestyle, alpha) in zip(labels, styles):
        ax.plot(
            list(runs),
            [entry["wall"] for entry in cases[label]],
            marker=marker,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            linewidth=2,
            markersize=7,
            label=SHORT_LABELS.get(label, label).replace("\n", " "),
        )

    ax.set_xticks(list(runs))
    setup_plot(
        ax.figure,
        ax,
        "Wall Clock per Run (the machine drifts, so the top panel normalizes within a run)",
        xlabel="run",
        ylabel="seconds",
    )
    legend = ax.legend(
        loc="best", framealpha=0.85, fontsize=STYLE_CONFIG["font_sizes"]["text"]
    )
    legend.get_frame().set_facecolor(STYLE_CONFIG["background_color"])
    for text in legend.get_texts():
        text.set_color(STYLE_CONFIG["text_color"])


def main() -> int:
    if not REPORT_PATH.exists():
        print(f"no report at {REPORT_PATH}, run bench_dialogue.py first")
        return 1

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    cases = load_cases(report)
    if not cases:
        print("report has no runs")
        return 1
    baseline_label = report["runs"][0][0]["label"]
    run_count = len(report["runs"])

    plt.style.use("dark_background")
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor(STYLE_CONFIG["background_color"])

    plot_relative(top, cases, baseline_label)
    plot_per_run(bottom, cases, baseline_label)

    fig.text(
        0.5,
        0.015,
        f"{report['words']} words per case, {run_count} run{'s' if run_count != 1 else ''}. "
        f"Top panel: one dot per run, cyan bar is the mean.",
        ha="center",
        color=STYLE_CONFIG["text_color"],
        fontsize=STYLE_CONFIG["font_sizes"]["text"],
        alpha=0.7,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    print(f"wrote {PLOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
