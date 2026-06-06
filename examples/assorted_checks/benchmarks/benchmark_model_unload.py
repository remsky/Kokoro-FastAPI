#!/usr/bin/env python3
"""Measure VRAM reclaimed by POST /dev/unload, and the lazy reload after it.

Runs two scenarios (short + longform), each: baseline -> request -> unload ->
request (triggers lazy reload). A background monitor samples whole-GPU VRAM and
system RAM the whole time, so the plots show the drop on unload and the climb
back on reload, exactly as the host sees it (nvidia-smi view).

Usage (from examples/, model server already up on :8880):
    uv run --extra benchmarks assorted_checks/benchmarks/benchmark_model_unload.py
    uv run --extra benchmarks assorted_checks/benchmarks/benchmark_model_unload.py --long-lines 60

Note: reclaim scales with how hard the model has been worked (the activation
pool grows under load and is fully returned on unload), so a bigger --long-lines
yields a bigger longform reclaim.
"""
import os
import sys
import time
import queue
import argparse
import threading
from datetime import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.shared_utils import (  # noqa: E402
    get_system_metrics,
    save_json_results,
    save_audio_file,
    get_audio_length,
    write_benchmark_stats,
)
from lib.shared_plotting import STYLE_CONFIG, setup_plot  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_FILE = os.path.join(SCRIPT_DIR, "the_time_machine_hg_wells.txt")
SAMPLE_INTERVAL = 0.2  # seconds between metric samples

SHORT_TEXT = (
    "The model has been released from GPU memory, "
    "and will reload automatically on the next request."
)


class SystemMonitor:
    """Background sampler of system + GPU metrics (whole-GPU via nvidia-smi)."""

    def __init__(self, interval=SAMPLE_INTERVAL):
        self.interval = interval
        self.metrics_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.metrics_timeline = []
        self.start_time = None

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            metrics = get_system_metrics()
            metrics["relative_time"] = time.time() - self.start_time
            self.metrics_queue.put(metrics)
            time.sleep(self.interval)

    def start(self):
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=2)
        while True:
            try:
                self.metrics_timeline.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return self.metrics_timeline


def load_longform(n_lines: int) -> str:
    lines = [ln.strip() for ln in open(TEXT_FILE, encoding="utf-8") if ln.strip()]
    return " ".join(lines[:n_lines])


def speech(base_url: str, text: str, voice: str, fmt: str) -> tuple[bytes, float]:
    """Fire one /v1/audio/speech request, return (audio_bytes, gen_seconds)."""
    t0 = time.time()
    r = requests.post(
        f"{base_url}/v1/audio/speech",
        json={"model": "kokoro", "input": text, "voice": voice, "response_format": fmt},
        timeout=600,
    )
    r.raise_for_status()
    return r.content, time.time() - t0


def unload(base_url: str) -> None:
    r = requests.post(f"{base_url}/dev/unload", timeout=60)
    r.raise_for_status()


def run_scenario(label, text, args) -> dict:
    """Normalize -> load -> warm -> unload -> reload, monitored throughout.

    A leading unload clears any reserved pool inherited from a prior scenario,
    so each run is reproducible from the same floor regardless of history.
    """
    print(f"\n=== scenario: {label} ({len(text)} chars / {len(text.split())} words) ===")
    monitor = SystemMonitor()
    events = []

    def mark(name):
        events.append({"time": time.time() - monitor.start_time, "label": name})
        print(f"  t={events[-1]['time']:6.2f}s  {name}")

    monitor.start()

    mark("unload (reset)")          # normalize: drop any inherited pool
    unload(args.url)
    time.sleep(args.settle)

    mark("load (cold)")             # first request reloads the model lazily
    audio1, gen_cold1 = speech(args.url, text, args.voice, args.format)
    mark("warm request")           # second request runs warm
    _, gen_warm = speech(args.url, text, args.voice, args.format)
    mark("loaded")
    time.sleep(args.settle)         # steady loaded window

    mark("unload")                  # the drop we care about
    unload(args.url)
    time.sleep(args.settle)         # floor window

    mark("reload (cold)")           # request after unload -> lazy reload
    audio2, gen_cold2 = speech(args.url, text, args.voice, args.format)
    mark("reloaded")
    time.sleep(args.settle)

    timeline = monitor.stop()

    # save audio so runs are inspectable, like the other benchmarks
    audio_dir = os.path.join(SCRIPT_DIR, "output_audio")
    p1 = save_audio_file(audio1, f"model_unload_{label}_loaded", audio_dir)
    save_audio_file(audio2, f"model_unload_{label}_reloaded", audio_dir)
    alen = get_audio_length(audio1, audio_dir) if args.format == "wav" else None

    df = pd.DataFrame(timeline)
    gpu_gb = df["gpu_memory_used"] / 1024
    t = lambda name: next(e["time"] for e in events if e["label"] == name)

    def window(lo, hi):
        return gpu_gb[(df["relative_time"] >= lo) & (df["relative_time"] < hi)]

    # steady loaded level (after warm gen settled) vs floor after the unload
    loaded = window(t("loaded"), t("unload")).median()
    floor = window(t("unload"), t("reload (cold)")).median()
    reclaim = loaded - floor
    cold_gen = (gen_cold1 + gen_cold2) / 2

    stats = {
        "loaded_gpu_gb": round(loaded, 3),
        "floor_after_unload_gb": round(floor, 3),
        "reclaimed_gb": round(reclaim, 3),
        "reclaimed_mib": round(reclaim * 1024, 1),
        "warm_gen_seconds": round(gen_warm, 2),
        "cold_reload_gen_seconds": round(cold_gen, 2),
        "reload_penalty_seconds": round(cold_gen - gen_warm, 2),
    }
    if alen:
        stats["audio_seconds"] = round(alen, 1)
        stats["warm_rtf"] = round(gen_warm / alen, 3)
    print(f"  -> reclaimed {stats['reclaimed_mib']} MiB on unload; "
          f"reload cost +{stats['reload_penalty_seconds']}s")

    plot_path = os.path.join(SCRIPT_DIR, "output_plots", f"model_unload_{label}.png")
    plot_timeline(df, events, stats, loaded, floor, plot_path, label)

    return {"label": label, "stats": stats, "events": events, "timeline": timeline,
            "audio_sample": p1}


def plot_timeline(df, events, stats, loaded, floor, output_path, label):
    """Single-panel dark GPU timeline: loaded plateau, unload drop, reload climb.

    The story is VRAM, so the plot is GPU-only. Phases are shown as shaded
    windows with large horizontal labels instead of cramped rotated text.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.style.use("dark_background")
    t = df["relative_time"]
    gpu_gb = df["gpu_memory_used"] / 1024
    ev = {e["label"]: e["time"] for e in events}
    fs = STYLE_CONFIG["font_sizes"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(STYLE_CONFIG["background_color"])

    # ── phase shading: loaded window (cyan tint) vs unloaded floor (grey tint)
    if "load (cold)" in ev and "unload" in ev:
        ax.axvspan(ev["load (cold)"], ev["unload"],
                   color=STYLE_CONFIG["secondary_color"], alpha=0.07)
    if "unload" in ev and "reload (cold)" in ev:
        ax.axvspan(ev["unload"], ev["reload (cold)"],
                   color=STYLE_CONFIG["text_color"], alpha=0.06)

    # ── gpu trace
    ax.plot(t, gpu_gb, color=STYLE_CONFIG["primary_color"], linewidth=2.5)

    # ── loaded / floor reference lines + reclaim band
    ax.axhline(loaded, color=STYLE_CONFIG["secondary_color"], linestyle="--",
               alpha=0.7, linewidth=1.5, label=f"loaded  {loaded:.2f} GB")
    ax.axhline(floor, color=STYLE_CONFIG["text_color"], linestyle="--",
               alpha=0.55, linewidth=1.5, label=f"unloaded floor  {floor:.2f} GB")
    ax.fill_between(t, floor, loaded, color=STYLE_CONFIG["secondary_color"], alpha=0.08)
    ax.annotate(
        f"reclaimed {stats['reclaimed_mib']:.0f} MiB",
        xy=(0.5, (loaded + floor) / 2), xycoords=("axes fraction", "data"),
        ha="center", va="center", color=STYLE_CONFIG["text_color"],
        fontsize=fs["title"], fontweight="bold",
        bbox=dict(facecolor=STYLE_CONFIG["background_color"],
                  edgecolor=STYLE_CONFIG["secondary_color"], alpha=0.85, pad=6),
    )

    # ── big phase labels centered over each window (no rotation)
    ytop = gpu_gb.max()

    def phase_label(x0, x1, txt):
        ax.text((x0 + x1) / 2, ytop + 0.04, txt, ha="center", va="bottom",
                fontsize=fs["label"], fontweight="bold",
                color=STYLE_CONFIG["text_color"], alpha=0.9)

    if "load (cold)" in ev and "unload" in ev:
        phase_label(ev["load (cold)"], ev["unload"], "model loaded")
    if "unload" in ev and "reload (cold)" in ev:
        phase_label(ev["unload"], ev["reload (cold)"], "unloaded (idle)")

    # ── boundary markers at the two moments that matter (split sides so the
    #    labels don't collide when the idle window is short)
    for name, disp, ha in [
        ("unload", "unload ", "right"),
        ("reload (cold)", " request → reload", "left"),
    ]:
        if name in ev:
            ax.axvline(ev[name], color=STYLE_CONFIG["text_color"], alpha=0.45,
                       linestyle=":", linewidth=1.3)
            ax.text(ev[name], floor - 0.06, disp, ha=ha, va="top",
                    fontsize=fs["text"] + 1, color=STYLE_CONFIG["text_color"],
                    alpha=0.85)

    ax.set_ylim(floor - 0.22, ytop + 0.22)
    setup_plot(fig, ax, f"Model Unload / Reload VRAM ({label})",
               xlabel="Time (seconds)", ylabel="GPU Memory (GB)")
    ax.legend(loc="center right", fontsize=fs["text"] + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  plot -> {output_path}")


def replot_from_saved():
    """Re-render plots from output_data/model_unload_results.json, no server."""
    import json
    path = os.path.join(SCRIPT_DIR, "output_data", "model_unload_results.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data["scenarios"]:
        df = pd.DataFrame(r["timeline"])
        stats = r["stats"]
        plot_path = os.path.join(SCRIPT_DIR, "output_plots", f"model_unload_{r['label']}.png")
        plot_timeline(df, r["events"], stats, stats["loaded_gpu_gb"],
                      stats["floor_after_unload_gb"], plot_path, r["label"])
    print("\nreplotted.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8880")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--format", default="wav", help="wav enables audio-length/RTF")
    ap.add_argument("--long-lines", type=int, default=30,
                    help="paragraphs of the source text for the longform run")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds to hold between phases (plot readability)")
    ap.add_argument("--replot", action="store_true",
                    help="re-render plots from saved results json (no server needed)")
    args = ap.parse_args()

    if args.replot:
        return replot_from_saved()

    results = [
        run_scenario("short", SHORT_TEXT, args),
        run_scenario("longform", load_longform(args.long_lines), args),
    ]

    save_json_results(
        {"timestamp": datetime.now().isoformat(), "scenarios": results},
        os.path.join(SCRIPT_DIR, "output_data", "model_unload_results.json"),
    )
    write_benchmark_stats(
        [{"title": f"Model Unload - {r['label']}", "stats": r["stats"]} for r in results],
        os.path.join(SCRIPT_DIR, "output_data", "model_unload_stats.txt"),
    )
    print("\ndone.")


if __name__ == "__main__":
    main()
