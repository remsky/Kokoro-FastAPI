#!/usr/bin/env python3
import os
import sys
import json
import time
import queue
import threading
from datetime import datetime

import pandas as pd
from lib.shared_utils import (
    real_time_factor,
    save_json_results,
    get_system_metrics,
    write_benchmark_stats,
)
from lib.shared_plotting import plot_correlation, plot_system_metrics
from lib.shared_benchmark_utils import (
    enc,
    make_tts_request,
    get_text_for_tokens,
    generate_token_sizes,
)


class SystemMonitor:
    def __init__(self, interval=1.0):
        """Rough system tracker: Not always accurate"""
        self.interval = interval
        self.metrics_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.metrics_timeline = []
        self.start_time = None

    def _monitor_loop(self):
        """Background thread function to collect system metrics."""
        while not self.stop_event.is_set():
            metrics = get_system_metrics()
            metrics["relative_time"] = time.time() - self.start_time
            self.metrics_queue.put(metrics)
            time.sleep(self.interval)

    def start(self):
        """Start the monitoring thread."""
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        """Stop the monitoring thread and collect final metrics."""
        self.stop_event.set()
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=2)

        # Collect all metrics from queue
        while True:
            try:
                metrics = self.metrics_queue.get_nowait()
                self.metrics_timeline.append(metrics)
            except queue.Empty:
                break

        return self.metrics_timeline


def main():
    # Initialize system monitor
    monitor = SystemMonitor(interval=1.0)  # 1 second interval
    # Set prefix for output files (e.g. "gpu", "cpu", "onnx", etc.)
    prefix = os.environ.get("BENCH_PREFIX", "gpu")
    # Generate token sizes
    if "gpu" in prefix:
        token_sizes = generate_token_sizes(
            max_tokens=1000, dense_step=150, dense_max=1000, sparse_step=1000
        )
    elif "cpu" in prefix:
        token_sizes = generate_token_sizes(
            max_tokens=1000, dense_step=100, dense_max=500, sparse_step=250
        )
    else:
        token_sizes = generate_token_sizes(max_tokens=3000)

    # Set up paths relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output_audio")
    output_data_dir = os.path.join(script_dir, "output_data")
    output_plots_dir = os.path.join(script_dir, "output_plots")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    # Function to prefix filenames
    def prefix_path(path: str, filename: str) -> str:
        if prefix:
            filename = f"{prefix}_{filename}"
        return os.path.join(path, filename)

    with open(
        os.path.join(script_dir, "the_time_machine_hg_wells.txt"), "r", encoding="utf-8"
    ) as f:
        text = f.read()

    total_tokens = len(enc.encode(text))
    print(f"Total tokens in file: {total_tokens}")

    print(f"Testing sizes: {token_sizes}")

    results = []
    test_start_time = time.time()

    # Start system monitoring
    monitor.start()

    for num_tokens in token_sizes:
        chunk = get_text_for_tokens(text, num_tokens)
        actual_tokens = len(enc.encode(chunk))

        print(f"\nProcessing chunk with {actual_tokens} tokens:")
        print(f"Text preview: {chunk[:100]}...")

        processing_time, audio_length = make_tts_request(
            chunk,
            output_dir=output_dir,
            prefix=prefix,
            stream=False,  # Use non-streaming mode for RTF benchmarking
        )
        if processing_time is None or audio_length is None:
            print("Breaking loop due to error")
            break

        # Calculate RTF using the correct formula
        rtf = real_time_factor(processing_time, audio_length)
        print(f"Real-Time Factor: {rtf:.5f}")

        results.append(
            {
                "tokens": actual_tokens,
                "processing_time": processing_time,
                "output_length": audio_length,
                "rtf": rtf,
                "elapsed_time": round(time.time() - test_start_time, 5),
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        print("No data to plot")
        return

    df["tokens_per_second"] = df["tokens"] / df["processing_time"]

    # Write benchmark stats
    stats = [
        {
            "title": "Benchmark Statistics (with correct RTF)",
            "stats": {
                "Total tokens processed": df["tokens"].sum(),
                "Total audio generated (s)": df["output_length"].sum(),
                "Total test duration (s)": df["elapsed_time"].max(),
                "Average processing rate (tokens/s)": df["tokens_per_second"].mean(),
                "Average RTF": df["rtf"].mean(),
                "Average Real Time Speed": 1 / df["rtf"].mean(),
            },
        },
        {
            "title": "Per-chunk Stats",
            "stats": {
                "Average chunk size (tokens)": df["tokens"].mean(),
                "Min chunk size (tokens)": df["tokens"].min(),
                "Max chunk size (tokens)": df["tokens"].max(),
                "Average processing time (s)": df["processing_time"].mean(),
                "Average output length (s)": df["output_length"].mean(),
            },
        },
        {
            "title": "Performance Ranges",
            "stats": {
                "Processing rate range (tokens/s)": f"{df['tokens_per_second'].min():.2f} - {df['tokens_per_second'].max():.2f}",
                "RTF range": f"{df['rtf'].min():.2f}x - {df['rtf'].max():.2f}x",
                "Real Time Speed range": f"{1/df['rtf'].max():.2f}x - {1/df['rtf'].min():.2f}x",
            },
        },
    ]
    write_benchmark_stats(
        stats, prefix_path(output_data_dir, "benchmark_stats_rtf.txt")
    )

    # Plot Processing Time vs Token Count
    plot_correlation(
        df,
        "tokens",
        "processing_time",
        "Processing Time vs Input Size",
        "Number of Input Tokens",
        "Processing Time (seconds)",
        prefix_path(output_plots_dir, "processing_time_rtf.png"),
    )

    # Plot RTF vs Token Count
    plot_correlation(
        df,
        "tokens",
        "rtf",
        "Real-Time Factor vs Input Size",
        "Number of Input Tokens",
        "Real-Time Factor (processing time / audio length)",
        prefix_path(output_plots_dir, "realtime_factor_rtf.png"),
    )

    # Stop monitoring and get final metrics
    final_metrics = monitor.stop()

    # Convert metrics timeline to DataFrame for stats
    metrics_df = pd.DataFrame(final_metrics)

    # Add system usage stats
    if not metrics_df.empty:
        stats.append(
            {
                "title": "System Usage Statistics",
                "stats": {
                    "Peak CPU Usage (%)": metrics_df["cpu_percent"].max(),
                    "Avg CPU Usage (%)": metrics_df["cpu_percent"].mean(),
                    "Peak RAM Usage (%)": metrics_df["ram_percent"].max(),
                    "Avg RAM Usage (%)": metrics_df["ram_percent"].mean(),
                    "Peak RAM Used (GB)": metrics_df["ram_used_gb"].max(),
                    "Avg RAM Used (GB)": metrics_df["ram_used_gb"].mean(),
                },
            }
        )
        if "gpu_memory_used" in metrics_df:
            stats[-1]["stats"].update(
                {
                    "Peak GPU Memory (MB)": metrics_df["gpu_memory_used"].max(),
                    "Avg GPU Memory (MB)": metrics_df["gpu_memory_used"].mean(),
                }
            )

    # Plot system metrics
    plot_system_metrics(
        final_metrics, prefix_path(output_plots_dir, "system_usage_rtf.png")
    )

    # Save final results
    save_json_results(
        {
            "results": results,
            "system_metrics": final_metrics,
            "test_duration": time.time() - test_start_time,
        },
        prefix_path(output_data_dir, "benchmark_results_rtf.json"),
    )

    print("\nResults saved to:")
    print(f"- {prefix_path(output_data_dir, 'benchmark_results_rtf.json')}")
    print(f"- {prefix_path(output_data_dir, 'benchmark_stats_rtf.txt')}")
    print(f"- {prefix_path(output_plots_dir, 'processing_time_rtf.png')}")
    print(f"- {prefix_path(output_plots_dir, 'realtime_factor_rtf.png')}")
    print(f"- {prefix_path(output_plots_dir, 'system_usage_rtf.png')}")
    print(f"\nAudio files saved in {output_dir} with prefix: {prefix or '(none)'}")


if __name__ == "__main__":
    main()
