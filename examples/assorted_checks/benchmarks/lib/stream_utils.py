#!/usr/bin/env python3
import os
import time
import wave
from typing import Any, Dict, List, Callable, Optional

import pandas as pd
import scipy.io.wavfile as wavfile

from .shared_utils import save_json_results
from .shared_plotting import plot_timeline, plot_correlation
from .shared_benchmark_utils import enc, get_text_for_tokens


def check_audio_silence(audio_path: str) -> bool:
    """Check if audio file contains only silence"""
    sample_rate, audio_data = wavfile.read(audio_path)
    # Convert to float for RMS calculation
    audio_float = audio_data.astype(float)
    # Calculate RMS value
    rms = (audio_float**2).mean() ** 0.5
    # Define silence threshold (adjust if needed)
    SILENCE_THRESHOLD = 50.0
    return rms < SILENCE_THRESHOLD


def process_benchmark_results(
    all_results: List[Dict[str, Any]], token_sizes: List[int]
) -> Dict[str, Any]:
    """Process benchmark results and generate summary"""
    summary = {}
    for tokens in token_sizes:
        matching_results = [
            r for r in all_results if r["target_tokens"] == tokens and not r["error"]
        ]
        if matching_results:
            avg_first_chunk = sum(
                r["time_to_first_chunk"] for r in matching_results
            ) / len(matching_results)
            avg_total = sum(r["total_time"] for r in matching_results) / len(
                matching_results
            )
            avg_audio_length = sum(r["audio_length"] for r in matching_results) / len(
                matching_results
            )
            summary[tokens] = {
                "avg_time_to_first_chunk": round(avg_first_chunk, 3),
                "avg_total_time": round(avg_total, 3),
                "avg_audio_length": round(avg_audio_length, 3),
                "num_successful_runs": len(matching_results),
            }
    return summary


def save_benchmark_results(
    all_results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_data_dir: str,
    output_plots_dir: str,
    suffix: str,
    plot_title_suffix: str,
    prefix: str = "",
):
    """Save benchmark results and generate plots"""
    # Save results
    results_data = {
        "individual_runs": all_results,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json_results(
        results_data,
        os.path.join(output_data_dir, f"{prefix}first_token_benchmark{suffix}.json"),
    )

    # Create DataFrame for plotting
    df = pd.DataFrame(all_results)

    # Create plots
    plot_correlation(
        df,
        "target_tokens",
        "time_to_first_chunk",
        f"Time to First Audio vs Input Size {plot_title_suffix}",
        "Number of Input Tokens",
        "Time to First Audio (seconds)",
        os.path.join(output_plots_dir, f"{prefix}first_token_latency{suffix}.png"),
    )

    plot_correlation(
        df,
        "target_tokens",
        "total_time",
        f"Total Time vs Input Size {plot_title_suffix}",
        "Number of Input Tokens",
        "Total Time (seconds)",
        os.path.join(output_plots_dir, f"{prefix}total_time_latency{suffix}.png"),
    )

    plot_timeline(
        df,
        os.path.join(output_plots_dir, f"{prefix}first_token_timeline{suffix}.png"),
        suffix=plot_title_suffix,
    )


def run_benchmark(
    measure_func: Callable,
    output_dir: str,
    output_data_dir: str,
    output_plots_dir: str,
    suffix: str = "",
    plot_title_suffix: str = "",
    num_runs: int = 5,
    client=None,
    prefix="",
):
    """Run benchmark with the given measurement function"""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    # Load sample text
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(
        os.path.join(script_dir, "the_time_machine_hg_wells.txt"), "r", encoding="utf-8"
    ) as f:
        text = f.read()

    # Test specific token counts
    token_sizes = [10, 50, 100, 250, 500]
    all_results = []
    silent_files = []

    for tokens in token_sizes:
        print(
            f"\nTesting {tokens} tokens{' ' + plot_title_suffix if plot_title_suffix else ''}"
        )
        test_text = get_text_for_tokens(text, tokens)
        actual_tokens = len(enc.encode(test_text))
        print(f"Text preview: {test_text[:50]}...")

        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...")
            result = measure_func(test_text, output_dir, tokens, i + 1)
            result["target_tokens"] = tokens
            result["actual_tokens"] = actual_tokens
            result["run_number"] = i + 1

            # Handle time to first audio
            first_chunk = result.get("time_to_first_chunk")
            print(
                f"Time to First Audio: {f'{first_chunk:.3f}s' if first_chunk is not None else 'N/A'}"
            )

            # Handle total time
            total_time = result.get("total_time")
            print(
                f"Time to Save Complete: {f'{total_time:.3f}s' if total_time is not None else 'N/A'}"
            )

            # Handle audio length
            audio_length = result.get("audio_length")
            print(
                f"Audio length: {f'{audio_length:.3f}s' if audio_length is not None else 'N/A'}"
            )
            # Calculate streaming overhead only if both values exist
            if total_time is not None and first_chunk is not None:
                print(f"Streaming overhead: {(total_time - first_chunk):.3f}s")
            else:
                print("Streaming overhead: N/A")

            if result["error"]:
                print(f"Error: {result['error']}")
            elif result["audio_path"] and check_audio_silence(result["audio_path"]):
                silent_files.append(result["audio_path"])

            all_results.append(result)

    # Process and save results
    summary = process_benchmark_results(all_results, token_sizes)
    save_benchmark_results(
        all_results,
        summary,
        output_data_dir,
        output_plots_dir,
        suffix,
        plot_title_suffix,
    )

    # Print paths
    print("\nResults and plots saved to:")
    print(
        f"- {os.path.join(output_data_dir, f'{prefix}first_token_benchmark{suffix}.json')}"
    )
    print(
        f"- {os.path.join(output_plots_dir, f'{prefix}first_token_latency{suffix}.png')}"
    )
    print(
        f"- {os.path.join(output_plots_dir, f'{prefix}total_time_latency{suffix}.png')}"
    )
    print(
        f"- {os.path.join(output_plots_dir, f'{prefix}first_token_timeline{suffix}.png')}"
    )

    # Print silence check summary
    if silent_files:
        print("\nWARNING: The following files contain only silence:")
        for file in silent_files:
            print(f"- {file}")
    else:
        print("\nAll generated audio files contain valid audio content.")
