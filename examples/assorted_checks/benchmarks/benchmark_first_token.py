#!/usr/bin/env python3
import os
import json
import time

import numpy as np
import pandas as pd
import requests
from lib.shared_utils import save_json_results
from lib.shared_plotting import plot_timeline, plot_correlation
from lib.shared_benchmark_utils import enc, get_text_for_tokens


def measure_first_token(
    text: str, output_dir: str, tokens: int, run_number: int
) -> dict:
    """Measure time to audio via API calls and save the audio output"""
    results = {
        "text_length": len(text),
        "token_count": len(enc.encode(text)),
        "total_time": None,
        "time_to_first_chunk": None,
        "error": None,
        "audio_path": None,
        "audio_length": None,  # Length of output audio in seconds
    }

    try:
        start_time = time.time()

        # Make request without streaming
        response = requests.post(
            "http://localhost:8880/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": "af_heart",
                "response_format": "wav",
                "stream": False,
            },
            timeout=1800,
        )
        response.raise_for_status()

        # Save complete audio
        audio_filename = f"benchmark_tokens{tokens}_run{run_number}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        results["audio_path"] = audio_path

        content = response.content
        with open(audio_path, "wb") as f:
            f.write(content)

        # Calculate audio length using scipy
        import scipy.io.wavfile as wavfile

        sample_rate, audio_data = wavfile.read(audio_path)
        results["audio_length"] = len(audio_data) / sample_rate  # Length in seconds
        results["time_to_first_chunk"] = time.time() - start_time

        results["total_time"] = time.time() - start_time
        return results

    except Exception as e:
        results["error"] = str(e)
        return results


def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output_audio")
    output_data_dir = os.path.join(script_dir, "output_data")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

    # Load sample text
    with open(
        os.path.join(script_dir, "the_time_machine_hg_wells.txt"), "r", encoding="utf-8"
    ) as f:
        text = f.read()

    # Test specific token counts
    token_sizes = [10, 25, 50, 100, 200, 500]
    all_results = []

    for tokens in token_sizes:
        print(f"\nTesting {tokens} tokens")
        test_text = get_text_for_tokens(text, tokens)
        actual_tokens = len(enc.encode(test_text))
        print(f"Text preview: {test_text[:50]}...")

        # Run test 3 times for each size to get average
        for i in range(5):
            print(f"Run {i+1}/3...")
            result = measure_first_token(test_text, output_dir, tokens, i + 1)
            result["target_tokens"] = tokens
            result["actual_tokens"] = actual_tokens
            result["run_number"] = i + 1

            print(f"Time to Audio: {result.get('time_to_first_chunk', 'N/A'):.3f}s")
            print(f"Total time: {result.get('total_time', 'N/A'):.3f}s")

            if result["error"]:
                print(f"Error: {result['error']}")

            all_results.append(result)

    # Calculate averages per token size
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

    # Save results
    # Save results
    results_data = {
        "individual_runs": all_results,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json_results(
        results_data, os.path.join(output_data_dir, "first_token_benchmark.json")
    )

    # Create plot directory if it doesn't exist
    output_plots_dir = os.path.join(script_dir, "output_plots")
    os.makedirs(output_plots_dir, exist_ok=True)

    # Create DataFrame for plotting
    df = pd.DataFrame(all_results)

    # Create both plots
    plot_correlation(
        df,
        "target_tokens",
        "time_to_first_chunk",
        "Time to Audio vs Input Size",
        "Number of Input Tokens",
        "Time to Audio (seconds)",
        os.path.join(output_plots_dir, "first_token_latency.png"),
    )

    plot_timeline(df, os.path.join(output_plots_dir, "first_token_timeline.png"))

    print("\nResults and plots saved to:")
    print(f"- {os.path.join(output_data_dir, 'first_token_benchmark.json')}")
    print(f"- {os.path.join(output_plots_dir, 'first_token_latency.png')}")
    print(f"- {os.path.join(output_plots_dir, 'first_token_timeline.png')}")


if __name__ == "__main__":
    main()
