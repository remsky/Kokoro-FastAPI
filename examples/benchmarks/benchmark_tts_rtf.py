import os
import json
import time
import subprocess
from datetime import datetime

import pandas as pd
import psutil
import seaborn as sns
import requests
import tiktoken
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

enc = tiktoken.get_encoding("cl100k_base")


def setup_plot(fig, ax, title):
    """Configure plot styling"""
    ax.grid(True, linestyle="--", alpha=0.3, color="#ffffff")
    ax.set_title(title, pad=20, fontsize=16, fontweight="bold", color="#ffffff")
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight="medium", color="#ffffff")
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight="medium", color="#ffffff")
    ax.tick_params(labelsize=12, colors="#ffffff")

    for spine in ax.spines.values():
        spine.set_color("#ffffff")
        spine.set_alpha(0.3)
        spine.set_linewidth(0.5)

    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    return fig, ax


def get_text_for_tokens(text: str, num_tokens: int) -> str:
    """Get a slice of text that contains exactly num_tokens tokens"""
    tokens = enc.encode(text)
    if num_tokens > len(tokens):
        return text
    return enc.decode(tokens[:num_tokens])


def get_audio_length(audio_data: bytes) -> float:
    """Get audio length in seconds from bytes data"""
    temp_path = "examples/benchmarks/output/temp.wav"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(audio_data)

    try:
        rate, data = wavfile.read(temp_path)
        return len(data) / rate
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_gpu_memory():
    """Get GPU memory usage using nvidia-smi"""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return float(result.decode("utf-8").strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_system_metrics():
    """Get current system metrics"""
    # Get per-CPU percentages and calculate average
    cpu_percentages = psutil.cpu_percent(percpu=True)
    avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": round(avg_cpu, 2),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": psutil.virtual_memory().used / (1024**3),
    }

    gpu_mem = get_gpu_memory()
    if gpu_mem is not None:
        metrics["gpu_memory_used"] = gpu_mem

    return metrics


def real_time_factor(processing_time: float, audio_length: float, decimals: int = 2) -> float:
    """Calculate Real-Time Factor (RTF) as processing-time / length-of-audio"""
    rtf = processing_time / audio_length
    return round(rtf, decimals)


def make_tts_request(text: str, timeout: int = 1800) -> tuple[float, float]:
    """Make TTS request using OpenAI-compatible endpoint and return processing time and output length"""
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8880/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": "af",
                "response_format": "wav",
            },
            timeout=timeout,
        )
        response.raise_for_status()

        processing_time = round(time.time() - start_time, 2)
        audio_length = round(get_audio_length(response.content), 2)

        # Save the audio file
        token_count = len(enc.encode(text))
        output_file = f"examples/benchmarks/output/chunk_{token_count}_tokens.wav"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Saved audio to {output_file}")

        return processing_time, audio_length

    except requests.exceptions.RequestException as e:
        print(f"Error making request for text: {text[:50]}... Error: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {str(e)}")
        return None, None


def plot_system_metrics(metrics_data):
    """Create plots for system metrics over time"""
    df = pd.DataFrame(metrics_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    elapsed_time = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    baseline_cpu = df["cpu_percent"].iloc[0]
    baseline_ram = df["ram_used_gb"].iloc[0]
    baseline_gpu = df["gpu_memory_used"].iloc[0] / 1024 if "gpu_memory_used" in df.columns else None

    if "gpu_memory_used" in df.columns:
        df["gpu_memory_gb"] = df["gpu_memory_used"] / 1024

    plt.style.use("dark_background")

    has_gpu = "gpu_memory_used" in df.columns
    num_plots = 3 if has_gpu else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots))
    fig.patch.set_facecolor("#1a1a2e")

    window = min(5, len(df) // 2)

    # Plot CPU Usage
    smoothed_cpu = df["cpu_percent"].rolling(window=window, center=True).mean()
    sns.lineplot(x=elapsed_time, y=smoothed_cpu, ax=axes[0], color="#ff2a6d", linewidth=2)
    axes[0].axhline(y=baseline_cpu, color="#05d9e8", linestyle="--", alpha=0.5, label="Baseline")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("CPU Usage (%)")
    axes[0].set_title("CPU Usage Over Time")
    axes[0].set_ylim(0, max(df["cpu_percent"]) * 1.1)
    axes[0].legend()

    # Plot RAM Usage
    smoothed_ram = df["ram_used_gb"].rolling(window=window, center=True).mean()
    sns.lineplot(x=elapsed_time, y=smoothed_ram, ax=axes[1], color="#05d9e8", linewidth=2)
    axes[1].axhline(y=baseline_ram, color="#ff2a6d", linestyle="--", alpha=0.5, label="Baseline")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("RAM Usage (GB)")
    axes[1].set_title("RAM Usage Over Time")
    axes[1].set_ylim(0, max(df["ram_used_gb"]) * 1.1)
    axes[1].legend()

    # Plot GPU Memory if available
    if has_gpu:
        smoothed_gpu = df["gpu_memory_gb"].rolling(window=window, center=True).mean()
        sns.lineplot(x=elapsed_time, y=smoothed_gpu, ax=axes[2], color="#ff2a6d", linewidth=2)
        axes[2].axhline(y=baseline_gpu, color="#05d9e8", linestyle="--", alpha=0.5, label="Baseline")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("GPU Memory (GB)")
        axes[2].set_title("GPU Memory Usage Over Time")
        axes[2].set_ylim(0, max(df["gpu_memory_gb"]) * 1.1)
        axes[2].legend()

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_color("#ffffff")
            spine.set_alpha(0.3)

    plt.tight_layout()
    plt.savefig("examples/benchmarks/system_usage_rtf.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("examples/benchmarks/output", exist_ok=True)

    with open("examples/benchmarks/the_time_machine_hg_wells.txt", "r", encoding="utf-8") as f:
        text = f.read()

    total_tokens = len(enc.encode(text))
    print(f"Total tokens in file: {total_tokens}")

    # Generate token sizes with dense sampling at start
    dense_range = list(range(100, 1001, 100))
    token_sizes = sorted(list(set(dense_range)))
    print(f"Testing sizes: {token_sizes}")

    results = []
    system_metrics = []
    test_start_time = time.time()

    for num_tokens in token_sizes:
        chunk = get_text_for_tokens(text, num_tokens)
        actual_tokens = len(enc.encode(chunk))

        print(f"\nProcessing chunk with {actual_tokens} tokens:")
        print(f"Text preview: {chunk[:100]}...")

        system_metrics.append(get_system_metrics())

        processing_time, audio_length = make_tts_request(chunk)
        if processing_time is None or audio_length is None:
            print("Breaking loop due to error")
            break

        system_metrics.append(get_system_metrics())

        # Calculate RTF using the correct formula
        rtf = real_time_factor(processing_time, audio_length)
        
        results.append({
            "tokens": actual_tokens,
            "processing_time": processing_time,
            "output_length": audio_length,
            "rtf": rtf,
            "elapsed_time": round(time.time() - test_start_time, 2),
        })

        with open("examples/benchmarks/benchmark_results_rtf.json", "w") as f:
            json.dump({"results": results, "system_metrics": system_metrics}, f, indent=2)

    df = pd.DataFrame(results)
    if df.empty:
        print("No data to plot")
        return

    df["tokens_per_second"] = df["tokens"] / df["processing_time"]

    with open("examples/benchmarks/benchmark_stats_rtf.txt", "w") as f:
        f.write("=== Benchmark Statistics (with correct RTF) ===\n\n")

        f.write("Overall Stats:\n")
        f.write(f"Total tokens processed: {df['tokens'].sum()}\n")
        f.write(f"Total audio generated: {df['output_length'].sum():.2f}s\n")
        f.write(f"Total test duration: {df['elapsed_time'].max():.2f}s\n")
        f.write(f"Average processing rate: {df['tokens_per_second'].mean():.2f} tokens/second\n")
        f.write(f"Average RTF: {df['rtf'].mean():.2f}x\n\n")

        f.write("Per-chunk Stats:\n")
        f.write(f"Average chunk size: {df['tokens'].mean():.2f} tokens\n")
        f.write(f"Min chunk size: {df['tokens'].min():.2f} tokens\n")
        f.write(f"Max chunk size: {df['tokens'].max():.2f} tokens\n")
        f.write(f"Average processing time: {df['processing_time'].mean():.2f}s\n")
        f.write(f"Average output length: {df['output_length'].mean():.2f}s\n\n")

        f.write("Performance Ranges:\n")
        f.write(f"Processing rate range: {df['tokens_per_second'].min():.2f} - {df['tokens_per_second'].max():.2f} tokens/second\n")
        f.write(f"RTF range: {df['rtf'].min():.2f}x - {df['rtf'].max():.2f}x\n")

    plt.style.use("dark_background")

    # Plot Processing Time vs Token Count
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x="tokens", y="processing_time", s=100, alpha=0.6, color="#ff2a6d")
    sns.regplot(data=df, x="tokens", y="processing_time", scatter=False, color="#05d9e8", line_kws={"linewidth": 2})
    corr = df["tokens"].corr(df["processing_time"])
    plt.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax.transAxes, fontsize=10, color="#ffffff",
             bbox=dict(facecolor="#1a1a2e", edgecolor="#ffffff", alpha=0.7))
    setup_plot(fig, ax, "Processing Time vs Input Size")
    ax.set_xlabel("Number of Input Tokens")
    ax.set_ylabel("Processing Time (seconds)")
    plt.savefig("examples/benchmarks/processing_time_rtf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot RTF vs Token Count
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x="tokens", y="rtf", s=100, alpha=0.6, color="#ff2a6d")
    sns.regplot(data=df, x="tokens", y="rtf", scatter=False, color="#05d9e8", line_kws={"linewidth": 2})
    corr = df["tokens"].corr(df["rtf"])
    plt.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax.transAxes, fontsize=10, color="#ffffff",
             bbox=dict(facecolor="#1a1a2e", edgecolor="#ffffff", alpha=0.7))
    setup_plot(fig, ax, "Real-Time Factor vs Input Size")
    ax.set_xlabel("Number of Input Tokens")
    ax.set_ylabel("Real-Time Factor (processing time / audio length)")
    plt.savefig("examples/benchmarks/realtime_factor_rtf.png", dpi=300, bbox_inches="tight")
    plt.close()

    plot_system_metrics(system_metrics)

    print("\nResults saved to:")
    print("- examples/benchmarks/benchmark_results_rtf.json")
    print("- examples/benchmarks/benchmark_stats_rtf.txt")
    print("- examples/benchmarks/processing_time_rtf.png")
    print("- examples/benchmarks/realtime_factor_rtf.png")
    print("- examples/benchmarks/system_usage_rtf.png")
    print("\nAudio files saved in examples/benchmarks/output/")


if __name__ == "__main__":
    main()