"""Test script to generate and analyze different audio formats"""

import os
import time
from pathlib import Path

import numpy as np
import openai
import requests
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile

SAMPLE_TEXT = """
That is the germ of my great discovery. But you are wrong to say that we cannot move about in Time.
"""

# Configure OpenAI client
client = openai.OpenAI(
    timeout=60,
    api_key="notneeded",  # API key not required for our endpoint
    base_url="http://localhost:8880/v1",  # Point to our local server with v1 prefix
)


def setup_plot(fig, ax, title):
    """Configure plot styling"""
    # Improve grid
    ax.grid(True, linestyle="--", alpha=0.3, color="#ffffff")

    # Set title and labels with better fonts and more padding
    ax.set_title(title, pad=40, fontsize=16, fontweight="bold", color="#ffffff")
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight="medium", color="#ffffff")
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight="medium", color="#ffffff")

    # Improve tick labels
    ax.tick_params(labelsize=12, colors="#ffffff")

    # Style spines
    for spine in ax.spines.values():
        spine.set_color("#ffffff")
        spine.set_alpha(0.3)
        spine.set_linewidth(0.5)

    # Set background colors
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    return fig, ax


def plot_format_comparison(stats: list, output_dir: str):
    """Plot audio format comparison"""
    plt.style.use("dark_background")

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 16))  # Taller figure to accommodate bottom legend
    fig.patch.set_facecolor("#1a1a2e")

    # Create subplot grid with balanced spacing for waveforms
    gs_waves = plt.GridSpec(
        len(stats), 1, left=0.15, right=0.85, top=0.9, bottom=0.35, hspace=0.4
    )

    # Plot waveforms for each format
    for i, stat in enumerate(stats):
        format_name = stat["format"].upper()
        try:
            file_path = os.path.join(output_dir, f"test_audio.{stat['format']}")

            if stat["format"] == "wav":
                # Use scipy.io.wavfile for WAV files
                sr, data = wavfile.read(file_path)
                data = data.astype(np.float32) / 32768.0  # Convert to float [-1, 1]
            elif stat["format"] == "pcm":
                # Read raw 16-bit signed little-endian PCM data at 24kHz
                data = np.frombuffer(
                    open(file_path, "rb").read(), dtype="<i2"
                )  # '<i2' means little-endian 16-bit signed int
                data = data.astype(np.float32) / 32768.0  # Convert to float [-1, 1]
                sr = 24000  # Known sample rate for our endpoint
            else:
                # Use soundfile for other formats (mp3, opus, flac)
                data, sr = sf.read(file_path)

            # Plot waveform with consistent normalization
            ax = plt.subplot(gs_waves[i])
            time = np.arange(len(data)) / sr
            plt.plot(time, data, linewidth=0.5, color="#ff2a6d")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("")
            ax.set_ylim(-1.1, 1.1)
            setup_plot(fig, ax, f"Waveform: {format_name}")
        except Exception as e:
            print(f"Error plotting waveform for {format_name}: {e}")

    # Colors for formats
    colors = ["#ff2a6d", "#05d9e8", "#d1f7ff", "#ff9e00", "#8c1eff"]

    # Create three subplots for metrics with more space at bottom for legend
    gs_bottom = plt.GridSpec(
        1,
        3,
        left=0.15,
        right=0.85,
        bottom=0.15,
        top=0.25,  # More bottom space for legend
        wspace=0.3,
    )

    # File Size subplot
    ax1 = plt.subplot(gs_bottom[0])
    metrics1 = [("File Size", [s["file_size_kb"] for s in stats], "KB")]

    # Duration and Gen Time subplot
    ax2 = plt.subplot(gs_bottom[1])
    metrics2 = [
        ("Duration", [s["duration_seconds"] for s in stats], "s"),
        ("Gen Time", [s["generation_time"] for s in stats], "s"),
    ]

    # Sample Rate subplot
    ax3 = plt.subplot(gs_bottom[2])
    metrics3 = [("Sample Rate", [s["sample_rate"] / 1000 for s in stats], "kHz")]

    def plot_grouped_bars(ax, metrics, show_legend=True):
        n_groups = len(metrics)
        n_formats = len(stats)
        # Use wider bars for time metrics
        bar_width = 0.175 if metrics == metrics2 else 0.1

        indices = np.arange(n_groups)

        # Get max value for y-axis scaling
        max_val = max(max(m[1]) for m in metrics)

        for i, (stat, color) in enumerate(zip(stats, colors)):
            values = [m[1][i] for m in metrics]
            # Reduce spacing between bars for time metrics
            spacing = 1.1 if metrics == metrics2 else 1.0
            offset = (i - n_formats / 2 + 0.5) * bar_width * spacing
            bars = ax.bar(
                indices + offset,
                values,
                bar_width,
                label=stat["format"].upper(),
                color=color,
                alpha=0.8,
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    color="white",
                    fontsize=10,
                )

        ax.set_xticks(indices)
        ax.set_xticklabels([f"{m[0]}\n({m[2]})" for m in metrics])

        # Set y-axis limits with some padding
        ax.set_ylim(0, max_val * 1.2)

        if show_legend:
            # Place legend at the bottom
            ax.legend(
                bbox_to_anchor=(1.8, -0.8),
                loc="center",
                facecolor="#1a1a2e",
                edgecolor="#ffffff",
                ncol=len(stats),
            )  # Show all formats in one row

    # Plot all three subplots with shared legend
    plot_grouped_bars(ax1, metrics1, show_legend=True)
    plot_grouped_bars(ax2, metrics2, show_legend=False)
    plot_grouped_bars(ax3, metrics3, show_legend=False)

    # Style all subplots
    setup_plot(fig, ax1, "File Size")
    setup_plot(fig, ax2, "Time Metrics")
    setup_plot(fig, ax3, "Sample Rate")

    # Add y-axis labels
    ax1.set_ylabel("Value")
    ax2.set_ylabel("Value")
    ax3.set_ylabel("Value")

    # Save the plot
    plt.savefig(os.path.join(output_dir, "format_comparison.png"), dpi=300)
    print(f"\nSaved format comparison plot to {output_dir}/format_comparison.png")


def get_audio_stats(file_path: str) -> dict:
    """Get audio file statistics"""
    file_size = os.path.getsize(file_path)
    file_size_kb = file_size / 1024  # Convert to KB
    format_name = Path(file_path).suffix[1:]

    if format_name == "wav":
        # Use scipy.io.wavfile for WAV files
        sample_rate, data = wavfile.read(file_path)
        data = data.astype(np.float32) / 32768.0  # Convert to float [-1, 1]
        duration = len(data) / sample_rate
        channels = 1 if len(data.shape) == 1 else data.shape[1]
    elif format_name == "pcm":
        # For PCM, read raw 16-bit signed little-endian PCM data at 24kHz
        data = np.frombuffer(
            open(file_path, "rb").read(), dtype="<i2"
        )  # '<i2' means little-endian 16-bit signed int
        data = data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        sample_rate = 24000  # Known sample rate for our endpoint
        duration = len(data) / sample_rate
        channels = 1
    else:
        # Use soundfile for other formats (mp3, opus, flac)
        data, sample_rate = sf.read(file_path)
        duration = len(data) / sample_rate
        channels = 1 if len(data.shape) == 1 else data.shape[1]

    # Calculate audio statistics
    stats = {
        "format": format_name,
        "file_size_kb": round(file_size_kb, 2),
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "min_amplitude": float(np.min(data)),
        "max_amplitude": float(np.max(data)),
        "mean_amplitude": float(np.mean(np.abs(data))),
        "rms_amplitude": float(np.sqrt(np.mean(np.square(data)))),
    }
    return stats


def main():
    """Generate and analyze audio in different formats"""
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "test_formats"
    output_dir.mkdir(exist_ok=True, parents=True)

    # First generate audio in each format using the API
    voice = "af_heart"  # Using default voice
    formats = ["wav", "mp3", "opus", "flac", "pcm"]
    stats = []

    for fmt in formats:
        output_path = output_dir / f"test_audio.{fmt}"
        print(f"\nGenerating {fmt.upper()} audio...")

        # Generate and save
        start_time = time.time()

        # Use requests with stream=False for consistent data handling
        response = requests.post(
            "http://localhost:8880/v1/audio/speech",
            json={
                "model": "kokoro",
                "voice": voice,
                "input": SAMPLE_TEXT,
                "response_format": fmt,
                "stream": False,  # Explicitly disable streaming to get single complete chunk
            },
            stream=False,
            headers={"Accept": f"audio/{fmt}"},  # Explicitly request audio format
        )
        generation_time = time.time() - start_time

        print(f"\nResponse headers for {fmt}:")
        for header, value in response.headers.items():
            print(f"{header}: {value}")
        print(f"Content length: {len(response.content)} bytes")
        print(f"First few bytes: {response.content[:20].hex()}")

        # Write the file and verify it was written correctly
        try:
            with open(output_path, "wb") as f:
                f.write(response.content)

            # Verify file was written
            if not output_path.exists():
                raise Exception(f"Failed to write {fmt} file")

            # Check file size matches content length
            written_size = output_path.stat().st_size
            if written_size != len(response.content):
                raise Exception(
                    f"File size mismatch: expected {len(response.content)} bytes, got {written_size}"
                )

            print(f"Successfully wrote {fmt} file")

        except Exception as e:
            print(f"Error writing {fmt} file: {e}")
            continue

        # Get stats
        file_stats = get_audio_stats(str(output_path))
        file_stats["generation_time"] = round(generation_time, 3)
        stats.append(file_stats)

    # Generate comparison plot
    plot_format_comparison(stats, str(output_dir))

    # Print detailed statistics
    print("\nDetailed Audio Statistics:")
    print("=" * 100)
    for stat in stats:
        print(f"\n{stat['format'].upper()} Format:")
        for key, value in sorted(stat.items()):
            if key not in ["format"]:  # Skip format as it's in the header
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
