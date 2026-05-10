import argparse
from typing import Any, Dict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def validate_tts(wav_path: str) -> dict:
    """
    Validation checks for TTS-generated audio files to detect common artifacts.
    """
    try:
        # Load and process audio
        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / sr
        issues = []

        # Basic quality checks
        abs_audio = np.abs(audio)
        stats = {
            "rms": float(np.sqrt(np.mean(audio**2))),
            "peak": float(np.max(abs_audio)),
            "dc_offset": float(np.mean(audio)),
        }

        clip_count = np.sum(abs_audio >= 0.99)
        clip_percent = (clip_count / len(audio)) * 100

        if duration < 0.1:
            issues.append(
                "WARNING: Audio is suspiciously short - possible failed generation"
            )

        if stats["peak"] >= 1.0:
            if clip_percent > 1.0:
                issues.append(
                    f"WARNING: Significant clipping detected ({clip_percent:.2e}% of samples)"
                )
            elif clip_percent > 0.01:
                issues.append(
                    f"INFO: Minor peak limiting detected ({clip_percent:.2e}% of samples)"
                )

        if stats["rms"] < 0.01:
            issues.append("WARNING: Audio is very quiet - possible failed generation")

        if abs(stats["dc_offset"]) > 0.1:
            issues.append(f"WARNING: High DC offset ({stats['dc_offset']:.3f})")

        # Check for long silence gaps
        eps = np.finfo(float).eps
        db = 20 * np.log10(abs_audio + eps)
        silence_threshold = -45  # dB
        min_silence = 2.0  # seconds
        window_size = int(min_silence * sr)
        silence_count = 0
        last_silence = -1

        start_idx = int(0.2 * sr)  # Skip first 0.2s
        for i in tqdm(
            range(start_idx, len(db) - window_size, window_size),
            desc="Checking for silence",
        ):
            window = db[i : i + window_size]
            if np.mean(window) < silence_threshold:
                silent_ratio = np.mean(window < silence_threshold)
                if silent_ratio > 0.9:
                    if last_silence == -1 or (i / sr - last_silence) > 2.0:
                        silence_count += 1
                        last_silence = i / sr
                        issues.append(
                            f"WARNING: Long silence detected at {i/sr:.2f}s (duration: {min_silence:.1f}s)"
                        )

        if silence_count > 2:
            issues.append(
                f"WARNING: Multiple long silences found ({silence_count} total)"
            )

        # Detect audio artifacts
        diff = np.diff(audio)
        abs_diff = np.abs(diff)
        window_size = min(int(0.005 * sr), 256)
        window = np.ones(window_size) / window_size
        local_avg_diff = np.convolve(abs_diff, window, mode="same")

        spikes = (abs_diff > (10 * local_avg_diff)) & (abs_diff > 0.1)
        artifact_indices = np.nonzero(spikes)[0]

        artifacts = []
        if len(artifact_indices) > 0:
            gaps = np.diff(artifact_indices)
            min_gap = int(0.005 * sr)
            break_points = np.nonzero(gaps > min_gap)[0] + 1
            groups = np.split(artifact_indices, break_points)

            for group in groups:
                if len(group) >= 5:
                    severity = np.max(abs_diff[group])
                    if severity > 0.2:
                        center_idx = group[len(group) // 2]
                        artifacts.append(
                            {
                                "time": float(
                                    center_idx / sr
                                ),  # Ensure float for consistent timing
                                "severity": float(severity),
                            }
                        )
                        issues.append(
                            f"WARNING: Audio discontinuity at {center_idx/sr:.3f}s "
                            f"(severity: {severity:.3f})"
                        )

        # Check for repeated speech segments
        for chunk_duration in tqdm(
            [0.5, 2.5, 5.0, 10.0], desc="Checking for repeated speech"
        ):
            chunk_size = int(chunk_duration * sr)
            overlap = int(0.2 * chunk_size)

            for i in range(0, len(audio) - 2 * chunk_size, overlap):
                chunk1 = audio[i : i + chunk_size]
                chunk2 = audio[i + chunk_size : i + 2 * chunk_size]

                if np.mean(np.abs(chunk1)) < 0.01 or np.mean(np.abs(chunk2)) < 0.01:
                    continue

                try:
                    correlation = np.corrcoef(chunk1, chunk2)[0, 1]
                    if not np.isnan(correlation) and correlation > 0.92:
                        issues.append(
                            f"WARNING: Possible repeated speech at {i/sr:.1f}s "
                            f"(~{int(chunk_duration*160/60):d} words, correlation: {correlation:.3f})"
                        )
                        break
                except:
                    continue

        return {
            "file": wav_path,
            "duration": f"{duration:.2f}s",
            "sample_rate": sr,
            "peak_amplitude": f"{stats['peak']:.3f}",
            "rms_level": f"{stats['rms']:.3f}",
            "dc_offset": f"{stats['dc_offset']:.3f}",
            "artifact_count": len(artifacts),
            "artifact_locations": [a["time"] for a in artifacts],
            "artifact_severities": [a["severity"] for a in artifacts],
            "issues": issues,
            "valid": len(issues) == 0,
        }

    except Exception as e:
        return {"file": wav_path, "error": str(e), "valid": False}


def generate_analysis_plots(
    wav_path: str, output_dir: str, validation_result: Dict[str, Any]
):
    """
    Generate analysis plots for audio file with time-aligned visualizations.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    # Load audio
    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Create figure with shared x-axis
    fig = plt.figure(figsize=(15, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[1.2, 0.8], hspace=0.1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Calculate spectrogram
    nperseg = 2048
    noverlap = 1536
    f, t, Sxx = spectrogram(
        audio, sr, nperseg=nperseg, noverlap=noverlap, window="hann", scaling="spectrum"
    )

    # Plot spectrogram
    im = ax1.pcolormesh(
        t,
        f,
        10 * np.log10(Sxx + 1e-10),
        shading="gouraud",
        cmap="viridis",
        vmin=-100,
        vmax=-20,
    )
    ax1.set_ylabel("Frequency [Hz]", fontsize=10)
    cbar = plt.colorbar(im, ax=ax1, label="dB")
    ax1.set_title("Spectrogram", pad=10, fontsize=12)

    # Plot waveform with exact time alignment
    times = np.arange(len(audio)) / sr
    ax2.plot(times, audio, color="#2E5596", alpha=0.7, linewidth=0.5, label="Audio")
    ax2.set_ylabel("Amplitude", fontsize=10)
    ax2.set_xlabel("Time [sec]", fontsize=10)
    ax2.grid(True, alpha=0.2)

    # Add artifact markers
    if (
        "artifact_locations" in validation_result
        and validation_result["artifact_locations"]
    ):
        for loc in validation_result["artifact_locations"]:
            ax1.axvline(x=loc, color="red", alpha=0.7, linewidth=2)
            ax2.axvline(
                x=loc, color="red", alpha=0.7, linewidth=2, label="Detected Artifacts"
            )

        # Add legend to both plots
        if len(validation_result["artifact_locations"]) > 0:
            ax1.plot([], [], color="red", linewidth=2, label="Detected Artifacts")
            ax1.legend(loc="upper right", fontsize=8)
            # Only add unique labels to legend
            handles, labels = ax2.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            ax2.legend(
                unique_labels.values(),
                unique_labels.keys(),
                loc="upper right",
                fontsize=8,
            )

    # Set common x limits
    xlim = (0, len(audio) / sr)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    og_filename = Path(wav_path).name.split(".")[0]
    # Save plot
    plt.savefig(
        Path(output_dir) / f"{og_filename}_audio_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    wav_file = str(Path(__file__).parent / "benchmarks" / "output_audio" / "chunk_600_tokens.wav")
    silent = False

    print(f"\n\n Processing:\n\t{wav_file}")
    result = validate_tts(wav_file)
    if not silent:
        wav_root_dir = Path(wav_file).parent
        generate_analysis_plots(wav_file, wav_root_dir, result)

    print(f"\nValidating: {result['file']}")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Duration: {result['duration']}")
        print(f"Sample Rate: {result['sample_rate']} Hz")
        print(f"Peak Amplitude: {result['peak_amplitude']}")
        print(f"RMS Level: {result['rms_level']}")
        print(f"DC Offset: {result['dc_offset']}")
        print(f"Detected Artifacts: {result['artifact_count']}")

        if result["issues"]:
            print("\nIssues Found:")
            for issue in result["issues"]:
                print(f"- {issue}")
        else:
            print("\nNo issues found")
