"""Shared plotting utilities for benchmarks and tests."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Common style configurations
STYLE_CONFIG = {
    "background_color": "#1a1a2e",
    "primary_color": "#ff2a6d",
    "secondary_color": "#05d9e8",
    "grid_color": "#ffffff",
    "text_color": "#ffffff",
    "font_sizes": {"title": 16, "label": 14, "tick": 12, "text": 10},
}


def setup_plot(fig, ax, title, xlabel=None, ylabel=None):
    """Configure plot styling with consistent theme.

    Args:
        fig: matplotlib figure object
        ax: matplotlib axis object
        title: str, plot title
        xlabel: str, optional x-axis label
        ylabel: str, optional y-axis label

    Returns:
        tuple: (fig, ax) with applied styling
    """
    # Grid styling
    ax.grid(True, linestyle="--", alpha=0.3, color=STYLE_CONFIG["grid_color"])

    # Title and labels
    ax.set_title(
        title,
        pad=20,
        fontsize=STYLE_CONFIG["font_sizes"]["title"],
        fontweight="bold",
        color=STYLE_CONFIG["text_color"],
    )

    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=STYLE_CONFIG["font_sizes"]["label"],
            fontweight="medium",
            color=STYLE_CONFIG["text_color"],
        )
    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=STYLE_CONFIG["font_sizes"]["label"],
            fontweight="medium",
            color=STYLE_CONFIG["text_color"],
        )

    # Tick styling
    ax.tick_params(
        labelsize=STYLE_CONFIG["font_sizes"]["tick"], colors=STYLE_CONFIG["text_color"]
    )

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(STYLE_CONFIG["text_color"])
        spine.set_alpha(0.3)
        spine.set_linewidth(0.5)

    # Background colors
    ax.set_facecolor(STYLE_CONFIG["background_color"])
    fig.patch.set_facecolor(STYLE_CONFIG["background_color"])

    return fig, ax


def plot_system_metrics(metrics_data, output_path):
    """Create plots for system metrics over time.

    Args:
        metrics_data: list of dicts containing system metrics
        output_path: str, path to save the output plot
    """
    df = pd.DataFrame(metrics_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    elapsed_time = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    # Get baseline values
    baseline_cpu = df["cpu_percent"].iloc[0]
    baseline_ram = df["ram_used_gb"].iloc[0]
    baseline_gpu = (
        df["gpu_memory_used"].iloc[0] / 1024
        if "gpu_memory_used" in df.columns
        else None
    )

    # Convert GPU memory to GB if present
    if "gpu_memory_used" in df.columns:
        df["gpu_memory_gb"] = df["gpu_memory_used"] / 1024

    plt.style.use("dark_background")

    # Create subplots based on available metrics
    has_gpu = "gpu_memory_used" in df.columns
    num_plots = 3 if has_gpu else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots))
    fig.patch.set_facecolor(STYLE_CONFIG["background_color"])

    # Smoothing window
    window = min(5, len(df) // 2)

    # Plot CPU Usage
    smoothed_cpu = df["cpu_percent"].rolling(window=window, center=True).mean()
    sns.lineplot(
        x=elapsed_time,
        y=smoothed_cpu,
        ax=axes[0],
        color=STYLE_CONFIG["primary_color"],
        linewidth=2,
    )
    axes[0].axhline(
        y=baseline_cpu,
        color=STYLE_CONFIG["secondary_color"],
        linestyle="--",
        alpha=0.5,
        label="Baseline",
    )
    setup_plot(
        fig,
        axes[0],
        "CPU Usage Over Time",
        xlabel="Time (seconds)",
        ylabel="CPU Usage (%)",
    )
    axes[0].set_ylim(0, max(df["cpu_percent"]) * 1.1)
    axes[0].legend()

    # Plot RAM Usage
    smoothed_ram = df["ram_used_gb"].rolling(window=window, center=True).mean()
    sns.lineplot(
        x=elapsed_time,
        y=smoothed_ram,
        ax=axes[1],
        color=STYLE_CONFIG["secondary_color"],
        linewidth=2,
    )
    axes[1].axhline(
        y=baseline_ram,
        color=STYLE_CONFIG["primary_color"],
        linestyle="--",
        alpha=0.5,
        label="Baseline",
    )
    setup_plot(
        fig,
        axes[1],
        "RAM Usage Over Time",
        xlabel="Time (seconds)",
        ylabel="RAM Usage (GB)",
    )
    axes[1].set_ylim(0, max(df["ram_used_gb"]) * 1.1)
    axes[1].legend()

    # Plot GPU Memory if available
    if has_gpu:
        smoothed_gpu = df["gpu_memory_gb"].rolling(window=window, center=True).mean()
        sns.lineplot(
            x=elapsed_time,
            y=smoothed_gpu,
            ax=axes[2],
            color=STYLE_CONFIG["primary_color"],
            linewidth=2,
        )
        axes[2].axhline(
            y=baseline_gpu,
            color=STYLE_CONFIG["secondary_color"],
            linestyle="--",
            alpha=0.5,
            label="Baseline",
        )
        setup_plot(
            fig,
            axes[2],
            "GPU Memory Usage Over Time",
            xlabel="Time (seconds)",
            ylabel="GPU Memory (GB)",
        )
        axes[2].set_ylim(0, max(df["gpu_memory_gb"]) * 1.1)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_timeline(df, output_path, suffix="", prefix=""):
    """Create timeline plot showing latency for each run.

    Args:
        df: pandas DataFrame containing run data with columns:
            - target_tokens: number of tokens
            - run_number: run iteration
            - time_to_first_chunk: latency to first token
        output_path: str, path to save the output plot
    """
    plt.style.use("dark_background")

    # Sort by tokens and run number
    df = df.sort_values(["target_tokens", "run_number"])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate y positions for each run with tighter grouping
    unique_tokens = sorted(df["target_tokens"].unique())
    y_positions = {}
    current_y = 0
    group_spacing = 0.8  # Space between groups
    run_spacing = 0.2  # Space between runs in a group

    for tokens in unique_tokens:
        runs = df[df["target_tokens"] == tokens]
        base_y = current_y
        for i, (_, run) in enumerate(runs.iterrows()):
            y_positions[(tokens, run["run_number"])] = base_y + (i * run_spacing)
        current_y = base_y + (len(runs) * run_spacing) + group_spacing

    # Plot bars and points with more transparency
    bar_height = 0.15
    for _, row in df.iterrows():
        y = y_positions[(row["target_tokens"], row["run_number"])]
        latency = row["time_to_first_chunk"]

        # Latency bar
        ax.add_patch(
            patches.Rectangle(
                (0, y - bar_height / 2),
                latency,
                bar_height,
                facecolor=STYLE_CONFIG["primary_color"],
                alpha=0.3,
            )
        )

        # End point
        ax.plot(
            latency,
            y,
            "o",
            color=STYLE_CONFIG["secondary_color"],
            markersize=4,
            alpha=0.5,
        )

    # Add mean lines and values for each token group
    for tokens in unique_tokens:
        token_runs = df[df["target_tokens"] == tokens]
        mean_latency = token_runs["time_to_first_chunk"].mean()
        y_positions_for_token = [
            y_positions[(tokens, run["run_number"])] for _, run in token_runs.iterrows()
        ]
        min_y = min(y_positions_for_token)
        max_y = max(y_positions_for_token)
        group_center = (min_y + max_y) / 2

        # Plot mean line with gradient alpha
        gradient = np.linspace(0.2, 0.8, 100)
        for i in range(len(gradient) - 1):
            y1 = (
                min_y
                - bar_height
                + (max_y - min_y + 2 * bar_height) * (i / len(gradient))
            )
            y2 = (
                min_y
                - bar_height
                + (max_y - min_y + 2 * bar_height) * ((i + 1) / len(gradient))
            )
            ax.plot(
                [mean_latency, mean_latency],
                [y1, y2],
                "-",
                color=STYLE_CONFIG["secondary_color"],
                linewidth=3,
                alpha=gradient[i],
            )

        # Add mean value label with background
        label_text = f"Mean: {mean_latency:.3f}s"
        bbox_props = dict(
            facecolor=STYLE_CONFIG["background_color"],
            edgecolor=STYLE_CONFIG["secondary_color"],
            alpha=0.8,
            pad=3,
            linewidth=1,
        )
        ax.text(
            mean_latency + 0.02,
            group_center,
            label_text,
            color=STYLE_CONFIG["secondary_color"],
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=bbox_props,
        )

    # Customize plot
    ax.set_ylim(-1, current_y)
    ax.set_xlim(0, df["time_to_first_chunk"].max() * 1.3)  # Extra space for labels

    # Add labels for token groups with tighter spacing
    group_positions = {}
    for tokens in unique_tokens:
        runs = df[df["target_tokens"] == tokens]
        y_positions_for_token = [
            y_positions[(tokens, run["run_number"])] for _, run in runs.iterrows()
        ]
        group_positions[tokens] = sum(y_positions_for_token) / len(
            y_positions_for_token
        )
        plt.axhline(
            y=min(y_positions_for_token) - bar_height,
            color="white",
            alpha=0.1,
            linestyle="-",
        )

    # Calculate mean audio length for each token group
    audio_lengths = {}
    for tokens in unique_tokens:
        token_runs = df[df["target_tokens"] == tokens]
        audio_lengths[tokens] = token_runs["audio_length"].mean()

    # Set y-ticks at group centers with token counts and audio lengths
    plt.yticks(
        list(group_positions.values()),
        [
            f"{tokens} tokens\n({audio_lengths[tokens]:.1f}s)"
            for tokens in group_positions.keys()
        ],
        fontsize=10,
    )

    # Customize appearance
    setup_plot(
        fig,
        ax,
        prefix.upper() + " Time-To-Audio Latency " + suffix,
        xlabel="Time (seconds)",
        ylabel="Input Size",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation(df, x, y, title, xlabel, ylabel, output_path, show_trend=True):
    """Create scatter plot summarizing a metric against input size.

    Args:
        df: pandas DataFrame containing the data
        x: str, column name for x-axis
        y: str, column name for y-axis
        title: str, plot title
        xlabel: str, x-axis label
        ylabel: str, y-axis label
        output_path: str, path to save the output plot
        show_trend: bool, fit a regression line and annotate correlation when the
            relationship is genuinely linear (e.g. processing time vs tokens). Set
            False for ratio metrics like RTF that hover around a constant, where a
            fitted slope over-reads noise; a mean line with a std band is drawn instead.
    """
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    sns.scatterplot(
        data=df, x=x, y=y, s=100, alpha=0.6, color=STYLE_CONFIG["primary_color"]
    )

    if show_trend:
        # Regression line
        sns.regplot(
            data=df,
            x=x,
            y=y,
            scatter=False,
            color=STYLE_CONFIG["secondary_color"],
            line_kws={"linewidth": 2},
        )
        corr = df[x].corr(df[y])
        annotation = f"Correlation: {corr:.2f}"
    else:
        # Mean line with std band, no fitted slope
        mean = df[y].mean()
        std = df[y].std()
        ax.axhline(
            y=mean, color=STYLE_CONFIG["secondary_color"], linewidth=2, alpha=0.9
        )
        ax.axhspan(
            mean - std,
            mean + std,
            color=STYLE_CONFIG["secondary_color"],
            alpha=0.15,
        )
        annotation = f"Mean: {mean:.4f} ± {std:.4f}"

    plt.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        fontsize=STYLE_CONFIG["font_sizes"]["text"],
        color=STYLE_CONFIG["text_color"],
        bbox=dict(
            facecolor=STYLE_CONFIG["background_color"],
            edgecolor=STYLE_CONFIG["text_color"],
            alpha=0.7,
        ),
    )

    setup_plot(fig, ax, title, xlabel=xlabel, ylabel=ylabel)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
