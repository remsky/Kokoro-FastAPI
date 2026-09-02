#!/usr/bin/env python3
"""Measure unload and reload timings against a running Kokoro service.

The service must already be running with ALLOW_DEV_UNLOAD=true. This script
does not start or stop containers; it follows the convention of the other
benchmark scripts in this directory and only calls HTTP endpoints.

Usage, from repo root with a service already running:
    uv run --extra benchmarks \
        examples/assorted_checks/benchmarks/benchmark_model_unload_strategies.py \
        --trials 10

To benchmark both strategies manually from repo root, build the image once,
run the service with one strategy, run this benchmark, then repeat with the
other strategy:

    docker build -f docker/gpu/Dockerfile.optimized -t kokoro-benchmark-gpu .

    docker rm -f kokoro-benchmark 2>/dev/null || true
    docker run -d \
        --name kokoro-benchmark \
        --gpus '"device=1"' \
        -p 8880:8880 \
        -e PYTHONPATH=/app:/app/api \
        -e USE_GPU=true \
        -e PYTHONUNBUFFERED=1 \
        -e API_LOG_LEVEL=DEBUG \
        -e DOWNLOAD_MODEL=true \
        -e ALLOW_DEV_UNLOAD=true \
        -e MODEL_AUTO_UNLOAD_TIMEOUT_SECONDS=0 \
        -e MODEL_UNLOAD_STRATEGY=move_to_cpu \
        -v "$PWD/api:/app/api" \
        -v "$PWD/web:/app/web" \
        --user 1001:1001 \
        kokoro-benchmark-gpu
    uv run --extra benchmarks \
        examples/assorted_checks/benchmarks/benchmark_model_unload_strategies.py \
        --trials 10 --strategy move_to_cpu --output-prefix model_unload_move_to_cpu

    docker rm -f kokoro-benchmark
    docker run -d \
        --name kokoro-benchmark \
        --gpus '"device=1"' \
        -p 8880:8880 \
        --env-file .env \
        -e PYTHONPATH=/app:/app/api \
        -e USE_GPU=true \
        -e PYTHONUNBUFFERED=1 \
        -e API_LOG_LEVEL=DEBUG \
        -e DOWNLOAD_MODEL=true \
        -e ALLOW_DEV_UNLOAD=true \
        -e MODEL_AUTO_UNLOAD_TIMEOUT_SECONDS=0 \
        -e MODEL_UNLOAD_STRATEGY=destroy \
        -v "$PWD/api:/app/api" \
        -v "$PWD/web:/app/web" \
        --user 1001:1001 \
        kokoro-benchmark-gpu
    uv run --extra benchmarks \
        examples/assorted_checks/benchmarks/benchmark_model_unload_strategies.py \
        --trials 10 --strategy destroy --output-prefix model_unload_destroy

    docker rm -f kokoro-benchmark

"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from datetime import datetime
from typing import Any

import psutil
import requests


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URL = "http://127.0.0.1:8880"


def save_json_results(results: dict[str, Any], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def write_benchmark_stats(stats: list[dict[str, Any]], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        for section in stats:
            handle.write(f"=== {section['title']} ===\n\n")
            for label, value in section["stats"].items():
                if isinstance(value, float):
                    handle.write(f"{label}: {value:.2f}\n")
                else:
                    handle.write(f"{label}: {value}\n")
            handle.write("\n")


def request(
    method: str, base_url: str, path: str, timeout: int = 600
) -> tuple[bytes, float]:
    start = time.perf_counter()
    response = requests.request(method, f"{base_url}{path}", timeout=timeout)
    response.raise_for_status()
    return response.content, time.perf_counter() - start


def model_status(base_url: str) -> dict:
    body, _ = request("GET", base_url, "/dev/model", timeout=60)
    return json.loads(body.decode("utf-8"))


def optional_json_request(base_url: str, path: str, timeout: int = 60) -> dict | None:
    try:
        body, _ = request("GET", base_url, path, timeout=timeout)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in {403, 404}:
            return None
        raise
    except requests.RequestException:
        return None
    return json.loads(body.decode("utf-8"))


def gpu_memory_used_mb(gpu_index: str | None) -> float | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,nounits,noheader",
    ]
    if gpu_index:
        command.extend(["--id", gpu_index])
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [float(line.strip()) for line in output.splitlines() if line.strip()]
    if not values:
        return None
    return sum(values)


def memory_snapshot(args: argparse.Namespace) -> dict[str, float | None]:
    virtual_memory = psutil.virtual_memory()
    debug_system = optional_json_request(args.url, "/debug/system", timeout=10)
    process_rss_mb = None
    if debug_system:
        memory_percent = debug_system.get("process", {}).get("memory_percent")
        total_memory_gb = debug_system.get("memory", {}).get("virtual", {}).get(
            "total_gb"
        )
        if memory_percent is not None and total_memory_gb is not None:
            process_rss_mb = total_memory_gb * 1024 * (memory_percent / 100)

    return {
        "gpu_memory_used_mb": gpu_memory_used_mb(args.gpu_index),
        "system_ram_used_mb": virtual_memory.used / (1024**2),
        "system_ram_available_mb": virtual_memory.available / (1024**2),
        "service_rss_mb": process_rss_mb,
    }


def first_generation(args: argparse.Namespace) -> float:
    payload = {
        "model": args.speech_model,
        "input": args.speech_text,
        "voice": args.speech_voice,
        "response_format": args.speech_format,
        "speed": 1.0,
    }
    start = time.perf_counter()
    response = requests.post(
        f"{args.url}/v1/audio/speech", json=payload, timeout=args.endpoint_timeout
    )
    response.raise_for_status()
    return time.perf_counter() - start


def run_trial(strategy: str, trial: int, args: argparse.Namespace) -> dict:
    print(f"\n=== {strategy} trial {trial}/{args.trials} ===")

    before_unload = memory_snapshot(args)
    print("  unload")
    _, unload_seconds = request(
        "POST", args.url, "/dev/unload", timeout=args.endpoint_timeout
    )
    time.sleep(args.settle)
    after_unload = memory_snapshot(args)

    print("  reload")
    _, reload_seconds = request(
        "POST", args.url, "/dev/reload", timeout=args.endpoint_timeout
    )
    time.sleep(args.settle)
    after_reload = memory_snapshot(args)

    print("  first generation")
    first_generation_seconds = first_generation(args)
    time.sleep(args.settle)

    row = {
        "strategy": strategy,
        "trial": trial,
        "unload_seconds": unload_seconds,
        "reload_seconds": reload_seconds,
        "first_generation_seconds": first_generation_seconds,
        "vram_before_unload_mb": before_unload["gpu_memory_used_mb"],
        "vram_after_unload_mb": after_unload["gpu_memory_used_mb"],
        "vram_after_reload_mb": after_reload["gpu_memory_used_mb"],
        "system_ram_before_unload_mb": before_unload["system_ram_used_mb"],
        "system_ram_after_unload_mb": after_unload["system_ram_used_mb"],
        "system_ram_after_reload_mb": after_reload["system_ram_used_mb"],
        "service_rss_before_unload_mb": before_unload["service_rss_mb"],
        "service_rss_after_unload_mb": after_unload["service_rss_mb"],
        "service_rss_after_reload_mb": after_reload["service_rss_mb"],
    }
    print(
        "  -> unload={unload:.3f}s reload={reload:.3f}s first_generation={first:.3f}s".format(
            unload=unload_seconds,
            reload=reload_seconds,
            first=first_generation_seconds,
        )
    )
    return row


def stats_for(values: list[float]) -> dict:
    return {
        "average": statistics.mean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(results: list[dict]) -> list[dict]:
    summary = []
    for strategy in sorted({row["strategy"] for row in results}):
        rows = [row for row in results if row["strategy"] == strategy]
        unload = [row["unload_seconds"] for row in rows]
        reload = [row["reload_seconds"] for row in rows]
        first_generation = [row["first_generation_seconds"] for row in rows]
        summary.append(
            {
                "strategy": strategy,
                "trials": len(rows),
                "unload_seconds": stats_for(unload),
                "reload_seconds": stats_for(reload),
                "first_generation_seconds": stats_for(first_generation),
            }
        )
    return summary


def write_trials_csv(results: list[dict], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def fmt_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}"


def write_report(summary: list[dict], results: list[dict], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("# Model Unload Strategy Timing\n\n")
        handle.write(f"Generated: {datetime.now().isoformat()}\n\n")
        handle.write("Standard deviation is sample standard deviation across trials.\n\n")
        handle.write(
            "| strategy | unload avg | unload stddev | reload avg | reload stddev | "
            "first generation avg | first generation stddev |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            unload = row["unload_seconds"]
            reload = row["reload_seconds"]
            first_generation = row["first_generation_seconds"]
            handle.write(
                f"| {row['strategy']} | {fmt(unload['average'])} | "
                f"{fmt(unload['standard_deviation'])} | "
                f"{fmt(reload['average'])} | {fmt(reload['standard_deviation'])} | "
                f"{fmt(first_generation['average'])} | "
                f"{fmt(first_generation['standard_deviation'])} |\n"
            )

        handle.write("\n## Trial Data\n\n")
        handle.write("| strategy | trial | unload | reload | first generation |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in results:
            handle.write(
                f"| {row['strategy']} | {row['trial']} | "
                f"{fmt(row['unload_seconds'])} | {fmt(row['reload_seconds'])} | "
                f"{fmt(row['first_generation_seconds'])} |\n"
            )

        handle.write("\n## Memory Data\n\n")
        handle.write(
            "| strategy | trial | VRAM before unload MB | VRAM after unload MB | "
            "VRAM after reload MB | system RAM before unload MB | "
            "system RAM after unload MB | system RAM after reload MB | "
            "service RSS before unload MB | service RSS after unload MB | "
            "service RSS after reload MB |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            handle.write(
                f"| {row['strategy']} | {row['trial']} | "
                f"{fmt_mb(row['vram_before_unload_mb'])} | "
                f"{fmt_mb(row['vram_after_unload_mb'])} | "
                f"{fmt_mb(row['vram_after_reload_mb'])} | "
                f"{fmt_mb(row['system_ram_before_unload_mb'])} | "
                f"{fmt_mb(row['system_ram_after_unload_mb'])} | "
                f"{fmt_mb(row['system_ram_after_reload_mb'])} | "
                f"{fmt_mb(row['service_rss_before_unload_mb'])} | "
                f"{fmt_mb(row['service_rss_after_unload_mb'])} | "
                f"{fmt_mb(row['service_rss_after_reload_mb'])} |\n"
            )


def write_stats_file(summary: list[dict], output_file: str) -> None:
    stats = []
    for row in summary:
        values = {}
        for label in ("unload_seconds", "reload_seconds", "first_generation_seconds"):
            section = row[label]
            values[f"{label}_average"] = section["average"]
            values[f"{label}_standard_deviation"] = section["standard_deviation"]
        stats.append(
            {"title": f"Model Unload Strategy - {row['strategy']}", "stats": values}
        )
    write_benchmark_stats(stats, output_file)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--strategy", help="expected strategy; defaults to /dev/model")
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--endpoint-timeout", type=int, default=600)
    ap.add_argument("--gpu-index", help="GPU id to pass to nvidia-smi --id")
    ap.add_argument("--speech-model", default="kokoro")
    ap.add_argument("--speech-voice", default="af_heart")
    ap.add_argument("--speech-format", default="mp3")
    ap.add_argument("--speech-text", default="Warm model restore benchmark.")
    ap.add_argument("--output-json")
    ap.add_argument("--output-prefix", default="model_unload_strategy")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    status = model_status(args.url)
    strategy = args.strategy or status.get("unload_strategy")
    if not strategy:
        raise RuntimeError("could not determine strategy from /dev/model")
    if status.get("unload_strategy") != strategy:
        raise RuntimeError(
            f"expected unload_strategy={strategy!r}, got {status.get('unload_strategy')!r}"
        )

    results = [run_trial(strategy, trial, args) for trial in range(1, args.trials + 1)]
    summary = summarize(results)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "url": args.url,
        "model": status,
        "results": results,
        "summary": summary,
    }

    output_data_dir = os.path.join(SCRIPT_DIR, "output_data")
    json_path = args.output_json or os.path.join(
        output_data_dir, f"{args.output_prefix}_results.json"
    )
    save_json_results(payload, json_path)

    if not args.no_report:
        write_trials_csv(
            results,
            os.path.join(output_data_dir, f"{args.output_prefix}_trials.csv"),
        )
        write_report(
            summary,
            results,
            os.path.join(output_data_dir, f"{args.output_prefix}_report.md"),
        )
        write_stats_file(
            summary,
            os.path.join(output_data_dir, f"{args.output_prefix}_stats.txt"),
        )

    print("\ndone.")
    print(f"- {json_path}")
    if not args.no_report:
        print(f"- {os.path.join(output_data_dir, f'{args.output_prefix}_trials.csv')}")
        print(f"- {os.path.join(output_data_dir, f'{args.output_prefix}_report.md')}")
        print(f"- {os.path.join(output_data_dir, f'{args.output_prefix}_stats.txt')}")


if __name__ == "__main__":
    main()
