#!/usr/bin/env python3
"""Measure unload and warm timings against a running Kokoro service.

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
    rm -f \
        examples/assorted_checks/benchmarks/output_data/model_unload_strategies_results.json \
        examples/assorted_checks/benchmarks/output_data/model_unload_strategies_stats.txt

    docker rm -f kokoro-benchmark 2>/dev/null || true
    docker run -d \
        --name kokoro-benchmark \
        --gpus '"device=0"' \
        -p 8880:8880 \
        -e PYTHONPATH=/app:/app/api \
        -e USE_GPU=true \
        -e PYTHONUNBUFFERED=1 \
        -e API_LOG_LEVEL=DEBUG \
        -e DOWNLOAD_MODEL=true \
        -e ALLOW_DEV_UNLOAD=true \
        -e MODEL_AUTO_UNLOAD_TIMEOUT_SECONDS=0 \
        -e MODEL_UNLOAD_STRATEGY=move_to_cpu \
        -e ENABLE_DEBUG_ENDPOINTS=true \
        --user 1001:1001 \
        kokoro-benchmark-gpu
    uv run --extra benchmarks \
        examples/assorted_checks/benchmarks/benchmark_model_unload_strategies.py \
        --trials 5 --strategy move_to_cpu

    docker rm -f kokoro-benchmark
    docker run -d \
        --name kokoro-benchmark \
        --gpus '"device=0"' \
        -p 8880:8880 \
        -e PYTHONPATH=/app:/app/api \
        -e USE_GPU=true \
        -e PYTHONUNBUFFERED=1 \
        -e API_LOG_LEVEL=DEBUG \
        -e DOWNLOAD_MODEL=true \
        -e ALLOW_DEV_UNLOAD=true \
        -e MODEL_AUTO_UNLOAD_TIMEOUT_SECONDS=0 \
        -e MODEL_UNLOAD_STRATEGY=destroy \
        -e ENABLE_DEBUG_ENDPOINTS=true \
        --user 1001:1001 \
        kokoro-benchmark-gpu
    uv run --extra benchmarks \
        examples/assorted_checks/benchmarks/benchmark_model_unload_strategies.py \
        --trials 5 --strategy destroy

    docker rm -f kokoro-benchmark

"""
import argparse
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
OUTPUT_DATA_DIR = os.path.join(SCRIPT_DIR, "output_data")
DEFAULT_RESULTS_FILE = os.path.join(
    OUTPUT_DATA_DIR, "model_unload_strategies_results.json"
)
DEFAULT_STATS_FILE = os.path.join(
    OUTPUT_DATA_DIR, "model_unload_strategies_stats.txt"
)


def save_json_results(results: dict[str, Any], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def format_stat_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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


def wait_for_model_status(base_url: str, timeout: float, interval: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while True:
        try:
            return model_status(base_url)
        except requests.RequestException as exc:
            last_error = exc
        except json.JSONDecodeError as exc:
            last_error = exc

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"service did not become ready at {base_url}/dev/model "
                f"within {timeout:g}s"
            ) from last_error

        time.sleep(min(interval, remaining))


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

    print("  warm")
    _, warm_seconds = request(
        "POST", args.url, "/dev/warm", timeout=args.endpoint_timeout
    )
    time.sleep(args.settle)
    after_warm = memory_snapshot(args)

    print("  first generation")
    first_generation_seconds = first_generation(args)
    time.sleep(args.settle)

    row = {
        "strategy": strategy,
        "trial": trial,
        "unload_seconds": unload_seconds,
        "warm_seconds": warm_seconds,
        "first_generation_seconds": first_generation_seconds,
        "vram_before_unload_mb": before_unload["gpu_memory_used_mb"],
        "vram_after_unload_mb": after_unload["gpu_memory_used_mb"],
        "vram_after_warm_mb": after_warm["gpu_memory_used_mb"],
        "system_ram_before_unload_mb": before_unload["system_ram_used_mb"],
        "system_ram_after_unload_mb": after_unload["system_ram_used_mb"],
        "system_ram_after_warm_mb": after_warm["system_ram_used_mb"],
        "service_rss_before_unload_mb": before_unload["service_rss_mb"],
        "service_rss_after_unload_mb": after_unload["service_rss_mb"],
        "service_rss_after_warm_mb": after_warm["service_rss_mb"],
    }
    print(
        "  -> unload={unload:.3f}s warm={warm:.3f}s first_generation={first:.3f}s".format(
            unload=unload_seconds,
            warm=warm_seconds,
            first=first_generation_seconds,
        )
    )
    return row


def stats_for(values: list[float]) -> dict:
    return {
        "average": statistics.mean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def optional_stats_for(values: list[float]) -> dict:
    if not values:
        return {"average": None, "standard_deviation": None}
    return stats_for(values)


def values_for(rows: list[dict], key: str) -> list[float]:
    return [row[key] for row in rows if is_number(row.get(key))]


def deltas_for(rows: list[dict], before_key: str, after_key: str) -> list[float]:
    values = []
    for row in rows:
        before = row.get(before_key)
        after = row.get(after_key)
        if is_number(before) and is_number(after):
            values.append(after - before)
    return values


def summarize(results: list[dict]) -> list[dict]:
    summary = []
    for strategy in sorted({row["strategy"] for row in results}):
        rows = [row for row in results if row["strategy"] == strategy]
        unload = [row["unload_seconds"] for row in rows]
        warm = [row["warm_seconds"] for row in rows]
        first_generation = [row["first_generation_seconds"] for row in rows]
        summary.append(
            {
                "strategy": strategy,
                "trials": len(rows),
                "unload_seconds": stats_for(unload),
                "warm_seconds": stats_for(warm),
                "first_generation_seconds": stats_for(first_generation),
                "vram_before_unload_mb": optional_stats_for(
                    values_for(rows, "vram_before_unload_mb")
                ),
                "vram_after_unload_mb": optional_stats_for(
                    values_for(rows, "vram_after_unload_mb")
                ),
                "vram_delta_after_unload_mb": optional_stats_for(
                    deltas_for(rows, "vram_before_unload_mb", "vram_after_unload_mb")
                ),
                "vram_after_warm_mb": optional_stats_for(
                    values_for(rows, "vram_after_warm_mb")
                ),
                "vram_delta_after_warm_mb": optional_stats_for(
                    deltas_for(rows, "vram_after_unload_mb", "vram_after_warm_mb")
                ),
                "service_rss_before_unload_mb": optional_stats_for(
                    values_for(rows, "service_rss_before_unload_mb")
                ),
                "service_rss_after_unload_mb": optional_stats_for(
                    values_for(rows, "service_rss_after_unload_mb")
                ),
                "service_rss_delta_after_unload_mb": optional_stats_for(
                    deltas_for(
                        rows,
                        "service_rss_before_unload_mb",
                        "service_rss_after_unload_mb",
                    )
                ),
                "service_rss_after_warm_mb": optional_stats_for(
                    values_for(rows, "service_rss_after_warm_mb")
                ),
                "service_rss_delta_after_warm_mb": optional_stats_for(
                    deltas_for(
                        rows,
                        "service_rss_after_unload_mb",
                        "service_rss_after_warm_mb",
                    )
                ),
                "system_ram_before_unload_mb": optional_stats_for(
                    values_for(rows, "system_ram_before_unload_mb")
                ),
                "system_ram_after_unload_mb": optional_stats_for(
                    values_for(rows, "system_ram_after_unload_mb")
                ),
                "system_ram_delta_after_unload_mb": optional_stats_for(
                    deltas_for(
                        rows,
                        "system_ram_before_unload_mb",
                        "system_ram_after_unload_mb",
                    )
                ),
                "system_ram_after_warm_mb": optional_stats_for(
                    values_for(rows, "system_ram_after_warm_mb")
                ),
                "system_ram_delta_after_warm_mb": optional_stats_for(
                    deltas_for(
                        rows,
                        "system_ram_after_unload_mb",
                        "system_ram_after_warm_mb",
                    )
                ),
            }
        )
    return summary


def load_existing_payload(output_file: str) -> dict[str, Any]:
    if not os.path.exists(output_file):
        return {"strategies": []}
    with open(output_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "strategies" in payload:
        return payload

    strategy = None
    if payload.get("summary"):
        strategy = payload["summary"][0].get("strategy")
    elif payload.get("results"):
        strategy = payload["results"][0].get("strategy")
    if strategy:
        return {
            "timestamp": payload.get("timestamp"),
            "url": payload.get("url"),
            "strategies": [
                {
                    "strategy": strategy,
                    "timestamp": payload.get("timestamp"),
                    "url": payload.get("url"),
                    "model": payload.get("model"),
                    "results": payload.get("results", []),
                    "summary": payload.get("summary", [{}])[0],
                }
            ],
            "summary": payload.get("summary", []),
        }
    raise RuntimeError(f"cannot merge unrecognized results file: {output_file}")


def merge_strategy_payload(
    output_file: str,
    strategy_payload: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    payload = load_existing_payload(output_file)
    strategies = [
        row
        for row in payload.get("strategies", [])
        if row.get("strategy") != strategy_payload["strategy"]
    ]
    strategies.append(strategy_payload)
    strategies.sort(key=lambda row: row["strategy"])
    for row in strategies:
        row_results = row.get("results", [])
        if row_results:
            row["summary"] = summarize(row_results)[0]

    all_results = [
        result
        for strategy_row in strategies
        for result in strategy_row.get("results", [])
    ]
    return {
        "timestamp": generated_at,
        "url": strategy_payload["url"],
        "strategies": strategies,
        "summary": summarize(all_results),
    }


def write_stats_file(summary: list[dict], output_file: str) -> None:
    def write_metric(handle, row: dict, label: str) -> None:
        section = row[label]
        handle.write(
            f"{label}_average: {format_stat_value(section['average'])}\n"
            f"{label}_standard_deviation: "
            f"{format_stat_value(section['standard_deviation'])}\n\n"
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        for row in summary:
            handle.write(f"=== Model Unload Strategy - {row['strategy']} ===\n\n")

            handle.write("# timing\n\n")
            for label in (
                "unload_seconds",
                "warm_seconds",
                "first_generation_seconds",
            ):
                write_metric(handle, row, label)

            handle.write("# memory usage\n\n")
            handle.write("## vram\n\n")
            for label in (
                "vram_before_unload_mb",
                "vram_after_unload_mb",
                "vram_delta_after_unload_mb",
                "vram_after_warm_mb",
                "vram_delta_after_warm_mb",
            ):
                write_metric(handle, row, label)

            handle.write("## service rss\n\n")
            for label in (
                "service_rss_before_unload_mb",
                "service_rss_after_unload_mb",
                "service_rss_delta_after_unload_mb",
                "service_rss_after_warm_mb",
                "service_rss_delta_after_warm_mb",
            ):
                write_metric(handle, row, label)

            handle.write("## system ram\n\n")
            for label in (
                "system_ram_before_unload_mb",
                "system_ram_after_unload_mb",
                "system_ram_delta_after_unload_mb",
                "system_ram_after_warm_mb",
                "system_ram_delta_after_warm_mb",
            ):
                write_metric(handle, row, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--strategy", help="expected strategy; defaults to /dev/model")
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--endpoint-timeout", type=int, default=600)
    ap.add_argument("--ready-timeout", type=float, default=60.0)
    ap.add_argument("--ready-interval", type=float, default=1.0)
    ap.add_argument("--gpu-index", help="GPU id to pass to nvidia-smi --id")
    ap.add_argument("--speech-model", default="kokoro")
    ap.add_argument("--speech-voice", default="af_heart")
    ap.add_argument("--speech-format", default="mp3")
    ap.add_argument("--speech-text", default="Warm model restore benchmark.")
    ap.add_argument("--output-json", default=DEFAULT_RESULTS_FILE)
    ap.add_argument("--output-stats", default=DEFAULT_STATS_FILE)
    args = ap.parse_args()

    status = wait_for_model_status(args.url, args.ready_timeout, args.ready_interval)
    strategy = args.strategy or status.get("unload_strategy")
    if not strategy:
        raise RuntimeError("could not determine strategy from /dev/model")
    if status.get("unload_strategy") != strategy:
        raise RuntimeError(
            f"expected unload_strategy={strategy!r}, got {status.get('unload_strategy')!r}"
        )

    results = [run_trial(strategy, trial, args) for trial in range(1, args.trials + 1)]
    summary = summarize(results)
    generated_at = datetime.now().isoformat()
    strategy_payload = {
        "strategy": strategy,
        "timestamp": generated_at,
        "url": args.url,
        "model": status,
        "results": results,
        "summary": summary[0],
    }

    payload = merge_strategy_payload(args.output_json, strategy_payload, generated_at)
    save_json_results(payload, args.output_json)
    write_stats_file(payload["summary"], args.output_stats)

    print("\ndone.")
    print(f"- {args.output_json}")
    print(f"- {args.output_stats}")


if __name__ == "__main__":
    main()
