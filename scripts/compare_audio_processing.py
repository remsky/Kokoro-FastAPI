"""Reproduce PR #577's per-chunk baseline and save persistent-filter A/B WAVs.

Run from the repository root: python -m scripts.compare_audio_processing --output /-ab
Add --speech to synthesize local Kokoro voices (no model downloads).
"""

import argparse
import json
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import soundfile as sf

from api.src.services.audio import AudioFilterStream, _atempo_factors

RATE = 24000
SPEEDS = (0.7, 1.0, 1.5, 2.0)
TEXTS = {
    "short": "Please bring three fresh strawberries.",
    "punctuation": "Wait... Really? Yes! Dr. Smith prescribed three specific treatments; read each instruction carefully.",
    "long": "The rural railway rarely runs regularly. She sells seashells by the seashore. "
    "We should distinguish statistical significance from practical importance. "
    "After a short pause, the speaker continues at a steady pace. " * 3,
}


def legacy_stretch(audio, speed):
    """Original 457755e algorithm, retained only as a benchmark reference."""
    if speed == 1 or not len(audio):
        return audio
    graph = av.filter.Graph()
    nodes = [graph.add_abuffer(format="flt", sample_rate=RATE, layout="mono")]
    nodes += [graph.add("atempo", str(f)) for f in _atempo_factors(speed)]
    nodes.append(graph.add("abuffersink"))
    for source, sink in zip(nodes, nodes[1:]):
        source.link_to(sink)
    graph.configure()
    frame = av.AudioFrame.from_ndarray(
        np.pad(audio, (0, 4096)).reshape(1, -1), format="flt", layout="mono"
    )
    frame.sample_rate, frame.pts = RATE, 0
    graph.push(frame)
    graph.push(None)
    result = []
    while True:
        try:
            result.append(graph.pull().to_ndarray().reshape(-1))
        except (av.BlockingIOError, av.EOFError):
            break
    return np.concatenate(result)[: round(len(audio) / speed)]


def measure(chunks, speed, method):
    start, cpu = time.perf_counter(), time.process_time()
    tracemalloc.start()
    processor = (
        AudioFilterStream(speed, normalize=method == "normalized")
        if method != "legacy"
        else None
    )
    output, consumed, first_input, first_ms = [], 0, None, None
    try:
        for chunk in chunks:
            consumed += len(chunk)
            result = (
                processor.push(chunk) if processor else legacy_stretch(chunk, speed)
            )
            output.append(result)
            if len(result) and first_input is None:
                first_input = consumed / RATE
                first_ms = (time.perf_counter() - start) * 1000
        if processor:
            output.append(processor.flush())
    finally:
        if processor:
            processor.close()
    cpu_ms = (time.process_time() - cpu) * 1000
    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    audio = np.concatenate(output)
    return audio, {
        "method": method,
        "speed": speed,
        "input_seconds": consumed / RATE,
        "output_seconds": len(audio) / RATE,
        "duration_error_ms": (len(audio) - round(consumed / speed)) / RATE * 1000,
        "first_output_after_input_seconds": first_input,
        "first_output_processing_ms": first_ms,
        "processing_ms": elapsed_ms,
        "cpu_ms": cpu_ms,
        "python_peak_bytes_including_collected_output": peak,
        "rms": float(np.sqrt(np.mean(audio**2))),
        "peak": float(np.max(abs(audio))),
    }


def speech_cases(voices, model_speed=1.0):
    import torch
    from kokoro import KModel, KPipeline

    torch.set_num_threads(2)
    model_dir = Path("api/src/models/v1_0")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(model_dir / "config.json"),
        model=str(model_dir / "kokoro-v1_0.pth"),
    ).eval()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
    for voice in voices:
        for name, text in TEXTS.items():
            started = time.perf_counter()
            chunks = [
                r.audio.numpy()
                for r in pipeline(
                    text,
                    voice=f"api/src/voices/v1_0/{voice}.pt",
                    speed=model_speed,
                )
                if r.audio is not None
            ]
            print(
                f"Synthesized {voice}/{name} in {time.perf_counter() - started:.2f}s",
                flush=True,
            )
            yield f"{voice}-{name}", np.concatenate(chunks), model_speed


def normalization_probe(output):
    candidates = {
        "gain_limiter": [
            ("volume", "2"),
            ("alimiter", "limit=0.95:level=false:latency=true"),
        ],
        "dynaudnorm_default": [("dynaudnorm", "")],
        "dynaudnorm_short": [("dynaudnorm", "f=40:g=5:p=0.9:m=4:r=0.12:t=0.005")],
        "loudnorm": [("loudnorm", "I=-18:TP=-1:LRA=7")],
    }
    rows = []
    for name, filters in candidates.items():
        for amplitude in [0.02, 0.2, 0.9, 0.0, 0.00001]:
            graph = av.filter.Graph()
            nodes = [
                graph.add_abuffer(
                    format="flt",
                    sample_rate=RATE,
                    layout="mono",
                    time_base=Fraction(1, RATE),
                )
            ]
            nodes += [graph.add(k, v) for k, v in filters]
            nodes += [
                graph.add(
                    "aformat",
                    f"sample_fmts=flt:sample_rates={RATE}:channel_layouts=mono",
                ),
                graph.add("abuffersink"),
            ]
            for a, b in zip(nodes, nodes[1:]):
                a.link_to(b)
            graph.configure()
            result = []
            first = None
            start = time.perf_counter()

            def drain():
                while True:
                    try:
                        result.append(graph.pull().to_ndarray().reshape(-1))
                    except (av.BlockingIOError, av.EOFError):
                        break

            for pts in range(0, RATE * 4, 480):
                x = (
                    amplitude
                    * np.sin(2 * np.pi * 220 * np.arange(pts, pts + 480) / RATE)
                ).astype("float32")
                frame = av.AudioFrame.from_ndarray(x[None], format="flt", layout="mono")
                frame.sample_rate = RATE
                frame.pts = pts
                frame.time_base = Fraction(1, RATE)
                graph.push(frame)
                drain()
                if result and first is None:
                    first = (pts + 480) / RATE
            graph.push(None)
            drain()
            y = np.concatenate(result)
            row = dict(
                candidate=name,
                amplitude=amplitude,
                first_input_seconds=first,
                output_samples=len(y),
                rms=float(np.sqrt(np.mean(y**2))),
                peak=float(abs(y).max()),
                processing_ms=(time.perf_counter() - start) * 1000,
            )
            rows.append(row)
    (output / "normalization-probe.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech", action="store_true")
    parser.add_argument(
        "--hybrid-candidate",
        action="store_true",
        help="Also synthesize at an experimental 1.3x; does not enable a server mode",
    )
    parser.add_argument("--normalization-probe", action="store_true")
    parser.add_argument("--voices", nargs="+", default=["af_heart", "am_michael"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    t = np.arange(RATE * 4) / RATE
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    tone[RATE : RATE + RATE // 5] = 0
    cases = [("tone-with-pause", tone, 1.0)]
    if args.speech:
        cases.extend(speech_cases(args.voices))
    if args.hybrid_candidate:
        cases.extend(
            (name + "-hybrid-candidate", source, pace)
            for name, source, pace in speech_cases(args.voices, 1.3)
        )
    metrics = []
    for name, source, pace in cases:
        sf.write(args.output / f"{name}-source.wav", source, RATE, subtype="FLOAT")
        for requested in SPEEDS:
            if pace != 1.0 and requested <= 1.0:
                continue
            speed = requested / pace
            # Identical half-second partitions isolate filtering from model inference.
            chunks = [
                source[i : i + RATE // 2] for i in range(0, len(source), RATE // 2)
            ]
            for method in ("legacy", "persistent", "normalized"):
                audio, row = measure(chunks, speed, method)
                row.update(case=name, requested_speed=requested, model_speed=pace)
                metrics.append(row)
                sf.write(
                    args.output / f"{name}-{requested}-{method}.wav",
                    audio,
                    RATE,
                    subtype="FLOAT",
                )
    report = {
        "pyav": av.__version__,
        "ffmpeg": av.library_versions,
        "metrics": metrics,
        "notes": "No subjective listening assessment. CPU-only filter measurements exclude inference; "
        "Python allocation peak includes collected output and excludes native FFmpeg allocations.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.normalization_probe:
        normalization_probe(args.output)
    print(f"Saved {len(metrics)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
