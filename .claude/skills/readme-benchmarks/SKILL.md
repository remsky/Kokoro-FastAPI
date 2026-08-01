---
name: readme-benchmarks
description: "Running the Kokoro-FastAPI benchmark + transcription-roundtrip suites and regenerating the README plots. Use when asked to run/refresh benchmarks, RTF/first-token plots, transcription sanity checks, or the long-form baseline."
---

# Benchmarks + README plots

Two suites under `examples/assorted_checks/`:

- `test_transcription/` - synth with a running server, transcribe with faster-whisper, report WER/CER. Short, multilingual, and long-form.
- `benchmarks/` - RTF (processing time vs tokens) and first-token latency/timeline plots. These feed the README performance images.

## Prereqs

- A Kokoro server on `:8880`. GPU and CPU docker images both bind 8880, so swap images, never run both. Set `KOKORO_DEVICE` / `BENCH_PREFIX` to match whichever is up.
- **Warm the server first.** cuDNN autotune cold-start inflates the first GPU run (heavy voices ~1.2s vs ~0.3s warm). Hit a couple voices (e.g. `af_heart`, `zf_xiaobei`) before capturing, or discard the first pass.
- `examples/` has its own uv venv. Run everything from **inside `examples/`** (`cd examples`), not the root `.venv`. The first-token script also needs `examples/` as cwd for its audio path.

```bash
cd examples
uv sync --extra transcription --extra transcription-gpu --extra benchmarks
```

Drop `--extra transcription-gpu` to skip the ~1.2 GB cuDNN/cuBLAS download and transcribe on CPU.

## Run (from `examples/`)

```bash
# short English per-voice sanity (base.en, WER)
KOKORO_DEVICE=gpu uv run python assorted_checks/test_transcription/test_transcription.py

# multilingual (small model, CER for ja/zh)
KOKORO_DEVICE=gpu uv run python assorted_checks/test_transcription/test_transcription_multilingual.py

# long-form book roundtrip (the baseline). see BASELINE.md
LONGFORM_CHARS=65000 WHISPER_DEVICE=cuda KOKORO_DEVICE=gpu \
  uv run python assorted_checks/test_transcription/test_long_form.py

# RTF plots
BENCH_PREFIX=gpu uv run python assorted_checks/benchmarks/benchmark_tts_rtf.py

# first-token latency/timeline plots
BENCH_PREFIX=gpu uv run python assorted_checks/benchmarks/benchmark_first_token_stream_unified.py
```

For a CPU capture: swap to the CPU image, then rerun with `KOKORO_DEVICE=cpu` / `BENCH_PREFIX=cpu` (and `WHISPER_DEVICE=cpu` for long-form). Long-form Windows wrapper: `run_long_form.bat [short|full|synth|transcribe]`.

## Env vars

| Var | Applies to | Notes |
| --- | --- | --- |
| `KOKORO_DEVICE` | transcription | Label only (server doesn't self-report device). Sets report filename `report_{device}.json` + `meta` header so a cpu capture can't overwrite a gpu baseline. Default `gpu`. |
| `BENCH_PREFIX` | benchmarks | Prefixes RTF output files and picks the token sweep (gpu: dense to 1000; cpu: dense to 500). RTF default `gpu`, first-token default `cpu`. |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE` | transcription | Transcribe device is independent of synth device. Scripts default `cpu`/`int8`; long-form `.bat` uses `cuda`/`float16`. |
| `WHISPER_MODEL` | transcription | `base.en` short, `small` multilingual. |
| `LONGFORM_CHARS` / `LONGFORM_VOICE` / `LONGFORM_INPUT` | long-form | Char cap, voice, input book. |

## Gotchas

- **First-token plots save under GENERIC names**, not `BENCH_PREFIX`. `BENCH_PREFIX` only changes plot titles and the (misleading) "Results saved to" print. Actual files are `first_token_{timeline,latency}_stream{,_openai}.png`. So a GPU run overwrites the CPU generic file. To keep both, copy the generic file to its device-named asset **between** runs (CPU run to copy to GPU run).
- Reports are split `_gpu` / `_cpu` with a `meta` header. Don't let a cpu run clobber a gpu file. If you see a bare `report.json` / `long_form_report.json`, it predates the split.
- `transcribe_seconds` is constant across kokoro cpu/gpu unless you change `WHISPER_DEVICE`. Only `synth_seconds` tracks the kokoro device.
- Regression bands live in `test_transcription/BASELINE.md` (WER < 0.07, synth >= 25x rt warm, transcribe >= 40x CUDA / 13-17x CPU int8).
- **Replotting doesn't need a server.** `{prefix}_benchmark_results_rtf.json` keeps `processing_time` + `output_length` at full precision, so you can rebuild any RTF plot from the saved data (recompute `rtf = processing_time / output_length`, feed `plot_correlation`). No re-synth. Note the stored `rtf` field is rounded to 2 decimals for the stats text, too coarse to plot, recompute it.
- **RTF gets a mean line, not a regression.** `plot_correlation(..., show_trend=False)` for the RTF plot: rtf is ~constant vs input size, so a fitted slope over-reads single-run noise (each token count is n=1). Processing-time keeps `show_trend=True`, it's genuinely linear (corr ~0.99). First-token plots use `plot_timeline` and impose no trend at all.

## Output paths

| Suite | Files |
| --- | --- |
| short | `test_transcription/output/report_{device}.json` + wavs |
| multilingual | `test_transcription/output_multilingual/report_{device}.json` |
| long-form | `test_transcription/output_long_form/long_form_report_{device}.json` + wav + transcript + `*.synth_meta.json` sidecar |
| RTF | `benchmarks/output_plots/{prefix}_{processing_time,realtime_factor,system_usage}_rtf.png`; data in `benchmarks/output_data/{prefix}_benchmark_{results,stats}_rtf.*` |
| first-token | `benchmarks/output_plots/first_token_{timeline,latency}_stream{,_openai}.png`, `total_time_latency_stream{,_openai}.png` |

## README asset mapping

Copy plot outputs to `assets/` under the README's names (No stamp):

| README asset | Source plot |
| --- | --- |
| `assets/gpu_processing_time.png` | `output_plots/gpu_processing_time_rtf.png` |
| `assets/gpu_realtime_factor.png` | `output_plots/gpu_realtime_factor_rtf.png` |
| `assets/gpu_first_token_timeline_openai.png` | `output_plots/first_token_timeline_stream_openai.png` (GPU run; name drops "stream") |
| `assets/cpu_first_token_timeline_stream_openai.png` | `output_plots/first_token_timeline_stream_openai.png` (CPU run) |

The README displays these four in the perf blocks (`README.md` ~line 338 first-token, ~383 RTF). Other `gpu_first_token_*` assets exist but aren't shown there.
