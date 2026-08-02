# Multi-Speaker Dialogue Checks

Exercises multi-speaker input against a running Kokoro server, both forms:
inline `[voice:...]` tokens on `/v1/audio/speech` and the structured
`/dev/dialogue` endpoint.

- `test_dialogue.py`: functional checks. Speaker switching, per-speaker
  language pipelines, combined voice tags, `pause_between_turns`, unknown
  voice rejection, and that tags stay literal without `allow_voice_tags`.
  Exits non-zero if any check fails.
- `caption_dialogue.py`: builds speaker-labelled subtitles. Renders through
  `/dev/captioned_speech` with `allow_voice_tags`, groups the word timestamps
  into cues on speaker changes, writes an SRT beside the audio.
- `bench_dialogue.py`: timing. Renders a ~3.5 minute corpus four ways and
  reports throughput against a single-voice baseline.
- `plot_dialogue_bench.py`: renders `bench_report.json` in the shared benchmark
  theme. Needs the `benchmarks` extra for matplotlib.

## Run

From the repo root, with a server already up on `localhost:8880`:

```bash
uv sync --project examples
uv run --project examples python examples/assorted_checks/test_dialogue/test_dialogue.py
uv run --project examples python examples/assorted_checks/test_dialogue/caption_dialogue.py
uv run --project examples python examples/assorted_checks/test_dialogue/bench_dialogue.py
```

To plot the bench afterwards:

```bash
uv sync --project examples --extra benchmarks
uv run --project examples --extra benchmarks python examples/assorted_checks/test_dialogue/plot_dialogue_bench.py
```

The multilingual check needs the `j` and `z` pipelines, so run against an image
built with the Japanese and Chinese extras if you want that one to pass.

## Config (env vars)

| Var | Default | Notes |
| --- | --- | --- |
| `KOKORO_BASE_URL` | `http://localhost:8880` | Server root, no `/v1` suffix (the checks hit both `/v1/audio/speech` and `/dev/dialogue`) |
| `KOKORO_TIMEOUT` | `600` / `1800` | Per request seconds, checks / bench |
| `DIALOGUE_BENCH_RUNS` | `1` | Repeat the bench to see run to run spread |

## Output

Everything lands in `output/` (gitignored):

- one WAV per functional check, so the switches can be listened to
- `report.json`: pass/fail and timing per check
- `dialogue_captions.mp3` + `dialogue_captions.srt`: the labelled subtitle demo
- `bench_report.json`: wall clock, audio seconds and realtime factor per case
- `dialogue_throughput.png`: the bench plotted, throughput against the baseline
  band on top, per-run wall clock underneath

To refresh the README image, copy that plot to its asset name (same convention
as the other benchmark plots, see the `readme-benchmarks` skill):

```bash
cp examples/assorted_checks/test_dialogue/output/dialogue_throughput.png assets/cpu_dialogue_throughput.png
```

## Reading the bench

`4 speakers` against `2 speakers` is the load bearing comparison. Each distinct
speaker resolves once per request and the backend caches voice tensors and
pipelines, so a wider cast should cost nothing.

Six CPU runs of the 579 word corpus, as a percentage of that run's single-voice
baseline throughput:

| case | runs | mean |
| --- | --- | --- |
| 2 speakers, `/dev/dialogue` | 90.9 - 98.3% | 94.6% |
| 4 speakers, `/dev/dialogue` | 88.8 - 100.5% | 95.9% |
| 2 speakers, one call per turn | 87.8 - 96.4% | 91.0% |

Read that as no measurable per-switch cost rather than a 5% penalty. The
baseline itself varies about 9% run to run on the same workload, which is wider
than the spread between cases, and the 4 speaker case is marginally *faster*
than the 2 speaker case on average. Run with `DIALOGUE_BENCH_RUNS=3` before
drawing conclusions from any single number.

`one call per turn` is the client side workaround this replaces, and it is the
only case consistently below the others.
