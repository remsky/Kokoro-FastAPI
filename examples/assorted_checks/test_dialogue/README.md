# Multi-Speaker Dialogue Checks

Exercises multi-speaker input against a running Kokoro server, both forms:
inline `[voice:...]` tokens on `/v1/audio/speech` and the structured
`/dev/dialogue` endpoint.

- `test_dialogue.py`: functional checks. Speaker switching, per-speaker
  language pipelines, combined voice tags, `pause_between_turns`, and unknown
  voice rejection. Exits non-zero if any check fails.
- `bench_dialogue.py`: timing. Renders a ~3.5 minute corpus four ways and
  reports throughput against a single-voice baseline.

## Run

From the repo root, with a server already up on `localhost:8880`:

```bash
uv sync --project examples
uv run --project examples python examples/assorted_checks/test_dialogue/test_dialogue.py
uv run --project examples python examples/assorted_checks/test_dialogue/bench_dialogue.py
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
- `bench_report.json`: wall clock, audio seconds and realtime factor per case

## Reading the bench

`4 speakers` against `2 speakers` is the load bearing comparison. Each distinct
speaker resolves once per request and the backend caches voice tensors and
pipelines, so a wider cast should cost nothing. On CPU both land around 90-95%
of the single-voice baseline, and the gap is chunk granularity rather than
switching: per-turn rendering means many sentence sized inferences instead of a
few packed ones. `one call per turn` is the client side workaround this
replaces, for reference.
