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
- `bench_dialogue.py`: timing. Sweeps turn length across 2, 4 and 8 voices plus
  one request per turn, then sweeps text length at a realistic turn size and the
  worst case. Every case is normalized to a plain single-voice request of the
  same text measured in the same run. Warms every voice first.
- `plot_dialogue_bench.py`: renders `bench_report.json` in the shared benchmark
  theme, one figure per sweep on a shared y scale. Needs the `benchmarks` extra
  for matplotlib.

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
| `DIALOGUE_BENCH_RUNS` | `3` | Repeat the bench to see run to run spread. Use 5 for a published plot |
| `KOKORO_DEVICE` | `gpu` | Label only, stamped into the report and plot caption |

## Output

Everything lands in `output/` (gitignored):

- one WAV per functional check, so the switches can be listened to
- `report.json`: pass/fail and timing per check
- `dialogue_captions.mp3` + `dialogue_captions.srt`: the labelled subtitle demo
- `bench_report.json`: wall clock, audio seconds and realtime factor per case
- `dialogue_turn_length.png`: throughput against how long each speaker turn is,
  with 2, 4 and 8 voices and the previous method over the same axis
- `dialogue_text_length.png`: a voice change every 5 sentences against one every
  sentence, as the text grows to about 7 minutes of audio

Both render at the same figure size and y range so they can sit side by side in
the README and be compared by eye.

To refresh the README images, copy them to their asset names (same convention as
the other benchmark plots, see the `readme-benchmarks` skill):

```bash
cd examples/assorted_checks/test_dialogue/output
cp dialogue_turn_length.png ../../../../assets/gpu_dialogue_turn_length.png
cp dialogue_text_length.png ../../../../assets/gpu_dialogue_text_length.png
```

## Reading the bench

Only one thing moves the number: how long each speaker turn is. Every
`[voice:...]` change ends a segment and segments are chunked independently, so
chunk accumulation resets at each change and the tail flushes short. Turn length
becomes chunk length, and cost tracks chunk count.

Anything down to two sentences a turn sits at the single-voice baseline, inside
run to run noise. A change on every sentence costs roughly a third. Speaker count
is free, the voice counts cross each other and none leads.

`one call per turn` is the client side method this replaces. It tracks
`/dev/dialogue` while turns are long and falls away as they shorten.

The length sweep grows the text to about seven minutes of audio and both turn
sizes stay flat, so the cost is a constant ratio rather than something that
compounds.

The numbers apply to `/v1/audio/speech` with `allow_voice_tags` too, it is the
same code path. CPU tells a milder version of the same story, compute dominates
there so per-segment overhead is a smaller share.
