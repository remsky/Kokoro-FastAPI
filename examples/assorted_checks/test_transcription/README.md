# Transcription Roundtrip Check

Synthesizes a few phrases with a running Kokoro server, transcribes the audio
locally with [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper),
and reports word error rate (WER) against the expected text.

## Run

From the `examples/` directory:

```bash
uv sync --extra transcription
uv run python assorted_checks/test_transcription/test_transcription.py
```

First run downloads the Whisper model (~150 MB for `base.en`) into the HF cache.

## Config (env vars)

| Var | Default | Notes |
| --- | --- | --- |
| `KOKORO_BASE_URL` | `http://localhost:8880/v1` | Running Kokoro server |
| `WHISPER_MODEL` | `base.en` | Try `tiny.en` for speed, `small.en` for accuracy |
| `WER_THRESHOLD` | `0.2` | Per-clip pass cutoff |

## Output

WAVs and a `report.json` are written to `output/`. Exit code is `0` if every
case passed.
