# Transcription Roundtrip Check

Synthesizes phrases with a running Kokoro server, transcribes the audio with
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), and reports WER
against the expected text.

- `test_transcription.py`: short English clips per voice, fast sanity check
  (uses `base.en`, WER).
- `test_transcription_multilingual.py`: one short clip per non-English voice
  with native reference text. Defaults to multilingual `small` (~470MB), uses
  CER for ja/zh and WER otherwise. Output goes to `output_multilingual/`.
- `test_long_form.py`: multi-hour roundtrip on a full book. See `BASELINE.md`.
  Windows wrapper: `run_long_form.bat [short|full|synth|transcribe]`
  (default `short`).

## Run

From the repo root:

```bash
uv sync --project examples --extra transcription --extra transcription-gpu
uv run --project examples python examples/assorted_checks/test_transcription/test_transcription.py
```

Omit `--extra transcription-gpu` to skip the ~1.2 GB cuDNN/cuBLAS download and
run Whisper on CPU. First run also pulls the `base.en` model (~150 MB) into
the HF cache.

## Config (env vars)

| Var | Default | Notes |
| --- | --- | --- |
| `KOKORO_BASE_URL` | `http://localhost:8880/v1` | Running Kokoro server |
| `WHISPER_MODEL` | `base.en` | Try `tiny.en` for speed, `small.en` for accuracy |
| `WHISPER_DEVICE` | `cpu` (script) / `cuda` (bat) | Whisper device |
| `WHISPER_COMPUTE` | `int8` on cpu, `float16` on cuda | CTranslate2 compute type |
| `WER_THRESHOLD` | `0.2` | Per-clip pass cutoff (short test only) |

`test_long_form.py` also reads `LONGFORM_VOICE`, `LONGFORM_INPUT`, and
`LONGFORM_CHARS`; see its module docstring.

## Output

- Short test → `output/` (WAVs + `report.json`).
- Long-form → `output_long_form/` (WAV, transcript, `long_form_report.json`,
  and a `*.synth_meta.json` sidecar so `transcribe`-only runs inherit the
  exact char cap from the prior synth).
