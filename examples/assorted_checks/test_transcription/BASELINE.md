# Long-Form Baseline

Reference numbers from `test_long_form.py` so future runs have something to
compare against. All numbers are with `WHISPER_DEVICE=cuda` (the .bat default).

- Voice: `af_heart`
- Server: local, GPU
- Whisper: `base.en`, float16, VAD on, CUDA
- Source: `input/journey_all.txt.gz` (*A Journey to the Centre of the Earth*, Project Gutenberg)
- Short: `--chars 65000` (~end of chapter 7). Full: entire book.

| Metric | Short (warm) | Full |
| --- | --- | --- |
| Input words / chars | 11,468 / 64,996 | 88,884 / 502,766 |
| Audio length | 66m06s | 507m52s |
| Synth time / speedup | 1m49s, 36.4x rt | 11m06s, 45.7x rt |
| Transcribe time / speedup | 1m03s, 62.4x rt | 7m48s, 65.1x rt |
| Output size | 181.6 MB | 1394.9 MB |
| Transcript words | 11,518 | 89,173 |
| **WER (normalized)** | **0.0466** | **0.0334** |

Captured on cu126 GPU build, warm container.

## Cold vs warm

The short test is sensitive to first-run cuDNN autotune: a cold-container short
synth on the same setup landed at 17x rt (vs 36x warm). The full run is long
enough to amortize the autotune cost, so it lands at the same number cold or
warm. When comparing the short numbers, make sure the container has done at
least one prior synth — otherwise you'll be measuring autotune, not throughput.

## Regression bands

- WER < 0.07
- Transcript word count within +/-1% of cleaned input
- Synth >= 25x realtime, warm (depends on GPU; cold first-run on short can drop to ~15-20x)
- Transcribe >= 40x realtime on CUDA (CPU `base.en` int8 lands ~13-17x)
