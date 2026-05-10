# Long-Form Baseline

Reference numbers from `test_long_form.py` against a known-good build, so future
runs have something to compare against.

## Setup

- Voice: `af_heart`
- Server: local, GPU
- Whisper: `base.en`, CPU, int8, VAD on
- Output format: 24 kHz mono PCM WAV
- Source: `input/journey_all.txt.gz` — Project Gutenberg, *A Journey to the Centre of the Earth* (Jules Verne)

Both runs use the same source. The short run caps the cleaned input at 65,000 chars (`--chars 65000`), which lands at roughly end-of-chapter-7. The full run uses the entire book.

## Numbers

| Metric | Short (`--chars 65000`) | Full book |
| --- | --- | --- |
| Input words / chars | 11,468 / 64,996 | 88,884 / 502,766 |
| Synth time | 1m21s | 10m13s |
| Audio length | 66m06s | 507m52s |
| Synth speedup | 48.6× realtime | 49.7× realtime |
| Output size | 181.6 MB | 1394.9 MB |
| Transcribe time | 4m54s | 31m24s |
| Transcribe speedup | 13.5× realtime | 16.2× realtime |
| Transcript words | 11,546 | 89,300 |
| **WER (normalized)** | **0.0481** | **0.0339** |

## Regression thresholds

A fresh run on the same input + voice + Whisper config should land near the table above. Loose bands:

- WER < 0.06 (current: 0.034–0.048)
- Transcript word count within **±1%** of cleaned input
- Synth realtime factor in the 40–60× range (GPU-dependent; a 10× regression is worth a look)
- Transcribe realtime factor in the 13–17× range (CPU-dependent on `base.en` int8)
