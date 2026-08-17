# Kokoro GPU Unload Notes

## Question

Could reload after GPU unload be faster if the model stays resident in system RAM
and is moved back to CUDA on demand, instead of destroying and reconstructing the
backend from disk?

## Benchmark Setup

- Ran entirely inside the running container: `kokoro-tts-gpu-kokoro-tts-1`
- GPU: NVIDIA GeForce RTX 2070 SUPER
- PyTorch: `2.8.0+cu126`
- Model: `/app/api/src/models/v1_0/kokoro-v1_0.pth`
- Voice: `af_heart`
- Test text: short unload/reload sentence
- The API model was unloaded before benchmarking to free VRAM.

No project files were changed for the benchmark.

## Current API Behavior

Measured through the live API using `POST /dev/unload` followed by
`POST /v1/audio/speech`.

| Case | Time |
|---|---:|
| Cold request after unload, average | `1.835s` |
| Warm request, average | `0.135s` |
| Reload penalty, average | `1.700s` |

Interpretation: with the current destroy/reload mechanism, the next short request
after unload pays about `+1.7s`.

## Direct Model Load Benchmark

Compared fresh model construction plus CUDA load against keeping the model object
in CPU RAM and moving it back to CUDA.

| Case | Time |
|---|---:|
| Fresh model load, average | `1.692s` |
| CPU RAM to CUDA, average | `0.193s` |
| Estimated time saved | `1.499s` |

Interpretation: most of the reload time is CPU-side reconstruction/loading, not
the CUDA transfer itself.

## Backend-Level CPU Cache Benchmark

This benchmark kept the backend/model/pipeline/voice state alive in CPU RAM,
then moved the model back to CUDA before generating.

| Case | Time |
|---|---:|
| Fresh backend load plus cold generation, average | `2.247s` |
| CPU-cache reload plus generation, average | `0.334s` |
| Estimated time saved | `1.913s` |

Interpretation: in an optimistic implementation, reload plus first generation was
roughly `6x` to `7x` faster than the current fresh backend path.

## Takeaway

CPU-RAM caching looks worth implementing. A likely design is an unload strategy
that moves the loaded model to CPU and clears CUDA memory, while preserving enough
backend state to avoid reconstructing the model from disk on the next request.

Potential strategy setting:

```env
MODEL_UNLOAD_STRATEGY=destroy
# or
MODEL_UNLOAD_STRATEGY=cpu_cache
```

Expected tradeoff:

- `destroy`: maximum system RAM release, slower reload.
- `cpu_cache`: keeps more system RAM in use, much faster reload, still reclaims
  most model VRAM.

Important caveat: the backend-level benchmark is a best case because it preserved
pipeline and voice cache state. If an implementation preserves only model weights
but rebuilds pipeline or voice state, the speedup may be smaller.
