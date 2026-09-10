# Voice tuning (`/dev/tune`)

*Last updated: 2026-09-09*

`POST /dev/tune` takes a short reference clip and returns speech in a voice tuned toward it, or the voice pack itself. Backed by [inno-kokoro](https://github.com/remsky/inno-kokoro) and its [weights](https://huggingface.co/remsky/kokoro-inno-clone-tuner).

It is a tuner, not a cloner. The stock model is steered toward the reference's timbre, pitch, pitch range, and speaking rate. Expect a voice in the same neighbourhood, not a match. English only. Only tune voices you have permission to use.

Off by default, `ENABLE_INNO_TUNER=true` turns it on.

The web player has a Tune tab with the same controls.

## Request

Multipart form. `audio` is the only required field.

| Field | Default | |
|---|---|---|
| `audio` | required | Reference clip, any format libsndfile reads (wav, flac, ogg, mp3). 3 to 30 s, one speaker, mono or stereo, 8 to 96 kHz, 10 MB cap. Only the first 30 s are decoded |
| `request` | unset | JSON with the same fields as the `/v1/audio/speech` body, minus `voice` (`input`, `response_format`, `speed`, `stream`, `lang_code`, `normalization_options`, `return_download_link`, etc). When set, the clip's voice speaks it |
| `prosody_head` | `true` | Apply the prosody head. `false` uses the plain stock blend, closer to a stock voice |
| `fmax` | auto | Pitch tracking ceiling in Hz, 60 to 1000. Unset, the tuner picks one from the clip's harmonics. Set it if a band-limited clip reads an octave high |
| `return_voice_pack` | `false` | Return the tuned `.pt` instead of audio |
| `save_voice` | unset | Keep the pack as `VOICES_DIR/<name>_tuned.pt`. `af_`, `am_`, `bf_`, or `bm_` plus lowercase letters, digits, single underscores. Needs `ALLOW_LOCAL_VOICE_SAVING=true` |

`allow_voice_tags` and `ssml` inside `request` are ignored. The reference clip is the voice.

## Responses

| Call | Response |
|---|---|
| `request` set | Audio, same shape and headers as `/v1/audio/speech` (streamed by default, `X-Download-Path` with `return_download_link`) |
| `return_voice_pack=true` | The `[510, 1, 256]` pack as `<name>_tuned.pt` (`a_tune.pt` unnamed), wins over `request` |
| `save_voice` only | `{"voice": "<name>_tuned"}` |

Packs live in the OS temp dir for the length of the response and are removed after, along with their cached tensor. Nothing is kept unless `save_voice` is set.

```bash
curl -s http://localhost:8880/dev/tune -F audio=@ref.wav -F 'request={"input":"Hello there."}' -o out.mp3
curl -s http://localhost:8880/dev/tune -F audio=@ref.wav -F return_voice_pack=true -o ref.pt
curl -s http://localhost:8880/dev/tune -F audio=@ref.wav -F save_voice=am_ref   # saves am_ref_tuned
```

## Naming

Every voice is a flat `<accent><gender>_<name>` file in `VOICES_DIR`, a suffix says where it came from:

| | Example | |
|---|---|---|
| stock | `af_heart` | from the model card |
| shipped tuned | `am_price_inno` | four tuned voices bundled with the server: `af_amelia_inno`, `af_goodall_inno`, `am_price_inno`, `bm_atten_inno` |
| saved by you | `am_ref_tuned` | `save_voice=am_ref`, the suffix is added |
| in flight | `a_tune_<hex>` | the transient pack while a `/dev/tune` request runs, never listed |

`save_voice` must match `^[ab][a-z]?_[a-z0-9]+(_[a-z0-9]+)*$` (uppercase is lowered first; the first letter picks US or UK English, the optional second letter is naming convention only, the web player uses `x`) and gets `_tuned` appended, else 400. Hyphens are out because `-` is a combine operator. An existing name is a 409, delete the file to replace it.

A saved voice is a plain voice. It works anywhere a voice name does (`voice`, combine syntax, `[voice:]` tags, SSML) and shows up in `/v1/audio/voices`. The language code comes from the first letter, `a` American, `b` British.

Off by default. With `ALLOW_LOCAL_VOICE_SAVING=true` any caller can write into `VOICES_DIR`, so keep it off on a shared host.

## Settings

| Variable | Default | |
|---|---|---|
| `ENABLE_INNO_TUNER` | `false` | Load the tuner at startup and expose the route, otherwise 403 |
| `ALLOW_LOCAL_VOICE_SAVING` | `false` | Allow `save_voice` |
| `VOICES_DIR` | `/app/api/src/voices/v1_0` | Where saved packs go |
| `HF_HUB_OFFLINE` / `HF_HUB_DISABLE_TELEMETRY` | unset | Either one skips the startup version check below |

`docker/scripts/download_model.py` fetches the pinned tuner weights into `MODEL_DIR/v1_0/inno_tuner/` next to the Kokoro weights. The Docker images ship with them.

On load the package does one GET of the weights' `config.json` on Hugging Face and logs a line if newer weights exist. `HF_HUB_OFFLINE=1` skips it.

If the weights are missing or fail to load, the server still starts, logs `Inno voice tuner not loaded`, and the route answers 503.

## Errors

| Status | |
|---|---|
| 400 | Clip under 3 s, over 2 channels, outside 8 to 96 kHz, unreadable audio, bad `request` JSON, `save_voice` off the naming rule, nothing to do (no `request`, `return_voice_pack`, or `save_voice`) |
| 403 | `ENABLE_INNO_TUNER=false`, or `save_voice` without `ALLOW_LOCAL_VOICE_SAVING` |
| 409 | `<name>` already exists |
| 413 | Clip over 10 MB |
| 503 | Tuner not loaded, or busy with another clip |

## How it works

A Kokoro voice pack is 510 rows of 256 floats: the first 128 drive the decoder (timbre), the last 128 drive the predictor (prosody).

- Timbre: a speaker embedding of the clip goes through a learned style head, plus a shift along a learned direction for the clip's spectral tilt.
- Prosody: the stock packs' predictor halves are blended by least squares to the clip's pitch mean, pitch spread, and syllable rate. The prosody head then nudges the blend from the same measurements.

Enrollment takes well under a second on a GPU, a few seconds on CPU. Clips shorter than 5 s, or unusually dark or bright ones (archival, heavy filtering), tune less reliably.
