![FastKoko: Dockerized Kokoro-82M TTS, OpenAI-compatible API](assets/banner.png)
<br>

<a href="https://trendshift.io/repositories/13745?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-13745"><img src="https://trendshift.io/api/badge/repositories/13745" alt="remsky%2FKokoro-FastAPI | Trendshift" width="164" height="36"/></a>

[![Changelog](https://img.shields.io/badge/changelog-white)](./CHANGELOG.md) [![Tests](https://img.shields.io/badge/tests-192-darkgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-66%25-tan)]()

[![Kokoro](https://img.shields.io/badge/kokoro-0.9.4-BB5420)](https://github.com/hexgrad/kokoro)
[![Misaki](https://img.shields.io/badge/misaki-0.9.4-B8860B)](https://github.com/hexgrad/misaki)
[![Tested at Model Commit](https://img.shields.io/badge/model-1.0::9901c2b-blue)](https://huggingface.co/hexgrad/Kokoro-82M/commit/9901c2b79161b6e898b7ea857ae5298f47b8b0d6) 

[![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/Remsky/FastKoko) [![Downloads](https://img.shields.io/badge/downloads-1.9M%2B-2496ED?logo=docker&logoColor=white)](https://github.com/remsky?tab=packages&repo_name=Kokoro-FastAPI)


Dockerized FastAPI wrapper for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model. Generate hours of high quality speech in minutes.



- OpenAI-compatible Speech endpoint, multi-language support
  - English (US/GB), Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin Chinese
- Optional integrated WebUI; read-along long-generation
- Inline multi-speaker generation & voice mixing with weighted combinations
- Per-word, or per-chunk timestamped caption generation
- Phoneme endpoints: generate phonemes from text, or generate audio from phonemes
- Prebuilt multiplatform images
  - CPU and NVIDIA GPU (CUDA): linux/amd64 + linux/arm64
  - AMD GPU (ROCm, experimental): linux/amd64 only
- Apple Silicon (MPS) supported when running directly via UV (no image)


### Integration Guides
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/remsky/Kokoro-FastAPI) [![Ask CodeWiki](https://img.shields.io/badge/Ask%20CodeWiki-4285F4?logo=googlegemini&logoColor=white)](https://codewiki.google/github.com/remsky/kokoro-fastapi)

 [![Helm Chart](https://img.shields.io/badge/Helm%20Chart-black?style=flat&logo=helm&logoColor=white)](https://github.com/remsky/Kokoro-FastAPI/wiki/Setup-Kubernetes) [![DigitalOcean](https://img.shields.io/badge/DigitalOcean-black?style=flat&logo=digitalocean&logoColor=white)](https://github.com/remsky/Kokoro-FastAPI/wiki/Integrations-DigitalOcean) [![SillyTavern](https://img.shields.io/badge/SillyTavern-black?style=flat&color=red)](https://github.com/remsky/Kokoro-FastAPI/wiki/Integrations-SillyTavern)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-black?style=flat&color=white)](https://github.com/remsky/Kokoro-FastAPI/wiki/Integrations-OpenWebUi)

## Get Started

<details>
<summary>Quickest Start (docker run)</summary>

Pre-built multi-arch images with models baked in. 

`:latest` is available, but please pin to a release tag for stable usage.

**No GPU** (laptop, CPU-only server)
```bash
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

**NVIDIA** (GTX 900-series through RTX 40; ships cu126)
```bash
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

**NVIDIA RTX 50-series / Blackwell** (ships cu128)
```bash
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest-cu128
```

**NVIDIA arm64** (Jetson, GH200; same tag, ships cu129)
```bash
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

**AMD GPU** (ROCm, experimental, x86_64 only)
```bash
docker run --device=/dev/kfd --device=/dev/dri -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-rocm:latest
```

**Apple Silicon** (native MPS, from a clone; the CPU image also works in Docker)
```bash
./start-gpu_mac.sh
```

`gpu:latest` is the same image as `gpu:latest-cu126`. Configuration via environment variables, see `core/config.py`.

</details>

<details>

<summary>Quick Start (docker compose) </summary>

1. Install prerequisites, and start the service using Docker Compose (Full setup including UI):
   - Install [Docker](https://www.docker.com/products/docker-desktop/)
   - Clone the repository:
        ```bash
        git clone https://github.com/remsky/Kokoro-FastAPI.git
        cd Kokoro-FastAPI

        cd docker/gpu   # For NVIDIA GPU support
        # or cd docker/cpu   # For CPU support
        # or cd docker/rocm  # For AMD GPU (ROCm, experimental, amd64 only)
        docker compose up --build

        # *Note for Apple Silicon (M1/M2/M3) users:
        # The Docker GPU image is CUDA-only and won't run on Apple Silicon. With Docker, use `docker/cpu`.
        # For native MPS (Apple GPU) acceleration, run directly via UV with `./start-gpu_mac.sh`.

        cd ../..  # back to repo root for the paths below

        # Models will auto-download, but if needed you can manually download:
        python docker/scripts/download_model.py --output api/src/models/v1_0

        # Or run directly via UV:
        ./start-gpu.sh  # For GPU support
        ./start-cpu.sh  # For CPU support
        ```
</details>
<details>
<summary>Direct Run (via uv) </summary>

1. Install prerequisites:
   - Install [astral-uv](https://docs.astral.sh/uv/)
   - Install [espeak-ng](https://github.com/espeak-ng/espeak-ng) in your system if you want it available as a fallback for unknown words/sounds. The upstream libraries may attempt to handle this, but results have varied.
   - Clone the repository:
        ```bash
        git clone https://github.com/remsky/Kokoro-FastAPI.git
        cd Kokoro-FastAPI
        ```
        
        Run the [model download script](https://github.com/remsky/Kokoro-FastAPI/blob/master/docker/scripts/download_model.py) if you haven't already
     
        Start directly via UV (with hot-reload)
        
        Linux and macOS
        ```bash
        ./start-cpu.sh OR
        ./start-gpu.sh 
        ```

        Windows
        ```powershell
        .\start-cpu.ps1 OR
        .\start-gpu.ps1 
        ```

</details>

<details open>
<summary> Up and Running? </summary>


Run locally as an OpenAI-Compatible Speech Endpoint
    
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8880/v1", api_key="not-needed"
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_sky+af_bella", #single or multiple voicepack combo
    input="Hello world!"
  ) as response:
      response.stream_to_file("output.mp3")
```
  
- The API will be available at http://localhost:8880
- API Documentation: http://localhost:8880/docs

- Web Interface: http://localhost:8880/web

<div align="center">
  <img src="assets/webui-screenshot.png" width="52.9%" alt="Web UI Screenshot">
  <img src="assets/docs-screenshot.png" width="45.1%" alt="API Documentation">
</div>

</details>

## Features 

### Core

<details>
<summary>OpenAI-Compatible Speech Endpoint</summary>

```python
# Using OpenAI's Python library
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
response = client.audio.speech.create(
    model="kokoro",  
    voice="af_bella+af_sky", # see /api/src/core/openai_mappings.json to customize
    input="Hello world!",
    response_format="mp3"
)

response.stream_to_file("output.mp3")
```
Or Via Requests:
```python
import requests


response = requests.get("http://localhost:8880/v1/audio/voices")
voices = [v["id"] for v in response.json()["voices"]]

# Generate audio
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "model": "kokoro",  
        "input": "Hello world!",
        "voice": "af_bella",
        "response_format": "mp3",  # Supported: mp3, wav, opus, flac, aac, pcm
        "speed": 1.0
    }
)

# Save audio
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

Quick tests (run from another terminal):
```bash
python examples/assorted_checks/test_openai/test_openai_tts.py # Test OpenAI Compatibility
python examples/assorted_checks/test_voices/test_all_voices.py # Test all available voices
```
</details>

<details>
<summary>Streaming Support</summary>

```python
# OpenAI-compatible streaming
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8880/v1", api_key="not-needed")

# Stream to file
with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_bella",
    input="Hello world!"
) as response:
    response.stream_to_file("output.mp3")

# Stream to speakers (requires PyAudio)
import pyaudio
player = pyaudio.PyAudio().open(
    format=pyaudio.paInt16, 
    channels=1, 
    rate=24000, 
    output=True
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_bella",
    response_format="pcm",
    input="Hello world!"
) as response:
    for chunk in response.iter_bytes(chunk_size=1024):
        player.write(chunk)
```

Or via requests:
```python
import requests

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "af_bella",
        "response_format": "pcm"
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        # Process streaming chunks
        pass
```

<p align="center">
  <img src="assets/gpu_first_token_timeline_openai.png" width="45%" alt="GPU First Token Timeline" style="border: 2px solid #333; padding: 10px; margin-right: 1%;">
  <img src="assets/cpu_first_token_timeline_stream_openai.png" width="45%" alt="CPU First Token Timeline" style="border: 2px solid #333; padding: 10px;">
</p>

Key Streaming Metrics:
- First token latency @ chunksize
    - ~300ms  (GPU) @ 400 
    - ~3500ms (CPU) @ 200 (older i7)
    - ~<1s    (CPU) @ 200 (M3 Pro)
- Adjustable chunking settings for real-time playback 

*Note: Artifacts in intonation can increase with smaller chunks*
</details>

<details>
<summary>Multiple Output Audio Formats</summary>

- mp3
- wav
- opus 
- flac
- aac
- pcm

<p align="center">
<img src="assets/format_comparison.png" width="80%" alt="Audio Format Comparison" style="border: 2px solid #333; padding: 10px;">
</p>

</details>

### Voices

<details>
<summary>Voice Combination</summary>

- Weighted voice combinations using ratios (e.g., "af_bella(2)+af_heart(1)" for 67%/33% mix)
- Ratios are automatically normalized to sum to 100%
- Available through any endpoint by adding weights in parentheses
- Saves generated voicepacks for future use

Combine voices and generate audio:
```python
import requests
response = requests.get("http://localhost:8880/v1/audio/voices")
voices = [v["id"] for v in response.json()["voices"]]

# Example 1: Simple voice combination (50%/50% mix)
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "af_bella+af_sky",  # Equal weights
        "response_format": "mp3"
    }
)

# Example 2: Weighted voice combination (67%/33% mix)
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "af_bella(2)+af_sky(1)",  # 2:1 ratio = 67%/33%
        "response_format": "mp3"
    }
)

# Example 3: Download combined voice as .pt file
response = requests.post(
    "http://localhost:8880/v1/audio/voices/combine",
    json="af_bella(2)+af_sky(1)"  # 2:1 ratio = 67%/33%
)

# Save the .pt file
with open("combined_voice.pt", "wb") as f:
    f.write(response.content)

# Use the downloaded voice file
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "combined_voice",  # Use the saved voice file
        "response_format": "mp3"
    }
)

```
<p align="center">
  <img src="assets/voice_analysis.png" width="80%" alt="Voice Analysis Comparison" style="border: 2px solid #333; padding: 10px;">
</p>
</details>

<details>
<summary>Voice Aliases</summary>

Weighted mixes can get long fast. `voice_aliases` maps a short name per request, for both the `voice` field and `[voice:...]` tags:
- Aliases prefer to match case-insensitively (keep lowercase to avoid inconsistencies). 
- An alias pointing at a nonexistent voice returns a 400.
- The web UI's cast exports in the same format e.g. `{"voice_aliases": {...}}`; interchangeable for API calls.


```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

client.audio.speech.create(
    model="kokoro",
    voice="narrator",
    input="[voice:narrator] Once upon a time. [voice:villain] Never!",
    extra_body={
        "allow_voice_tags": True,
        "voice_aliases": {"narrator": "af_bella(2)+af_sky", "villain": "am_michael"},
    },
)
```

or

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "voice": "narrator",
    "input": "[voice:narrator] Once upon a time. [voice:villain] Never!",
    "allow_voice_tags": true,
    "voice_aliases": {"narrator": "af_bella(2)+af_sky", "villain": "am_michael"},
    "response_format": "mp3"
  }' --output aliased.mp3
```

</details>

<details>
<summary>Multi-Speaker / Dialogue</summary>

- `[voice:...]` tags work anywhere `input` is accepted when enabled (on by default). 
- OpenAI endpoint requires `allow_voice_tags=true` passed per request to enable (off by default)
- Opt-out entirely by setting `ENABLE_VOICE_TAGS=false` to refuse the parameter, and disable `/dev/dialogue` server-wide (403). 

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "voice": "af_heart",
    "input": "The narrator opens. [voice:af_bella] Did it land? [pause:0.3s] [voice:am_michael] It did.",
    "allow_voice_tags": true,
    "response_format": "mp3"
  }' --output dialogue.mp3
```

With the official OpenAI client, pass the param in `extra_body`:

```python
client.audio.speech.create(
    model="kokoro",
    voice="af_jadzia",
    input="The narrator opens. [voice:af_bella] Did it land?",
    extra_body={"allow_voice_tags": True},
)
```

`POST /dev/dialogue` uses structured turns, and allows `pause_between_turns` to be controlled by param. 

```bash
curl -X POST http://localhost:8880/dev/dialogue \
  -H "Content-Type: application/json" \
  -d '{
    "turns": [
      {"voice": "af_bella", "text": "Did the multi speaker support land?"},
      {"voice": "am_michael", "text": "It did. Turns switch voices inline."}
    ],
    "pause_between_turns": 0.4,
    "response_format": "mp3"
  }' --output dialogue.mp3
```

Notes:
- Any text before the first tag uses the request's `voice`/default. 
- Only the inline form on `/v1/audio/speech` requires `allow_voice_tags`, `/dev/dialogue` does so by default
- Each speaker keeps its own language pipeline based on voice prefix. An explicit `lang_code` will override every speaker.
- Consecutive turns sharing a voice are merged automatically.
- Tags accept short names from **Voice Aliases** above, instead of full weighted mixes.

<div align="center">
  <img src="assets/gpu_dialogue_turn_length.png" width="45%" alt="Throughput against the single-voice baseline as speaker turns get shorter" style="border: 2px solid #333; padding: 10px; margin-right: 1%;">
  <img src="assets/gpu_dialogue_text_length.png" width="45%" alt="Throughput at two voice change rates as the generation grows" style="border: 2px solid #333; padding: 10px;">
</div>

Number of voices has minimal impact on generation speed. For continuous swaps though, if each speaker gets less than about 2 sentences, chunking requirements slow generation down. Still faster than what was previously required, and it stays a flat cost rather than compounding as the text grows. Regenerate with `examples/assorted_checks/test_dialogue/`.
</details>

### Text Control

<details>
<summary>Inline Control Tokens</summary>

Four tokens can be embedded in the `input` text and are parsed server-side (API, WebUI, or any client):

- **Pause**: `[pause:1.5s]` inserts that much silence. Must be exactly this form (colon, trailing `s`, case-insensitive). `[pause=1.5]` and `[PAUSE 1.0]` are not recognized and get read aloud.
- **Pronunciation**: `[Worcester](/wˈʊstər/)` speaks the IPA between the slashes instead of the word. English only; use `/dev/phonemize` to find the IPA.
- **Voice**: `[voice:am_michael]` switches speaker for everything that follows.
  - Requires `allow_voice_tags: true` per request, and `ENABLE_VOICE_TAGS` server-side (on by default). Otherwise the tag is spoken as written.
  - Accepts the same combine syntax as the `voice` parameter (`[voice:af_bella(2)+af_sky]`), 
  - Short names/aliases can be defined in `voice_aliases`, 
  - Unknown values return a 400.
- **Rate**: `[rate:1.5]` multiplies the request `speed` until the next rate tag or voice change; `[rate:1.0]` reverts. Clamped to 0.25-4.0. Same gating as voice tags.
  - A voice alias can carry a natural pace: `{"grandpa": {"voice": "am_michael", "rate": 0.8}}` applies that rate whenever the alias speaks, as the `voice` parameter or in tags. Useful for voices that read fast or slow, and for named presets over one voice (`narrator_fast`, `narrator_slow`).
  - Rate belongs to the voice speaking it. Every `[voice:...]` tag re-asserts that voice's own rate, `1.0` when it has none, so a calibrated speaker cannot drag its pace onto the next one. Use `speed` for a pace over the whole request.

```text
The city of [Worcester](/wˈʊstər/) is easy. [pause:1s] See?
```
</details>

<details>
<summary>SSML Input (experimental)</summary>

`POST /dev/ssml` translates SSML into the tokens above; feed the result to the speech endpoints as normal input. The speech endpoints themselves never parse XML. Send `text` (the SSML) and optionally `voice` (the voice your speech request will use); with a voice set the translator emits `[voice:...]`/`[rate:...]` spans with reverts, so pass the result with `allow_voice_tags: true` to have them apply and get validated. Without a voice those elements are stripped and their content kept.

- `<break time="750ms"/>` or `strength="strong"` becomes a pause (none/x-weak: 0, weak: 0.25s, medium: 0.5s, strong: 1s, x-strong: 1.5s)
- `<voice name="am_michael">` switches speaker and reverts at the closing tag
- `<prosody rate="slow">` (or `80%`, or `1.2`) becomes a rate span, multiplying the request `speed` and reverting at the closing tag. Clamped to 0.25-4.0 like `[rate:]`. Other prosody attributes (`pitch`, `volume`) are ignored.
- `<phoneme alphabet="ipa" ph="wˈʊstər">Worcester</phoneme>` maps to the pronunciation token (IPA only, English only)
- `<sub alias="World Wide Web">WWW</sub>` speaks the alias
- Everything else (`<emphasis>`, `<say-as>`, `<p>`, `<s>`, `<lang>`, etc) is a no-op: the markup is dropped, the text is spoken. The model has no per-span control for these.
- Malformed SSML returns a 400; non-SSML input passes through unchanged.
- `GET /dev/ssml` returns the same list as data: supported elements, ignored ones, the break and rate keyword tables, and the rate bounds. It is built from the translator's own tables, so it cannot drift from what the build does.

```bash
curl -s http://localhost:8880/dev/ssml -H "Content-Type: application/json" \
  -d '{"text": "<speak>The city of <phoneme alphabet=\"ipa\" ph=\"wˈʊstər\">Worcester</phoneme> is easy.<break time=\"1s\"/>See?</speak>"}'
# {"text": "The city of [Worcester](/wˈʊstər/) is easy. [pause:1.0s] See?"}
```
</details>

<details>
<summary>Natural Boundary Detection</summary>

- Automatically splits and stitches at sentence boundaries 
- Reduces artifacts, and allows long-form output from a base model configured for roughly 30s at a time

The model takes up to 510 phonemized tokens per chunk, but running it that long tends to produce 'rushed' speech and other artifacts. The server adds its own chunking layer on top, sized by `TARGET_MIN_TOKENS`, `TARGET_MAX_TOKENS`, and `ABSOLUTE_MAX_TOKENS` (175, 250, 450 by default, set via environment variables).

</details>

<details>
<summary>Phoneme & Token Routes</summary>

Convert text to phonemes and/or generate audio directly from phonemes:
```python
import requests

def get_phonemes(text: str, language: str = "a"):
    """Get phonemes and tokens for input text"""
    response = requests.post(
        "http://localhost:8880/dev/phonemize",
        json={"text": text, "language": language}  # "a" for American English
    )
    response.raise_for_status()
    result = response.json()
    return result["phonemes"], result["tokens"]

def generate_audio_from_phonemes(phonemes: str, voice: str = "af_bella"):
    """Generate audio from phonemes"""
    response = requests.post(
        "http://localhost:8880/dev/generate_from_phonemes",
        json={"phonemes": phonemes, "voice": voice},
        headers={"Accept": "audio/wav"}
    )
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return None
    return response.content

# Example usage
text = "Hello world!"
try:
    # Convert text to phonemes
    phonemes, tokens = get_phonemes(text)
    print(f"Phonemes: {phonemes}")  # e.g. ðɪs ɪz ˈoʊnli ɐ tˈɛst
    print(f"Tokens: {tokens}")      # Token IDs including start/end tokens

    # Generate and save audio
    if audio_bytes := generate_audio_from_phonemes(phonemes):
        with open("speech.wav", "wb") as f:
            f.write(audio_bytes)
        print(f"Generated {len(audio_bytes)} bytes of audio")
except Exception as e:
    print(f"Error: {e}")
```

See `examples/phoneme_examples/generate_phonemes.py` for a sample script.
</details>

### Captions

<details>
<summary>Timestamps (word level)</summary>

Generate audio with word-level timestamps without streaming:
```python
import requests
import base64
import json

response = requests.post(
    "http://localhost:8880/dev/captioned_speech",
    json={
        "model": "kokoro",
        "input": "Hello world!",
        "voice": "af_bella",
        "speed": 1.0,
        "response_format": "mp3",
        "stream": False,
    },
    stream=False
)

with open("output.mp3","wb") as f:

    audio_json=json.loads(response.content)
    chunk_audio=base64.b64decode(audio_json["audio"].encode("utf-8"))
    f.write(chunk_audio)
    print(audio_json["timestamps"])
```

Generate audio with word-level timestamps with streaming:
```python
import requests
import base64
import json

response = requests.post(
    "http://localhost:8880/dev/captioned_speech",
    json={
        "model": "kokoro",
        "input": "Hello world!",
        "voice": "af_bella",
        "speed": 1.0,
        "response_format": "mp3",
        "stream": True,
    },
    stream=True
)

f=open("output.mp3","wb")
for chunk in response.iter_lines(decode_unicode=True):
    if chunk:
        chunk_json=json.loads(chunk)
        chunk_audio=base64.b64decode(chunk_json["audio"].encode("utf-8"))
        f.write(chunk_audio)
        print(chunk_json["timestamps"])
```

With `"allow_voice_tags": true`, each timestamp also carries the `voice` that spoke the word, so multi-speaker captions can be labelled without re-deriving the split client side. Without it the field is absent.
</details>

<details>
<summary>Timestamps (streaming chunks)</summary>

With `stream`, `return_download_link`, and `return_timing` set, the response carries an `X-Timing-Path` header pointing at a JSON sidecar of per-chunk timings. The audio body is unchanged and nothing extra is computed; this is what drives the web UI's read along.

```python
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world! [pause:1s] Again.",
        "voice": "af_bella",
        "stream": True,
        "return_download_link": True,
        "return_timing": True,
    },
    stream=True,
)
audio = b"".join(response.iter_content(1024))

timings = requests.get(f"http://localhost:8880/v1{response.headers['x-timing-path']}").json()
# {"chunks": [{"text": "Hello world!", "start": 0.0, "end": 0.64}, ...]}
```

- Chunk-level (a sentence group, roughly 10-20s of audio), not word-level. For word-level use `/dev/captioned_speech`.
- The header arrives up front, but the file is written when generation finishes. Fetch it after the stream ends; earlier is a 404.
- `start`/`end` are seconds in the final audio, so pauses and speed are already accounted for. `[pause:Ns]` gaps appear as `{"text": ""}` entries.
- `text` is normalized (numbers etc expanded), so align by words rather than exact match.
- With `allow_voice_tags`, each spoken chunk also carries its `voice`, same as captioned timestamps. Absent otherwise.
- The sidecar sits next to the download file and shares its temp lifetime.
</details>

## Performance & Operations

<details>
<summary>Performance & Benchmarks</summary>

### Hardware variants

```bash
# GPU: Requires NVIDIA driver with CUDA 12.6+ support (~35x-100x realtime speed)
cd docker/gpu
docker compose up --build

# CPU: PyTorch CPU inference
cd docker/cpu
docker compose up --build

# AMD GPU: ROCm 6.4 (experimental, amd64 only)
cd docker/rocm
docker compose up --build
```

### Throughput

Generation through the local API, text lengths up to feature-length books (~1.5 hours output), measuring processing time and realtime factor. Run on:
- Windows 11 Home w/ WSL2
- NVIDIA 4060Ti 16gb GPU @ CUDA 12.1
- 11th Gen i7-11700 @ 2.5GHz
- 64gb RAM
- WAV native output
- H.G. Wells - The Time Machine (full text)

<p align="center">
  <img src="assets/gpu_processing_time.png" width="45%" alt="Processing Time" style="border: 2px solid #333; padding: 10px; margin-right: 1%;">
  <img src="assets/gpu_realtime_factor.png" width="45%" alt="Realtime Factor" style="border: 2px solid #333; padding: 10px;">
</p>

Key Performance Metrics:
- Realtime Speed: Ranges between 35x-100x (generation time to output audio length)
- Average Processing Rate: 137.67 tokens/second (cl100k_base)

### Model Unload / VRAM Reclaim

`POST /dev/unload` frees the model from VRAM and reloads lazily on the next request. Reclaim scales with load (the activation pool, not just weights) but plateaus: chunks cap at 450 tokens. Long-form = ~30 paragraphs. Same setup as above.

<p align="center">
  <img src="assets/gpu_model_unload_short.png" width="45%" alt="Short workload" style="border: 2px solid #333; padding: 10px; margin-right: 1%;">
  <img src="assets/gpu_model_unload_longform.png" width="45%" alt="Long-form workload" style="border: 2px solid #333; padding: 10px;">
</p>

| Workload | Loaded | Floor | Reclaimed | Reload |
| --- | --- | --- | --- | --- |
| Short (6s audio) | 3.11 GB | 2.37 GB | 758 MiB | +4.9s |
| Long-form (7.5m) | 3.98 GB | 2.37 GB | 1,656 MiB | +5.1s |

Floor is host + CUDA context. Reproduce with `uv run --extra benchmarks assorted_checks/benchmarks/benchmark_model_unload.py` from `examples/`.

### Transcription roundtrip (WER/CER)

End-to-end roundtrip: synthesize with Kokoro, transcribe the result back with [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), compare to the source text. Scripts and data live under `examples/assorted_checks/test_transcription/`.

**Long-form English** (full book, *A Journey to the Centre of the Earth*, Project Gutenberg, voice `af_heart`, `base.en` Whisper on CUDA float16, baseline captured on cu126 GPU build):

| Run | Input chars | Audio length | Synth speedup | Transcribe speedup | WER |
| --- | --- | --- | --- | --- | --- |
| Short (~ch.7) | 64,996 | 66m 06s | 36.4x rt | 62.4x rt | **0.047** |
| Full book | 502,766 | 507m 52s | 45.7x rt | 65.1x rt | **0.033** |

See `examples/assorted_checks/test_transcription/BASELINE.md` for the full regression bands.

**Per-language check** (single-sentence per voice, multilingual Whisper `small`. WER for Latin scripts, CER for ja/zh/hi):

| Language | Voice | Metric | Score |
| --- | --- | --- | --- |
| English | `af_heart` | WER | 0.000 |
| English (UK) | `bf_emma` | WER | 0.111 |
| Spanish | `ef_dora` | WER | 0.000 |
| French | `ff_siwis` | WER | 0.000 |
| Italian | `if_sara` | WER | 0.000 |
| Portuguese | `pf_dora` | WER | 0.000 |
| Hindi | `hf_alpha` | CER | 0.059 |
| Japanese | `jf_alpha` | CER | 0.000 |
| Chinese | `zf_xiaobei` | CER | 0.143 |

*Caveat: these are single short sentences, not a comprehensive per-language quality benchmark. They confirm each voice produces transcribable audio in its target language; deeper quality evaluation per language is open work.*

To reproduce, see `examples/assorted_checks/test_transcription/README.md`.
</details>

<details>
<summary>Debug Endpoints</summary>

System state and resource usage, for debugging exhaustion or performance issues. The `/debug/*` routes expose host and process internals, so they are off by default; set `ENABLE_DEBUG_ENDPOINTS=true` to enable.

- `/debug/threads` - Get thread information and stack traces
- `/debug/storage` - Monitor temp file and output directory usage
- `/debug/system` - Get system information (CPU, memory, GPU)
- `POST /dev/unload` - Release model from VRAM; reloads lazily on next request. Off by default; set `ALLOW_DEV_UNLOAD=true` to enable

Stability: the `/v1/*` OpenAI-compatible routes are the stable API. `/dev/*` and `/debug/*` are operational helpers, and may change or move behind flags between minor releases.
</details>

<details>
<summary>Logging</summary>

Global API [loguru logging level](https://loguru.readthedocs.io/en/stable/api/logger.html#levels) can be set using the `API_LOG_LEVEL` environment variable. Defaults to `DEBUG`.

**Docker**

Modify the appropriate compose `yml` or append to command line.
```bash
docker run --env 'API_LOG_LEVEL=WARNING' ...
```

**Direct via UV**

Linux and macOS
```bash
export API_LOG_LEVEL=WARNING
./start-cpu.sh OR
./start-gpu.sh
```

Windows
```powershell
$env:API_LOG_LEVEL = 'WARNING'
.\start-cpu.ps1 OR
.\start-gpu.ps1
```
</details>

## Known Issues & Troubleshooting

<details>
<summary>Missing words & Missing some timestamps</summary>

The API normalizes input text, which can incorrectly remove or change some phrases. Disable it with `"normalization_options":{"normalize": false}` in the request json:
```python
import requests

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "af_heart",
        "response_format": "pcm",
        "normalization_options":
        {
            "normalize": False
        }
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        # Process streaming chunks
        pass
```
  
</details>

<details>
<summary>Linux GPU Permissions</summary>

Some Linux users may encounter GPU permission issues when running as non-root. 
Can't guarantee anything, but here are some common solutions, consider your security requirements carefully

### Option 1: Container Groups (Likely the best option)
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    group_add:
      - "video"
      - "render"
```

### Option 2: Host System Groups
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    user: "${UID}:${GID}"
    group_add:
      - "video"
```
Note: May require adding host user to groups: `sudo usermod -aG docker,video $USER` and system restart.

### Option 3: Device Permissions (Use with caution)
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
```
⚠️ Warning: Reduces system security. Use only in development environments.

Prerequisites: NVIDIA GPU, drivers, and container toolkit must be properly configured.

Visit [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for more detailed information

</details>

<details>
<summary>AMD GPU (ROCm) troubleshooting</summary>

The ROCm image is experimental, x86_64 only. Findings below are largely from [discussion #151](https://github.com/remsky/Kokoro-FastAPI/discussions/151).

### Native Linux host required

`/dev/kfd` and `/dev/dri` passthrough does not work through Docker Desktop on Windows, or through WSL2. Reports of it working are all on a native Linux host.

### "HIP error: invalid device function" / card not detected

Set `HSA_OVERRIDE_GFX_VERSION` to the LLVM target of the closest officially supported architecture. Common values:

| Card | Value |
| --- | --- |
| RX 7900 XTX / XT | `11.0.0` |
| RDNA 3 iGPU (780M, 7840HS) | `11.0.2` or `11.0.3` |
| RX 6700 XT / 6600 (gfx1031, gfx1032) | `10.3.0` |
| RX 5700 XT (unofficial, mixed reports) | `10.3.0` |

The RX 6800/6900 (gfx1030) are supported directly and need no override.

```yaml
services:
  kokoro-tts:
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Check what your card reports with `rocminfo | grep gfx`.

### Slow or unstable matmuls

hipBLASLt does not cover every architecture. Falling back to hipBLAS is slower on paper but more reliable on consumer cards:

```yaml
      - TORCH_BLAS_PREFER_HIPBLASLT=0
      - PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
```

### First request is slow

MIOpen searches for a kernel per unique tensor shape, which costs 5-60s a shape. The image ships `MIOPEN_FIND_MODE=2` and prebaked kernel databases, but only for the architectures listed in `docker/rocm/kdb_install.sh` (CDNA plus gfx1030). RDNA 3 has no prebaked database, so the search runs on first use.

To pre-populate the on-disk cache, which `docker/rocm/docker-compose.yml` persists in named volumes:

```bash
cd docker/rocm
docker compose run --rm \
  -e MIOPEN_FIND_MODE=3 -e MIOPEN_FIND_ENFORCE=3 \
  kokoro-tts python docker/rocm/warmup_miopen.py
```

This sweeps every phoneme length up to 340 and takes hours (~2 on Strix Halo). Run it once per ROCm or PyTorch upgrade. Then start normally: the default `MIOPEN_FIND_MODE=2` reuses the cache. `docker compose down -v` clears it.

Generating audio for a few paragraphs of varied length under the same overrides is the cheaper, partial version.

</details>

<details>
<summary>WAV duration reported as nonsense in some readers</summary>

WAV responses ship with streaming-sentinel (`0xFFFFFFFF`) size fields in the header. Most readers (`soundfile`, `pydub`/ffmpeg, browsers, OS players) handle this fine. Python's stdlib `wave` does not, and reports a bogus duration. Use `soundfile.info(path).duration` or `ffprobe` for exact length.

</details>

## Project

<details>
<summary>Versioning & Development</summary>

**Branching Strategy:**
*   **`release` branch:** Contains the latest stable build, recommended for production use. Docker images tagged with specific versions are built from this branch.
*   **`master` branch:** Active development. Experimental features, ongoing changes, and fixes not yet released. Use it for the newest code, expect less stability. No images are published from here; `:latest` is built from `release` like every other tag.

Note: This is a *development* focused project at its core.

If you run into trouble, you may have to roll back a version on the release tags if something comes up, or build up from source and/or troubleshoot + submit a PR.

Free and open source is a community effort, and there's only really so many hours in a day. If you'd like to support the work, feel free to open a PR, buy me a coffee, or report any bugs/features/etc you find during use.

Working on the code, or pointing an AI agent at it? [AGENTS.md](AGENTS.md) covers the repo layout, commands, and conventions.

  <a href="https://www.buymeacoffee.com/remsky" target="_blank">
    <img
      src="https://cdn.buymeacoffee.com/buttons/v2/default-violet.png"
      alt="Buy Me A Coffee"
      style="height: 30px !important;width: 110px !important;"
    >
  </a>


</details>

<details open>
<summary>Model</summary>

This API uses the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model from HuggingFace. 

Visit the model page for more details about training, architecture, and capabilities. I have no affiliation with any of their work, and produced this wrapper for ease of use and personal projects.
</details>
<details>
<summary>License</summary>
This project is licensed under the Apache License 2.0 - see below for details:

- The Kokoro model weights are licensed under Apache 2.0 (see [model page](https://huggingface.co/hexgrad/Kokoro-82M))
- The FastAPI wrapper code in this repository is licensed under Apache 2.0 to match
- The inference code adapted from StyleTTS2 is MIT licensed

The full Apache 2.0 license text can be found at: https://www.apache.org/licenses/LICENSE-2.0
</details>

<details>
<summary>Project Structure and Churn</summary>

![repoglyph](https://repoglyph.net/remsky/Kokoro-FastAPI.svg?palette=light&commits=40&detail=15&branch=master&prefix=1&border=1&skip_dirs=ui%2Cexamples%2Cscripts%2Cdev%2Cdepr_tests)

</details>


## Contributors

<a href="https://github.com/remsky/Kokoro-FastAPI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=remsky/Kokoro-FastAPI" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
