<p align="center">
  <img src="githubbanner.png" alt="Kokoro TTS Banner">
</p>

# Kokoro TTS API
[![Tests](https://img.shields.io/badge/tests-117%20passed-darkgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-75%25-darkgreen)]()
[![Tested at Model Commit](https://img.shields.io/badge/last--tested--model--commit-a67f113-blue)](https://huggingface.co/hexgrad/Kokoro-82M/tree/c3b0d86e2a980e027ef71c28819ea02e351c2667) [![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/Remsky/Kokoro-TTS-Zero)

Dockerized FastAPI wrapper for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model
- OpenAI-compatible Speech endpoint, with inline voice combination functionality
- NVIDIA GPU accelerated or CPU Onnx inference 
- very fast generation time
  - 100x+ real time speed via HF A100
  - 35-50x+ real time speed via 4060Ti
  - 5x+ real time speed via M3 Pro CPU
- streaming support w/ variable chunking to control latency & artifacts
- simple audio generation web ui utility
- (new) phoneme endpoints for conversion and generation


## Quick Start

The service can be accessed through either the API endpoints or the Gradio web interface.

1. Install prerequisites:
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) + [Git](https://git-scm.com/downloads)
   - Clone and start the service:
        ```bash
        git clone https://github.com/remsky/Kokoro-FastAPI.git
        cd Kokoro-FastAPI
        docker compose up --build
        ```
2. Run locally as an OpenAI-Compatible Speech Endpoint
    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8880/v1",
        api_key="not-needed"
        )

    response = client.audio.speech.create(
        model="kokoro", 
        voice="af_sky+af_bella", #single or multiple voicepack combo
        input="Hello world!",
        response_format="mp3"
    )
    response.stream_to_file("output.mp3")
    ```

    or visit http://localhost:7860
    <p align="center">
    <img src="ui\GradioScreenShot.png" width="80%" alt="Voice Analysis Comparison" style="border: 2px solid #333; padding: 10px;">
    </p>

## Features 
<details>
<summary>OpenAI-Compatible Speech Endpoint</summary>

```python
# Using OpenAI's Python library
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
response = client.audio.speech.create(
    model="kokoro",  # Not used but required for compatibility, also accepts library defaults
    voice="af_bella+af_sky",
    input="Hello world!",
    response_format="mp3"
)

response.stream_to_file("output.mp3")
```
Or Via Requests:
```python
import requests


response = requests.get("http://localhost:8880/v1/audio/voices")
voices = response.json()["voices"]

# Generate audio
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "model": "kokoro",  # Not used but required for compatibility
        "input": "Hello world!",
        "voice": "af_bella",
        "response_format": "mp3",  # Supported: mp3, wav, opus, flac
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
<summary>Voice Combination</summary>

- Averages model weights of any existing voicepacks
- Saves generated voicepacks for future use
- (new) Available through any endpoint, simply concatenate desired packs with "+"

Combine voices and generate audio:
```python
import requests
response = requests.get("http://localhost:8880/v1/audio/voices")
voices = response.json()["voices"]

# Create combined voice (saves locally on server)
response = requests.post(
    "http://localhost:8880/v1/audio/voices/combine",
    json=[voices[0], voices[1]]
)
combined_voice = response.json()["voice"]

# Generate audio with combined voice (or, simply pass multiple directly with `+` )
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": combined_voice, # or skip the above step with f"{voices[0]}+{voices[1]}"
        "response_format": "mp3"
    }
)
```
<p align="center">
  <img src="assets/voice_analysis.png" width="80%" alt="Voice Analysis Comparison" style="border: 2px solid #333; padding: 10px;">
</p>
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

<details>
<summary>Gradio Web Utility</summary>

Access the interactive web UI at http://localhost:7860 after starting the service. Features include:
- Voice/format/speed selection
- Audio playback and download
- Text file or direct input

If you only want the API, just comment out everything in the docker-compose.yml under and including `gradio-ui`

Currently, voices created via the API are accessible here, but voice combination/creation has not yet been added

*Note: Recent updates for streaming could lead to temporary glitches. If so, pull from the most recent stable release v0.0.2 to restore*
</details>

<details>
<summary>Streaming Support</summary>

```python
# OpenAI-compatible streaming
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8880", api_key="not-needed")

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

## Processing Details
<details>
<summary>Performance Benchmarks</summary>

Benchmarking was performed on generation via the local API using text lengths up to feature-length books (~1.5 hours output), measuring processing time and realtime factor. Tests were run on: 
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
- Realtime Speed: Ranges between 25-50x (generation time to output audio length)
- Average Processing Rate: 137.67 tokens/second (cl100k_base)
</details>
<details>
<summary>GPU Vs. CPU</summary>

```bash
# GPU: Requires NVIDIA GPU with CUDA 12.1 support (~35x realtime speed)
docker compose up --build

# CPU: ONNX optimized inference (~2.4x realtime speed)
docker compose -f docker-compose.cpu.yml up --build
```
*Note: Overall speed may have reduced somewhat with the structural changes to accomodate streaming. Looking into it* 
</details>

<details>
<summary>Natural Boundary Detection</summary>

- Automatically splits and stitches at sentence boundaries 
- Helps to reduce artifacts and allow long form processing as the base model is only currently configured for approximately 30s output 
</details>

<details>
<summary>Phoneme & Token Routes</summary>

Convert text to phonemes and/or generate audio directly from phonemes:
```python
import requests

# Convert text to phonemes
response = requests.post(
    "http://localhost:8880/dev/phonemize",
    json={
        "text": "Hello world!",
        "language": "a"  # "a" for American English
    }
)
result = response.json()
phonemes = result["phonemes"]  # Phoneme string e.g  ðɪs ɪz ˈoʊnli ɐ tˈɛst
tokens = result["tokens"]      # Token IDs including start/end tokens 

# Generate audio from phonemes
response = requests.post(
    "http://localhost:8880/dev/generate_from_phonemes",
    json={
        "phonemes": phonemes,
        "voice": "af_bella",
        "speed": 1.0
    }
)

# Save WAV audio
with open("speech.wav", "wb") as f:
    f.write(response.content)
```

See `examples/phoneme_examples/generate_phonemes.py` for a sample script.
</details>

## Model and License

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
