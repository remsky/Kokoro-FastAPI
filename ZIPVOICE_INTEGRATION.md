# ZipVoice Integration Guide

This document describes the ZipVoice integration in the FastAPI TTS service, enabling zero-shot voice cloning capabilities.

## Overview

ZipVoice is a fast, high-quality zero-shot text-to-speech system that uses flow matching for audio generation. Unlike Kokoro which uses pre-made voice tensors, ZipVoice allows you to clone any voice by providing a short audio sample (prompt_wav) and its transcription (prompt_text).

## Architecture

### Multi-Backend System

The FastAPI service now supports multiple TTS backends:

- **Kokoro** (default): Original backend with pre-made voice tensors
- **ZipVoice**: Zero-shot voice cloning with prompt-based generation

Backends are managed by `ModelManager` and can be selected per-request or configured as default.

### Key Components

```
api/src/
├── inference/
│   ├── base.py                    # Abstract backend interface
│   ├── kokoro_v1.py              # Kokoro backend
│   ├── zipvoice.py               # ZipVoice backend (NEW)
│   ├── model_manager.py          # Multi-backend manager (UPDATED)
│   └── voice_prompt_manager.py   # Voice prompt caching (NEW)
├── routers/
│   ├── openai_compatible.py      # OpenAI-compatible endpoints
│   └── zipvoice.py               # ZipVoice-specific endpoints (NEW)
└── structures/
    └── schemas.py                # API schemas (UPDATED)
```

## Installation

### Basic Installation

```bash
# Install with CPU support
pip install -e .[cpu]

# Install with GPU support
pip install -e .[gpu]
```

### ZipVoice Installation

```bash
# Install ZipVoice optional dependencies
pip install -e .[zipvoice]

# Or install manually
pip install k2>=1.24.4 -f https://k2-fsa.github.io/k2/cuda.html

# Install ZipVoice itself (models auto-download on first use)
pip install zipvoice
```

### Configuration

Create or update `.env`:

```env
# Backend Configuration
enable_kokoro=true
enable_zipvoice=true
default_backend=kokoro

# ZipVoice Settings
zipvoice_model=zipvoice              # or zipvoice_distill for faster inference
zipvoice_num_steps=8                 # Lower = faster (range: 1-32)
zipvoice_cache_dir=api/src/voices/zipvoice_prompts
zipvoice_max_prompt_duration=3.0     # Maximum prompt audio length
zipvoice_remove_long_silence=true
zipvoice_allow_url_download=true     # Allow downloading prompts from URLs
zipvoice_allow_base64=true           # Allow base64 encoded prompts
```

## API Usage

### 1. Voice Prompt Input Methods

ZipVoice supports **4 ways** to provide voice prompts:

#### Method A: Pre-registered Voice (Recommended)

First, register a voice:

```bash
curl -X POST "http://localhost:8880/v1/zipvoice/voices/register" \
  -F "name=my_voice" \
  -F "transcription=This is a sample of my voice." \
  -F "audio_file=@my_voice.wav"
```

Then use it:

```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Hello, this is generated speech!",
    "voice": "my_voice",
    "prompt_text": "This is a sample of my voice."
  }'
```

#### Method B: URL Download

```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Hello, this is generated speech!",
    "voice": "url+https://example.com/voice_sample.wav",
    "prompt_text": "This is the transcription of the voice sample."
  }'
```

#### Method C: Base64 Encoded Audio

```python
import base64
import requests

# Read and encode audio
with open('voice.wav', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    'http://localhost:8880/v1/zipvoice/audio/speech',
    json={
        'model': 'zipvoice',
        'input': 'Hello, this is generated speech!',
        'voice': f'base64+{audio_b64}',
        'prompt_text': 'This is my voice sample.'
    }
)
```

#### Method D: File Upload (Multipart)

```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -F "model=zipvoice" \
  -F "input=Hello, this is generated speech!" \
  -F "voice=file+voice_sample" \
  -F "prompt_text=This is my voice sample." \
  -F "prompt_wav_file=@voice_sample.wav"
```

### 2. Generation Parameters

```json
{
  "model": "zipvoice",              // Model variant
  "input": "Text to synthesize",    // Input text
  "voice": "my_voice",              // Voice identifier
  "prompt_text": "Transcription",   // Required for voice cloning

  // Audio format options
  "response_format": "mp3",         // mp3, wav, opus, flac, aac, pcm
  "speed": 1.0,                     // 0.25 to 4.0
  "stream": true,                   // Streaming response

  // ZipVoice-specific
  "num_steps": 8,                   // Inference steps (lower = faster)
  "remove_long_silence": true,      // Remove silences
  "max_duration": null,             // Max duration constraint
  "volume_multiplier": 1.0          // Volume adjustment
}
```

### 3. Voice Management API

#### List Registered Voices

```bash
curl http://localhost:8880/v1/zipvoice/voices
```

Response:
```json
{
  "voices": {
    "my_voice": {
      "audio_path": "/path/to/cached/audio.wav",
      "transcription": "This is my voice sample."
    }
  },
  "count": 1
}
```

#### Get Voice Info

```bash
curl http://localhost:8880/v1/zipvoice/voices/my_voice
```

Response:
```json
{
  "name": "my_voice",
  "transcription": "This is my voice sample.",
  "audio_info": {
    "duration": 2.5,
    "samplerate": 24000,
    "channels": 1,
    "samples": 60000,
    "format": ".wav"
  }
}
```

#### Delete Voice

```bash
curl -X DELETE http://localhost:8880/v1/zipvoice/voices/my_voice
```

#### Clear Cache

```bash
# Clear URL download cache
curl -X POST "http://localhost:8880/v1/zipvoice/voices/cache/clear?cache_type=url"

# Clear base64 cache
curl -X POST "http://localhost:8880/v1/zipvoice/voices/cache/clear?cache_type=base64"

# Clear all caches
curl -X POST "http://localhost:8880/v1/zipvoice/voices/cache/clear?cache_type=all"
```

## Python Client Examples

### Basic Usage

```python
import httpx
import asyncio

async def generate_speech():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:8880/v1/zipvoice/audio/speech',
            json={
                'model': 'zipvoice',
                'input': 'Hello world! This is a test of voice cloning.',
                'voice': 'my_voice',
                'prompt_text': 'Sample transcription.',
                'response_format': 'mp3',
                'stream': False
            }
        )

        # Save audio
        with open('output.mp3', 'wb') as f:
            f.write(response.content)

asyncio.run(generate_speech())
```

### Streaming Response

```python
import httpx
import asyncio

async def stream_speech():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            'POST',
            'http://localhost:8880/v1/zipvoice/audio/speech',
            json={
                'model': 'zipvoice',
                'input': 'Long text to synthesize...',
                'voice': 'my_voice',
                'prompt_text': 'Sample.',
                'stream': True
            }
        ) as response:
            with open('output.mp3', 'wb') as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

asyncio.run(stream_speech())
```

### Voice Registration

```python
import httpx
import asyncio

async def register_voice():
    async with httpx.AsyncClient() as client:
        files = {'audio_file': open('my_voice.wav', 'rb')}
        data = {
            'name': 'my_custom_voice',
            'transcription': 'This is a sample of my voice.'
        }

        response = await client.post(
            'http://localhost:8880/v1/zipvoice/voices/register',
            files=files,
            data=data
        )

        print(response.json())

asyncio.run(register_voice())
```

## Model Variants

### Available Models

1. **zipvoice** (default)
   - Highest quality
   - ~8 inference steps
   - Best for production

2. **zipvoice_distill**
   - Faster inference
   - ~4-6 steps recommended
   - Minimal quality loss

3. **zipvoice_dialog**
   - Multi-speaker dialogue
   - Mono output

4. **zipvoice_dialog_stereo**
   - Multi-speaker dialogue
   - Stereo output (speakers in separate channels)

### Changing Models

```env
# In .env
zipvoice_model=zipvoice_distill
```

Or per-request:
```json
{
  "model": "zipvoice_distill",
  ...
}
```

## Performance Optimization

### Speed Optimization

1. **Reduce inference steps**:
   ```json
   {"num_steps": 4}  // Faster, slight quality trade-off
   ```

2. **Use distilled model**:
   ```json
   {"model": "zipvoice_distill"}
   ```

3. **Shorter prompts**:
   - Keep prompt_wav under 3 seconds
   - Shorter prompts = faster inference

4. **Enable GPU**:
   ```env
   use_gpu=true
   device_type=cuda
   ```

### Memory Management

- **Voice prompt caching**: Reuse registered voices to avoid re-processing
- **Cache cleanup**: Periodically clear URL/base64 caches
- **Temp file cleanup**: Automatic cleanup on startup

## Best Practices

### Voice Prompt Quality

1. **Audio Requirements**:
   - Format: WAV (16-bit or 24-bit)
   - Sample rate: 16kHz - 48kHz (24kHz recommended)
   - Duration: 1-3 seconds
   - Channels: Mono or stereo

2. **Recording Tips**:
   - Clear, noise-free audio
   - Natural speaking pace
   - Representative of target voice
   - Good microphone quality

3. **Transcription Accuracy**:
   - Exact word-for-word transcription
   - Include punctuation
   - Match audio timing

### Error Handling

```python
try:
    response = await client.post(...)
    response.raise_for_status()
except httpx.HTTPStatusError as e:
    if e.response.status_code == 404:
        print("Voice not found")
    elif e.response.status_code == 400:
        print("Invalid request:", e.response.json())
    elif e.response.status_code == 503:
        print("ZipVoice backend not available")
```

## Comparison: Kokoro vs ZipVoice

| Feature | Kokoro | ZipVoice |
|---------|--------|----------|
| **Voice Input** | Pre-made .pt tensors | Audio + transcription |
| **Voice Variety** | ~100 pre-made voices | Infinite (any voice) |
| **Setup Time** | Instant | ~1-3s per new voice |
| **Streaming** | Native chunk-by-chunk | Pseudo-streaming |
| **Speed** | Very fast | Fast (tunable) |
| **Quality** | High | High |
| **Use Case** | Consistent voices | Custom voice cloning |
| **Memory** | Low | Medium |

## Troubleshooting

### ZipVoice backend not available

**Cause**: ZipVoice package not installed

**Solution**:
```bash
pip install zipvoice
pip install k2>=1.24.4 -f https://k2-fsa.github.io/k2/cuda.html
```

### "Voice not found in registry"

**Cause**: Voice not registered or typo in name

**Solution**:
1. List available voices: `GET /v1/zipvoice/voices`
2. Register voice: `POST /v1/zipvoice/voices/register`

### "Invalid audio file or duration exceeds maximum"

**Cause**: Audio file corrupt or too long

**Solution**:
- Check audio file plays correctly
- Reduce prompt_wav to <3 seconds
- Adjust `zipvoice_max_prompt_duration` in settings

### Generation is slow

**Solution**:
1. Reduce `num_steps` (try 4-6)
2. Use `zipvoice_distill` model
3. Enable GPU acceleration
4. Use shorter input text

## Advanced Features

### Multi-Backend Switching

```python
# Use Kokoro for fast generation
response1 = await client.post('/v1/audio/speech', json={
    'model': 'kokoro',
    'input': 'Fast generation',
    'voice': 'af_heart'
})

# Use ZipVoice for custom voice
response2 = await client.post('/v1/zipvoice/audio/speech', json={
    'model': 'zipvoice',
    'input': 'Custom voice cloning',
    'voice': 'my_voice',
    'prompt_text': '...'
})
```

### Dialogue Generation

Coming soon: Support for `zipvoice_dialog` models with multi-speaker conversations.

## API Reference

### Endpoints

- `POST /v1/zipvoice/audio/speech` - Generate speech with voice cloning
- `POST /v1/zipvoice/voices/register` - Register a reusable voice
- `GET /v1/zipvoice/voices` - List all registered voices
- `GET /v1/zipvoice/voices/{name}` - Get voice information
- `DELETE /v1/zipvoice/voices/{name}` - Delete a registered voice
- `POST /v1/zipvoice/voices/cache/clear` - Clear voice caches

### Environment Variables

```env
# Backend Management
enable_kokoro=true                    # Enable Kokoro backend
enable_zipvoice=true                  # Enable ZipVoice backend
default_backend=kokoro                # Default backend

# ZipVoice Configuration
zipvoice_model=zipvoice               # Model variant
zipvoice_num_steps=8                  # Inference steps
zipvoice_cache_dir=...                # Voice prompt cache directory
zipvoice_max_prompt_duration=3.0      # Max prompt length (seconds)
zipvoice_remove_long_silence=true     # Remove silences
zipvoice_allow_url_download=true      # Allow URL downloads
zipvoice_allow_base64=true            # Allow base64 input
zipvoice_max_download_size_mb=10.0    # Max download size
```

## Contributing

To add support for additional TTS backends:

1. Create new backend class extending `BaseModelBackend`
2. Implement required methods: `load_model()`, `generate()`, `unload()`
3. Register backend in `ModelManager.initialize()`
4. Add configuration in `config.py`
5. Create router if needed for backend-specific endpoints

## License

Same as the main project (check main README for license information).

## References

- **ZipVoice Paper**: https://arxiv.org/pdf/2506.13053
- **ZipVoice Repository**: https://github.com/k2-fsa/ZipVoice
- **Kokoro TTS**: https://huggingface.co/hexgrad/Kokoro-82M

---

**Questions or issues?** Please open an issue on GitHub.
