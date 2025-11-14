# 🎙️ ZipVoice FastAPI - Zero-Shot Voice Cloning API

**Production-ready FastAPI service for zero-shot text-to-speech with voice cloning using ZipVoice.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## ✨ Features

### 🎯 Core Features
- **Zero-Shot Voice Cloning** - Clone any voice with just a 1-3 second audio sample
- **Multiple Input Methods** - Upload files, URLs, base64, or pre-registered voices
- **Streaming Support** - Real-time audio generation with progressive delivery
- **Multiple Formats** - MP3, WAV, OPUS, FLAC, AAC, PCM
- **OpenAPI Compatible** - Full Swagger/OpenAPI documentation

### 🤖 Smart Features (NEW!)
- **Auto-Transcription** - Automatic voice prompt transcription with Whisper
- **Quality Detection** - AI-powered audio quality analysis with recommendations
- **Smart Parameter Tuning** - Auto-optimize settings based on input text
- **ONNX/TensorRT Optimization** - Up to 2-3x faster inference
- **Intelligent Caching** - Smart voice prompt caching and prefetching

### 🚀 Performance
- **GPU Acceleration** - CUDA, MPS (Apple Silicon), and CPU support
- **Optimized Inference** - ONNX and TensorRT backends for maximum speed
- **Concurrent Requests** - Handle multiple requests simultaneously
- **Memory Efficient** - Smart chunking and cleanup

---

## 📦 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Fabul8/ZipVoice-FastAPI.git
cd ZipVoice-FastAPI

# Install with GPU support (recommended)
pip install -e .[gpu,zipvoice]

# Or install with CPU only
pip install -e .[cpu,zipvoice]

# Install ZipVoice
pip install zipvoice
```

### Docker Installation (Recommended for Production)

**GPU Setup:**
```bash
cd docker/gpu
docker-compose up -d

# Check logs
docker-compose logs -f

# Access API at http://localhost:8880
```

**CPU Setup:**
```bash
cd docker/cpu
docker-compose up -d

# Access API at http://localhost:8880
```

See [docker/README.md](docker/README.md) for complete Docker documentation including:
- Performance optimization (ONNX, TensorRT)
- Configuration options
- Troubleshooting
- Production deployment

### Basic Usage

```bash
# Start server
uvicorn api.src.main:app --host 0.0.0.0 --port 8880

# Server will be available at:
# - API: http://localhost:8880
# - Docs: http://localhost:8880/docs
# - OpenAPI: http://localhost:8880/openapi.json
```

### First Voice Clone

```bash
# 1. Register a voice (with auto-transcription!)
curl -X POST "http://localhost:8880/v1/zipvoice/voices/register" \
  -F "name=my_voice" \
  -F "audio_file=@voice_sample.wav" \
  -F "auto_transcribe=true"

# 2. Generate speech
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Hello! This is my cloned voice speaking.",
    "voice": "my_voice",
    "prompt_text": "Auto-transcribed text from registration"
  }' --output output.mp3
```

---

## 🎨 Smart Features Usage

### 1. Auto-Transcription with Whisper

**No more manual transcription!** The API can automatically transcribe your voice samples:

```bash
# Register voice WITH auto-transcription
curl -X POST "http://localhost:8880/v1/zipvoice/voices/register" \
  -F "name=auto_voice" \
  -F "audio_file=@sample.wav" \
  -F "auto_transcribe=true"  # ← Enable auto-transcription

# Or use in speech generation
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech?auto_transcribe=true" \
  -F "model=zipvoice" \
  -F "input=Generate this text" \
  -F "voice=file+temp" \
  -F "prompt_wav_file=@voice.wav"
  # No prompt_text needed! ✨
```

**Configuration:**
```env
enable_auto_transcription=true
whisper_model_size=base  # tiny, base, small, medium, large
auto_transcribe_on_upload=true
```

### 2. Quality Detection & Analysis

**Get instant feedback on your voice samples:**

```bash
# Analyze voice quality
curl http://localhost:8880/v1/zipvoice/voices/my_voice/quality
```

**Response:**
```json
{
  "voice_name": "my_voice",
  "analysis": {
    "quality_score": 0.85,
    "passes_threshold": true,
    "metrics": {
      "rms_level": 0.15,
      "peak_level": 0.62,
      "snr_db": 28.5,
      "silence_ratio": 0.08,
      "clipping_ratio": 0.0
    },
    "recommendations": [
      "Audio quality is good",
      "Duration optimal (2.3s)"
    ],
    "warnings": []
  }
}
```

**Quality warnings appear in response headers:**
```
X-Quality-Warnings: Audio too quiet - increase volume; High background noise detected
```

### 3. Smart Parameter Auto-Tuning

**Let the API optimize parameters for you:**

```bash
# Get tuning recommendations
curl -X POST "http://localhost:8880/v1/zipvoice/tune" \
  -F "text=Long complex text with numbers like 12345 and special characters" \
  -F "priority=balanced"  # speed | balanced | quality
```

**Response:**
```json
{
  "text_analysis": {
    "word_count": 42,
    "complexity": 0.65,
    "has_numbers": true
  },
  "recommendations": {
    "model": "zipvoice",
    "num_steps": 10,
    "remove_long_silence": true
  },
  "estimated_time_seconds": 8.5
}
```

**Auto-apply tuning:**
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech?auto_tune=true&priority=speed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your text here",
    "voice": "my_voice"
  }'
# Parameters auto-optimized! ⚡
```

### 4. ONNX/TensorRT Optimization

**Enable optimized inference for 2-3x speedup:**

```env
# Enable ONNX (works on CPU and GPU)
enable_onnx=true
onnx_cache_dir=api/src/models/onnx_cache

# Enable TensorRT (GPU only, fastest)
enable_tensorrt=true
tensorrt_cache_dir=api/src/models/tensorrt_cache
```

**Installation:**
```bash
# ONNX Runtime
pip install onnxruntime-gpu  # For GPU
pip install onnxruntime       # For CPU

# TensorRT (Linux + NVIDIA GPU)
pip install tensorrt
```

**The API automatically uses the fastest available backend:**
```
TensorRT (fastest) > ONNX (faster) > PyTorch (standard)
```

---

## 📚 API Documentation

Full API documentation available at:
- **Swagger UI:** http://localhost:8880/docs
- **ReDoc:** http://localhost:8880/redoc
- **OpenAPI JSON:** http://localhost:8880/openapi.json

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/zipvoice/audio/speech` | POST | Generate speech with voice cloning |
| `/v1/zipvoice/voices/register` | POST | Register a reusable voice |
| `/v1/zipvoice/voices` | GET | List all registered voices |
| `/v1/zipvoice/voices/{name}` | GET | Get voice information |
| `/v1/zipvoice/voices/{name}/quality` | GET | Analyze voice quality |
| `/v1/zipvoice/voices/{name}` | DELETE | Delete a voice |
| `/v1/zipvoice/tune` | POST | Get tuning recommendations |

---

## 🎛️ Configuration

Create a `.env` file:

```env
# ============================================
# ZIPVOICE API CONFIGURATION
# ============================================

# API Settings
api_title="ZipVoice TTS API"
api_version="2.0.0"
host="0.0.0.0"
port=8880

# Backend Settings (Kokoro DISABLED - ZipVoice only)
enable_kokoro=false
enable_zipvoice=true
default_backend="zipvoice"

# ZipVoice Core Settings
zipvoice_model="zipvoice"  # zipvoice | zipvoice_distill | zipvoice_dialog
zipvoice_num_steps=8       # 1-32, lower = faster
zipvoice_remove_long_silence=true
zipvoice_cache_dir="api/src/voices/zipvoice_prompts"

# Voice Prompt Settings
zipvoice_max_prompt_duration=3.0
zipvoice_allow_url_download=true
zipvoice_allow_base64=true
zipvoice_max_download_size_mb=10.0

# Auto-Transcription (Whisper)
enable_auto_transcription=true
whisper_model_size="base"  # tiny | base | small | medium | large
auto_transcribe_on_upload=true
whisper_device=null  # null = auto-detect

# Optimization Settings
enable_onnx=false      # Enable ONNX optimization
enable_tensorrt=false  # Enable TensorRT (GPU only)
onnx_cache_dir="api/src/models/onnx_cache"
tensorrt_cache_dir="api/src/models/tensorrt_cache"

# Smart Features
enable_smart_tuning=true        # Auto-tune parameters
enable_quality_detection=true   # Analyze audio quality
enable_intelligent_caching=true # Smart caching
quality_threshold=0.7           # Min quality score (0-1)

# Device Settings
use_gpu=true        # Enable GPU acceleration
device_type=null    # null | cuda | mps | cpu (auto-detect if null)

# CORS
cors_enabled=true
cors_origins=["*"]
```

---

## 🐍 Python Client Examples

See `examples/zipvoice_example.py` for comprehensive examples.

### Basic Generation
```python
import httpx
import asyncio

async def generate_speech():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:8880/v1/zipvoice/audio/speech',
            json={
                'model': 'zipvoice',
                'input': 'Hello, world!',
                'voice': 'my_voice',
                'prompt_text': 'Sample text',
                'auto_tune': True,
                'priority': 'balanced'
            }
        )

        with open('output.mp3', 'wb') as f:
            f.write(response.content)

asyncio.run(generate_speech())
```

---

## 🧪 Testing

```bash
# Run all tests
pytest api/tests/test_zipvoice_integration.py -v

# Run with coverage
pytest api/tests/ --cov=api --cov-report=html

# Run examples
python examples/zipvoice_example.py
```

---

## 📊 Performance Benchmarks

**Hardware:** NVIDIA RTX 3090, AMD Ryzen 9 5900X

| Configuration | Speed (s) | Quality | Speedup |
|---------------|-----------|---------|---------|
| PyTorch (8 steps) | 5.2 | ★★★★★ | 1.0x |
| PyTorch + Distilled (4 steps) | 2.8 | ★★★★☆ | 1.9x |
| ONNX (8 steps) | 3.1 | ★★★★★ | 1.7x |
| TensorRT (8 steps) | 1.9 | ★★★★★ | 2.7x |

---

## 🐛 Troubleshooting

### Common Issues

**"ZipVoice backend not available"**
```bash
pip install zipvoice
pip install k2>=1.24.4 -f https://k2-fsa.github.io/k2/cuda.html
```

**"Whisper not available"**
```bash
pip install openai-whisper
```

**"Quality score too low"**
- Record in quiet environment
- Use better microphone
- Ensure 1-3 second duration

**"Generation too slow"**
- Enable ONNX: `enable_onnx=true`
- Use distilled model: `model=zipvoice_distill`
- Reduce steps: `num_steps=4`

---

## 📁 Project Structure

```
ZipVoice-FastAPI/
├── api/
│   ├── src/
│   │   ├── inference/           # TTS backends
│   │   ├── services/            # Smart services
│   │   ├── routers/             # API endpoints
│   │   └── core/                # Configuration
│   └── tests/                   # E2E tests
├── examples/                    # Usage examples
├── docs/                        # Documentation
└── README.md                    # This file
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

Apache 2.0 - See LICENSE for details.

---

## 🙏 Acknowledgments

- **ZipVoice** - [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice)
- **Whisper** - [OpenAI Whisper](https://github.com/openai/whisper)
- **FastAPI** - [tiangolo/fastapi](https://github.com/tiangolo/fastapi)

---

**Made with ❤️ for the open-source community**
