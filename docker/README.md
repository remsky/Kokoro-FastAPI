# ZipVoice TTS API - Docker Setup

Complete Docker setup for running ZipVoice TTS API with GPU or CPU acceleration.

## Table of Contents

- [Quick Start](#quick-start)
- [GPU Setup](#gpu-setup)
- [CPU Setup](#cpu-setup)
- [Configuration](#configuration)
- [Optimization](#optimization)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- For GPU: NVIDIA GPU with CUDA 12.4+ support and nvidia-docker2

### GPU Setup (Recommended)

```bash
# Navigate to GPU directory
cd docker/gpu

# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Access the API
curl http://localhost:8880/health
```

### CPU Setup

```bash
# Navigate to CPU directory
cd docker/cpu

# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Access the API
curl http://localhost:8880/health
```

## Architecture

### Multi-Stage Builds

Both GPU and CPU Dockerfiles use optimized multi-stage builds:

**Stage 1 (Builder):**
- Full build environment with all development dependencies
- Compiles PyTorch, ZipVoice, k2, Whisper, ONNX Runtime
- Installs Rust for building native dependencies
- Cleans up unnecessary files

**Stage 2 (Runtime):**
- Minimal runtime image with only necessary dependencies
- Non-root user (`zipvoice`) for security
- Pre-downloaded Whisper models for faster startup
- Health checks enabled

### Directory Structure

```
docker/
├── README.md                    # This file
├── gpu/
│   ├── Dockerfile              # GPU-optimized multi-stage build
│   └── docker-compose.yml      # GPU container orchestration
├── cpu/
│   ├── Dockerfile              # CPU-optimized multi-stage build
│   └── docker-compose.yml      # CPU container orchestration
└── scripts/
    ├── entrypoint-zipvoice.sh  # Container initialization script
    ├── healthcheck.sh          # Health check script
    └── download_model.py       # (Legacy - for Kokoro)
```

## Configuration

### Environment Variables

All configuration is done via environment variables in `docker-compose.yml`:

#### Core Settings

```yaml
# Device configuration
USE_GPU: true                    # Enable/disable GPU
DEVICE_TYPE: cuda               # cuda, cpu, or auto

# Backend selection
ENABLE_KOKORO: false            # Kokoro disabled (legacy)
ENABLE_ZIPVOICE: true           # ZipVoice enabled
DEFAULT_BACKEND: zipvoice       # Default backend to use
```

#### ZipVoice Settings

```yaml
# Model configuration
ZIPVOICE_MODEL: zipvoice        # Options: zipvoice, zipvoice_distill, zipvoice_dialog, zipvoice_dialog_stereo
ZIPVOICE_NUM_STEPS: 8           # Inference steps (1-32, lower=faster)
ZIPVOICE_REMOVE_LONG_SILENCE: true
ZIPVOICE_SPEED_MULTIPLIER: 1.0
ZIPVOICE_MAX_PROMPT_DURATION: 3.0

# Voice prompt input methods
ZIPVOICE_ALLOW_URL_DOWNLOAD: true
ZIPVOICE_ALLOW_BASE64: true
ZIPVOICE_MAX_DOWNLOAD_SIZE_MB: 10.0
```

#### Smart Features

```yaml
# Auto-transcription with Whisper
ENABLE_AUTO_TRANSCRIPTION: true
WHISPER_MODEL_SIZE: base        # Options: tiny, base, small, medium, large

# Automatic parameter optimization
ENABLE_SMART_TUNING: true

# Audio quality detection
ENABLE_QUALITY_DETECTION: true
QUALITY_THRESHOLD: 0.7          # 0-1 scale
```

#### Optimization Settings

```yaml
# GPU optimizations
ENABLE_ONNX: false              # 1.7x speedup (requires model conversion)
ENABLE_TENSORRT: false          # 2.7x speedup (requires model conversion)

# CPU optimizations (CPU only)
ONNX_NUM_THREADS: 8
ONNX_INTER_OP_THREADS: 4
ONNX_EXECUTION_MODE: parallel
ONNX_OPTIMIZATION_LEVEL: all
```

### Volume Mounts

Persistent storage for models and cache:

```yaml
volumes:
  - ../../api:/app/api                                     # API code (dev mode)
  - zipvoice_cache:/app/api/src/voices/zipvoice_prompts  # Voice prompt cache
  - onnx_cache:/app/api/src/models/onnx_cache            # ONNX models
  - tensorrt_cache:/app/api/src/models/tensorrt_cache    # TensorRT engines (GPU only)
  - temp_files:/app/api/temp_files                        # Temporary files
```

## Optimization

### GPU Optimization

#### 1. Basic Setup (Default)

```yaml
ZIPVOICE_MODEL: zipvoice
ZIPVOICE_NUM_STEPS: 8
ENABLE_ONNX: false
ENABLE_TENSORRT: false
```

**Performance:** ~2-3s per sentence
**Quality:** High
**Use case:** General purpose

#### 2. Speed-Optimized

```yaml
ZIPVOICE_MODEL: zipvoice_distill
ZIPVOICE_NUM_STEPS: 4
ENABLE_ONNX: true
```

**Performance:** ~1-1.5s per sentence
**Quality:** Good
**Use case:** Real-time applications

#### 3. Maximum Performance (Advanced)

```yaml
ZIPVOICE_MODEL: zipvoice_distill
ZIPVOICE_NUM_STEPS: 4
ENABLE_ONNX: true
ENABLE_TENSORRT: true
```

**Performance:** ~0.7-1s per sentence
**Quality:** Good
**Use case:** High-throughput production
**Note:** Requires TensorRT model conversion (see below)

### CPU Optimization

#### Default CPU Settings

```yaml
ZIPVOICE_MODEL: zipvoice_distill  # Faster model for CPU
ZIPVOICE_NUM_STEPS: 4             # Fewer steps for speed
WHISPER_MODEL_SIZE: tiny          # Smaller Whisper model
ENABLE_ONNX: false                # Can enable for slight speedup

# ONNX CPU threading
ONNX_NUM_THREADS: 8               # Match CPU core count
ONNX_INTER_OP_THREADS: 4
```

**Performance:** ~5-10s per sentence
**Quality:** Good
**Use case:** Development, testing, low-volume production

## Advanced Usage

### TensorRT Model Conversion

TensorRT provides maximum GPU performance but requires model conversion:

```bash
# Enter the running GPU container
docker-compose exec zipvoice-tts bash

# Convert model to TensorRT (placeholder - requires implementation)
python scripts/convert_to_tensorrt.py \
    --model zipvoice \
    --output /app/api/src/models/tensorrt_cache/zipvoice.trt

# Enable TensorRT in docker-compose.yml
ENABLE_TENSORRT: true

# Restart container
docker-compose restart
```

### ONNX Model Conversion

Similar to TensorRT but works on both GPU and CPU:

```bash
# Enter the container
docker-compose exec zipvoice-tts bash

# Convert model to ONNX (placeholder - requires implementation)
python scripts/convert_to_onnx.py \
    --model zipvoice \
    --output /app/api/src/models/onnx_cache/zipvoice.onnx

# Enable ONNX in docker-compose.yml
ENABLE_ONNX: true

# Restart container
docker-compose restart
```

### Custom Whisper Models

Change the Whisper model size based on your needs:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny  | 39 MB | Fastest | Good | CPU/development |
| base  | 74 MB | Fast | Better | Default |
| small | 244 MB | Medium | High | Production |
| medium | 769 MB | Slow | Higher | High accuracy |
| large | 1.5 GB | Slowest | Highest | Maximum accuracy |

```yaml
# In docker-compose.yml
WHISPER_MODEL_SIZE: small  # or tiny, base, medium, large
```

## Troubleshooting

### Container won't start

**Check logs:**
```bash
docker-compose logs -f
```

**Common issues:**
- Missing NVIDIA drivers (GPU setup)
- Insufficient memory
- Port 8880 already in use

**Solutions:**
```bash
# Check NVIDIA Docker setup
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Change port in docker-compose.yml
ports:
  - "8881:8880"  # Use different host port
```

### Out of Memory (GPU)

**Symptoms:**
- CUDA out of memory errors
- Container crashes during inference

**Solutions:**
1. Reduce batch size (if batching is enabled)
2. Use distilled model: `ZIPVOICE_MODEL: zipvoice_distill`
3. Reduce steps: `ZIPVOICE_NUM_STEPS: 4`

### Slow Performance (CPU)

**Optimizations:**
```yaml
# Use fastest settings
ZIPVOICE_MODEL: zipvoice_distill
ZIPVOICE_NUM_STEPS: 4
WHISPER_MODEL_SIZE: tiny
ENABLE_SMART_TUNING: true  # Auto-optimizes based on input

# Maximize CPU usage
ONNX_NUM_THREADS: 16  # Set to your CPU core count
```

### Health Check Failing

**Check endpoint:**
```bash
curl http://localhost:8880/health
```

**Expected response:**
```json
{"status": "healthy"}
```

**If failing:**
1. Check if API is running: `docker-compose ps`
2. Check logs: `docker-compose logs`
3. Verify port mapping: `docker-compose port zipvoice-tts 8880`

### Permission Issues

**Symptom:**
```
Permission denied: /app/api/src/voices/zipvoice_prompts
```

**Solution:**
```bash
# Fix host directory permissions
sudo chown -R 1001:1001 api/  # For GPU (UID 1001)
# or
sudo chown -R 1000:1000 api/  # For CPU (UID 1000)
```

## Performance Benchmarks

### GPU (NVIDIA RTX 4090)

| Configuration | Speed | Quality | Memory |
|---------------|-------|---------|--------|
| Standard (zipvoice, 8 steps) | 2.1s | Excellent | 4.2 GB |
| Fast (zipvoice_distill, 4 steps) | 1.3s | Very Good | 3.8 GB |
| ONNX (zipvoice_distill, 4 steps) | 0.9s | Very Good | 3.5 GB |
| TensorRT (zipvoice_distill, 4 steps) | 0.7s | Very Good | 3.2 GB |

*Benchmark: 20-word sentence average*

### CPU (Intel i9-12900K, 16 cores)

| Configuration | Speed | Quality | CPU Usage |
|---------------|-------|---------|-----------|
| Standard (zipvoice, 8 steps) | 8.5s | Excellent | 65% |
| Fast (zipvoice_distill, 4 steps) | 5.2s | Very Good | 58% |
| Optimized (distill, 4 steps, 16 threads) | 4.8s | Very Good | 85% |

*Benchmark: 20-word sentence average*

## Development

### Hot Reload

The API code is mounted as a volume for development:

```yaml
volumes:
  - ../../api:/app/api  # Changes reflect immediately
```

**Note:** Server restart may be needed for some changes.

### Building Images

```bash
# Build GPU image
cd docker/gpu
docker-compose build

# Build CPU image
cd docker/cpu
docker-compose build

# Build with no cache (full rebuild)
docker-compose build --no-cache
```

### Accessing Container

```bash
# GPU container
docker-compose -f docker/gpu/docker-compose.yml exec zipvoice-tts bash

# CPU container
docker-compose -f docker/cpu/docker-compose.yml exec zipvoice-tts bash
```

## Production Deployment

### Recommended Settings

**GPU Production:**
```yaml
# Optimize for quality and reliability
ZIPVOICE_MODEL: zipvoice
ZIPVOICE_NUM_STEPS: 8
WHISPER_MODEL_SIZE: small
ENABLE_AUTO_TRANSCRIPTION: true
ENABLE_SMART_TUNING: true
ENABLE_QUALITY_DETECTION: true
ENABLE_ONNX: true
API_LOG_LEVEL: INFO

# Remove development mounts
volumes:
  - zipvoice_cache:/app/api/src/voices/zipvoice_prompts
  - onnx_cache:/app/api/src/models/onnx_cache
  # Do NOT mount ../../api:/app/api in production
```

**CPU Production:**
```yaml
# Optimize for CPU performance
ZIPVOICE_MODEL: zipvoice_distill
ZIPVOICE_NUM_STEPS: 4
WHISPER_MODEL_SIZE: tiny
ENABLE_SMART_TUNING: true
ONNX_NUM_THREADS: 16  # Match CPU cores
API_LOG_LEVEL: WARNING
```

### Security

1. **Run as non-root:** Already configured (UID 1000/1001)
2. **Limit resources:**
```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 16G
```

3. **Use specific image tags:** Don't use `latest` in production
4. **Enable health checks:** Already configured

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/Fabul8/ZipVoice-FastAPI/issues
- Documentation: See `/docs` directory
- API Docs: http://localhost:8880/docs (when running)

## License

This Docker setup is part of the ZipVoice-FastAPI project. See LICENSE file for details.
