#!/bin/bash
# ============================================================================
# ZipVoice TTS API - Container Entrypoint Script
# ============================================================================
# This script initializes the container and starts the FastAPI server
# ============================================================================

set -e

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Initialization
# ============================================================================

log_info "Starting ZipVoice TTS API initialization..."

# Check required environment variables
if [ -z "$PYTHONPATH" ]; then
    log_warning "PYTHONPATH not set, using default: /app:/app/api"
    export PYTHONPATH=/app:/app/api
fi

# Set default log level if not specified
if [ -z "$API_LOG_LEVEL" ]; then
    export API_LOG_LEVEL=info
fi

# Convert log level to lowercase for uvicorn
LOG_LEVEL=$(echo "$API_LOG_LEVEL" | tr '[:upper:]' '[:lower:]')

# ============================================================================
# Directory Setup
# ============================================================================

log_info "Setting up directories..."

# Create necessary directories
mkdir -p /app/api/src/voices/zipvoice_prompts
mkdir -p /app/api/src/models/onnx_cache
mkdir -p /app/api/src/models/tensorrt_cache
mkdir -p /app/api/temp_files
mkdir -p /app/api/logs

log_success "Directories created"

# ============================================================================
# Environment Information
# ============================================================================

log_info "Environment Configuration:"
echo "  - Backend: ${DEFAULT_BACKEND:-zipvoice}"
echo "  - Device: ${DEVICE_TYPE:-auto}"
echo "  - GPU Enabled: ${USE_GPU:-false}"
echo "  - ZipVoice Model: ${ZIPVOICE_MODEL:-zipvoice}"
echo "  - Inference Steps: ${ZIPVOICE_NUM_STEPS:-8}"
echo "  - Auto-Transcription: ${ENABLE_AUTO_TRANSCRIPTION:-false}"
echo "  - Whisper Model: ${WHISPER_MODEL_SIZE:-base}"
echo "  - Smart Tuning: ${ENABLE_SMART_TUNING:-false}"
echo "  - Quality Detection: ${ENABLE_QUALITY_DETECTION:-false}"
echo "  - ONNX Optimization: ${ENABLE_ONNX:-false}"
echo "  - TensorRT Optimization: ${ENABLE_TENSORRT:-false}"
echo "  - Log Level: $LOG_LEVEL"

# ============================================================================
# Optional: Pre-download Whisper Model
# ============================================================================

if [ "$ENABLE_AUTO_TRANSCRIPTION" = "true" ]; then
    log_info "Checking Whisper model cache..."
    WHISPER_SIZE=${WHISPER_MODEL_SIZE:-base}

    # Try to pre-load the model (will use cache if available)
    python3 -c "
import whisper
try:
    model = whisper.load_model('$WHISPER_SIZE')
    print('✓ Whisper model loaded successfully')
except Exception as e:
    print(f'⚠ Whisper model loading failed: {e}')
    print('  (will be downloaded on first use)')
" 2>/dev/null || log_warning "Whisper model will be downloaded on first use"
fi

# ============================================================================
# Optional: Verify GPU Access
# ============================================================================

if [ "$USE_GPU" = "true" ]; then
    log_info "Verifying GPU access..."

    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  Device count: {torch.cuda.device_count()}')
else:
    print('⚠ GPU not available, will use CPU')
" || log_warning "Could not verify GPU access"
fi

# ============================================================================
# Start FastAPI Server
# ============================================================================

log_success "Initialization complete!"
log_info "Starting FastAPI server on port ${PORT:-8880}..."
echo ""

# Determine which extra to use based on device
EXTRA_FLAG="zipvoice"
if [ "$USE_GPU" = "true" ]; then
    EXTRA_FLAG="gpu,zipvoice"
else
    EXTRA_FLAG="cpu,zipvoice"
fi

# Start the server
exec python -m uvicorn api.src.main:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8880} \
    --log-level "$LOG_LEVEL"
