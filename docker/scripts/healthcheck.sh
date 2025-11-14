#!/bin/bash
# ============================================================================
# ZipVoice TTS API - Health Check Script
# ============================================================================
# This script performs a health check on the running API server
# Returns 0 (success) if healthy, 1 (failure) if unhealthy
# ============================================================================

set -e

# Configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8880}"
TIMEOUT=5

# Health check endpoint
HEALTH_URL="http://localhost:${PORT}/health"
DOCS_URL="http://localhost:${PORT}/docs"

# Check if curl is available, fallback to wget
if command -v curl &> /dev/null; then
    USE_CURL=1
elif command -v wget &> /dev/null; then
    USE_CURL=0
else
    echo "ERROR: Neither curl nor wget is available"
    exit 1
fi

# Function to make HTTP request
http_get() {
    local url=$1
    if [ "$USE_CURL" -eq 1 ]; then
        curl -f -s -m "$TIMEOUT" "$url" > /dev/null 2>&1
    else
        wget -q -O /dev/null -T "$TIMEOUT" "$url" > /dev/null 2>&1
    fi
    return $?
}

# Try to reach the health endpoint
if http_get "$HEALTH_URL"; then
    exit 0
fi

# If /health endpoint doesn't exist, try /docs (OpenAPI)
if http_get "$DOCS_URL"; then
    exit 0
fi

# If both fail, check if the process is at least listening on the port
if command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":${PORT} "; then
        # Port is listening, might be starting up
        exit 0
    fi
elif command -v ss &> /dev/null; then
    if ss -tuln | grep -q ":${PORT} "; then
        # Port is listening, might be starting up
        exit 0
    fi
fi

# All checks failed
exit 1
