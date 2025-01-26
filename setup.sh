#!/bin/bash

# Ensure models directory exists
mkdir -p api/src/models

# Function to download a file
download_file() {
    local url="$1"
    local filename=$(basename "$url")
    echo "Downloading $filename..."
    curl -L "$url" -o "api/src/models/$filename"
}

# Default PTH model if no arguments provided
DEFAULT_MODELS=(
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.pth"
)

# Use provided models or default
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Download all models
for model in "${MODELS[@]}"; do
    download_file "$model"
done

echo "PyTorch model download complete!"