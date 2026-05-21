#!/bin/bash

# Find project root by looking for api directory
find_project_root() {
    local current_dir="$PWD"
    local max_steps=5
    local steps=0
    
    while [ $steps -lt $max_steps ]; do
        if [ -d "$current_dir/api" ]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
        ((steps++))
    done
    
    echo "Error: Could not find project root (no api directory found)" >&2
    exit 1
}

# Function to verify files exist and are valid
verify_files() {
    local model_path="$1"
    local config_path="$2"
    
    # Check files exist
    if [ ! -f "$model_path" ] || [ ! -f "$config_path" ]; then
        return 1
    fi
    
    # Check files are not empty
    if [ ! -s "$model_path" ] || [ ! -s "$config_path" ]; then
        return 1
    fi
    
    # Try to parse config.json
    if ! jq . "$config_path" >/dev/null 2>&1; then
        return 1
    fi
    
    return 0
}

# Function to download a file
download_file() {
    local url="$1"
    local output_path="$2"
    local filename=$(basename "$output_path")
    
    echo "Downloading $filename..."
    mkdir -p "$(dirname "$output_path")"
    if curl -L "$url" -o "$output_path"; then
        echo "Successfully downloaded $filename"
        return 0
    else
        echo "Error downloading $filename" >&2
        return 1
    fi
}

# Find project root and ensure models directory exists
PROJECT_ROOT=$(find_project_root)
if [ $? -ne 0 ]; then
    exit 1
fi

MODEL_DIR="$PROJECT_ROOT/api/src/models/v1_0"
echo "Model directory: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

# Define file paths
MODEL_FILE="kokoro-v1_0.pth"
CONFIG_FILE="config.json"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
CONFIG_PATH="$MODEL_DIR/$CONFIG_FILE"

# Check if files already exist and are valid
if verify_files "$MODEL_PATH" "$CONFIG_PATH"; then
    echo "Model files already exist and are valid"
    exit 0
fi

# Define URLs
BASE_URL="https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.4"
MODEL_URL="$BASE_URL/$MODEL_FILE"
CONFIG_URL="$BASE_URL/$CONFIG_FILE"

# Download files
success=true

if ! download_file "$MODEL_URL" "$MODEL_PATH"; then
    success=false
fi

if ! download_file "$CONFIG_URL" "$CONFIG_PATH"; then
    success=false
fi

# Verify downloaded files
if [ "$success" = true ] && verify_files "$MODEL_PATH" "$CONFIG_PATH"; then
    echo "✓ Model files prepared in $MODEL_DIR"
    exit 0
else
    echo "Failed to download or verify model files" >&2
    exit 1
fi