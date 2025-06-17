#!/usr/bin/env python3
"""
1. Uses huggingface_hub.snapshot_download() with HF_TOKEN to fetch your gated model.
2. Launches NVIDIA NIM microservice via Docker, mounting the downloaded model.
"""

import os
import subprocess
import sys
from huggingface_hub import snapshot_download

# ——— Configuration ———
HF_TOKEN = os.environ.get("HF_TOKEN")  # Expect you exported this already
if not HF_TOKEN:
    print("Error: Please set your HF_TOKEN environment variable.", file=sys.stderr)
    sys.exit(1)

# Replace with your HF org/model path
HF_REPO_ID = "yuiuo/alex-65_BATCH-gradient_65"  
# Where to cache the downloaded model
LOCAL_MODEL_DIR = os.path.expanduser("~/nim_models/your-gated-model")  

# NIM Docker image for your base model type; adjust if using a different microservice
NIM_IMAGE = "ghcr.io/nvidia/nim/text-generation-base:latest"  
# The environment variable inside the container pointing to your model store
NIM_MODEL_PATH = "/models/your-gated-model"  

# GPU flags (remove --gpus if CPU-only)
DOCKER_GPU_FLAG = "--gpus all"  

def download_model():
    """
    Pulls the gated HF model to LOCAL_MODEL_DIR using your read token.
    """
    print(f"Downloading '{HF_REPO_ID}' to '{LOCAL_MODEL_DIR}' …")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        cache_dir=LOCAL_MODEL_DIR,
        library_name="nim-integration",
        token=HF_TOKEN,
        local_files_only=False,  # ensure it pulls from remote
    )
    print("Download complete.\n")

def serve_with_nim():
    """
    Launches the NIM microservice Docker container, mounting the model dir.
    """
    cmd = [
        "docker", "run", "-d",
        DOCKER_GPU_FLAG,
        "--name", "nim-gated-model",
        "-p", "8000:8000",                         # expose port 8000 for inference
        "-e", f"NIM_MODEL_STORE={NIM_MODEL_PATH}", # tells NIM where to find models
        "-e", "NIM_REFRESH_INTERVAL=3600",         # reload every hour
        "-v", f"{LOCAL_MODEL_DIR}:{NIM_MODEL_PATH}:ro",  # mount your model readonly
        NIM_IMAGE
    ]
    print("Starting NIM container with command:")
    print("  " + " ".join(cmd))
    subprocess.check_call(cmd)
    print("\nNIM microservice is now running on http://localhost:8000")

def main():
    download_model()
    serve_with_nim()

if __name__ == "__main__":
    main()
