# Kokoro-FastAPI: Intel GPU (XPU) Setup

This directory contains the configuration to run Kokoro-FastAPI leveraging Intel GPUs (Arc, Data Center, and Integrated Graphics) through the Intel Extension for PyTorch (IPEX).

## Features
- **Intel XPU Acceleration**: Uses IPEX 2.5.10 + PyTorch 2.5.1 for optimized inference.
- **Automated Driver Stack**: Integrated with Intel's reference driver installation script.
- **Secure Non-Root User**: Runs as a standard `appuser` with correct GPU group permissions.

## Requirements
- **Host Drivers**: Ensure your host has the Intel GPU drivers installed (Compute Runtime, Level Zero).
- **Docker**: Version 20.10+ with `device_cgroup_rules` support.

## Usage

### 1. Build and Start
Run the following command from the project root:
```bash
docker compose -f docker/intel/docker-compose.yml up --build
```

### 2. Verify GPU Access
Check the container logs for XPU registration messages or run:
```bash
docker compose -f docker/intel/docker-compose.yml exec kokoro-tts python -c "import torch; import intel_extension_for_pytorch; print(f'XPU Available: {torch.xpu.is_available()}')"
```

## Technical Details
- **Base Image**: Ubuntu 24.04 (required for glibc compatibility).
- **Versions**:
  - Python: 3.12
  - Torch: 2.5.1
  - IPEX: 2.5.10+xpu
- **Groups**: The container uses a default `RENDER_GID=992`. If your host's `render` group has a different ID, pass it as a build argument:
  ```bash
  docker compose -f docker/intel/docker-compose.yml build --build-arg RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
  ```

## Troubleshooting
If you see `RuntimeError: Native API failed`, check:
1. That `/dev/dri` is correctly mapped.
2. That your host user has permissions to access `/dev/dri` (usually by being in the `video` or `render` groups).
3. That the `RENDER_GID` build argument matches your host's GID.
