# Troubleshooting

*Last updated: 2026-08-11*

## Common Base URLs

The base URL for accessing the API endpoints can vary depending on how you are accessing the service, your docker setup, etc. Here are the common base URLs:

- **From the host machine**: `http://localhost:8880`
- **From another Docker container**: `http://host.docker.internal:8880`

The second resolves to the host machine's IP address from within the Docker network.

If neither is reaching the service, check that it's running on the expected port, that the containers can see the host, and that nothing is filtering the traffic.

## Missing models

The prebuilt images have the model baked in, so this only comes up when running from source.

```bash
python docker/scripts/download_model.py --output api/src/models/v1_0
```

Or the bash version, which finds the project root and output path on its own:

```bash
./docker/scripts/download_model.sh
```

Both pull `kokoro-v1_0.pth` and `config.json`, and exit early if valid copies are already in place.

## Linux GPU Permissions

Some Linux users may encounter GPU permission issues when running as non-root.
Can't guarantee anything, but here are some common solutions, consider your security requirements carefully

### Option 1: Container Groups (Likely the best option)
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    group_add:
      - "video"
      - "render"
```

### Option 2: Host System Groups
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    user: "${UID}:${GID}"
    group_add:
      - "video"
```
Note: May require adding host user to groups: `sudo usermod -aG docker,video $USER` and system restart.

### Option 3: Device Permissions (Use with caution)
```yaml
services:
  kokoro-tts:
    # ... existing config ...
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
```
Warning: Reduces system security. Use only in development environments.

Prerequisites: NVIDIA GPU, drivers, and container toolkit must be properly configured.

Visit [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for more detailed information

## AMD GPU (ROCm)

The ROCm image is experimental, x86_64 only. Findings below are largely from [discussion #151](https://github.com/remsky/Kokoro-FastAPI/discussions/151).

### Native Linux host required

`/dev/kfd` and `/dev/dri` passthrough does not work through Docker Desktop on Windows, or through WSL2. Reports of it working are all on a native Linux host.

### "HIP error: invalid device function" / card not detected

Set `HSA_OVERRIDE_GFX_VERSION` to the LLVM target of the closest officially supported architecture. Common values:

| Card | Value |
| --- | --- |
| RX 7900 XTX / XT | `11.0.0` |
| RDNA 3 iGPU (780M, 7840HS) | `11.0.2` or `11.0.3` |
| RX 6700 XT / 6600 (gfx1031, gfx1032) | `10.3.0` |
| RX 5700 XT (unofficial, mixed reports) | `10.3.0` |

The RX 6800/6900 (gfx1030) are supported directly and need no override.

```yaml
services:
  kokoro-tts:
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Check what your card reports with `rocminfo | grep gfx`.

### Slow or unstable matmuls

hipBLASLt does not cover every architecture. Falling back to hipBLAS is slower on paper but more reliable on consumer cards:

```yaml
      - TORCH_BLAS_PREFER_HIPBLASLT=0
      - PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
```

### First request is slow

MIOpen compiles a kernel per unique tensor shape, which costs 5-60s a shape. Kokoro's decoder length is derived from predicted durations, so it changes with the input text and most requests hit a shape that has never been seen.

Because of that, MIOpen is **disabled by default on ROCm**; PyTorch's shape-independent kernels are used instead. Measured on gfx1100 / ROCm 7.2, six novel texts: 2433ms per generation with MIOpen, 268ms without. If first requests are still slow, check the startup log for `MIOpen disabled`.

To go back to MIOpen, set `ENABLE_MIOPEN=true`. The image ships `MIOPEN_FIND_MODE=2` and prebaked kernel databases, but only for the architectures listed in `docker/rocm/kdb_install.sh` (CDNA plus gfx1030). RDNA 3 has no prebaked database, so the search runs on first use.

To pre-populate the on-disk cache, which `docker/rocm/docker-compose.yml` persists in named volumes:

```bash
cd docker/rocm
docker compose run --rm \
  -e MIOPEN_FIND_MODE=3 -e MIOPEN_FIND_ENFORCE=3 \
  kokoro-tts python docker/rocm/warmup_miopen.py
```

This sweeps every phoneme length up to 340 and takes hours (~2 on Strix Halo). Run it once per ROCm or PyTorch upgrade. Then start with `ENABLE_MIOPEN=true` and the default `MIOPEN_FIND_MODE=2`, which reuses the cache. `docker compose down -v` clears it.

Note the sweep covers phoneme counts, and shape is not a function of phoneme count: on gfx1100, 20 random texts of the same word count produced 20 distinct output lengths. It therefore cannot cover every shape.

Generating audio for a few paragraphs of varied length under the same overrides is the cheaper, partial version.

## Missing words & timestamps

The API normalizes input text, which can incorrectly remove or change some phrases. Disable it with `"normalization_options":{"normalize": false}` in the request json:
```python
import requests

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "af_heart",
        "response_format": "pcm",
        "normalization_options":
        {
            "normalize": False
        }
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        pass
```

## WAV duration reported as nonsense in some readers

WAV responses ship with streaming-sentinel (`0xFFFFFFFF`) size fields in the header. Most readers (`soundfile`, `pydub`/ffmpeg, browsers, OS players) handle this fine. Python's stdlib `wave` does not, and reports a bogus duration. Use `soundfile.info(path).duration` or `ffprobe` for exact length.
