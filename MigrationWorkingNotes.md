# UV Setup
Deprecated notes for myself
## Structure
```
docker/
  ├── cpu/
  │   ├── pyproject.toml     # CPU deps (torch CPU)
  │   └── requirements.lock  # CPU lockfile
  ├── gpu/
  │   ├── pyproject.toml     # GPU deps (torch CUDA)
  │   └── requirements.lock  # GPU lockfile
  ├── rocm/
  │   ├── pyproject.toml     # ROCM deps (torch ROCM)
  │   └── requirements.lock  # ROCM lockfile
  └── shared/
      └── pyproject.toml     # Common deps
```

## Regenerate Lock Files

### CPU
```bash
cd docker/cpu
uv pip compile pyproject.toml ../shared/pyproject.toml --output-file requirements.lock
```

### GPU
```bash
cd docker/gpu
uv pip compile pyproject.toml ../shared/pyproject.toml --output-file requirements.lock
```

### ROCM
```bash
cd docker/rocm
uv pip compile pyproject.toml ../shared/pyproject.toml --output-file requirements.lock
```

## Local Dev Setup

### CPU
```bash
cd docker/cpu
uv venv
.venv\Scripts\activate  # Windows
uv pip sync requirements.lock
```

### GPU
```bash
cd docker/gpu
uv venv
.venv\Scripts\activate  # Windows
uv pip sync requirements.lock --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
```

### ROCM
```bash
cd docker/rocm
uv venv
source .venv/bin/activate
# not tested on Windows
#.venv\Scripts\activate  # Windows
uv pip sync requirements.lock --extra-index-url https://download.pytorch.org/whl/rocm6.2
```

### Run Server
```bash
# From project root with venv active:
uvicorn api.src.main:app --reload
```

## Docker

### CPU
```bash
cd docker/cpu
docker compose up
```

### GPU
```bash
cd docker/gpu
docker compose up
```

### ROCM
```bash
cd docker/rocm
docker compose up
```

## Known Issues
- Module imports: Run server from project root
- PyTorch CUDA: Always use --extra-index-url and --index-strategy for GPU env
