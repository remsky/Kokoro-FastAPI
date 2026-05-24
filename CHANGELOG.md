# Changelog

Notable changes to this project will be documented in this file.

Per-PR attribution and contributor credits are published automatically on the corresponding GitHub release page; this file is the curated, human-readable summary.

## [v0.4.0] - Unreleased
### Added
- GPU image variants for Blackwell / RTX 50-series (`:latest-cu128`, `:vX.Y.Z-cu128`, amd64 only) with PyTorch cu128 wheels (#443). Default `:latest` and new `:latest-cu126` alias stay on cu126 for Maxwell/Pascal compatibility.
- Integration test suite (`api/tests/integration/`, opt-in `integration` marker) and a `tts-api-test-client` image that round-trips speech through faster-whisper against a live server. Run via `docker/docker-compose.test.yml`.
- Web UI footer badge showing the server version from `/config`.

### Changed
- `/v1/audio/voices` default response shape changed to `[{"id", "name"}, ...]` so OpenAI-compatible clients like Open WebUI see the full voice catalog (#462). Pass `?legacy=true` to restore the old `string[]` shape.
- `api_version` now read from the `VERSION` file instead of hardcoded.
- Removed the legacy `docker/{cpu,gpu}/Dockerfile`; the `.optimized` variants are the only build files now.

### Fixed
- WAV responses drop junk size-field trailer that decoded as a click at chunk end. (#463)
- cpu/gpu composes set `DOWNLOAD_MODEL=true` for an idempotent model fetch on startup.
- Silence trimming no longer treats full-scale-negative samples as silent (`int16` `abs()` overflow).
- Fixed invalid escape sequences in the text-normalizer URL regex.
- CI test job uses the CPU PyTorch build and excludes integration tests by default.

## [v0.3.0] - 2026-05-15
### Added
- AMD GPU support via ROCm (`docker/rocm/` build, `rocm` extra in `pyproject.toml`). Also explored/proposed via @asheghi in #393.
- `gpt-4o-mini-tts` model alias for OpenAI-compatible clients.
- Reverse-proxy support for the Web UI (new `/config` endpoint exposing `UVICORN_ROOT_PATH`).
- Configurable logging level via the `API_LOG_LEVEL` environment variable.
- `INCLUDE_JAPANESE` Docker build flag for opt-in Japanese support.
- Transcription accuracy test harness under `examples/assorted_checks/test_transcription/` (baselines, multilingual reports, long-form runner).
- Override of `docker-bake.hcl` variables through GitHub Actions environment variables.

### Changed
- PyTorch bumped to 2.8.0 (x86_64: cu126, aarch64: cu129). x86_64 settled on cu126 to keep Maxwell/Pascal cards working, which drops native Blackwell (RTX 50-series) kernel support. Blackwell users need to override the torch index manually. See #443.
- `kokoro` bumped to 0.9.4 and `misaki` to 0.9.4 (proposed by @jcheek in #371, superceded).
- New optimized multi-stage Dockerfiles (`docker/{cpu,gpu}/Dockerfile.optimized`) become the default bake target. Reported image sizes: CPU 5.6 → 4.9 GB, GPU 14.8 → 9.9 GB.
- Parallelized Docker bake targets per architecture for faster CI.
- ROCBlas version pinned; ROCm docker-compose now builds locally.
- CI/release workflow hardening: pinned BuildKit/runners, branch-tagged builds, manifest fixes, `workflow_dispatch` ref and tag-check race fixed, `latest` tag gated.

### Fixed
- OGG/Opus audio truncation where the final page was lost during `write_chunk` finalize.
- Voice tensor loading hardened with `weights_only=True` (avoids unsafe pickle in `torch.load`).
- Per-request voice-tensor memory leak resolved via caching (#453), with cache cleared on unload.
- Custom phoneme handling made significantly more robust.
- Firefox Web UI playback falls back gracefully when `audio/mpeg` MSE is unsupported; waveform rendering bugfix bundled in the same web rewrite.
- CPU Docker builds: Rust now installed for `appuser` with proper `PATH` and longer `uv` timeouts.
- `cmake` added to CI deps to unblock `pyopenjtalk` builds (proposed by @jcheek in #371; superceded).
- `start-gpu.sh` uses `#!/usr/bin/env bash` for broader compatibility.
- Apple Silicon: `test_initial_state()` no longer fails.

## [v0.2.4] - 2025-06-18
### Added
- Apple Silicon (MPS) acceleration support for macOS users.
- Voice subtraction capability for creating unique voice effects.
- Windows PowerShell start scripts (`start-cpu.ps1`, `start-gpu.ps1`).
- Automatic model downloading integrated into all start scripts.
- Example Helm chart values for Azure AKS and Nvidia GPU Operator deployments.
- Volume multiplier setting.
- Chinese punctuation-based sentence splitting.
- `CONTRIBUTING.md` guidelines for developers.

### Changed
- Version bump of underlying Kokoro and Misaki libraries.
- Default API port reverted to 8880.
- Docker containers now run as a non-root user.
- Improved text normalization for numbers, currency, and time formats.
- Improved MP3 encoding and audio-pause handling.
- Updated and improved Helm chart configurations and documentation.
- Enhanced temporary file management with better error tracking.
- Web UI dependencies (Siriwave) are now served locally.
- Standardized environment variable handling across shell/PowerShell scripts.
- Rust installed in Dockerfile for builds requiring it.

### Fixed
- Download links no longer dropped when `streaming=false` and `return_download_link=true`.
- Windows PowerShell start scripts fixed around virtual-environment activation order.
- Potential segfaults during inference addressed.
- Helm chart issues around health checks, ingress, and default values.
- Audio-quality degradation from incorrect bitrate settings in some paths.
- Custom phonemes provided in input text are now preserved end-to-end.
- 'MediaSource' error affecting playback stability in the web player.
- CRLF line endings in `custom_responses.py` converted to LF.
- Money parsing and related tests.
- Additional safety checks on captioned-speech generation.
- Phoneme handling fixes.

### Removed
- Obsolete GitHub Actions build workflow; build and publish now occurs on merge to `Release` branch.

## [v0.2.3] - 2025-03-06
### Added
- Streaming word timestamps.
- `.gitattributes` for consistent line endings.

### Changed
- Text normalization improvements.

### Fixed
- Audio-quality regression caused by lower-bitrate encoding.
- Disabled uvicorn/FastAPI `--reload` to avoid pegging a CPU core.

## [v0.2.2] - 2025-02-13
### Added
- Helm chart.
- Settings-based override of the default `lang_code`.
- Advanced normalization settings.

### Fixed
- Speech not engaging reliably on the CPU image fallback.
- Audio quality bumped via adjusted compression settings.
- Web UI format-selection bug.

## [v0.2.1] - 2025-02-10
### Added
- Dummy `/v1/models` endpoint for OpenAI compatibility (#144).

### Changed
- Caption flow now streams audio with tempfile download at completion, removing duplicate captions (#139).

### Fixed
- Compatibility with the `espeak-loader` dependency on misaki (#127).
- Build system and model-download issues.

## [v0.2.0post1] - 2025-02-07
- Fix: Building Kokoro from source with adjustments, to avoid CUDA lock 
- Fixed ARM64 compatibility on Spacy dep to avoid emulation slowdown
- Added g++ for Japanese language support
- Temporarily disabled Vietnamese language support due to ARM64 compatibility issues

## [v0.2.0-pre] - 2025-02-06
### Added
- Complete Model Overhaul:
  - Upgraded to Kokoro v1.0 model architecture
  - Pre-installed multi-language support from Misaki:
    - English (en), Japanese (ja), Korean (ko),Chinese (zh), Vietnamese (vi)
  - All voice packs included for supported languages, along with the original versions.
- Enhanced Audio Generation Features:
  - Per-word timestamped caption generation
  - Phoneme-based audio generation capabilities
  - Detailed phoneme generation
- Web UI Improvements:
  - Improved voice mixing with weighted combinations
  - Text file upload support
  - Enhanced formatting and user interface
  - Cleaner UI (in progress)
  - Integration with https://github.com/hexgrad/kokoro and https://github.com/hexgrad/misaki packages

### Removed
- Deprecated support for Kokoro v0.19 model

### Changes
- Combine Voices endpoint now returns a .pt file, with generation combinations generated on the fly otherwise 


## [v0.1.4] - 2025-01-30
### Added
- Smart Chunking System:
  - New text_processor with smart_split for improved sentence boundary detection
  - Dynamically adjusts chunk sizes based on sentence structure, using phoneme/token information in an intial pass
  - Should avoid ever going over the 510 limit per chunk, while preserving natural cadence
- Web UI Added (To Be Replacing Gradio):
  - Integrated streaming with tempfile generation
  - Download links available in X-Download-Path header
  - Configurable cleanup triggers for temp files
- Debug Endpoints:
  - /debug/threads for thread information and stack traces
  - /debug/storage for temp file and output directory monitoring
  - /debug/system for system resource information
  - /debug/session_pools for ONNX/CUDA session status
- Automated Model Management:
  - Auto-download from releases page
  - Included download scripts for manual installation
  - Pre-packaged voice models in repository

### Changed
- Significant architectural improvements:
  - Multi-model architecture support
  - Enhanced concurrency handling
  - Improved streaming header management
  - Better resource/session pool management


## [v0.1.2] - 2025-01-23
### Structural Improvements
- Models can be manually download and placed in api/src/models, or use included script
- TTSGPU/TPSCPU/STTSService classes replaced with a ModelManager service
  - CPU/GPU of each of ONNX/PyTorch (Note: Only Pytorch GPU, and ONNX CPU/GPU have been tested)
  - Should be able to improve new models as they become available, or new architectures, in a more modular way
- Converted a number of internal processes to async handling to improve concurrency
- Improving separation of concerns towards plug-in and modular structure, making PR's and new features easier

### Web UI (test release)
- An integrated simple web UI has been added on the FastAPI server directly
  - This can be disabled via core/config.py or ENV variables if desired. 
  - Simplifies deployments, utility testing, aesthetics, etc 
  - Looking to deprecate/collaborate/hand off the Gradio UI


## [v0.1.0] - 2025-01-13
### Changed
- Major Docker improvements:
  - Baked model directly into Dockerfile for improved deployment reliability
  - Switched to uv for dependency management
  - Streamlined container builds and reduced image sizes
- Dependency Management:
  - Migrated from pip/poetry to uv for faster, more reliable package management
  - Added uv.lock for deterministic builds
  - Updated dependency resolution strategy

## [v0.0.5post1] - 2025-01-11
### Fixed
- Docker image tagging and versioning improvements (-gpu, -cpu, -ui)
- Minor vram management improvements
- Gradio bugfix causing crashes and errant warnings
- Updated GPU and UI container configurations

## [v0.0.5] - 2025-01-10
### Fixed
- Stabilized issues with images tagging and structures from v0.0.4
- Added automatic master to develop branch synchronization
- Improved release tagging and structures
- Initial CI/CD setup

## 2025-01-04
### Added
- ONNX Support:
  - Added single batch ONNX support for CPU inference
  - Roughly 0.4 RTF (2.4x real-time speed)

### Modified
- Code Refactoring:
  - Work on modularizing phonemizer and tokenizer into separate services
  - Incorporated these services into a dev endpoint
- Testing and Benchmarking:
  - Cleaned up benchmarking scripts
  - Cleaned up test scripts
  - Added auto-WAV validation scripts

## 2025-01-02
- Audio Format Support:
  - Added comprehensive audio format conversion support (mp3, wav, opus, flac)

## 2025-01-01
### Added
- Gradio Web Interface:
  - Added simple web UI utility for audio generation from input or txt file

### Modified
#### Configuration Changes
- Updated Docker configurations:
  - Changes to `Dockerfile`:
    - Improved layer caching by separating dependency and code layers
  - Updates to `docker-compose.yml` and `docker-compose.cpu.yml`:
    - Removed commit lock from model fetching to allow automatic model updates from HF
    - Added git index lock cleanup

#### API Changes
- Modified `api/src/main.py`
- Updated TTS service implementation in `api/src/services/tts.py`:
  - Added device management for better resource control:
    - Voices are now copied from model repository to api/src/voices directory for persistence
  - Refactored voice pack handling:
    - Removed static voice pack dictionary
    - On-demand voice loading from disk
  - Added model warm-up functionality:
    - Model now initializes with a dummy text generation
    - Uses default voice (af.pt) for warm-up
    - Model is ready for inference on first request
