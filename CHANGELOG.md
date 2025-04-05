# Changelog

Notable changes to this project will be documented in this file.

## [v0.3.0] - 2025-04-04
### Added
- Apple Silicon (MPS) acceleration support for macOS users.
- Voice subtraction capability for creating unique voice effects.
- Windows PowerShell start scripts (`start-cpu.ps1`, `start-gpu.ps1`).
- Automatic model downloading integrated into all start scripts.
- Example Helm chart values for Azure AKS and Nvidia GPU Operator deployments.
- `CONTRIBUTING.md` guidelines for developers.

### Changed
- Version bump of underlying Kokoro and Misaki libraries
- Default API port reverted to 8880.
- Docker containers now run as a non-root user for enhanced security.
- Improved text normalization for numbers, currency, and time formats.
- Updated and improved Helm chart configurations and documentation.
- Enhanced temporary file management with better error tracking.
- Web UI dependencies (Siriwave) are now served locally.
- Standardized environment variable handling across shell/PowerShell scripts.

### Fixed
- Corrected an issue preventing download links from being returned when `streaming=false`.
- Resolved errors in Windows PowerShell scripts related to virtual environment activation order.
- Addressed potential segfaults during inference.
- Fixed various Helm chart issues related to health checks, ingress, and default values.
- Corrected audio quality degradation caused by incorrect bitrate settings in some cases.
- Ensured custom phonemes provided in input text are preserved.
- Fixed a 'MediaSource' error affecting playback stability in the web player.

### Removed
- Obsolete GitHub Actions build workflow, build and publish now occurs on merge to `Release` branch

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
