# Changelog

Notable changes to this project will be documented in this file.

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
