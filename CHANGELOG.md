# Changelog

Notable changes to this project will be documented in this file.

Per-PR attribution and contributor credits are published automatically on the corresponding GitHub release page; this file is the curated, human-readable summary.

## [v0.8.2] - 2026-09-05
### Added
- Optional model auto-unload after an idle timeout (`MODEL_AUTO_UNLOAD_TIMEOUT_SECONDS`, default off) to release VRAM. Reloads on the next request. `/dev/model` reports load/idle state and `POST /dev/reload` pre-warms the model, both behind `ALLOW_DEV_UNLOAD`.
- `/v1/audio/voices` entries carry the per-voice grades from [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md), where graded; added in web player dropdown w/ hover for the target quality and training duration.
- Web player: `Normalize text` toggle in settings, off sends input as written (#391, #523 by @webdevsamran).

### Changed
- Sentence splitting via UAX #29 segmentation instead of regex; e.g: `etc.`, `Dr.`, and CJK punctuation split more accurately (#308, adapts #415 by @lionel-rowe).
- ROCm: MIOpen off by default for better performance re: tensor shape recompilation (2.4s vs 0.27s on gfx1100). `ENABLE_MIOPEN=true` restores it (#518 by @s-kerdel).
- Compose files no longer mount `api/` or set defaults already baked into the image (`DOWNLOAD_MODEL`, `PYTHONPATH`, etc).

### Fixed
- `DEFAULT_VOICE_CODE` now applies on the speech endpoints and `/dev/generate_from_phonemes` (#514, #515 by @Christian-Sidak, #516).
- Entrypoint and `start-*.sh` honour `HOST` / `PORT`, so `HOST=::` binds IPv6-only (#408, reported by @felixls).

## [v0.8.1] - 2026-08-24
### Fixed
- `/v1/audio/voices/combine` accepts weighted syntax (`af_bella(2)+af_sky(1)`), matching the speech endpoints (#285).
- Oversized or pause-heavy requests now return 400 to avoid exhausting memory (reported by @sshpie, GHSA-f64g-9jmv-pm22). Two new configurable limits:
  - `MAX_INPUT_LENGTH` (default 1_000_000) caps characters of text per request.
  - `MAX_TOTAL_PAUSE_S` (default 300) caps total `[pause:Ns]` / SSML `<break>` silence per request.
- Native Windows installs (`start-cpu.ps1` etc) no longer need a C++ toolchain: `pyopenjtalk-plus` (a drop-in fork with prebuilt Windows wheels) replaces `pyopenjtalk` on win32 only (#508, proposed by @siliconfps). Needs a recent `uv`. Linux, macOS, and Docker are unchanged.

### Changed
- Documented `WEB_CONCURRENCY` (uvicorn worker count) in `docs/configuration.md` for parallel model loads/concurrency (#115, #358).
- Improved time-to-first-audio; sentence phonemization emits to avoid a first full-request pass. Some gradual latency growth at larger input sizes due to normalization pass.

<div align="center">

|   Input    | v0.8.0 (eager) | v0.8.1 (lazy) |     |
|:-----------|---------------:|--------------:|----:|
| 5k chars   |         0.31 s |        0.26 s | -16% |
| 10k chars  |         0.30 s |        0.27 s | -10% |
| 50k chars  |         0.45 s |        0.29 s | -36% |
| 100k chars |         0.73 s |        0.31 s | -58% |
| 250k chars |         1.41 s |        0.45 s | -68% |
| 500k chars |         2.78 s |        0.56 s | -80% |
| 1M chars   |         5.21 s |        0.85 s | -84% |

</div>

## [v0.8.0] - 2026-08-14
### Added
- Multi-speaker input on `/v1/audio/speech` and `/dev/captioned_speech` (#294). Opt in per request with `allow_voice_tags: true`; disable server-wide with `ENABLE_VOICE_TAGS=false`.
  - Inline `[voice:name]` tags switch speaker mid-text.
  - `voice_aliases` mapping for named weighted voice mixes, with optional per-alias `rate`.
  - `/dev/captioned_speech` timestamps carry the resolved `voice` per word; the field is absent unless `allow_voice_tags` is on, so existing responses are unchanged.
- `POST /dev/dialogue` for ordered multi-speaker turns.
- SSML input (experimental). Disable server-wide with `ENABLE_SSML=false`.
  - `ssml: true` on `/v1/audio/speech` and `/dev/captioned_speech` translates and speaks in one call. Requires `allow_voice_tags: true`, since the translation emits `[voice:]` and `[rate:]` spans.
  - `POST /dev/ssml` returns the translated tokens as text instead, for inspecting them before synthesis.
- `return_timing` on `/v1/audio/speech`: per-chunk `{text, start, end}` JSON sidecar next to the download (powers the web reader).
- `MAX_PAUSE_DURATION_S` (default 60) caps a single `[pause:Ns]` tag or SSML `<break>`.
- Web UI:
  - Voice alias/tag cast builder with import/export, pinning, and per-alias rate, synced with the editor (re: parallel work by @radzrader, [#272](https://github.com/remsky/Kokoro-FastAPI/discussions/272)).
  - Read-along mode: sentence highlighting synced to playback, bidirectional click to seek.
  - Find/replace across pages, direct page-number entry, download menu (audio / timings / both).
- Wiki pages moved into `docs/`, versioned alongside the code.

### Changed
- Docker images compile to bytecode at build, ~40% faster startup.
- Containers launch uvicorn directly rather than through `uv run`, which resolves startup permission failures on Unraid and similar hosts.
- `[rate:]` tags scale the speaking voice's alias rate instead of replacing it, so a voice calibrated to 0.8 stays proportionally slower under `[rate:1.1]`. Matches how SSML engines treat rate.
- Speed bounds (0.25 to 4.0) shared across speed fields and SSML.
- Unrecognized `.env` keys warn at startup instead of refusing to boot.
- README config table now covers every setting.

### Fixed
- Long generations swap from the live stream to the finished file as soon as it lands, so the scrubber shows true duration and seeking works mid-run.
- Volume control state reconnected to the player.

### Removed
- Unused `ffmpeg` from all images (~600MB); audio encoding already runs through PyAV's bundled copy.
- Dead `pydub` dependency.
- Unreachable list form of `voice` from the speech parser and unused `VoiceCombineRequest` schema.
- Legacy Gradio UI (`ui/`) code cruft; superseded by the web player since ~v0.2.0
- Legacy ONNX config compose vars, endpoints e.g `/debug/session_pools`.
- `OUTPUT_DIR`, `OUTPUT_DIR_SIZE_LIMIT_MB`, `SAMPLE_RATE` settings, never read.

## [v0.7.2] 2026-08-06
### Security
- `fastapi>=0.128.8`, `starlette>=1.3.1` to close CVE-2025-62727 (quadratic `Range` header parsing in `FileResponse`, reachable through the audio download path) (#500).

### Changed
- CORS `allow_credentials` now defaults off. Starlette 1.x echoes the caller's origin with `allow-credentials: true` where 0.47 returned `*`; nothing here uses cookies or auth, so this keeps the prior behavior.
- Docker build cache moved from GHA to the GHCR registry so forks and local builds can pull it, plus uv cache mounts and reordered test-client layers (#501).
- `response_format` docs (correctly) now list `aac` as supported.

### Fixed
- FLAC and WAV no longer lose the tail end of the audio; better muxer header patching at finalize (#497, covers #448 and #463). Diagnosis by @Technologicat.

## [v0.7.1] - 2026-08-02
### Added
- `/v1/download/{filename}` takes an optional `?name=` save-as name (sanitized, stored extension kept) and sets it in `Content-Disposition`. Omitting it keeps the previous name.
- Web UI keyboard navigation and ARIA labeling across header, player controls, and editor.

### Changed
- `Content-Disposition` is now built by `FileResponse` rather than by hand, so the filename comes back quoted (`filename="x.mp3"`) instead of bare. The name itself is unchanged when `?name=` is omitted.
- Web UI restyle: better use of space, responsive down to slim widths, playbar pinned to the bottom on narrow viewports.
- Waveform slowed and softened, made framerate-independent, respects `prefers-reduced-motion`.
- README: AMD GPU (ROCm) troubleshooting, clarified docker-compose comments.

### Fixed
- Downloads save as `{voice}_{timestamp}.{format}`, not the temp name (#338). Covers right-click "Save audio as" too, since `Content-Disposition` outranks the link's `download` attribute.
- Aborted streams no longer surface as playback failures; a user-initiated `MEDIA_ERR_ABORTED` is told apart from a real error.
- Stream-to-file swap settles pending buffer operations instead of leaving the feeder awaiting forever.

## [v0.7.0] - 2026-07-31
### Added
- `AGENTS.md` contributor guidelines, plus `SKILL.md` notes for the API, benchmarks, and web areas.

### Changed / Optimizations
- Docker images build on Python 3.12 (project floor stays 3.10 for local installs). Rust dropped from the CPU builder.
- Runtime dependencies trimmed to remove deprecated imports
- bumped `requests`,`python-dotenv`, capped `transformers<6`
- Builds now explicitly require BuildKit (default since Docker 23, ~Jan 2023); utilizing `COPY --exclude`
- Model bake reworked to ensure weights land exactly once (whether prexisting or downloaded at build)
- ROCm image now bakes the model at build like CPU/GPU (instead of a first-run fetch)
- GPU runtime now only uses torch shipped cuDNN/etc via pip wheels (#482). (see table below for size changes)
- Transcription benchmark reports split by device; RTF and first-token baselines refreshed.

Compressed image sizes + new bases:
<div align="center">

| Image       |  v0.6.0  |  v0.7.0  | Runtime base                                        |
|:------------|---------:|---------:|:----------------------------------------------------|
| cpu         |  1.66 GB |  1.56 GB | `python:3.10-slim` -> `python:3.12-slim`             |
| gpu         |  6.81 GB |  4.68 GB | `cuda:12.6.3-cudnn-runtime` -> `cuda:12.6.3-base`    |
| gpu (cu128) |  8.11 GB |  5.23 GB | `cuda:12.8.1-cudnn-runtime` -> `cuda:12.8.1-base`    |
| rocm        | 13.08 GB | 13.36 GB | unchanged; model now baked in                        |

</div>

### Fixed
- Model validation checksums against downloaded release artifact to ensure consistency, downloads first to a temp dir to avoid clobbering pre-existing models in the case of network issues/corrupted downloads.
- Model validation also rejects any custom files under 100MB to avoids false pass results (e.g. a 9-byte "Not found"), allowing a re-download instead of passing (#301).
- `.dockerignore` Fixed pycache ignore pattern to `**/`  to ensure nested .pyc/etc stay out of build contexts.
- Removed dead `pydub` imports from the audio services.
- `_find_file` rejects lookups that escape its search roots (`../` sequences, absolute paths) to avoid unintentional exposure of files outside the voices/models/web dirs. Symlinks placed inside those dirs resolve as before.
- Text normalizer: anchored decimal regex to prevent quadratic backtracking on digit floods, reordered range substitution so `NUMBER_PATTERN` no longer swallows hyphens meant as range separators, added version-number handling (`2.0.1` renders as "two point zero point one" instead of being split).

## [v0.6.0] - 2026-07-12
### Breaking changes
- `POST /dev/unload` is off by default; set `ALLOW_DEV_UNLOAD=true` to enable, otherwise returns 403. Shipped open in v0.5.0, now opt-in (#483).
- `/debug/*` routes also set off by default, continuation of above; to avoid unintentional exposure of internals (stack traces, temp storage, CPU/mem/GPU); set `ENABLE_DEBUG_ENDPOINTS=true` to enable, otherwise 403's.
- Removed lingering deprecated `/debug/session_pools`

### Changed
- Documented API stability: `/v1/*` is the stable surface; `/dev/*` and `/debug/*` are operational helpers that may change or move behind flags between minor releases.

### Fixed
- OpenAI voice aliases pointed at legacy v0.19 voicepacks that sound degraded on the v1.0 model. Added the proper v1.0 `bf_isabella` and repointed `nova` (`bf_v0isabella` -> `bf_isabella`), `alloy`, `ash`, `coral`, `echo` to their v1.0 voices. The `v0*` voices stay available by explicit name. (#479)
- `/dev/captioned_speech` returned `timestamps: null` for non-English espeak voices (es/fr/it/hi/pt). Word timestamps are now derived from the model's own phoneme durations (`pred_dur`), so they match the audio exactly; falls back to the old behavior when word counts can't be reconciled. English path unchanged, ja/zh keep previous behavior. (#484)

## [v0.5.0] - 2026-06-06
### Added
- `POST /dev/unload` release model from VRAM without stopping container; lazy reload on next request. For freeing a shared GPU while idle. Reclaim scale with load (~0.7 GB; ~1.6 GB via long-form test on 4060Ti). (#474)
### Fixed
- Web UI long-playback bugfix around the 10-minute mark; in-browser audio buffer is now bounded ahead of `currentTime` with trailing eviction behind it, so long generations stop overflowing the SourceBuffer.
- Web UI stays responsive on extended sessions; waveform animation is transition-gated and `PlayerState` short-circuits no-op updates, so controls don't drift into lag after 10+ minutes of playback.
- Web UI MP3 seek/scrub works after stream completes; pausing or playback end auto-swaps to the full server file, allowing timeline navigation.

## [v0.4.0] - 2026-05-24
### Added
- GPU image variants for Blackwell / RTX 50-series (`:latest-cu128`, `:vX.Y.Z-cu128`, amd64 only) with PyTorch cu128 wheels (#443). Default `:latest` and new `:latest-cu126` alias stay on cu126 for Maxwell/Pascal compatibility.
- Integration test suite (`api/tests/integration/`, opt-in `integration` marker) and a `tts-api-test-client` image that round-trips speech through faster-whisper against a live server. Run via `docker/docker-compose.test.yml`.
- Web UI footer badge showing the server version from `/config`.

### Breaking changes
- `/v1/audio/voices` items in the `voices` array changed from plain strings to `{"id", "name"}` objects (#462) to match OpenWebUI/similar clients, and allow metadata in the response. Clients reading entries as strings will break; pass `?legacy=true` to restore the old item shape.
  - Old: `{"voices": ["af_heart", ...]}`
  - New: `{"voices": [{"id": "af_heart", "name": "af_heart"}, ...]}`

### Changed
- `api_version` now read from the `VERSION` file instead of hardcoded.
- Removed the legacy `docker/{cpu,gpu}/Dockerfile`; the `.optimized` variants are the only build files now.
- Docker images carry OCI metadata so GHCR pages render properly. Integration compose defaults to the published test-client image.
- ROCm image defaults to `MIOPEN_FIND_MODE=2` so the on-disk kernel cache is reused instead of re-searched per process, and ships an opt-in warmup script at `docker/rocm/warmup_miopen.py` to pre-populate it. Recipe and benchmarks from @realugbun in #454.

### Fixed
- WAV responses drop junk size-field trailer that decoded as a click at chunk end. (#463)
- ROCm MIOpen cache set to persist across compose restarts; switched bind mounts to named volumes at the path MIOpen writes to (prior mounts targeted an inaccessible location).
- cpu/gpu composes set `DOWNLOAD_MODEL=true` for an idempotent model fetch on startup.
- `VERSION` shipped into images so `/config` reports the real server version.
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
