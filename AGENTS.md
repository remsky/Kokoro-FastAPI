# AGENTS.md

Guidance for coding agents working on Kokoro-FastAPI. 
Agent-assisted contributions are welcome, with caveats:
    - All judgement and accountability falls to the named human contributor as the reviewer. 
    - If the user doesn't demonstrate understanding a change, it's not ready to merge. 
    - Code style here must still match house style. 
    - To the above, explicitly: 
        - wanton em-dash usage is not permitted.
        - expositionary and excessive commenting is not permitted.
        - when in doubt, do less. Comments are reserved for decision points, or unexpected gotchas, etc.  


The general instructions must also still be followed, same as any contributor, and live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

- `api/src/routers/` - HTTP endpoints. `openai_compatible.py` is the main API surface.
- `api/src/services/` - TTS orchestration (`tts_service.py`), audio encoding and streaming (`audio.py`, `streaming_audio_writer.py`), `text_processing/` (normalization, phonemization).
- `api/src/inference/` - model backends (`kokoro_v1.py`), model and voice managers.
- `api/src/core/` - `config.py` (env-driven settings), `paths.py`, `openai_mappings.json` (OpenAI model/voice name mapping).
- `web/` - vanilla JS web player, served by the API. No framework, no build step.
- `docker/{cpu,gpu,rocm}/` - per-accelerator images and compose files. `docker-bake.hcl` defines the CI build targets.
- `examples/` - standalone samples with their own uv venv. Run `uv run` from inside `examples/`, not the root env.

## Commands

- Python tests: `uv run pytest` (runs `api/tests/`; integration tests are excluded by default, opt in with `-m integration`).
- Format and lint: `ruff format .` then `ruff check . --fix`.
- Web unit tests: `npm run test:web`. E2e: `npm run test:e2e`.
- Container smoke test: `npm run cpu:up` (or `gpu:up`, `rocm:up`).

## Conventions

- Commit and PR titles use conventional style with a scope: `fix(audio): ...`, `feat(api): ...`, `chore(docker): ...`.
- Every behavior change ships with a test (`api/tests/test_*.py` or `web/tests/`).
- Add a `CHANGELOG.md` entry for user-visible changes. Don't bump `VERSION`; that happens at release time.
- Endpoints that surface host, process, or model internals go behind an opt-in setting in `api/src/core/config.py` (see `enable_debug_endpoints`, `allow_dev_unload`) to avoid unintentional exposure on shared deployments.
- Keep changes small and modular; match the surrounding style.

## Gotchas

- Dockerfiles are tuned for multi-arch CI. Don't remove build tools or restructure stages without a tested build for every target. 
- Trust me on the above point. The dependencies are touchy for this stack. 
- Local (non-Docker) runs need espeak-ng installed and on PATH; the Docker images handle this separately.
- Japanese support (`misaki[ja]`) requires the full UniDic dictionary (~526MB), not `unidic-lite`. CI caches it.
- The release/publish workflow triggers itself on push to the `release` branch. It should never never be triggered manually. Tag @remsky to do so. 
