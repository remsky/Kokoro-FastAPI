# TTS API test client image.
#
# A portable validator that drives a running OpenAI-compatible TTS server
# over its HTTP API and asserts the responses meet a documented contract:
# format-correct audio bytes, sensible streaming, valid phonemes, coherent
# word timestamps, and (for transcription tests) intelligible speech.
#
# Test code is NOT baked in. Mount the suite at /tests/integration and the
# entrypoint discovers it. This lets the image stay stable while suites
# iterate. Currently driven from api/tests/integration in this repo.
#
# Whisper weights ARE baked in (~470MB) so CI never depends on Hugging Face
# at test time.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/test-client/.venv \
    WHISPER_MODEL=/opt/whisper/small

# libsndfile1 is required by `soundfile` for decoding flac/ogg/wav formats.
# faster-whisper's own pyav dependency handles mp3/opus/aac. curl fetches
# the prebaked Whisper weights and the uv installer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /opt/test-client
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Pre-fetch the Whisper small CT2 weights into a known path. Loading via
# directory path skips Hugging Face entirely at test time.
RUN mkdir -p /opt/whisper/small \
    && curl -fSL https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.4/faster-whisper-small-ct2.tar.gz \
       | tar -xz -C /opt/whisper/small \
    && echo 'Whisper small weights extracted to /opt/whisper/small'

# Put the venv bin first so `pytest` resolves to the locked one.
ENV PATH="/opt/test-client/.venv/bin:$PATH"

WORKDIR /tests
COPY pytest.ini /tests/pytest.ini
ENTRYPOINT ["pytest", "-m", "integration", "-v", "--tb=short", "-s"]
CMD ["integration/"]
