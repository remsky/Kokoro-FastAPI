"""Request bodies are bounded before parsing, and every text field has the input limit."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.src.core.config import settings
from api.src.main import app
from api.src.structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    SsmlRequest,
)

client = TestClient(app)
MB = 1024 * 1024


@pytest.fixture
def one_mb_limit(monkeypatch):
    monkeypatch.setattr(settings, "max_request_body_mb", 1)


def test_declared_oversize_body_is_413(one_mb_limit):
    response = client.post(
        "/v1/audio/speech",
        content=b"x" * (MB + 1),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["error"] == "too_large"


def test_chunked_oversize_body_is_413(one_mb_limit):
    def chunks():
        for _ in range(3):
            yield b"x" * (MB // 2)

    response = client.post(
        "/dev/phonemize",
        content=chunks(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_body_under_limit_reaches_the_route(one_mb_limit):
    response = client.post("/dev/phonemize", json={"text": 5})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "build",
    [
        lambda text: PhonemeRequest(text=text),
        lambda text: GenerateFromPhonemesRequest(phonemes=text, voice="af_heart"),
        lambda text: SsmlRequest(text=text),
    ],
)
def test_dev_text_fields_have_the_input_limit(build, monkeypatch):
    monkeypatch.setattr(settings, "max_input_length", 10)
    build("a" * 10)
    with pytest.raises(ValidationError):
        build("a" * 11)
