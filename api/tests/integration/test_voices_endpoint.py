"""Conformance for /v1/audio/voices shape + the OpenAI short-name mappings.

Default shape is `[{"id": ..., "name": ...}, ...]` for common
OpenAI-compatible clients (e.g. Open WebUI). `?legacy=true` preserves the
plain-string shape.
"""

from __future__ import annotations

import httpx
import pytest


pytestmark = pytest.mark.integration


def test_voices_default_shape(server_url: str):
    r = httpx.get(f"{server_url}/v1/audio/voices", timeout=10.0)
    r.raise_for_status()
    voices = r.json()["voices"]
    assert voices, "voice list is empty"
    assert all(isinstance(v, dict) for v in voices), (
        "default shape must be list[dict] for Open WebUI compatibility"
    )
    assert all(set(v.keys()) >= {"id", "name"} for v in voices), (
        "every entry needs id + name"
    )
    assert all(v["id"] == v["name"] for v in voices), (
        "id and name should match for Kokoro voices"
    )


def test_voices_legacy_shape(server_url: str):
    r = httpx.get(f"{server_url}/v1/audio/voices?legacy=true", timeout=10.0)
    r.raise_for_status()
    voices = r.json()["voices"]
    assert voices, "legacy voice list is empty"
    assert all(isinstance(v, str) for v in voices), (
        "?legacy=true must return plain strings"
    )


def test_nova_mapping_resolves(openai_client):
    """OpenAI short-names in openai_mappings.json must point at real voices."""
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input="Test.",
        response_format="wav",
    )
    assert response.content, "nova short-name produced empty audio"
    assert len(response.content) >= 1000, (
        "nova returned suspiciously small WAV; mapping likely broken again"
    )
