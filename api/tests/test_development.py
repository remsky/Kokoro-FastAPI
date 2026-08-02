import base64
import json
from unittest.mock import MagicMock, patch

import pytest
import requests


def test_generate_captioned_speech():
    """Test the generate_captioned_speech function with mocked responses"""
    # Mock the API responses
    mock_audio_response = MagicMock()
    mock_audio_response.status_code = 200

    mock_timestamps_response = MagicMock()
    mock_timestamps_response.status_code = 200
    mock_timestamps_response.content = json.dumps(
        {
            "audio": base64.b64encode(b"mock audio data").decode("utf-8"),
            "timestamps": [{"word": "test", "start_time": 0.0, "end_time": 1.0}],
        }
    )

    # Patch the HTTP requests
    with patch("requests.post", return_value=mock_timestamps_response):
        # Import here to avoid module-level import issues
        from examples.captioned_speech_example import generate_captioned_speech

        # Test the function
        audio, timestamps = generate_captioned_speech("test text")

        # Verify we got both audio and timestamps
        assert audio == b"mock audio data"
        assert timestamps == [{"word": "test", "start_time": 0.0, "end_time": 1.0}]


@pytest.mark.asyncio
async def test_phonemize_uses_configured_repo_id():
    """The phonemize endpoint must match synthesis, since repo_id picks the zh g2p version."""
    from api.src.core.config import settings
    from api.src.routers.development import phonemize_text
    from api.src.structures.text_schemas import PhonemeRequest

    mock_pipeline = MagicMock(return_value=iter([MagicMock(phonemes="ni˨˩ hao˨˩")]))
    with patch(
        "api.src.routers.development.KPipeline", return_value=mock_pipeline
    ) as mock_kpipeline:
        response = await phonemize_text(PhonemeRequest(text="你好", language="z"))

    mock_kpipeline.assert_called_once_with(
        lang_code="z", model=False, repo_id=settings.model_repo_id
    )
    assert response.phonemes == "ni˨˩ hao˨˩"
