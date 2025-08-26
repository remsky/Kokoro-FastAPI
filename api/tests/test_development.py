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
