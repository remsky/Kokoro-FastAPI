from unittest.mock import mock_open, patch

import pytest
import requests

from ui.lib import api


@pytest.fixture
def mock_response():
    class MockResponse:
        def __init__(self, json_data, status_code=200, content=b"audio data"):
            self._json = json_data
            self.status_code = status_code
            self.content = content

        def json(self):
            return self._json

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    return MockResponse


def test_check_api_status_success(mock_response):
    """Test successful API status check"""
    mock_data = {"voices": ["voice1", "voice2"]}
    with patch("requests.get", return_value=mock_response(mock_data)):
        status, voices = api.check_api_status()
        assert status is True
        assert voices == ["voice1", "voice2"]


def test_check_api_status_no_voices(mock_response):
    """Test API response with no voices"""
    with patch("requests.get", return_value=mock_response({"voices": []})):
        status, voices = api.check_api_status()
        assert status is False
        assert voices == []


def test_check_api_status_timeout():
    """Test API timeout"""
    with patch("requests.get", side_effect=requests.exceptions.Timeout):
        status, voices = api.check_api_status()
        assert status is False
        assert voices == []


def test_check_api_status_connection_error():
    """Test API connection error"""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
        status, voices = api.check_api_status()
        assert status is False
        assert voices == []


def test_text_to_speech_success(mock_response, tmp_path):
    """Test successful speech generation"""
    with (
        patch("requests.post", return_value=mock_response({})),
        patch("ui.lib.api.OUTPUTS_DIR", str(tmp_path)),
        patch("builtins.open", mock_open()) as mock_file,
    ):
        result = api.text_to_speech("test text", "voice1", "mp3", 1.0)

        assert result is not None
        assert "output_" in result
        assert result.endswith(".mp3")
        mock_file.assert_called_once()


def test_text_to_speech_empty_text():
    """Test speech generation with empty text"""
    result = api.text_to_speech("", "voice1", "mp3", 1.0)
    assert result is None


def test_text_to_speech_timeout():
    """Test speech generation timeout"""
    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        result = api.text_to_speech("test", "voice1", "mp3", 1.0)
        assert result is None


def test_text_to_speech_request_error():
    """Test speech generation request error"""
    with patch("requests.post", side_effect=requests.exceptions.RequestException):
        result = api.text_to_speech("test", "voice1", "mp3", 1.0)
        assert result is None


def test_get_status_html_available():
    """Test status HTML generation for available service"""
    html = api.get_status_html(True)
    assert "green" in html
    assert "Available" in html


def test_get_status_html_unavailable():
    """Test status HTML generation for unavailable service"""
    html = api.get_status_html(False)
    assert "red" in html
    assert "Unavailable" in html


def test_text_to_speech_api_params(mock_response, tmp_path):
    """Test correct API parameters are sent"""
    test_cases = [
        # Single voice as string
        ("voice1", "voice1"),
        # Multiple voices as list
        (["voice1", "voice2"], "voice1+voice2"),
        # Single voice as list
        (["voice1"], "voice1"),
    ]

    for input_voice, expected_voice in test_cases:
        with (
            patch("requests.post") as mock_post,
            patch("ui.lib.api.OUTPUTS_DIR", str(tmp_path)),
            patch("builtins.open", mock_open()),
        ):
            mock_post.return_value = mock_response({})
            api.text_to_speech("test text", input_voice, "mp3", 1.5)

            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args

            # Check request body
            assert kwargs["json"] == {
                "model": "kokoro",
                "input": "test text",
                "voice": expected_voice,
                "response_format": "mp3",
                "speed": 1.5,
            }

            # Check headers and timeout
            assert kwargs["headers"] == {"Content-Type": "application/json"}
            assert kwargs["timeout"] == 300


def test_text_to_speech_output_filename(mock_response, tmp_path):
    """Test output filename contains correct voice identifier"""
    test_cases = [
        # Single voice
        ("voice1", lambda f: "voice-voice1" in f),
        # Multiple voices
        (["voice1", "voice2"], lambda f: "voice-voice1+voice2" in f),
    ]

    for input_voice, filename_check in test_cases:
        with (
            patch("requests.post", return_value=mock_response({})),
            patch("ui.lib.api.OUTPUTS_DIR", str(tmp_path)),
            patch("builtins.open", mock_open()) as mock_file,
        ):
            result = api.text_to_speech("test text", input_voice, "mp3", 1.0)

            assert result is not None
            assert filename_check(result), (
                f"Expected voice pattern not found in filename: {result}"
            )
            mock_file.assert_called_once()
