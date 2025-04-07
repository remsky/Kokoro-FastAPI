from unittest.mock import MagicMock, PropertyMock, patch

import gradio as gr
import pytest

from ui.lib.interface import create_interface


@pytest.fixture
def mock_timer():
    """Create a mock timer with events property"""

    class MockEvent:
        def __init__(self, fn):
            self.fn = fn

    class MockTimer:
        def __init__(self):
            self._fn = None
            self.value = 5

        @property
        def events(self):
            return [MockEvent(self._fn)] if self._fn else []

        def tick(self, fn, outputs):
            self._fn = fn

    return MockTimer()


def test_create_interface_structure():
    """Test the basic structure of the created interface"""
    with patch("ui.lib.api.check_api_status", return_value=(False, [])):
        demo = create_interface()

        # Test interface type and theme
        assert isinstance(demo, gr.Blocks)
        assert demo.title == "Kokoro TTS Demo"
        assert isinstance(demo.theme, gr.themes.Monochrome)


def test_interface_html_links():
    """Test that HTML links are properly configured"""
    with patch("ui.lib.api.check_api_status", return_value=(False, [])):
        demo = create_interface()

        # Find HTML component
        html_components = [
            comp for comp in demo.blocks.values() if isinstance(comp, gr.HTML)
        ]
        assert len(html_components) > 0
        html = html_components[0]

        # Check for required links
        assert 'href="https://huggingface.co/hexgrad/Kokoro-82M"' in html.value
        assert 'href="https://github.com/remsky/Kokoro-FastAPI"' in html.value
        assert "Kokoro-82M HF Repo" in html.value
        assert "Kokoro-FastAPI Repo" in html.value


def test_update_status_available(mock_timer):
    """Test status update when service is available"""
    voices = ["voice1", "voice2"]
    with (
        patch("ui.lib.api.check_api_status", return_value=(True, voices)),
        patch("gradio.Timer", return_value=mock_timer),
    ):
        demo = create_interface()

        # Get the update function
        update_fn = mock_timer.events[0].fn

        # Test update with available service
        updates = update_fn()

        assert "Available" in updates[0]["value"]
        assert updates[1]["choices"] == voices
        assert updates[1]["value"] == voices[0]
        assert updates[2]["active"] is False  # Timer should stop


def test_update_status_unavailable(mock_timer):
    """Test status update when service is unavailable"""
    with (
        patch("ui.lib.api.check_api_status", return_value=(False, [])),
        patch("gradio.Timer", return_value=mock_timer),
    ):
        demo = create_interface()
        update_fn = mock_timer.events[0].fn

        updates = update_fn()

        assert "Waiting for Service" in updates[0]["value"]
        assert updates[1]["choices"] == []
        assert updates[1]["value"] is None
        assert updates[2]["active"] is True  # Timer should continue


def test_update_status_error(mock_timer):
    """Test status update when an error occurs"""
    with (
        patch("ui.lib.api.check_api_status", side_effect=Exception("Test error")),
        patch("gradio.Timer", return_value=mock_timer),
    ):
        demo = create_interface()
        update_fn = mock_timer.events[0].fn

        updates = update_fn()

        assert "Connection Error" in updates[0]["value"]
        assert updates[1]["choices"] == []
        assert updates[1]["value"] is None
        assert updates[2]["active"] is True  # Timer should continue


def test_timer_configuration(mock_timer):
    """Test timer configuration"""
    with (
        patch("ui.lib.api.check_api_status", return_value=(False, [])),
        patch("gradio.Timer", return_value=mock_timer),
    ):
        demo = create_interface()

        assert mock_timer.value == 5  # Check interval is 5 seconds
        assert len(mock_timer.events) == 1  # Should have one event handler


def test_interface_components_presence():
    """Test that all required components are present"""
    with patch("ui.lib.api.check_api_status", return_value=(False, [])):
        demo = create_interface()

        # Check for main component sections
        components = {
            comp.label
            for comp in demo.blocks.values()
            if hasattr(comp, "label") and comp.label
        }

        required_components = {
            "Text to speak",
            "Voice(s)",
            "Audio Format",
            "Speed",
            "Generated Speech",
            "Previous Outputs",
        }

        assert required_components.issubset(components)
