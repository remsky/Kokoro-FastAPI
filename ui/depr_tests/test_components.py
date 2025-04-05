import gradio as gr
import pytest

from ui.lib.components.model import create_model_column
from ui.lib.components.output import create_output_column
from ui.lib.config import AUDIO_FORMATS


def test_create_model_column_structure():
    """Test that create_model_column returns the expected structure"""
    voice_ids = ["voice1", "voice2"]
    column, components = create_model_column(voice_ids)

    # Test return types
    assert isinstance(column, gr.Column)
    assert isinstance(components, dict)

    # Test expected components presence
    expected_components = {"status_btn", "voice", "format", "speed"}
    assert set(components.keys()) == expected_components

    # Test component types
    assert isinstance(components["status_btn"], gr.Button)
    assert isinstance(components["voice"], gr.Dropdown)
    assert isinstance(components["format"], gr.Dropdown)
    assert isinstance(components["speed"], gr.Slider)


def test_model_column_default_values():
    """Test the default values of model column components"""
    voice_ids = ["voice1", "voice2"]
    _, components = create_model_column(voice_ids)

    # Test voice dropdown
    # Gradio Dropdown converts choices to (value, label) tuples
    expected_choices = [(voice_id, voice_id) for voice_id in voice_ids]
    assert components["voice"].choices == expected_choices
    # Value is not converted to tuple format for the value property
    assert components["voice"].value == [voice_ids[0]]
    assert components["voice"].interactive is True
    assert components["voice"].multiselect is True
    assert components["voice"].label == "Voice(s)"

    # Test format dropdown
    # Gradio Dropdown converts choices to (value, label) tuples
    expected_format_choices = [(fmt, fmt) for fmt in AUDIO_FORMATS]
    assert components["format"].choices == expected_format_choices
    assert components["format"].value == "mp3"

    # Test speed slider
    assert components["speed"].minimum == 0.5
    assert components["speed"].maximum == 2.0
    assert components["speed"].value == 1.0
    assert components["speed"].step == 0.1


def test_model_column_no_voices():
    """Test model column creation with no voice IDs"""
    _, components = create_model_column([])

    assert components["voice"].choices == []
    assert components["voice"].value is None


def test_create_output_column_structure():
    """Test that create_output_column returns the expected structure"""
    column, components = create_output_column()

    # Test return types
    assert isinstance(column, gr.Column)
    assert isinstance(components, dict)

    # Test expected components presence
    expected_components = {
        "audio_output",
        "output_files",
        "play_btn",
        "selected_audio",
        "clear_outputs",
    }
    assert set(components.keys()) == expected_components

    # Test component types
    assert isinstance(components["audio_output"], gr.Audio)
    assert isinstance(components["output_files"], gr.Dropdown)
    assert isinstance(components["play_btn"], gr.Button)
    assert isinstance(components["selected_audio"], gr.Audio)
    assert isinstance(components["clear_outputs"], gr.Button)


def test_output_column_configuration():
    """Test the configuration of output column components"""
    _, components = create_output_column()

    # Test audio output configuration
    assert components["audio_output"].label == "Generated Speech"
    assert components["audio_output"].type == "filepath"

    # Test output files dropdown
    assert components["output_files"].label == "Previous Outputs"
    assert components["output_files"].allow_custom_value is True

    # Test play button
    assert components["play_btn"].value == "▶️ Play Selected"
    assert components["play_btn"].size == "sm"

    # Test selected audio configuration
    assert components["selected_audio"].label == "Selected Output"
    assert components["selected_audio"].type == "filepath"
    assert components["selected_audio"].visible is False

    # Test clear outputs button
    assert components["clear_outputs"].size == "sm"
    assert components["clear_outputs"].variant == "secondary"
