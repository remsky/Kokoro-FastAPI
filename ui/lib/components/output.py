from typing import Tuple

import gradio as gr

from .. import files


def create_output_column() -> Tuple[gr.Column, dict]:
    """Create the output column with audio player and file list."""
    with gr.Column(scale=1) as col:
        gr.Markdown("### Latest Output")
        audio_output = gr.Audio(label="Generated Speech", type="filepath")

        gr.Markdown("### Generated Files")
        # Initialize dropdown with empty choices first
        output_files = gr.Dropdown(
            label="Previous Outputs",
            choices=[],
            value=None,
            allow_custom_value=True,
            interactive=True,
        )
        # Then update choices after component creation
        output_files.choices = files.list_output_files()

        play_btn = gr.Button("▶️ Play Selected", size="sm")

        selected_audio = gr.Audio(
            label="Selected Output", type="filepath", visible=False
        )

        clear_outputs = gr.Button(
            "⚠️ Delete All Previously Generated Output Audio 🗑️",
            size="sm",
            variant="secondary",
        )

    components = {
        "audio_output": audio_output,
        "output_files": output_files,
        "play_btn": play_btn,
        "selected_audio": selected_audio,
        "clear_outputs": clear_outputs,
    }

    return col, components
