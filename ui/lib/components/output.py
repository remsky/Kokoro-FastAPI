from typing import Tuple

import gradio as gr

from .. import files


def create_output_column(disable_local_saving: bool = False) -> Tuple[gr.Column, dict]:
    """Create the output column with audio player and file list."""
    with gr.Column(scale=1) as col:
        gr.Markdown("### Latest Output")
        audio_output = gr.Audio(
            label="Generated Speech",
            type="filepath",
            waveform_options={"waveform_color": "#4C87AB"},
        )

        # Create file-related components with visible=False when local saving is disabled
        gr.Markdown("### Generated Files", visible=not disable_local_saving)
        output_files = gr.Dropdown(
            label="Previous Outputs",
            choices=files.list_output_files() if not disable_local_saving else [],
            value=None,
            allow_custom_value=True,
            visible=not disable_local_saving,
        )

        play_btn = gr.Button(
            "▶️ Play Selected",
            size="sm",
            visible=not disable_local_saving,
        )

        selected_audio = gr.Audio(
            label="Selected Output",
            type="filepath",
            visible=False,  # Always initially hidden
        )

        clear_outputs = gr.Button(
            "⚠️ Delete All Previously Generated Output Audio 🗑️",
            size="sm",
            variant="secondary",
            visible=not disable_local_saving,
        )

    components = {
        "audio_output": audio_output,
        "output_files": output_files,
        "play_btn": play_btn,
        "selected_audio": selected_audio,
        "clear_outputs": clear_outputs,
    }

    return col, components
