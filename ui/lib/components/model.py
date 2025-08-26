from typing import Optional, Tuple

import gradio as gr

from .. import api, config


def create_model_column(voice_ids: Optional[list] = None) -> Tuple[gr.Column, dict]:
    """Create the model settings column."""
    if voice_ids is None:
        voice_ids = []

    with gr.Column(scale=1) as col:
        gr.Markdown("### Model Settings")

        # Status button starts in waiting state
        status_btn = gr.Button(
            "⌛ TTS Service: Waiting for Service...", variant="secondary"
        )

        voice_input = gr.Dropdown(
            choices=voice_ids,
            label="Voice(s)",
            value=voice_ids[0] if voice_ids else None,
            interactive=True,
            multiselect=True,
        )
        format_input = gr.Dropdown(
            choices=config.AUDIO_FORMATS, label="Audio Format", value="mp3"
        )
        speed_input = gr.Slider(
            minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed"
        )

    components = {
        "status_btn": status_btn,
        "voice": voice_input,
        "format": format_input,
        "speed": speed_input,
    }

    return col, components
