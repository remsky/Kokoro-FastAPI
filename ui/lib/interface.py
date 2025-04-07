import os

import gradio as gr

from . import api
from .components import create_input_column, create_model_column, create_output_column
from .handlers import setup_event_handlers


def create_interface():
    """Create the main Gradio interface."""
    # Skip initial status check - let the timer handle it
    is_available, available_voices = False, []

    # Check if local saving is disabled
    disable_local_saving = os.getenv("DISABLE_LOCAL_SAVING", "false").lower() == "true"

    with gr.Blocks(title="Kokoro TTS Demo", theme=gr.themes.Monochrome()) as demo:
        gr.HTML(
            value='<div style="display: flex; gap: 0;">'
            '<a href="https://huggingface.co/hexgrad/Kokoro-82M" target="_blank" style="color: #2196F3; text-decoration: none; margin: 2px; border: 1px solid #2196F3; padding: 4px 8px; height: 24px; box-sizing: border-box; display: inline-flex; align-items: center;">Kokoro-82M HF Repo</a>'
            '<a href="https://github.com/remsky/Kokoro-FastAPI" target="_blank" style="color: #2196F3; text-decoration: none; margin: 2px; border: 1px solid #2196F3; padding: 4px 8px; height: 24px; box-sizing: border-box; display: inline-flex; align-items: center;">Kokoro-FastAPI Repo</a>'
            "</div>",
            show_label=False,
        )

        # Main interface
        with gr.Row():
            # Create columns
            input_col, input_components = create_input_column(disable_local_saving)
            model_col, model_components = create_model_column(
                available_voices
            )  # Pass initial voices
            output_col, output_components = create_output_column(disable_local_saving)

            # Collect all components
            components = {
                "input": input_components,
                "model": model_components,
                "output": output_components,
            }

            # Set up event handlers
            setup_event_handlers(components, disable_local_saving)

        # Add periodic status check with Timer
        def update_status():
            try:
                is_available, voices = api.check_api_status()
                status = "Available" if is_available else "Waiting for Service..."

                if is_available and voices:
                    # Service is available, update UI and stop timer
                    current_voice = components["model"]["voice"].value
                    default_voice = (
                        current_voice if current_voice in voices else voices[0]
                    )
                    # Return values in same order as outputs list
                    return [
                        gr.update(
                            value=f"🔄 TTS Service: {status}",
                            interactive=True,
                            variant="secondary",
                        ),
                        gr.update(choices=voices, value=default_voice),
                        gr.update(active=False),  # Stop timer
                    ]

                # Service not available yet, keep checking
                return [
                    gr.update(
                        value=f"⌛ TTS Service: {status}",
                        interactive=True,
                        variant="secondary",
                    ),
                    gr.update(choices=[], value=None),
                    gr.update(active=True),
                ]
            except Exception as e:
                print(f"Error in status update: {str(e)}")
                # On error, keep the timer running but show error state
                return [
                    gr.update(
                        value="❌ TTS Service: Connection Error",
                        interactive=True,
                        variant="secondary",
                    ),
                    gr.update(choices=[], value=None),
                    gr.update(active=True),
                ]

        timer = gr.Timer(value=5)  # Check every 5 seconds
        timer.tick(
            fn=update_status,
            outputs=[
                components["model"]["status_btn"],
                components["model"]["voice"],
                timer,
            ],
        )

    return demo
