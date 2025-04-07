from typing import Tuple

import gradio as gr

from .. import files


def create_input_column(disable_local_saving: bool = False) -> Tuple[gr.Column, dict]:
    """Create the input column with text input and file handling."""
    with gr.Column(scale=1) as col:
        text_input = gr.Textbox(
            label="Text to speak", placeholder="Enter text here...", lines=4
        )

        # Always show file upload but handle differently based on disable_local_saving
        file_upload = gr.File(label="Upload Text File (.txt)", file_types=[".txt"])

        if not disable_local_saving:
            # Show full interface with tabs when saving is enabled
            with gr.Tabs() as tabs:
                # Set first tab as selected by default
                tabs.selected = 0
                # Direct Input Tab
                with gr.TabItem("Direct Input"):
                    text_submit_direct = gr.Button(
                        "Generate Speech", variant="primary", size="lg"
                    )

                # File Input Tab
                with gr.TabItem("From File"):
                    # Existing files dropdown
                    input_files_list = gr.Dropdown(
                        label="Select Existing File",
                        choices=files.list_input_files(),
                        value=None,
                    )

                    file_preview = gr.Textbox(
                        label="File Content Preview", interactive=False, lines=4
                    )

                    with gr.Row():
                        file_submit = gr.Button(
                            "Generate Speech", variant="primary", size="lg"
                        )
                        clear_files = gr.Button(
                            "Clear Files", variant="secondary", size="lg"
                        )
        else:
            # Just show the generate button when saving is disabled
            text_submit_direct = gr.Button(
                "Generate Speech", variant="primary", size="lg"
            )
            tabs = None
            input_files_list = None
            file_preview = None
            file_submit = None
            clear_files = None

    # Initialize components based on disable_local_saving
    if disable_local_saving:
        components = {
            "tabs": None,
            "text_input": text_input,
            "text_submit": text_submit_direct,
            "file_select": None,
            "file_upload": file_upload,  # Keep file upload even when saving is disabled
            "file_preview": None,
            "file_submit": None,
            "clear_files": None,
        }
    else:
        components = {
            "tabs": tabs,
            "text_input": text_input,
            "text_submit": text_submit_direct,
            "file_select": input_files_list,
            "file_upload": file_upload,
            "file_preview": file_preview,
            "file_submit": file_submit,
            "clear_files": clear_files,
        }

    return col, components
