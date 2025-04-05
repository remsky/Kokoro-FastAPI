import os
import shutil

import gradio as gr

from . import api, files


def setup_event_handlers(components: dict, disable_local_saving: bool = False):
    """Set up all event handlers for the UI components."""

    def refresh_status():
        try:
            is_available, voices = api.check_api_status()
            status = "Available" if is_available else "Waiting for Service..."

            if is_available and voices:
                # Preserve current voice selection if it exists and is still valid
                current_voice = components["model"]["voice"].value
                default_voice = current_voice if current_voice in voices else voices[0]
                return [
                    gr.update(
                        value=f"🔄 TTS Service: {status}",
                        interactive=True,
                        variant="secondary",
                    ),
                    gr.update(choices=voices, value=default_voice),
                ]
            return [
                gr.update(
                    value=f"⌛ TTS Service: {status}",
                    interactive=True,
                    variant="secondary",
                ),
                gr.update(choices=[], value=None),
            ]
        except Exception as e:
            print(f"Error in refresh status: {str(e)}")
            return [
                gr.update(
                    value="❌ TTS Service: Connection Error",
                    interactive=True,
                    variant="secondary",
                ),
                gr.update(choices=[], value=None),
            ]

    def handle_file_select(filename):
        if filename:
            try:
                text = files.read_text_file(filename)
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    return gr.update(value=preview)
            except Exception as e:
                print(f"Error reading file: {e}")
        return gr.update(value="")

    def handle_file_upload(file):
        if file is None:
            return (
                ""
                if disable_local_saving
                else [gr.update(choices=files.list_input_files())]
            )

        try:
            # Read the file content
            with open(file.name, "r", encoding="utf-8") as f:
                text_content = f.read()

            if disable_local_saving:
                # When saving is disabled, put content directly in text input
                # Normalize whitespace by replacing newlines with spaces
                normalized_text = " ".join(text_content.split())
                return normalized_text
            else:
                # When saving is enabled, save file and update dropdown
                filename = os.path.basename(file.name)
                target_path = os.path.join(files.INPUTS_DIR, filename)

                # Handle duplicate filenames
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(target_path):
                    new_name = f"{base}_{counter}{ext}"
                    target_path = os.path.join(files.INPUTS_DIR, new_name)
                    counter += 1

                shutil.copy2(file.name, target_path)
                return [gr.update(choices=files.list_input_files())]

        except Exception as e:
            print(f"Error handling file: {e}")
            return (
                ""
                if disable_local_saving
                else [gr.update(choices=files.list_input_files())]
            )

    def generate_from_text(text, voice, format, speed):
        """Generate speech from direct text input"""
        is_available, _ = api.check_api_status()
        if not is_available:
            gr.Warning("TTS Service is currently unavailable")
            return [None, gr.update(choices=files.list_output_files())]

        if not text or not text.strip():
            gr.Warning("Please enter text in the input box")
            return [None, gr.update(choices=files.list_output_files())]

        # Only save text if local saving is enabled
        if not disable_local_saving:
            files.save_text(text)

        result = api.text_to_speech(text, voice, format, speed)
        if result is None:
            gr.Warning("Failed to generate speech. Please try again.")
            return [None, gr.update(choices=files.list_output_files())]

        return [
            result,
            gr.update(
                choices=files.list_output_files(), value=os.path.basename(result)
            ),
        ]

    def generate_from_file(selected_file, voice, format, speed):
        """Generate speech from selected file"""
        is_available, _ = api.check_api_status()
        if not is_available:
            gr.Warning("TTS Service is currently unavailable")
            return [None, gr.update(choices=files.list_output_files())]

        if not selected_file:
            gr.Warning("Please select a file")
            return [None, gr.update(choices=files.list_output_files())]

        text = files.read_text_file(selected_file)
        result = api.text_to_speech(text, voice, format, speed)
        if result is None:
            gr.Warning("Failed to generate speech. Please try again.")
            return [None, gr.update(choices=files.list_output_files())]

        return [
            result,
            gr.update(
                choices=files.list_output_files(), value=os.path.basename(result)
            ),
        ]

    def play_selected(file_path):
        if file_path and os.path.exists(file_path):
            return gr.update(value=file_path, visible=True)
        return gr.update(visible=False)

    def clear_files(voice, format, speed):
        """Delete all input files and clear UI components while preserving model settings"""
        files.delete_all_input_files()
        return [
            gr.update(value=None, choices=[]),  # file_select
            None,  # file_upload
            gr.update(value=""),  # file_preview
            None,  # audio_output
            gr.update(choices=files.list_output_files()),  # output_files
            gr.update(value=voice),  # voice
            gr.update(value=format),  # format
            gr.update(value=speed),  # speed
        ]

    def clear_outputs():
        """Delete all output audio files and clear audio components"""
        files.delete_all_output_files()
        return [
            None,  # audio_output
            gr.update(choices=[], value=None),  # output_files
            gr.update(visible=False),  # selected_audio
        ]

    # Connect event handlers
    components["model"]["status_btn"].click(
        fn=refresh_status,
        outputs=[components["model"]["status_btn"], components["model"]["voice"]],
    )

    # Connect text submit button (always present)
    components["input"]["text_submit"].click(
        fn=generate_from_text,
        inputs=[
            components["input"]["text_input"],
            components["model"]["voice"],
            components["model"]["format"],
            components["model"]["speed"],
        ],
        outputs=[
            components["output"]["audio_output"],
            components["output"]["output_files"],
        ],
    )

    # Only connect file-related handlers if components exist
    if components["input"]["file_select"] is not None:
        components["input"]["file_select"].change(
            fn=handle_file_select,
            inputs=[components["input"]["file_select"]],
            outputs=[components["input"]["file_preview"]],
        )

    if components["input"]["file_upload"] is not None:
        # File upload handler - output depends on disable_local_saving
        components["input"]["file_upload"].upload(
            fn=handle_file_upload,
            inputs=[components["input"]["file_upload"]],
            outputs=[
                components["input"]["text_input"]
                if disable_local_saving
                else components["input"]["file_select"]
            ],
        )

    if components["output"]["play_btn"] is not None:
        components["output"]["play_btn"].click(
            fn=play_selected,
            inputs=[components["output"]["output_files"]],
            outputs=[components["output"]["selected_audio"]],
        )

    if components["input"]["clear_files"] is not None:
        components["input"]["clear_files"].click(
            fn=clear_files,
            inputs=[
                components["model"]["voice"],
                components["model"]["format"],
                components["model"]["speed"],
            ],
            outputs=[
                components["input"]["file_select"],
                components["input"]["file_upload"],
                components["input"]["file_preview"],
                components["output"]["audio_output"],
                components["output"]["output_files"],
                components["model"]["voice"],
                components["model"]["format"],
                components["model"]["speed"],
            ],
        )

    if components["output"]["clear_outputs"] is not None:
        components["output"]["clear_outputs"].click(
            fn=clear_outputs,
            outputs=[
                components["output"]["audio_output"],
                components["output"]["output_files"],
                components["output"]["selected_audio"],
            ],
        )

    if components["input"]["file_submit"] is not None:
        components["input"]["file_submit"].click(
            fn=generate_from_file,
            inputs=[
                components["input"]["file_select"],
                components["model"]["voice"],
                components["model"]["format"],
                components["model"]["speed"],
            ],
            outputs=[
                components["output"]["audio_output"],
                components["output"]["output_files"],
            ],
        )
