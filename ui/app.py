import warnings

# Filter out Gradio Dropdown warnings about values not in choices
#TODO: Warning continues to be displayed, though it isn't breaking anything
warnings.filterwarnings('ignore', category=UserWarning, module='gradio.components.dropdown')

from lib.interface import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
