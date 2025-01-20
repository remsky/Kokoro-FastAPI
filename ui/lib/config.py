import os

# API Configuration
API_HOST = os.getenv("API_HOST", "kokoro-tts")
API_PORT = os.getenv("API_PORT", "8880")
API_URL = f"http://{API_HOST}:{API_PORT}"

# File paths
INPUTS_DIR = "app/ui/data/inputs"
OUTPUTS_DIR = "app/ui/data/outputs"

# Create directories if they don't exist

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Audio formats
AUDIO_FORMATS = ["mp3", "wav", "opus", "flac"]

# UI Theme
THEME = "monochrome"
CSS = """
.gradio-container {
    max-width: 1000px;
    margin: auto;
}

.banner-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin-bottom: 2rem;
}

.banner-container img {
    width: 100%;
    max-width: 600px;
    border-radius: 10px;
    margin: 20px auto;
    display: block;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
"""
