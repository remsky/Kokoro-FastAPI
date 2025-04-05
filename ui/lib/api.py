import datetime
import os
from typing import List, Optional, Tuple

import requests

from .config import API_URL, OUTPUTS_DIR


def check_api_status() -> Tuple[bool, List[str]]:
    """Check TTS service status and get available voices."""
    try:
        # Use a longer timeout during startup
        response = requests.get(
            f"{API_URL}/v1/audio/voices",
            timeout=30,  # Increased timeout for initial startup period
        )
        response.raise_for_status()
        voices = response.json().get("voices", [])
        if voices:
            return True, voices
        print("No voices found in response")
        return False, []
    except requests.exceptions.Timeout:
        print("API request timed out (waiting for service startup)")
        return False, []
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error (service may be starting up): {str(e)}")
        return False, []
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return False, []
    except Exception as e:
        print(f"Unexpected error checking API status: {str(e)}")
        return False, []


def text_to_speech(
    text: str, voice_id: str | list, format: str, speed: float
) -> Optional[str]:
    """Generate speech from text using TTS API."""
    if not text.strip():
        return None

    # Handle multiple voices
    voice_str = voice_id if isinstance(voice_id, str) else "+".join(voice_id)

    # Create output filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"output_{timestamp}_voice-{voice_str}_speed-{speed}.{format}"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    try:
        response = requests.post(
            f"{API_URL}/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": voice_str,
                "response_format": format,
                "speed": float(speed),
            },
            headers={"Content-Type": "application/json"},
            timeout=300,  # Longer timeout for speech generation
        )
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path

    except requests.exceptions.Timeout:
        print("Speech generation request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Speech generation request failed: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error generating speech: {str(e)}")
        return None


def get_status_html(is_available: bool) -> str:
    """Generate HTML for status indicator."""
    color = "green" if is_available else "red"
    status = "Available" if is_available else "Unavailable"
    return f"""
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {color};"></div>
            <span>TTS Service: {status}</span>
        </div>
    """
