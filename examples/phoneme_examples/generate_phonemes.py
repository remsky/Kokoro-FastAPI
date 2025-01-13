import json
from typing import Tuple, Optional
from pathlib import Path

import requests

# Get the directory this script is in
SCRIPT_DIR = Path(__file__).parent.absolute()


def get_phonemes(text: str, language: str = "a") -> Tuple[str, list[int]]:
    """Get phonemes and tokens for input text.

    Args:
        text: Input text to convert to phonemes
        language: Language code (defaults to "a" for American English)

    Returns:
        Tuple of (phonemes string, token list)
    """
    # Create the request payload
    payload = {"text": text, "language": language}

    # Make POST request to the phonemize endpoint
    response = requests.post("http://localhost:8880/text/phonemize", json=payload)

    # Raise exception for error status codes
    response.raise_for_status()

    # Parse the response
    result = response.json()
    return result["phonemes"], result["tokens"]


def generate_audio_from_phonemes(
    phonemes: str, voice: str = "af_bella", speed: float = 1.0
) -> Optional[bytes]:
    """Generate audio from phonemes.

    Args:
        phonemes: Phoneme string to synthesize
        voice: Voice ID to use (defaults to af_bella)
        speed: Speed factor (defaults to 1.0)

    Returns:
        WAV audio bytes if successful, None if failed
    """
    # Create the request payload
    payload = {"phonemes": phonemes, "voice": voice, "speed": speed}

    # Make POST request to generate audio
    response = requests.post(
        "http://localhost:8880/text/generate_from_phonemes", json=payload
    )

    # Raise exception for error status codes
    response.raise_for_status()

    return response.content


def main():
    # Example texts to convert
    examples = [
        "Hello world! Welcome to the phoneme generation system.",
        "How are you today? I am doing reasonably well, thank you for asking",
        """This is a test of the phoneme generation system. Do not be alarmed.
        This is only a test. If this were a real phoneme emergency, '
        you would be instructed to a phoneme shelter in your area.""",
    ]

    print("Generating phonemes and audio for example texts...\n")

    # Create output directory in same directory as script
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    for i, text in enumerate(examples):
        print(f"{len(text)}: Input text: {text}")
        try:
            # Get phonemes
            phonemes, tokens = get_phonemes(text)
            print(f"{len(phonemes)} Phonemes: {phonemes}")
            print(f"{len(tokens)} Tokens: {tokens}")

            # Generate audio from phonemes
            print("Generating audio...")
            audio_bytes = generate_audio_from_phonemes(phonemes)

            if audio_bytes:
                # Save audio file
                output_path = output_dir / f"example_{i+1}.wav"
                with output_path.open("wb") as f:
                    f.write(audio_bytes)
                print(f"Audio saved to: {output_path}")

            print()

        except requests.RequestException as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
