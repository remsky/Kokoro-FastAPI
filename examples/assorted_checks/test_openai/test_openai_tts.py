from pathlib import Path

import openai

# Configure OpenAI client to use our local endpoint
client = openai.OpenAI(
    timeout=30,
    api_key="notneeded",  # API key not required for our endpoint
    base_url="http://localhost:8880/v1",  # Point to our local server with v1 prefix
)

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


def test_format(
    format: str, text: str = "The quick brown fox jumped over the lazy dog."
):
    speech_file = output_dir / f"speech_{format}.{format}"
    print(f"\nTesting {format} format...")
    print(f"Making request to {client.base_url}/audio/speech...")

    try:
        response = client.audio.speech.create(
            model="tts-1", voice="af_heart", input=text, response_format=format
        )

        print("Got response, saving to file...")
        with open(speech_file, "wb") as f:
            f.write(response.content)
        print(f"Success! Saved to: {speech_file}")

    except Exception as e:
        print(f"Error: {str(e)}")


def test_speed(speed: float):
    speech_file = output_dir / f"speech_speed_{speed}.wav"
    print(f"\nTesting speed {speed}x...")
    print(f"Making request to {client.base_url}/audio/speech...")

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="af_heart",
            input="The quick brown fox jumped over the lazy dog.",
            response_format="wav",
            speed=speed,
        )

        print("Got response, saving to file...")
        with open(speech_file, "wb") as f:
            f.write(response.content)
        print(f"Success! Saved to: {speech_file}")

    except Exception as e:
        print(f"Error: {str(e)}")


# Test different formats
for format in ["wav", "mp3", "opus", "aac", "flac", "pcm"]:
    test_format(format)  # aac and pcm should fail as they are not supported

# Test different speeds
for speed in [0.25, 1.0, 2.0, 4.0]:  # 5.0 should fail as it's out of range
    test_speed(speed)

# Test long text
test_format(
    "wav",
    """
That is the germ of my great discovery. But you are wrong to say that we cannot move about in Time. For instance, if I am recalling an incident very vividly I go back to the instant of its occurrence: I become absent-minded, as you say. I jump back for a moment. 
""",
)
