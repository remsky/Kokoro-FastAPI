from pathlib import Path

import openai
import requests

SAMPLE_TEXT = """
That is the germ of my great discovery. But you are wrong to say that we cannot move about in Time. For instance, if I am recalling an incident very vividly I go back to the instant of its occurrence: I become absent-minded, as you say. I jump back for a moment. 
"""

# Configure OpenAI client to use our local endpoint
client = openai.OpenAI(
    timeout=60,
    api_key="notneeded",  # API key not required for our endpoint
    base_url="http://localhost:8880/v1",  # Point to our local server with v1 prefix
)

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


def test_voice(voice: str):
    speech_file = output_dir / f"speech_{voice}.mp3"
    print(f"\nTesting voice: {voice}")
    print(f"Making request to {client.base_url}/audio/speech...")

    try:
        response = client.audio.speech.create(
            model="kokoro", voice=voice, input=SAMPLE_TEXT, response_format="mp3"
        )

        print("Got response, saving to file...")
        with open(speech_file, "wb") as f:
            f.write(response.content)
        print(f"Success! Saved to: {speech_file}")

    except Exception as e:
        print(f"Error with voice {voice}: {str(e)}")


# First, get list of available voices using requests
print("Getting list of available voices...")
try:
    # Convert base_url to string and ensure no double slashes
    base_url = str(client.base_url).rstrip("/")
    response = requests.get(f"{base_url}/audio/voices")
    if response.status_code != 200:
        raise Exception(f"Failed to get voices: {response.text}")
    data = response.json()
    if "voices" not in data:
        raise Exception(f"Unexpected response format: {data}")
    voices = [v["id"] if isinstance(v, dict) else v for v in data["voices"]]
    print(f"Found {len(voices)} voices: {', '.join(voices)}")

    # Test each voice
    for voice in voices:
        test_voice(voice)

except Exception as e:
    print(f"Error getting voices: {str(e)}")
