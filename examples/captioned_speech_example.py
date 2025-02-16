import json
from typing import Tuple, Optional, Dict, List
from pathlib import Path

import base64
import requests

# Get the directory this script is in
SCRIPT_DIR = Path(__file__).absolute().parent

def generate_captioned_speech(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "mp3"
) -> Tuple[Optional[bytes], Optional[List[Dict]]]:
    """Generate audio with word-level timestamps."""
    response = requests.post(
        "http://localhost:8880/dev/captioned_speech",
        json={
            "model": "kokoro",
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format,
            "stream": False
        }
    )
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        return None, None
        
    try:
        audio_json=json.loads(response.content)
    
        # Decode base 64 stream to bytes
        chunk_audio=base64.b64decode(audio_json["audio"].encode("utf-8"))
        
        # Print word level timestamps
        print(audio_json["timestamps"])

        if not chunk_audio:
            print("Error: Empty audio content")
            return None, None
            
        return chunk_audio, audio_json["timestamps"]
    except json.JSONDecodeError as e:
        print(f"Error parsing timestamps: {e}")
        return None, None
    except requests.RequestException as e:
        print(f"Error retrieving timestamps: {e}")
        return None, None

def main():
    # Example texts to convert
    examples = [
        "Hello world! Welcome to the captioned speech system.",
        "The quick brown fox jumps over the lazy dog.",
        """Of course if you come to the place fresh from New York, you are deceived. Your standard of vision is all astray, You do think the place is quiet. You do imagine that Mr. Smith is asleep merely because he closes his eyes as he stands. But live in Mariposa for six months or a year and then you will begin to understand it better; the buildings get higher and higher; the Mariposa House grows more and more luxurious; McCarthy's block towers to the sky; the 'buses roar and hum to the station; the trains shriek; the traffic multiplies; the people move faster and faster; a dense crowd swirls to and fro in the post-office and the five and ten cent store—and amusements! well, now! lacrosse, baseball, excursions, dances, the Fireman's Ball every winter and the Catholic picnic every summer; and music—the town band in the park every Wednesday evening, and the Oddfellows' brass band on the street every other Friday; the Mariposa Quartette, the Salvation Army—why, after a few months' residence you begin to realize that the place is a mere mad round of gaiety."""
    ]

    print("Generating captioned speech for example texts...\n")

    # Create output directory in same directory as script
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    for i, text in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Input text: {text}")
        try:
            # Generate audio and get timestamps
            audio_bytes, word_timestamps = generate_captioned_speech(text)
            
            if not audio_bytes or not word_timestamps:
                print("Error: No audio data or timestamps generated")
                continue

            # Save audio file
            audio_path = output_dir / f"captioned_example_{i+1}.wav"
            with audio_path.open("wb") as f:
                f.write(audio_bytes)
            print(f"Audio saved to: {audio_path}")

            # Save timestamps to JSON
            timestamps_path = output_dir / f"captioned_example_{i+1}_timestamps.json"
            with timestamps_path.open("w") as f:
                json.dump(word_timestamps, f, indent=2)
            print(f"Timestamps saved to: {timestamps_path}")

            # Print timestamps
            print("\nWord-level timestamps:")
            for ts in word_timestamps:
                print(f"{ts['word']}: {ts['start_time']:.3f}s - {ts['end_time']:.3f}s")

        except requests.RequestException as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()