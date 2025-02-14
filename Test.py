import requests


response = requests.get("http://localhost:8880/v1/audio/voices")
voices = response.json()["voices"]

# Generate audio
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "model": "kokoro",  
        "input": "http://localhost:8880/web/",
        "voice": "af_heart",
        "response_format": "mp3",  # Supported: mp3, wav, opus, flac
        "speed": 1.0,
        "stream":False,
    }
)

# Save audio
with open("output.mp3", "wb") as f:
    f.write(response.content)