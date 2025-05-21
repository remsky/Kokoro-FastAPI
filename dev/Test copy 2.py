import base64
import json

import pydub
import requests

text = """Running on localhost:7860"""


Type = "wav"
response = requests.post(
    "http://localhost:8880/dev/captioned_speech",
    json={
        "model": "kokoro",
        "input": text,
        "voice": "af_heart+af_sky",
        "speed": 1.0,
        "response_format": Type,
        "stream": True,
    },
    stream=True,
)

f = open(f"outputstream.{Type}", "wb")
for chunk in response.iter_lines(decode_unicode=True):
    if chunk:
        temp_json = json.loads(chunk)
        if temp_json["timestamps"] != []:
            chunk_json = temp_json

        # Decode base 64 stream to bytes
        chunk_audio = base64.b64decode(temp_json["audio"].encode("utf-8"))

        # Process streaming chunks
        f.write(chunk_audio)

        # Print word level timestamps
        print(chunk_json["timestamps"])
