import base64
import json

import requests

text = """the administration has offered up a platter of repression for more than a year and is still slated to lose $400 million.

Columbia is the largest private landowner in New York City and boasts an endowment of $14.8 billion;"""


Type = "wav"

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "model": "kokoro",
        "input": text,
        "voice": "af_heart+af_sky",
        "speed": 1.0,
        "response_format": Type,
        "stream": False,
    },
    stream=True,
)

with open(f"outputnostreammoney.{Type}", "wb") as f:
    f.write(response.content)
