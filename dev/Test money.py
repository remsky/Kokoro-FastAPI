import base64
import json

import requests

text = """奶酪芝士很浓郁！臭豆腐芝士有争议？陈年奶酪价格昂贵。"""


Type = "wav"

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "model": "kokoro",
        "input": text,
        "voice": "zf_xiaobei",
        "speed": 1.0,
        "response_format": Type,
        "stream": False,
    },
    stream=True,
)

with open(f"outputnostreammoney.{Type}", "wb") as f:
    f.write(response.content)
