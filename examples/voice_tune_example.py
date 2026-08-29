"""Tune a voice from a reference clip, then speak with it.

    uv run voice_tune_example.py path/to/reference.wav [voice_name] [strength]

Needs a server with TUNE_ADAPTER (on by default). The voice name's first
letter picks the language pipeline like any Kokoro voice, so af_/am_ for
American English. strength (default 1.0) pushes the voice away from the
adapter's mean voice; above 1 exaggerates it at a quality cost.
Writes <voice_name>.mp3 next to this script.
"""

import base64
import sys
import time
from pathlib import Path

import requests
from openai import OpenAI

BASE = "http://localhost:8880"
TEXT = (
    "The sun had set behind the hills, and the small town settled into its usual "
    "quiet evening. Nobody expected the phone to ring."
)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ref = Path(sys.argv[1])
    name = (
        sys.argv[2] if len(sys.argv) > 2 else "af_" + ref.stem.lower().replace("-", "_")
    )

    t0 = time.time()
    r = requests.post(
        f"{BASE}/dev/voices/tune",
        json={
            "name": name,
            "audio": base64.b64encode(ref.read_bytes()).decode(),
            "strength": float(sys.argv[3]) if len(sys.argv) > 3 else 1.0,
        },
    )
    r.raise_for_status()
    info = r.json()
    print(
        f"enrolled {info['voice']} in {time.time() - t0:.1f}s: speed {info['speed']}, f0 {info['f0_mean_st']} st"
    )

    voices = requests.get(f"{BASE}/v1/audio/voices?legacy=true").json()["voices"]
    assert name in voices, f"{name} missing from /v1/audio/voices"

    client = OpenAI(base_url=f"{BASE}/v1", api_key="not-needed")
    out = Path(__file__).parent / f"{name}.mp3"
    t0 = time.time()
    with client.audio.speech.with_streaming_response.create(
        model="kokoro", voice=name, input=TEXT, response_format="mp3"
    ) as response:
        response.stream_to_file(out)
    print(
        f"wrote {out} in {time.time() - t0:.1f}s; the web player at {BASE}/web lists {name} too"
    )


if __name__ == "__main__":
    main()
