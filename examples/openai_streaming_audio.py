
#!/usr/bin/env rye run python

import time
from pathlib import Path

from openai import OpenAI

# gets OPENAI_API_KEY from your environment variables
openai = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed-for-local")

speech_file_path = Path(__file__).parent / "speech.mp3"


def main() -> None:
    stream_to_speakers()

    # Create text-to-speech audio file
    with openai.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af",
        input="the quick brown fox jumped over the lazy dogs",
    ) as response:
        response.stream_to_file(speech_file_path)



def stream_to_speakers() -> None:
    import pyaudio

    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    start_time = time.time()

    with openai.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_0p0_n2p0",
        response_format="pcm",  # similar to WAV, but without a header chunk at the start.
        input="""My dear sir, that is just where you are wrong. That is just where the whole world has gone wrong. We are always getting away from the present moment. Our mental existences, which are immaterial and have no dimensions, are passing along the Time-Dimension with a uniform velocity from the cradle to the grave. Just as we should travel down if we began our existence fifty miles above the earth’s surface""",
    ) as response:
        print(f"Time to first byte: {int((time.time() - start_time) * 1000)}ms")
        for chunk in response.iter_bytes(chunk_size=1024):
            player_stream.write(chunk)

    print(f"Done in {int((time.time() - start_time) * 1000)}ms.")


if __name__ == "__main__":
    main()
