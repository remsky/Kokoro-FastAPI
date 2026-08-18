## OpenAI Integration Guide

*Last updated: 2025-02-01*

### Using Kokoro as the Model

You can use Kokoro as the model with OpenAI. Here's how you can integrate it:

1. **Install OpenAI Python Library**:
   - If you haven't already, install the OpenAI Python library:
     ```bash
     pip install openai
     ```

2. **Use the OpenAI Client**:
   - Here is an example of how to use the OpenAI client to generate speech:
     ```python
     from openai import OpenAI

     client = OpenAI(
         base_url="http://localhost:8880/v1", api_key="not-needed"
     )

     with client.audio.speech.with_streaming_response.create(
         model="kokoro",
         voice="af_sky+af_bella",  # Single or multiple voicepack combo
         input="Hello world!"
     ) as response:
         response.stream_to_file("output.mp3")
     ```
3. Or try streaming direct to speaker (may have to install and/or configure pyAudio)

```python

import time
from pathlib import Path

from openai import OpenAI

# note: it will try to grab OPENAI_API_KEY from your environment variables if not explicitly passed
openai = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed-for-local")


def stream_to_speakers() -> None:
    import pyaudio

    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    start_time = time.time()

    with openai.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_bella+af_nicole",
        response_format="pcm",  # similar to WAV, but without a header chunk at the start.
        input="""I see skies of blue and clouds of white
                The bright blessed days, the dark sacred nights
                And I think to myself
                What a wonderful world""",
    ) as response:
        print(f"Time to first byte: {int((time.time() - start_time) * 1000)}ms")
        for chunk in response.iter_bytes(chunk_size=1024):
            player_stream.write(chunk)

    print(f"Done in {int((time.time() - start_time) * 1000)}ms.")
