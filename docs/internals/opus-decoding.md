# Decoding OGG/Opus

*Last updated: 2025-02-13*

Streaming Opus out of the endpoint and piping it straight into ffplay.

```python
import time
import subprocess
from pathlib import Path
from io import BytesIO
from openai import OpenAI

# Initialize OpenAI client pointing to local Kokoro FastAPI server
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed-for-local")
def main():
    print("1. Starting Opus stream...")
    start_time = time.time()
    
    # Start ffplay process
    process = subprocess.Popen(
        ['ffplay', '-'],  # Read from stdin
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    try:
        # Create streaming response with Opus format
        with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_bella",
            response_format="opus",
            input="Testing Opus streaming and decoding"
        ) as response:
            # Process chunks as they arrive
            first_chunk = True
            
            for chunk in response.iter_bytes(chunk_size=1024):
                if first_chunk:
                    ttfb = time.time() - start_time
                    print(f"Time to first byte: {ttfb*1000:.0f}ms")
                    first_chunk = False
                
                # Write chunk to ffplay
                process.stdin.write(chunk)
                process.stdin.flush()
    
    finally:
        # Cleanup
        process.stdin.close()
        process.wait()
    
    print(f"\nTotal processing time: {(time.time() - start_time)*1000:.0f}ms")

if __name__ == "__main__":
    main()
```

Each page starts with the "OggS" marker, then the OGG Page Structure:

```

┌─────────────────────┐
│ OGG Page Header     │ 27 bytes, fixed
├─────────────────────┤
│ Segment Table       │ 1 byte per segment
├─────────────────────┤
│ Opus Packet Data    │ varies with content
└─────────────────────┘
```