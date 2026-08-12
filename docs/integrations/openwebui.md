# OpenWebUI Integration Guide

*Last updated: 2025-02-02*

## Setup -> Setting Base URL

- If you are running OpenWebUI as a Docker container, you need to change the base URL in OpenWebUI settings to
  - `http://host.docker.internal:8880/v1`
- If not, you will need to point it at 
  - `http://localhost:8880/v1`

This ensures that OpenWebUI can communicate with the Kokoro FastAPI wrapper running in another Docker container.

***

## Configure TTS

### Using Kokoro as the Model

You can use Kokoro as the model in OpenWebUI. Here's how you can integrate it:

### Set the Model
   - In OpenWebUI, go to the settings or configuration section.
   - Set the model to `kokoro`, api key to 'not-needed'

### Use Mapped OpenAI Voices
   - OpenWebUI supports mapped OpenAI voices. You can use any of the mapped voices directly.
     - e.g. `alloy,echo,fable,onyx,nova,shimmer`
   - You can see how it's currently mapped in api/src/core/openai_mappings.json, and modify (if running from compose or uv), or see the current mapping (at time of this post) as:
```json
    "voices": {
        "alloy": "am_adam",
        "ash": "af_nicole",
        "coral": "bf_emma",
        "echo": "af_bella",
        "fable": "af_sarah",
        "onyx": "bm_george",
        "nova": "bf_isabella",
        "sage": "am_michael",
        "shimmer": "af_sky"
}
```

### Use Direct Voice Names
   - If you prefer, you can use the direct voice names provided by Kokoro.
   - Set the `voice` parameter to the desired voice name (without the file extension) 
   - e.g `af_nicole','am_michael', etc
