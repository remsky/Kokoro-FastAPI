## Introduction
This is a simple PSR test written in locust. It simulates users that execute the following scenario:
- call API to convert text to speech using streaming API
- play the generated audio file
- repeat

## Get Started
Install the packages required by the PSR test: 
```
uv pip install -e ".[psr]"
```

## Run the test
Start Kokoro-FastAPI first, then start Locust:
```
set HOST=http://localhost:8880
locust -f ./api/psr/openai_load_test.py -H %HOST% --users 1 --spawn-rate 0.1 --run-time 5m KokoroOpenAPiClient
```

1. Use the Locust UI on http://localhost:8089 to start the test.
2. Set the number of users to a value supported by your machine (e.g.: Intel i9 + nvidia 3050 could support 10 users).
3. Set the user ramp up to a desired value (e.g.: 0.1 means one user every 10 seconds)
4. Click Advanced and set the desired test duration (Locust will run user simulations for that duration).
5. Click Start.
