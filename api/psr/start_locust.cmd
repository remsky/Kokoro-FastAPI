@echo off

set HOST=http://localhost:8880

locust -f ./openai_load_test.py -H %HOST% --users 1 --spawn-rate 0.1 --run-time 5m KokoroOpenAPiClient
