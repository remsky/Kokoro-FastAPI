# ZipVoice FastAPI - End-to-End Testing Checklist

Complete testing guide to verify the ZipVoice integration with all smart features.

## ✅ Pre-Testing Setup

### Environment Setup
- [ ] Python 3.10+ installed
- [ ] For GPU: NVIDIA GPU with CUDA 12.4+ support
- [ ] For Docker: Docker and Docker Compose installed
- [ ] For GPU Docker: nvidia-docker2 installed

### Installation Options

**Option 1: Native Installation (Development)**
```bash
# GPU
pip install -e .[gpu,zipvoice]

# CPU
pip install -e .[cpu,zipvoice]
```

**Option 2: Docker (Recommended)**
```bash
# GPU
cd docker/gpu && docker-compose up -d

# CPU
cd docker/cpu && docker-compose up -d
```

## 🔍 Validation Tests (Automated)

### 1. Code Structure Validation
```bash
python3 scripts/validate_integration.py
```

**Expected:** All validations PASSED (31 successes, 0 warnings, 0 errors)

## 🚀 Startup Tests

### 2. API Server Startup

**Native:**
```bash
uvicorn api.src.main:app --host 0.0.0.0 --port 8880
```

**Docker:**
```bash
docker-compose logs -f
```

**Verify:**
- [ ] Server starts without errors
- [ ] Logs show "ZipVoice" initialization
- [ ] No import errors
- [ ] GPU detected (if applicable)

### 3. Health Check
```bash
curl http://localhost:8880/health
```

**Expected:**
```json
{"status": "healthy"}
```

### 4. API Documentation
Open in browser: http://localhost:8880/docs

**Verify:**
- [ ] Documentation loads successfully
- [ ] ZipVoice endpoints visible under `/v1/zipvoice/`
- [ ] Schemas show `ZipVoiceSpeechRequest`

## 🎤 Voice Registration Tests

### 5. Basic Voice Registration
```bash
# Create a test audio file (or use your own)
curl -X POST "http://localhost:8880/v1/zipvoice/voices/register" \
  -F "name=test_voice" \
  -F "audio_file=@path/to/voice_sample.wav" \
  -F "transcription=Hello, this is a test voice sample"
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Voice registered successfully
- [ ] Response contains voice info

### 6. Voice Registration with Auto-Transcription
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/voices/register?auto_transcribe=true" \
  -F "name=auto_voice" \
  -F "audio_file=@path/to/voice_sample.wav"
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Transcription automatically generated
- [ ] Response includes transcribed text
- [ ] Logs show Whisper model loading (first time)

### 7. List Registered Voices
```bash
curl http://localhost:8880/v1/zipvoice/voices
```

**Expected:**
- [ ] HTTP 200 response
- [ ] JSON array with registered voices
- [ ] Both `test_voice` and `auto_voice` present

### 8. Get Voice Info
```bash
curl http://localhost:8880/v1/zipvoice/voices/test_voice
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Voice details including path and transcription

## 🔊 Speech Generation Tests

### 9. Basic Speech Generation (Pre-registered Voice)
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Hello! This is a test of ZipVoice text to speech synthesis.",
    "voice": "test_voice"
  }' --output test_output.mp3
```

**Expected:**
- [ ] HTTP 200 response
- [ ] MP3 file created
- [ ] Audio plays correctly
- [ ] Voice matches the registered sample

### 10. Speech Generation with File Upload
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Testing file upload method",
    "voice": "file+/path/to/voice_sample.wav",
    "prompt_text": "Voice sample transcription"
  }' --output file_upload_test.mp3
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Audio generated with uploaded voice

### 11. Speech Generation with URL
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Testing URL download method",
    "voice": "url+https://example.com/voice_sample.wav",
    "prompt_text": "Voice sample from URL"
  }' --output url_test.mp3
```

**Expected:**
- [ ] HTTP 200 response (if URL is valid)
- [ ] Voice downloaded and cached
- [ ] Audio generated successfully

### 12. Speech Generation with Base64
First, encode a voice sample:
```bash
BASE64_AUDIO=$(base64 -w 0 voice_sample.wav)
```

Then generate:
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"zipvoice\",
    \"input\": \"Testing base64 encoding method\",
    \"voice\": \"base64+${BASE64_AUDIO}\",
    \"prompt_text\": \"Voice sample from base64\"
  }" --output base64_test.mp3
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Audio generated from base64 data

## 🤖 Smart Features Tests

### 13. Quality Detection
```bash
curl http://localhost:8880/v1/zipvoice/voices/test_voice/quality
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Quality score (0-1)
- [ ] Metrics: RMS level, peak level, SNR, etc.
- [ ] Recommendations (if quality is low)
- [ ] Warnings (if issues detected)

### 14. Smart Parameter Tuning
```bash
# Short text (should use more steps for quality)
curl -X POST "http://localhost:8880/v1/zipvoice/tune" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world!",
    "priority": "quality"
  }'

# Long text (should use fewer steps for speed)
curl -X POST "http://localhost:8880/v1/zipvoice/tune" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a much longer piece of text that contains many words and should trigger the smart tuning system to optimize for speed rather than maximum quality because it would take too long otherwise...",
    "priority": "speed"
  }'
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Different recommendations based on text length
- [ ] Different recommendations based on priority
- [ ] Estimated generation time

### 15. Speech with Auto-Tuning
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech?auto_tune=true&priority=balanced" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "This is a test with automatic parameter tuning enabled.",
    "voice": "test_voice"
  }' --output auto_tuned.mp3
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Parameters automatically optimized
- [ ] Audio generated successfully

### 16. Quality Detection in Speech Generation
```bash
curl -v -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Testing quality detection",
    "voice": "test_voice"
  }' --output quality_test.mp3 2>&1 | grep -i "X-Quality"
```

**Expected:**
- [ ] Response headers include `X-Quality-Score`
- [ ] If low quality voice: `X-Quality-Warnings` header present
- [ ] Warnings visible in verbose output

## ⚡ Performance Tests

### 17. Different Models
```bash
# Standard model
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Standard model test",
    "voice": "test_voice"
  }' --output standard.mp3

# Distilled model (faster)
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice_distill",
    "input": "Distilled model test",
    "voice": "test_voice"
  }' --output distilled.mp3
```

**Expected:**
- [ ] Both generate successfully
- [ ] Distilled model is noticeably faster
- [ ] Quality difference is minimal

### 18. Inference Steps Variation
```bash
# Fewer steps (faster, lower quality)
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Testing with 4 steps",
    "voice": "test_voice",
    "num_steps": 4
  }' --output steps_4.mp3

# More steps (slower, higher quality)
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Testing with 16 steps",
    "voice": "test_voice",
    "num_steps": 16
  }' --output steps_16.mp3
```

**Expected:**
- [ ] 4 steps: Faster generation
- [ ] 16 steps: Slower but potentially better quality
- [ ] Both produce audible speech

### 19. Speed Control
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Testing speech speed control",
    "voice": "test_voice",
    "speed": 1.5
  }' --output fast_speech.mp3
```

**Expected:**
- [ ] HTTP 200 response
- [ ] Audio plays at 1.5x speed
- [ ] Pitch maintained (time-stretch, not pitch shift)

## 🐳 Docker-Specific Tests

### 20. Docker Health Check
```bash
docker-compose ps
```

**Expected:**
- [ ] Container status: `healthy`
- [ ] Health check passing

### 21. Docker Logs
```bash
docker-compose logs
```

**Verify:**
- [ ] No critical errors
- [ ] ZipVoice initialization successful
- [ ] GPU detection (GPU containers)
- [ ] Whisper model loaded

### 22. Docker Environment Variables
```bash
docker-compose exec zipvoice-tts env | grep -E "ZIPVOICE|ENABLE_"
```

**Expected:**
- [ ] `ENABLE_ZIPVOICE=true`
- [ ] `ENABLE_KOKORO=false`
- [ ] `DEFAULT_BACKEND=zipvoice`
- [ ] Smart features enabled

### 23. Volume Persistence
```bash
# Generate speech to create cache
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Testing cache persistence",
    "voice": "test_voice"
  }' --output cached_test.mp3

# Restart container
docker-compose restart

# Verify voice still exists
curl http://localhost:8880/v1/zipvoice/voices
```

**Expected:**
- [ ] Voice persists after restart
- [ ] Faster generation on second run (cache hit)

## 🔧 Error Handling Tests

### 24. Missing Voice
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Test with non-existent voice",
    "voice": "nonexistent_voice"
  }'
```

**Expected:**
- [ ] HTTP 404 or 400 error
- [ ] Helpful error message

### 25. Invalid Audio File
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/voices/register" \
  -F "name=bad_voice" \
  -F "audio_file=@README.md" \
  -F "transcription=Test"
```

**Expected:**
- [ ] HTTP 400 error
- [ ] Error message about invalid audio format

### 26. Missing Required Fields
```bash
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "Missing voice parameter"
  }'
```

**Expected:**
- [ ] HTTP 422 validation error
- [ ] Error indicates missing required field

## 📊 Performance Benchmarks

### 27. Generation Speed Test
```bash
# Time a generation
time curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "The quick brown fox jumps over the lazy dog. This is a test sentence for benchmarking.",
    "voice": "test_voice"
  }' --output benchmark.mp3
```

**Expected Performance (GPU):**
- [ ] Standard (8 steps): ~2-3 seconds
- [ ] Fast (4 steps): ~1-2 seconds

**Expected Performance (CPU):**
- [ ] Standard (8 steps): ~5-10 seconds
- [ ] Fast (4 steps): ~3-5 seconds

### 28. Concurrent Requests
```bash
# Generate 5 concurrent requests
for i in {1..5}; do
  curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"zipvoice\",
      \"input\": \"Concurrent request $i\",
      \"voice\": \"test_voice\"
    }" --output "concurrent_$i.mp3" &
done
wait
```

**Expected:**
- [ ] All 5 requests complete successfully
- [ ] No crashes or errors
- [ ] Total time reasonable (queue processing)

## 📝 Documentation Tests

### 29. README Accuracy
- [ ] Installation instructions work
- [ ] Quick start guide accurate
- [ ] All examples run successfully
- [ ] Docker instructions correct

### 30. API Documentation
Browse http://localhost:8880/docs

**Verify:**
- [ ] All endpoints documented
- [ ] Request/response schemas accurate
- [ ] "Try it out" feature works
- [ ] Example values are helpful

## ✅ Final Integration Check

### 31. Complete Workflow Test
```bash
# 1. Register voice with auto-transcription
curl -X POST "http://localhost:8880/v1/zipvoice/voices/register?auto_transcribe=true" \
  -F "name=final_test" \
  -F "audio_file=@voice_sample.wav"

# 2. Check quality
curl http://localhost:8880/v1/zipvoice/voices/final_test/quality

# 3. Get smart tuning recommendation
curl -X POST "http://localhost:8880/v1/zipvoice/tune" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a comprehensive final test of all features working together.",
    "priority": "balanced"
  }'

# 4. Generate with auto-tuning
curl -X POST "http://localhost:8880/v1/zipvoice/audio/speech?auto_tune=true" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zipvoice",
    "input": "This is a comprehensive final test of all features working together.",
    "voice": "final_test"
  }' --output final_test.mp3

# 5. Verify result
file final_test.mp3
mpg123 final_test.mp3  # or your audio player
```

**Expected:**
- [ ] All steps complete without errors
- [ ] Quality score calculated automatically
- [ ] Parameters auto-tuned based on input
- [ ] Final audio plays correctly
- [ ] Voice matches the sample

## 📋 Summary

**Total Tests:** 31
**Critical Tests:** 1-15, 31
**Performance Tests:** 17-19, 27-28
**Optional Tests:** 16, 20-26, 29-30

### Sign-Off
- [ ] All critical tests pass
- [ ] Performance meets expectations
- [ ] Docker containers work correctly
- [ ] Documentation is accurate
- [ ] Ready for production use

---

**Testing completed by:** _______________
**Date:** _______________
**Environment:** GPU / CPU / Docker
**Notes:** _______________
