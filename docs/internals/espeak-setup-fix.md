# ESpeak-NG Setup Fix

## Issue Description

Users are reporting two distinct errors:

1. Missing espeak-ng-data/phontab file:
```
Error processing file '/home/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data/phontab': No such file or directory.
```

2. Invalid pipeline state:
```
Error generating speech: The object is in an invalid state.
```

## Root Cause Analysis

### 1. ESpeak-NG Data Issue

The dependency chain has changed:
```
Before:
kokoro-fastapi (phonemizer 3.3.0) -> kokoro -> misaki -> phonemizer

After:
kokoro-fastapi -> kokoro -> misaki -> phonemizer-fork + espeakng-loader
```

The issue arises because:
1. misaki now uses espeakng-loader to manage espeak paths
2. espeakng-loader looks for data in its package directory
3. We have a direct dependency on phonemizer 3.3.0 that conflicts

### 2. Pipeline State Issue
The "invalid state" error occurs due to device mismatch in pipeline creation.

## Solution

### 1. For ESpeak-NG Data

Update dependencies and environment:

1. Remove direct phonemizer dependency:
```diff
- "phonemizer==3.3.0",  # Remove this
```

2. Let misaki handle phonemizer-fork and espeakng-loader

3. Set environment variable in Dockerfile:
```dockerfile
ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data  # Add this
```

This approach:
- Works with misaki's new dependencies
- Maintains our working espeak setup
- Avoids complex file copying or path manipulation

### 2. For Pipeline State

Use kokoro_v1's pipeline management:
```python
# Instead of creating pipelines directly:
# pipeline = KPipeline(...)

# Use backend's pipeline management:
pipeline = backend._get_pipeline(pipeline_lang_code)
```

## Implementation Steps

1. Update pyproject.toml:
   - Remove direct phonemizer dependency
   - Keep misaki dependency as is

2. Update Dockerfiles:
   - Add ESPEAK_DATA_PATH environment variable
   - Keep existing espeak-ng setup

3. Update tts_service.py:
   - Use backend's pipeline management
   - Add proper error handling

## Testing

1. Test espeak-ng functionality:
   ```bash
   # Verify environment variables
   echo $ESPEAK_DATA_PATH
   echo $PHONEMIZER_ESPEAK_DATA
   
   # Check data directory
   ls /usr/share/espeak-ng-data
   ```

2. Test pipeline state:
   - Test on both CPU and GPU
   - Verify no invalid state errors
   - Test with different voice models

## Success Criteria

1. No espeak-ng-data/phontab file errors
2. No invalid state errors
3. Consistent behavior across platforms
4. Successful CI/CD pipeline runs

## Future Considerations

1. Potential PR to misaki:
   - Add fallback mechanism if espeakng-loader fails
   - Make path configuration more flexible
   - Add better error messages

2. Environment Variable Documentation:
   - Document ESPEAK_DATA_PATH requirement
   - Explain interaction with espeakng-loader
   - Provide platform-specific setup instructions

## Notes

- This solution works with misaki's new dependencies while maintaining our setup
- Environment variable approach is simpler than file copying