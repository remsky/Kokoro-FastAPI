---
name: web-contrib
description: "Contributing to the Kokoro-FastAPI web player: vanilla JS constraints, MSE/audio gotchas, unit and e2e test setup. Use when changing anything under web/."
---

# Web player contributions

## Constraints

- Vanilla JS modules, no framework, no build step. Files under `web/src/` are served as-is by the API (`/web`).
- `web/src/services/AudioService.js` owns playback: MSE (`audio/mpeg`) with a bounded buffer, falling back to blob playback where MSE mp3 is unsupported (Firefox). Any playback change must keep both paths working.
- Long-session behavior matters: the bounded buffer exists because unbounded MSE appends crashed ~10 min in. Don't reintroduce unbounded growth.

## Testing

- Unit: `npm run test:web` (node test runner). New test files go in `web/tests/unit/` and must be imported from `web/tests/unit/index.test.mjs`, it's a manual registry.
- E2e: `npm run test:e2e` (Playwright against a static fixture server, no TTS backend needed).
- Bundled Chromium has no mp3 codec, so real MSE playback needs system Chrome: launch with `channel: 'chrome'` in the spec/config when a test depends on real decoding.
- For manual testing against a real backend, mount `web/` over the image's copy in the compose file rather than rebuilding per edit.
