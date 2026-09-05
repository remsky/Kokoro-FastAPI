---
name: web-contrib
description: "Contributing to the Kokoro-FastAPI web player: vanilla JS constraints, MSE/audio gotchas, unit and e2e test setup. Use when changing anything under web/."
---

# Web player contributions

## Constraints

- Vanilla JS modules, no framework, no build step. Files under `web/src/` are served as-is by the API (`/web`).
- `web/src/services/AudioService.js` owns playback: MSE (`audio/mpeg`) with a bounded buffer, falling back to blob playback where MSE mp3 is unsupported (Firefox). Any playback change must keep both paths working.
- Long-session behavior matters: the bounded buffer exists because unbounded MSE appends crashed ~10 min in. Don't reintroduce unbounded growth.
- A finished MSE stream swaps to the full server-side file so scrubbing and true duration work (`canSwapToFileSource` / `swapToFileSource`). Swaps only fire at existing discontinuities: end of playback, pause, seek. Don't trigger one mid-playback.

## Standard patterns

- Popup menus (`role="menu"`) get full keyboard support: ArrowUp/Down walk items with wrap, Escape closes and restores focus to the trigger, `focusout` closes when focus leaves, first item focused when opened via keyboard. Reference implementations: cast menu in `VoiceSelector.js`, download menu in `App.js`. New menus copy that shape.
- File downloads go through `App.triggerDownload(url, name)`, don't hand-roll anchor clicks.
- Outside-click dismissal uses `closeOnOutsidePress` from `dismiss.js`.
- Fire-and-forget promises still need a `.catch` that surfaces via `showStatus`.
- Settings checkboxes copy `.autoplay-toggle` (player.css): one rule, no `:hover` split, no per-control `accent-color`, `:root` sets it.
- Services take request values as arguments (`buildRequestBody(..., responseFormat)`). Read a control once in the caller, don't re-query it inside the service.

## Testing

- Unit: `npm run test:web` (node test runner). New test files go in `web/tests/unit/` and must be imported from `web/tests/unit/index.test.mjs`, it's a manual registry.
- E2e: `npm run test:e2e` (Playwright against a static fixture server, no TTS backend needed).
- Bundled Chromium has no mp3 codec, so real MSE playback needs system Chrome: launch with `channel: 'chrome'` in the spec/config when a test depends on real decoding.
- For manual testing against a real backend, mount `web/` over the image's copy in the compose file rather than rebuilding per edit.
