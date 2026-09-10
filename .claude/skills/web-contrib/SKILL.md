---
name: web-contrib
description: "Contributing to the Kokoro-FastAPI web player: vanilla JS constraints, MSE/audio gotchas, unit and e2e test setup. Use when changing anything under web/."
---

Read [AGENTS.md](../../../AGENTS.md) and [CONTRIBUTING.md](../../../CONTRIBUTING.md) first, house rules live there.

# Web player contributions

## Constraints

- Vanilla JS modules, no framework, no build step. Files under `web/src/` are served as-is by the API (`/web`).
- `web/src/services/AudioService.js` picks the playback path and owns the file-source swap. `services/audio/MsePipeline.js` is the bounded MSE (`audio/mpeg`) state machine, `services/audio/BlockLoader.js` the blob fallback where MSE mp3 is unsupported (Firefox). Any playback change must keep both paths working.
- Long-session behavior matters: the bounded buffer exists because unbounded MSE appends crashed ~10 min in. Don't reintroduce unbounded growth.
- A finished MSE stream swaps to the full server-side file so scrubbing and true duration work (`canSwapToFileSource` / `swapToFileSource`). Swaps only fire at existing discontinuities: end of playback, pause, seek, and read-along activation when no duration is known yet (`ReadAlong.js`). Don't trigger one mid-playback.

## Standard patterns

- Popup menus (`role="menu"`) get full keyboard support: ArrowUp/Down walk items with wrap, Escape closes and restores focus to the trigger, `focusout` closes when focus leaves, first item focused when opened via keyboard. Reference implementations: markup in `web/index.html` (`#cast-menu` is a `popover="manual"`, `#download-menu` is `hidden`-toggled), handlers in `VoiceSelector.js` and `App.js`. New menus copy that shape.
- File downloads go through `App.triggerDownload(url, name)`, don't hand-roll anchor clicks.
- Outside-click dismissal uses `closeOnOutsidePress` from `dismiss.js`.
- Fire-and-forget promises still need a `.catch` that surfaces via `showStatus`.
- Settings checkboxes copy `.autoplay-toggle` (player.css, plus its mobile media-query override): no `:hover` split, no per-control `accent-color`, `:root` in base.css sets it.
- Services take request values as arguments (`buildRequestBody(..., responseFormat)`). Read a control once in the caller, don't re-query it inside the service.

## Testing

- Unit: `npm run test:web` (node test runner). New test files go in `web/tests/unit/` and must be imported from `web/tests/unit/index.test.mjs`, it's a manual registry.
- E2e: `npm run test:e2e` (Playwright against `web/tests/e2e/fixtures/static-server.mjs` on port 4173, auto-started by the config). Stub the two startup calls with `mockApi(page)` from `fixtures/mock-api.mjs`, no TTS backend needed. CI runs unit and e2e on bundled Chromium.
- Bundled Chromium has no mp3 codec, so real MSE playback needs system Chrome: launch with `channel: 'chrome'` and skip when it's unavailable, CI has only bundled Chromium. `long-playback.spec.mjs` mocks `MediaSource` instead.
- For manual testing against a real backend, the cpu and gpu compose files already mount `web/` over the image's copy, edits are live per request. rocm needs the mount added.
- Before committing, check the change against [modern-web-guidance](https://developer.chrome.com/docs/modern-web-guidance) for current HTML/CSS/JS and accesibility patterns.
  