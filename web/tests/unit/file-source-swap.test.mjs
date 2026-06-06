import assert from 'node:assert/strict';
import test from 'node:test';

const { AudioService } = await import('../../src/services/AudioService.js');

// Minimal stand-in for an HTMLAudioElement. load() reports metadata asynchronously
// (like a real browser parsing the new file), which is what resolves swapToFileSource.
class FakeAudio {
    constructor() {
        this.listeners = new Map();
        this._src = '';
        this.paused = true;
        this.currentTime = 0;
        this.duration = NaN;
        this.volume = 1;
        this.playbackRate = 1;
        this.error = null;
        this.loadCount = 0;
        this.nextDuration = 720; // 12:00, longer than the bounded MSE window
    }

    get src() {
        return this._src;
    }

    set src(value) {
        this._src = value;
    }

    addEventListener(event, cb, opts) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push({ cb, once: !!(opts && opts.once) });
    }

    removeEventListener(event, cb) {
        const arr = this.listeners.get(event);
        if (arr) {
            this.listeners.set(event, arr.filter((l) => l.cb !== cb));
        }
    }

    emit(event) {
        for (const l of (this.listeners.get(event) || []).slice()) {
            if (l.once) {
                this.removeEventListener(event, l.cb);
            }
            l.cb();
        }
    }

    load() {
        this.loadCount += 1;
        queueMicrotask(() => {
            this.duration = this.nextDuration;
            this.emit('loadedmetadata');
        });
    }

    play() {
        this.paused = false;
        return Promise.resolve();
    }

    pause() {
        this.paused = true;
    }
}

function finishedMseService(audio) {
    const service = new AudioService();
    service.audio = audio;
    service.mediaSource = {}; // non-null marks MSE mode
    service.streamFinished = true;
    service.serverDownloadPath = '/v1/download/test.mp3';
    service.objectUrl = 'blob:mse-url';
    return service;
}

function stubRevoke() {
    const original = global.URL.revokeObjectURL;
    const calls = [];
    global.URL.revokeObjectURL = (u) => calls.push(u);
    return { calls, restore: () => { global.URL.revokeObjectURL = original; } };
}

test('swapToFileSource switches a finished MSE stream to the server file', async () => {
    const audio = new FakeAudio();
    audio.currentTime = 42;
    const service = finishedMseService(audio);

    const revoke = stubRevoke();
    let readyFired = false;
    service.addEventListener('ready', () => { readyFired = true; });

    const result = await service.swapToFileSource();
    revoke.restore();

    assert.equal(result, true);
    assert.equal(audio.src, '/v1/download/test.mp3');
    assert.equal(service.usingFileSource, true);
    assert.equal(service.mediaSource, null);
    assert.equal(service.sourceBuffer, null);
    assert.deepEqual(revoke.calls, ['blob:mse-url']);
    assert.equal(readyFired, true);
    assert.equal(audio.currentTime, 42); // playhead preserved across the swap
});

test('swapToFileSource honors an explicit target time and resume flag', async () => {
    const audio = new FakeAudio();
    const service = finishedMseService(audio);
    const revoke = stubRevoke();

    await service.swapToFileSource(123, true);
    revoke.restore();

    assert.equal(audio.currentTime, 123);
    assert.equal(audio.paused, false); // resumed playback after the swap
});

test('swapToFileSource clamps a target past the end of the file', async () => {
    const audio = new FakeAudio();
    audio.nextDuration = 100;
    const service = finishedMseService(audio);
    const revoke = stubRevoke();

    await service.swapToFileSource(99999);
    revoke.restore();

    assert.ok(audio.currentTime <= 100 && audio.currentTime > 99);
});

test('swapToFileSource is a no-op once already on the file source', async () => {
    const audio = new FakeAudio();
    const service = finishedMseService(audio);
    const revoke = stubRevoke();

    assert.equal(await service.swapToFileSource(), true);
    assert.equal(await service.swapToFileSource(), false);
    revoke.restore();
});

test('canSwapToFileSource is false for block mode (no MediaSource)', () => {
    const audio = new FakeAudio();
    const service = finishedMseService(audio);
    service.mediaSource = null; // block mode plays a full-file blob already

    assert.equal(service.canSwapToFileSource(), false);
});

test('canSwapToFileSource is false before the stream finishes', () => {
    const audio = new FakeAudio();
    const service = finishedMseService(audio);
    service.streamFinished = false;

    assert.equal(service.canSwapToFileSource(), false);
});

test('pause() swaps a finished MSE stream to the file source', async () => {
    const audio = new FakeAudio();
    audio.paused = false;
    audio.currentTime = 10;
    const service = finishedMseService(audio);
    service.objectUrl = null;
    const revoke = stubRevoke();

    service.pause();
    await new Promise((r) => setTimeout(r, 0)); // let the async swap settle
    revoke.restore();

    assert.equal(audio.paused, true);
    assert.equal(audio.src, '/v1/download/test.mp3');
    assert.equal(service.usingFileSource, true);
    assert.equal(audio.currentTime, 10);
});

test('cleanup and cancel reset the swap flags', () => {
    const cleanupService = new AudioService();
    cleanupService.usingFileSource = true;
    cleanupService.swapInProgress = true;
    cleanupService.cleanup();
    assert.equal(cleanupService.usingFileSource, false);
    assert.equal(cleanupService.swapInProgress, false);

    const cancelService = new AudioService();
    cancelService.usingFileSource = true;
    cancelService.swapInProgress = true;
    cancelService.cancel();
    assert.equal(cancelService.usingFileSource, false);
    assert.equal(cancelService.swapInProgress, false);
});
