import assert from 'node:assert/strict';
import test from 'node:test';

const { AudioService } = await import('../../src/services/AudioService.js');
const { MsePipeline } = await import('../../src/services/audio/MsePipeline.js');

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
    const pipeline = new MsePipeline(audio);
    pipeline.mediaSource = {}; // non-null marks an unfinished handoff
    pipeline.streamFinished = true;
    pipeline.objectUrl = 'blob:mse-url';
    service.msePipeline = pipeline;
    service.serverDownloadPath = '/v1/download/test.mp3';
    return service;
}

// the swap preloads the file before touching the element, so fetch and the blob url factory both need standing in
function stubBrowserIo({ ok = true } = {}) {
    const original = {
        fetch: global.fetch,
        create: global.URL.createObjectURL,
        revoke: global.URL.revokeObjectURL,
    };
    const revoked = [];
    global.fetch = async () => ({ ok, blob: async () => ({}) });
    global.URL.createObjectURL = () => 'blob:file-url';
    global.URL.revokeObjectURL = (u) => revoked.push(u);
    return {
        revoked,
        restore: () => {
            global.fetch = original.fetch;
            global.URL.createObjectURL = original.create;
            global.URL.revokeObjectURL = original.revoke;
        },
    };
}

test('swapToFileSource switches a finished MSE stream to the preloaded file', async () => {
    const audio = new FakeAudio();
    audio.currentTime = 42;
    const service = finishedMseService(audio);

    const io = stubBrowserIo();
    let readyFired = false;
    service.addEventListener('ready', () => { readyFired = true; });

    const result = await service.swapToFileSource();
    io.restore();

    assert.equal(result, true);
    assert.equal(audio.src, 'blob:file-url');
    assert.equal(service.usingFileSource, true);
    assert.equal(service.msePipeline, null);
    assert.deepEqual(io.revoked, ['blob:mse-url']);
    assert.equal(readyFired, true);
    assert.equal(audio.currentTime, 42); // playhead preserved across the swap
});

test('swapToFileSource honors an explicit target time and keeps playing audio playing', async () => {
    const audio = new FakeAudio();
    audio.paused = false;
    const service = finishedMseService(audio);
    const io = stubBrowserIo();

    await service.swapToFileSource(123);
    io.restore();

    assert.equal(audio.currentTime, 123);
    assert.equal(audio.paused, false);
});

test('a pause during the preload is honored rather than overridden', async () => {
    const audio = new FakeAudio();
    audio.paused = false;
    const service = finishedMseService(audio);
    const io = stubBrowserIo();
    const realFetch = global.fetch;
    global.fetch = async (...args) => {
        audio.paused = true; // the user hits pause while the bytes are still coming
        return realFetch(...args);
    };

    await service.swapToFileSource();
    io.restore();

    assert.equal(audio.paused, true);
});

test('a failed preload leaves the stream alone rather than swapping onto a dead url', async () => {
    const audio = new FakeAudio();
    audio.paused = false;
    const service = finishedMseService(audio);
    const io = stubBrowserIo({ ok: false });

    const result = await service.swapToFileSource();
    io.restore();

    assert.equal(result, false);
    assert.equal(audio.src, '');
    assert.equal(audio.paused, false); // still playing off the stream buffer
    assert.equal(service.usingFileSource, false);
    assert.equal(service.swapInProgress, false);
    assert.ok(service.msePipeline);
    assert.deepEqual(io.revoked, []);
});

test('swapToFileSource clamps a target past the end of the file', async () => {
    const audio = new FakeAudio();
    audio.nextDuration = 100;
    const service = finishedMseService(audio);
    const io = stubBrowserIo();

    await service.swapToFileSource(99999);
    io.restore();

    assert.ok(audio.currentTime <= 100 && audio.currentTime > 99);
});

test('swapToFileSource is a no-op once already on the file source', async () => {
    const audio = new FakeAudio();
    const service = finishedMseService(audio);
    const io = stubBrowserIo();

    assert.equal(await service.swapToFileSource(), true);
    assert.equal(await service.swapToFileSource(), false);
    io.restore();
});

test('canSwapToFileSource is false for block mode (no MediaSource)', () => {
    const audio = new FakeAudio();
    const service = finishedMseService(audio);
    service.msePipeline = null; // block mode plays a full-file blob already

    assert.equal(service.canSwapToFileSource(), false);
});

test('canSwapToFileSource is false before the stream finishes', () => {
    const audio = new FakeAudio();
    const service = finishedMseService(audio);
    service.msePipeline.streamFinished = false;

    assert.equal(service.canSwapToFileSource(), false);
});

test('pause() swaps a finished MSE stream to the file source', async () => {
    const audio = new FakeAudio();
    audio.paused = false;
    audio.currentTime = 10;
    const service = finishedMseService(audio);
    service.msePipeline.objectUrl = null;
    const io = stubBrowserIo();

    service.pause();
    await new Promise((r) => setTimeout(r, 0)); // let the async swap settle
    io.restore();

    assert.equal(audio.paused, true);
    assert.equal(audio.src, 'blob:file-url');
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
