import assert from 'node:assert/strict';
import test from 'node:test';

const { AudioService } = await import('../../src/services/AudioService.js');

test('AudioService streams supported MP3 requests with MediaSource regardless of length', () => {
    const service = new AudioService();

    assert.equal(service.shouldUseMseStream('mp3', true), true);
});

test('AudioService does not use MediaSource for unsupported or non-MP3 output', () => {
    const service = new AudioService();

    assert.equal(service.shouldUseMseStream('mp3', false), false);
    assert.equal(service.shouldUseMseStream('wav', true), false);
    assert.equal(service.shouldUseMseStream('pcm', true), false);
});

test('download name is voice + timestamp with unsafe characters replaced', () => {
    const service = new AudioService();

    assert.match(
        service.buildDownloadName('af_bella', 'mp3'),
        /^af_bella_\d{4}-\d{2}-\d{2}T[\d-]+Z\.mp3$/
    );
    assert.match(service.buildDownloadName('af_bella(2)+af_sky(1)', 'wav'), /^af_bella_2_af_sky_1_\d/);
    assert.ok(service.buildDownloadName('', 'mp3').startsWith('speech_'));
});

test('download URL carries the save-as name for the server to echo back', async () => {
    const service = new AudioService();

    service.downloadName = 'af_bella_2026-08-01T12-30-00-000Z.mp3';
    await service.setDownloadPath('/download/tmprloey00i.mp3');
    assert.equal(
        service.getDownloadUrl(),
        '/v1/download/tmprloey00i.mp3?name=af_bella_2026-08-01T12-30-00-000Z.mp3'
    );

    service.downloadName = null;
    await service.setDownloadPath('/download/tmprloey00i.mp3');
    assert.equal(service.getDownloadUrl(), '/v1/download/tmprloey00i.mp3');
});

test('timing download URL reuses the audio save-as name', async () => {
    const service = new AudioService();

    service.timingPath = '/download/tmprloey00i.mp3.json';
    service.downloadName = 'af_bella_2026-08-01T12-30-00-000Z.mp3';
    assert.equal(
        await service.getTimingDownloadUrl(),
        '/v1/download/tmprloey00i.mp3.json?name=af_bella_2026-08-01T12-30-00-000Z.mp3'
    );

    service.timingPath = null;
    assert.equal(await service.getTimingDownloadUrl(), null);
});

// Stand-in for teardown only: a real element fires an error when its src is blanked.
class TeardownAudio {
    constructor() {
        this.listeners = new Map();
        this.attributes = { src: 'blob:mse-url' };
        this.paused = false;
        this.error = null;
        this.loadCount = 0;
    }

    get src() {
        return this.attributes.src;
    }

    set src(value) {
        this.attributes.src = value;
    }

    addEventListener(event, cb) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(cb);
    }

    removeAttribute(name) {
        delete this.attributes[name];
    }

    load() {
        this.loadCount += 1;
    }

    pause() {
        this.paused = true;
    }

    emitError(code) {
        this.error = { code, message: 'stub' };
        (this.listeners.get('error') || []).forEach((cb) => cb({ target: this }));
    }
}

test('tearing an element down empties it rather than pointing it at an empty src', () => {
    const audio = new TeardownAudio();
    const service = new AudioService();
    service.audio = audio;

    service.cleanup();

    assert.equal(service.audio, null);
    assert.equal(audio.paused, true);
    assert.equal('src' in audio.attributes, false);
    assert.equal(audio.loadCount, 1);
});

test('an error from a replaced element is not a playback failure', () => {
    const mediaError = global.MediaError;
    const consoleError = console.error;
    global.MediaError = { MEDIA_ERR_ABORTED: 1 };
    console.error = () => {};

    try {
        const service = new AudioService();
        let unavailable = 0;
        service.addEventListener('playbackUnavailable', () => { unavailable += 1; });

        // regenerating tears the old element down while the new one is still being built
        const stale = new TeardownAudio();
        service.audio = stale;
        service.attachAudioErrorEvents('stream');
        service.cleanup();
        stale.emitError(4);
        assert.equal(unavailable, 0);

        const current = new TeardownAudio();
        service.audio = current;
        service.attachAudioErrorEvents('stream');
        current.emitError(4);
        assert.equal(unavailable, 1);

        current.emitError(1); // an abort is user-initiated
        assert.equal(unavailable, 1);
    } finally {
        global.MediaError = mediaError;
        console.error = consoleError;
    }
});
