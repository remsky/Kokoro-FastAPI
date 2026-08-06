import assert from 'node:assert/strict';
import test from 'node:test';

class FakeClassList {
    constructor() {
        this.classes = new Set();
    }

    add(name) {
        this.classes.add(name);
    }

    remove(name) {
        this.classes.delete(name);
    }

    toggle(name, force) {
        if (force === undefined ? !this.classes.has(name) : force) {
            this.classes.add(name);
        } else {
            this.classes.delete(name);
        }
    }
}

class FakeElement {
    constructor() {
        this.listeners = new Map();
        this.classList = new FakeClassList();
        this.style = {};
        this.disabled = false;
        this.value = 0;
        this.textContent = '';
        this.dragging = false;
        this.attributes = new Map();
    }

    setAttribute(name, value) {
        this.attributes.set(name, value);
    }

    addEventListener(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }
}

class FakeAudioService {
    constructor() {
        this.listeners = new Map();
    }

    addEventListener(event, callback) {
        this.listeners.set(event, callback);
    }

    emit(event) {
        this.listeners.get(event)?.();
    }

    getCurrentTime() {
        return 0;
    }

    getDuration() {
        return 0;
    }

    isPlaying() {
        return false;
    }

    play() {}

    pause() {}

    seek() {}

    setVolume() {}
}

function setupDocument() {
    const elements = new Map();
    global.document = {
        getElementById(id) {
            if (!elements.has(id)) {
                elements.set(id, new FakeElement());
            }
            return elements.get(id);
        },
    };
    return elements;
}

test('PlayerControls enables playback when audio readiness fires without duration', async () => {
    const elements = setupDocument();
    const { PlayerState } = await import('../../src/state/PlayerState.js');
    const { PlayerControls } = await import('../../src/components/PlayerControls.js');

    const audioService = new FakeAudioService();
    const playerState = new PlayerState();
    const controls = new PlayerControls(audioService, playerState);

    playerState.reset();
    const playPauseBtn = elements.get('play-pause-btn');
    const seekSlider = elements.get('seek-slider');

    assert.equal(playPauseBtn.disabled, true);

    audioService.emit('ready');

    assert.equal(playPauseBtn.disabled, false);
    assert.equal(seekSlider.disabled, true);

    controls.cleanup();
});

test('formatTime rolls minutes into hours', async () => {
    setupDocument();
    const { PlayerState } = await import('../../src/state/PlayerState.js');
    const { PlayerControls } = await import('../../src/components/PlayerControls.js');

    const controls = new PlayerControls(new FakeAudioService(), new PlayerState());

    assert.equal(controls.formatTime(0), '0:00');
    assert.equal(controls.formatTime(75), '1:15');
    assert.equal(controls.formatTime(3600), '1:00:00');
    assert.equal(controls.formatTime(7524), '2:05:24');

    controls.cleanup();
});

test('PlayerControls swaps the play/pause icon and its label', async () => {
    const elements = setupDocument();
    const { PlayerState } = await import('../../src/state/PlayerState.js');
    const { PlayerControls } = await import('../../src/components/PlayerControls.js');

    const audioService = new FakeAudioService();
    const controls = new PlayerControls(audioService, new PlayerState());
    const playPauseBtn = elements.get('play-pause-btn');

    audioService.emit('play');
    assert.equal(playPauseBtn.classList.classes.has('playing'), true);
    assert.equal(playPauseBtn.attributes.get('aria-label'), 'Pause');

    audioService.emit('pause');
    assert.equal(playPauseBtn.classList.classes.has('playing'), false);
    assert.equal(playPauseBtn.attributes.get('aria-label'), 'Play');

    audioService.emit('play');
    audioService.emit('ended');
    assert.equal(playPauseBtn.classList.classes.has('playing'), false);

    controls.cleanup();
});
