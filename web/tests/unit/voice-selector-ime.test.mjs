import assert from 'node:assert/strict';
import test from 'node:test';
import { VoiceSelector } from '../../src/components/VoiceSelector.js';

function element() {
    return {
        listeners: {},
        addEventListener(type, listener) { this.listeners[type] = listener; },
        focus() {},
        select() {},
    };
}

function key(key, isComposing = true) {
    return { key, isComposing, prevented: false, preventDefault() { this.prevented = true; } };
}

function globals(t, values) {
    for (const [name, value] of Object.entries(values)) {
        const original = Object.getOwnPropertyDescriptor(globalThis, name);
        Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
        t.after(() => original
            ? Object.defineProperty(globalThis, name, original)
            : delete globalThis[name]);
    }
}

test('IME confirmation does not create a tag; a subsequent Enter does', () => {
    const row = element();
    const button = element();
    let clicks = 0;
    button.click = () => { clicks++; };
    VoiceSelector.prototype.setupCastListeners.call({
        elements: { createTagRow: row, createTagBtn: button },
    });

    const composing = key('Enter');
    row.listeners.keydown(composing);
    assert.equal(clicks, 0);
    assert.equal(composing.prevented, false);
    row.listeners.keydown(key('Enter', false));
    assert.equal(clicks, 1);
});

test('IME Enter and Escape do not settle an in-place rename', (t) => {
    const input = element();
    const renamed = [];
    globals(t, { document: { createElement: () => input }, CSS: { escape: value => value } });
    const selector = {
        settleRename() {},
        handlers: { onRename: (...args) => renamed.push(args) },
        elements: { voiceCastList: { querySelector: () => ({
            querySelector: () => ({ replaceWith() {} }),
        }) } },
    };
    VoiceSelector.prototype.startRename.call(selector, 'narrator');
    input.value = '声音';
    input.listeners.keydown(key('Enter'));
    input.listeners.keydown(key('Escape'));
    assert.deepEqual(renamed, []);
    input.listeners.keydown(key('Enter', false));
    assert.deepEqual(renamed, [['narrator', '声音']]);
});

test('voice search keeps its dropdown open during composition', (t) => {
    globals(t, { document: { addEventListener() {} } });
    const search = element();
    let closes = 0;
    VoiceSelector.prototype.setupEventListeners.call({ elements: {
        voiceSearch: search,
        voiceDropdown: { classList: { remove: () => { closes++; } } },
        voiceOptions: element(),
        selectedVoices: element(),
    } });
    for (const name of ['Enter', 'Escape']) {
        const event = key(name);
        search.listeners.keydown(event);
        assert.equal(event.prevented, false);
    }
    assert.equal(closes, 0);
    search.listeners.keydown(key('Enter', false));
    assert.equal(closes, 1);
});
