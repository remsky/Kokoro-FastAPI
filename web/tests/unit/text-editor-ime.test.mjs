import assert from 'node:assert/strict';
import test from 'node:test';
import TextEditor from '../../src/components/TextEditor.js';

function element() {
    const listeners = {};
    return {
        listeners,
        value: '',
        textContent: '',
        classList: { add() {}, remove() {}, toggle() {} },
        addEventListener(type, listener) { this.listeners[type] = listener; },
        focus() {},
        select() {},
        click() {},
        setAttribute() {},
        removeAttribute() {},
        open: false,
    };
}

function editor() {
    const names = [
        'pageContent', 'prevBtn', 'nextBtn', 'pageJump', 'findInput', 'replaceInput',
        'findNextBtn', 'replaceOneBtn', 'replaceAllBtn', 'findMenu', 'uploadBtn',
        'fileInput', 'clearBtn', 'charsPerPage', 'formatBtn',
    ];
    const elements = Object.fromEntries(names.map(name => [name, element()]));
    const calls = { findNext: 0, replaceOne: 0 };
    TextEditor.prototype.bindEvents.call({
        elements,
        container: { querySelector: () => elements.formatBtn },
        options: {},
        findNext() { calls.findNext++; },
        replaceOne() { calls.replaceOne++; },
    });
    return { elements, calls };
}

function key(key, isComposing) {
    return { key, isComposing, prevented: false, preventDefault() { this.prevented = true; } };
}

test('IME Enter does not run find next; a subsequent Enter does', () => {
    const { elements, calls } = editor();
    const composing = key('Enter', true);
    elements.findInput.listeners.keydown(composing);
    assert.equal(calls.findNext, 0);
    assert.equal(composing.prevented, false);
    elements.findInput.listeners.keydown(key('Enter', false));
    assert.equal(calls.findNext, 1);
});

test('IME Enter does not replace; a subsequent Enter does', () => {
    const { elements, calls } = editor();
    const composing = key('Enter', true);
    elements.replaceInput.listeners.keydown(composing);
    assert.equal(calls.replaceOne, 0);
    assert.equal(composing.prevented, false);
    elements.replaceInput.listeners.keydown(key('Enter', false));
    assert.equal(calls.replaceOne, 1);
});
