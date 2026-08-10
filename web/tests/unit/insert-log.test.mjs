import assert from 'node:assert/strict';
import test from 'node:test';

import { locateInsert } from '../../src/insertLog.js';

test('an untouched insert is found at its recorded offset, for sure', () => {
    const text = 'one [voice:af_bella] two';
    assert.deepEqual(locateInsert(text, { inserted: ' [voice:af_bella] ', offset: 3 }), { at: 3, sure: true });
});

test('a drifted insert with no twin is still found for sure', () => {
    const entry = { inserted: ' [voice:af_bella] ', offset: 3 };
    const text = 'one more words [voice:af_bella] two';
    assert.deepEqual(locateInsert(text, entry), { at: 14, sure: true });
});

test('a twin still sitting at its recorded offset is sure', () => {
    const entry = { inserted: ' [voice:af_bella] ', offset: 30 };
    const text = 'a [voice:af_bella] middle text [voice:af_bella] end';
    assert.deepEqual(locateInsert(text, entry), { at: 30, sure: true });
});

test('a drifted twin is only a guess at the nearest occurrence', () => {
    const entry = { inserted: ' [voice:af_bella] ', offset: 4 };
    const text = 'aa [voice:af_bella] mid [voice:af_bella] end';
    assert.deepEqual(locateInsert(text, entry), { at: 2, sure: false });
});

test('an insert edited out by hand is reported gone', () => {
    assert.deepEqual(locateInsert('no tags left here', { inserted: ' [voice:af_bella] ', offset: 3 }), { at: -1, sure: true });
});
