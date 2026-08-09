import assert from 'node:assert/strict';
import test from 'node:test';

import { INSERT_LOG_LIMIT, locateInsert, recordInsert } from '../../src/insertLog.js';

test('the log keeps the newest inserts and stays capped', () => {
    let log = [];
    for (let i = 1; i <= INSERT_LOG_LIMIT + 3; i++) {
        log = recordInsert(log, { id: i, inserted: ' [voice:af_bella] ', offset: i });
    }
    assert.equal(log.length, INSERT_LOG_LIMIT);
    assert.equal(log[0].id, INSERT_LOG_LIMIT + 3);
});

test('an untouched insert is found at its recorded offset', () => {
    const text = 'one [voice:af_bella] two';
    assert.equal(locateInsert(text, { inserted: ' [voice:af_bella] ', offset: 3 }), 3);
});

test('edits before the insert shift it to the nearest occurrence', () => {
    const entry = { inserted: ' [voice:af_bella] ', offset: 3 };
    const text = 'one more words [voice:af_bella] two';
    assert.equal(locateInsert(text, entry), 14);
});

test('duplicate tags resolve to the occurrence nearest the recorded offset', () => {
    const entry = { inserted: ' [voice:af_bella] ', offset: 30 };
    const text = 'a [voice:af_bella] middle text [voice:af_bella] end';
    assert.equal(locateInsert(text, entry), 30);
});

test('an insert edited out by hand is reported gone', () => {
    assert.equal(locateInsert('no tags left here', { inserted: ' [voice:af_bella] ', offset: 3 }), -1);
});
