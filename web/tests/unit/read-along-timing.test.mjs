import assert from 'node:assert/strict';
import test from 'node:test';

import { segmentSentences } from '../../src/readAlong.js';
import { alignChunks, sentenceIndexAtTime } from '../../src/readAlongTiming.js';

test('chunk starts become exact sentence times', () => {
    const text = 'The fox jumped over the fence. The hound slept in the sun. A crow watched them both.';
    const sentences = segmentSentences(text);
    const chunks = [
        { text: 'The fox jumped over the fence.', start: 0, end: 3 },
        { text: 'The hound slept in the sun.', start: 3, end: 5.5 },
        { text: 'A crow watched them both.', start: 5.5, end: 8 }
    ];

    const times = alignChunks(text, sentences, chunks);
    assert.deepEqual(times, [0, 3, 5.5]);
});

test('sentences inside a multi-sentence chunk interpolate between its bounds', () => {
    const text = 'One two three. Four five six. Seven eight nine. Ten eleven twelve.';
    const sentences = segmentSentences(text);
    const chunks = [
        { text: 'One two three. Four five six.', start: 0, end: 4 },
        { text: 'Seven eight nine. Ten eleven twelve.', start: 5, end: 9 }
    ];

    const times = alignChunks(text, sentences, chunks);
    assert.equal(times[0], 0);
    assert.ok(times[1] > 0 && times[1] < 4);
    assert.equal(times[2], 5);
    assert.ok(times[3] > 5 && times[3] < 9);
});

test('normalized chunk text still anchors to the raw words', () => {
    const text = 'Dr. Smith owed $50 to the baker. He paid it back on Tuesday morning. Everyone was satisfied with that.';
    const sentences = segmentSentences(text);
    const chunks = [
        { text: 'Doctor Smith owed fifty dollars to the baker.', start: 0, end: 4 },
        { text: 'He paid it back on Tuesday morning.', start: 4, end: 7 },
        { text: 'Everyone was satisfied with that.', start: 7, end: 10 }
    ];

    const times = alignChunks(text, sentences, chunks);
    const startOf = (prefix) => times[
        sentences.findIndex((s) => text.slice(s.start, s.end).startsWith(prefix))
    ];
    assert.equal(startOf('He paid'), 4);
    assert.equal(startOf('Everyone'), 7);
});

test('a pause chunk leaves a gap the highlight rides through', () => {
    const text = 'Before the pause. [pause:2s] After the pause.';
    const sentences = segmentSentences(text);
    const chunks = [
        { text: 'Before the pause.', start: 0, end: 2 },
        { text: '', start: 2, end: 4 },
        { text: 'After the pause.', start: 4, end: 6 }
    ];

    const times = alignChunks(text, sentences, chunks);
    const spokenTimes = times.filter((value) => value !== null);
    assert.deepEqual(spokenTimes, [0, 4]);
    assert.equal(sentenceIndexAtTime(times, 3), times.indexOf(0));
});

test('an unmatched chunk merges into the previous segment', () => {
    const text = 'Alpha bravo charlie delta. Something entirely rewritten here. Echo foxtrot golf hotel.';
    const sentences = segmentSentences(text);
    const chunks = [
        { text: 'Alpha bravo charlie delta.', start: 0, end: 3 },
        { text: 'Quux zzz yyy xxx www.', start: 3, end: 6 },
        { text: 'Echo foxtrot golf hotel.', start: 6, end: 9 }
    ];

    const times = alignChunks(text, sentences, chunks);
    assert.equal(times[0], 0);
    assert.equal(times[2], 6);
    assert.ok(times[1] >= 0 && times[1] < 6);
});

test('voice tags are invisible to anchoring', () => {
    const text = '[voice:narrator] The story begins at dawn. [voice:sam] I never liked mornings much.';
    const sentences = segmentSentences(text);
    const chunks = [
        { text: 'The story begins at dawn.', start: 0, end: 2.5 },
        { text: 'I never liked mornings much.', start: 2.5, end: 5 }
    ];

    const times = alignChunks(text, sentences, chunks);
    const spoken = sentences
        .map((sentence, i) => ({ sentence, i }))
        .filter(({ sentence }) => sentence.spoken > 0);
    assert.equal(times[spoken[0].i], 0);
    assert.equal(times[spoken[1].i], 2.5);
});

test('index tracks time and clamps at both ends', () => {
    const times = [null, 0, 2, null, 5];
    assert.equal(sentenceIndexAtTime(times, -1), 1);
    assert.equal(sentenceIndexAtTime(times, 0), 1);
    assert.equal(sentenceIndexAtTime(times, 1.9), 1);
    assert.equal(sentenceIndexAtTime(times, 3), 2);
    assert.equal(sentenceIndexAtTime(times, 99), 4);
    assert.equal(sentenceIndexAtTime([null, null], 1), -1);
});

test('empty inputs align to nothing', () => {
    assert.equal(alignChunks('', [], []), null);
    assert.equal(alignChunks('Hello.', segmentSentences('Hello.'), []), null);
    assert.equal(alignChunks('Hello.', segmentSentences('Hello.'), [{ text: '', start: 0, end: 2 }]), null);
});
