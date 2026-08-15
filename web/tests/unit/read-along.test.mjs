import assert from 'node:assert/strict';
import test from 'node:test';

import {
    segmentSentences,
    sentenceIndexAt,
    sentenceStartFraction,
    totalSpoken
} from '../../src/readAlong.js';

test('segments cover the full text contiguously', () => {
    const text = 'First sentence. Second one! And a third?\nA new line here.';
    const sentences = segmentSentences(text);

    assert.equal(sentences[0].start, 0);
    assert.equal(sentences[sentences.length - 1].end, text.length);
    for (let i = 1; i < sentences.length; i++) {
        assert.equal(sentences[i].start, sentences[i - 1].end);
    }
    assert.equal(sentences.map((s) => text.slice(s.start, s.end)).join(''), text);
});

test('splits on terminators and newlines', () => {
    const text = 'One. Two!\nThree';
    const sentences = segmentSentences(text);
    assert.equal(sentences.length, 3);
    assert.equal(sentences[0].end, 'One. '.length);
});

test('empty text yields nothing to follow', () => {
    assert.deepEqual(segmentSentences(''), []);
    assert.equal(sentenceIndexAt([], 0.5), -1);
});

test('voice tags take no spoken time', () => {
    const tagged = segmentSentences('[voice:narrator] Hello there.');
    const plain = segmentSentences('Hello there.');
    assert.equal(totalSpoken(tagged), totalSpoken(plain));
});

test('fraction maps to the sentence playing', () => {
    const sentences = segmentSentences('Aaaa. Bbbb. Cccc.');
    assert.equal(sentenceIndexAt(sentences, 0), 0);
    assert.equal(sentenceIndexAt(sentences, 0.5), 1);
    assert.equal(sentenceIndexAt(sentences, 1), 2);
    // out of range clamps rather than falling off the ends
    assert.equal(sentenceIndexAt(sentences, -1), 0);
    assert.equal(sentenceIndexAt(sentences, 2), 2);
});

test('a tag-only segment is never the active sentence', () => {
    const sentences = segmentSentences('[voice:a]\nHello.\n[voice:b]\nWorld.');
    for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
        const index = sentenceIndexAt(sentences, fraction);
        assert.ok(sentences[index].spoken > 0);
    }
});

test('start fraction inverts back to the same sentence', () => {
    const sentences = segmentSentences('Short. A much longer sentence than that one. End.');
    for (let i = 0; i < sentences.length; i++) {
        const fraction = sentenceStartFraction(sentences, i);
        assert.equal(sentenceIndexAt(sentences, fraction + 0.001), i);
    }
});
