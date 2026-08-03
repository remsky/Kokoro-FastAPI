import assert from 'node:assert/strict';
import test from 'node:test';

import {
    countVoiceTags,
    formatVoiceTag,
    hasVoiceTags,
    insertVoiceTag,
    seedVoiceTag,
    stripVoiceTags
} from '../../src/voiceTags.js';

test('a mixed voice survives the round trip into a tag', () => {
    const tag = formatVoiceTag('af_bella(2)+am_michael(1)');
    assert.equal(tag, '[voice:af_bella(2)+am_michael(1)]');
    assert.equal(countVoiceTags(tag), 1);
});

test('tags are counted wherever they sit in the text', () => {
    const text = '[voice:af_bella] Hello. [voice:am_michael] Hi.\n[voice:af_sky] Bye.';
    assert.equal(countVoiceTags(text), 3);
    assert.equal(hasVoiceTags(text), true);
    assert.equal(hasVoiceTags('no tags at all'), false);
});

test('a bracket that is not a voice tag is left alone', () => {
    const text = 'The witness [sic] agreed.';
    assert.equal(countVoiceTags(text), 0);
    assert.equal(stripVoiceTags(text), text);
});

test('counting does not carry regex state between calls', () => {
    const text = '[voice:af_bella] one [voice:af_sky] two';
    assert.equal(countVoiceTags(text), 2);
    assert.equal(countVoiceTags(text), 2);
    assert.equal(hasVoiceTags(text), true);
    assert.equal(hasVoiceTags(text), true);
});

test('stripping an inline tag leaves exactly one space', () => {
    assert.equal(stripVoiceTags('Hi [voice:af_bella] there'), 'Hi there');
    assert.equal(stripVoiceTags('[voice:af_bella] Hello'), 'Hello');
});

test('a tag that owned its own line takes the line with it', () => {
    const text = '[voice:af_bella]\nHello there.\n[voice:am_michael]\nHi.';
    assert.equal(stripVoiceTags(text), 'Hello there.\nHi.');
});

test('stripping leaves the surrounding paragraph breaks intact', () => {
    const text = '[voice:af_bella] First line.\n\n[voice:am_michael] Second line.';
    assert.equal(stripVoiceTags(text), 'First line.\n\nSecond line.');
});

test('seeding uses the mixer string and only fires once', () => {
    const first = seedVoiceTag('Hello there.', 'af_bella(2)+am_michael(1)');
    assert.equal(first.changed, true);
    assert.equal(first.text, '[voice:af_bella(2)+am_michael(1)] Hello there.');

    const again = seedVoiceTag(first.text, 'af_sky');
    assert.equal(again.changed, false);
    assert.equal(again.text, first.text);
});

test('seeding an empty editor still shows the pattern', () => {
    assert.equal(seedVoiceTag('', 'af_bella').text, '[voice:af_bella] ');
});

test('seeding without a voice is a no-op', () => {
    const result = seedVoiceTag('Hello there.', '');
    assert.equal(result.changed, false);
    assert.equal(result.text, 'Hello there.');
});

test('an existing space is reused rather than doubled', () => {
    const { text, cursor } = insertVoiceTag('Hello there', 5, 'af_sky');
    assert.equal(text, 'Hello [voice:af_sky] there');
    assert.equal(text.slice(0, cursor), 'Hello [voice:af_sky]');
});

test('a caret parked mid word does not split it', () => {
    // 'He|llo there' is 2 back against 3 forward, so the tag lands in front of the word
    assert.equal(insertVoiceTag('Hello there', 2, 'af_sky').text, '[voice:af_sky] Hello there');
    // 'Hell|o there' is the other way around
    assert.equal(insertVoiceTag('Hello there', 4, 'af_sky').text, 'Hello [voice:af_sky] there');
});

test('inserting at the end leaves room to keep typing', () => {
    const { text, cursor } = insertVoiceTag('Hello there.', 12, 'af_sky');
    assert.equal(text, 'Hello there. [voice:af_sky] ');
    assert.equal(cursor, text.length);
});

test('inserting into an empty editor does not lead with a space', () => {
    assert.equal(insertVoiceTag('', 0, 'af_bella').text, '[voice:af_bella] ');
});

test('an out of range caret is clamped rather than trusted', () => {
    assert.equal(insertVoiceTag('Hello', 999, 'af_sky').text, 'Hello [voice:af_sky] ');
    assert.equal(insertVoiceTag('Hello', -4, 'af_sky').text, '[voice:af_sky] Hello');
});

test('every inserted tag is one the stripper can find again', () => {
    let text = 'One two three four.';
    text = insertVoiceTag(text, 7, 'af_bella').text;
    text = insertVoiceTag(text, 0, 'am_michael(2)+af_sky(1)').text;

    assert.equal(countVoiceTags(text), 2);
    assert.equal(stripVoiceTags(text), 'One two three four.');
});
