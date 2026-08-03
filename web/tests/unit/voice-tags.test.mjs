import assert from 'node:assert/strict';
import test from 'node:test';

import {
    addToCast,
    castAliases,
    countVoiceTags,
    defaultCastName,
    formatVoiceTag,
    hasVoiceTags,
    insertVoiceTag,
    parseVoiceMix,
    removeFromCast,
    removeVoiceTagsFor,
    renameCastMember,
    renameVoiceTags,
    seedVoiceTag,
    stripVoiceTags,
    updateCastMix
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

test('a tag ending a line takes the space that preceded it', () => {
    assert.equal(stripVoiceTags('Hi [voice:af_bella]\nthere'), 'Hi\nthere');
    assert.equal(stripVoiceTags('Hi [voice:af_bella]'), 'Hi');
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

test('the cast keeps insertion order and refuses the same mix twice', () => {
    let cast = addToCast([], 'af_bella');
    cast = addToCast(cast, 'am_michael(2)+af_sky(1)');
    cast = addToCast(cast, 'af_bella');

    assert.deepEqual(cast, [
        { name: 'af_bella', mix: 'af_bella' },
        { name: 'michael', mix: 'am_michael(2)+af_sky(1)' }
    ]);
});

test('the same voice at a different weight is a different cast member', () => {
    const cast = addToCast(addToCast([], 'af_bella'), 'af_bella(2)');
    assert.deepEqual(cast.map((m) => m.mix), ['af_bella', 'af_bella(2)']);
    assert.deepEqual(cast.map((m) => m.name), ['af_bella', 'bella']);
});

test('an empty mix never reaches the cast', () => {
    assert.deepEqual(addToCast([], ''), []);
    assert.deepEqual(addToCast([], '   '), []);
});

test('removing a cast member leaves the rest in order', () => {
    const cast = [
        { name: 'af_bella', mix: 'af_bella' },
        { name: 'narrator', mix: 'am_michael(2)' },
        { name: 'af_sky', mix: 'af_sky' }
    ];
    assert.deepEqual(removeFromCast(cast, 'narrator').map((m) => m.name), ['af_bella', 'af_sky']);
    assert.deepEqual(removeFromCast(cast, 'not_there'), cast);
});

test('a plain voice names itself, a mix is named after its loudest member', () => {
    assert.equal(defaultCastName('af_bella'), 'af_bella');
    assert.equal(defaultCastName('af_bella(2)+am_michael(1)'), 'bella');
    assert.equal(defaultCastName('af_bella(1)+am_michael(3)'), 'michael');
    // a weight of its own is still a recipe, so it earns a short name
    assert.equal(defaultCastName('af_bella(2)'), 'bella');
});

test('a short name that is already spoken for takes a number', () => {
    assert.equal(defaultCastName('af_bella(2)+af_sky', ['bella']), 'bella2');
    assert.equal(defaultCastName('af_bella(2)+af_sky', ['bella', 'bella2']), 'bella3');
});

test('only names that stand for something else are sent as aliases', () => {
    const cast = [
        { name: 'af_bella', mix: 'af_bella' },
        { name: 'narrator', mix: 'am_michael(2)+af_sky(1)' }
    ];
    assert.deepEqual(castAliases(cast), { narrator: 'am_michael(2)+af_sky(1)' });
    assert.deepEqual(castAliases([]), {});
});

test('renaming a member follows through to the tags already placed', () => {
    const text = '[voice:bella] One.\nPlain line.\n[voice:bella] Two. [voice:af_sky] Three.';
    assert.equal(
        renameVoiceTags(text, 'bella', 'narrator'),
        '[voice:narrator] One.\nPlain line.\n[voice:narrator] Two. [voice:af_sky] Three.'
    );
});

test('a rename cannot be confused by a name that looks like a pattern', () => {
    assert.equal(renameVoiceTags('[voice:af_bella(2)] Hi.', 'af_bella(2)', 'bella'), '[voice:bella] Hi.');
});

test('one speaker can be dropped from the text without touching the others', () => {
    const text = '[voice:narrator] One.\n[voice:villain] Two.\n[voice:narrator] Three.';
    assert.equal(removeVoiceTagsFor(text, 'narrator'), 'One.\n[voice:villain] Two.\nThree.');
});

test('a member can have its mix retuned in place', () => {
    const cast = [{ name: 'narrator', mix: 'af_bella(2)' }];
    assert.deepEqual(updateCastMix(cast, 'narrator', 'af_bella(3)+af_sky(1)'), [
        { name: 'narrator', mix: 'af_bella(3)+af_sky(1)' }
    ]);
    assert.deepEqual(renameCastMember(cast, 'narrator', 'storyteller'), [
        { name: 'storyteller', mix: 'af_bella(2)' }
    ]);
});

test('a mix string parses back into the weights the mixer had', () => {
    assert.deepEqual(parseVoiceMix('am_michael(2)+af_sky(0.5)'), [
        { voice: 'am_michael', weight: 2 },
        { voice: 'af_sky', weight: 0.5 }
    ]);
});

test('an unweighted mix parses as full weight', () => {
    assert.deepEqual(parseVoiceMix('af_bella'), [{ voice: 'af_bella', weight: 1 }]);
    assert.deepEqual(parseVoiceMix(''), []);
});

test('a mix survives the round trip out of the mixer and back', () => {
    const mix = 'af_bella(2)+am_michael(1)';
    const restored = parseVoiceMix(mix)
        .map(({ voice, weight }) => `${voice}(${weight})`)
        .join('+');

    assert.equal(restored, mix);
});

test('every inserted tag is one the stripper can find again', () => {
    let text = 'One two three four.';
    text = insertVoiceTag(text, 7, 'af_bella').text;
    text = insertVoiceTag(text, 0, 'am_michael(2)+af_sky(1)').text;

    assert.equal(countVoiceTags(text), 2);
    assert.equal(stripVoiceTags(text), 'One two three four.');
});
