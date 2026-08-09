import assert from 'node:assert/strict';
import test from 'node:test';

import {
    addToCast,
    castAliases,
    countVoiceTags,
    exportCast,
    formatVoiceTag,
    hasVoiceTagFor,
    hasVoiceTags,
    insertVoiceTag,
    isSpeakableMix,
    leadingVoiceTag,
    parseCastFile,
    parseVoiceMix,
    removeFromCast,
    removeVoiceTagsFor,
    renameCastMember,
    renameVoiceTags,
    seedVoiceTag,
    stripVoiceTags,
    unspeakableTagNames,
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

test('the leading tag is the one the request speaks with', () => {
    assert.equal(leadingVoiceTag('[voice:af_bella] Hello.'), 'af_bella');
    assert.equal(leadingVoiceTag('\n  [voice:af_bella(2)+af_sky(1)] Hello.'), 'af_bella(2)+af_sky(1)');
});

test('a tag that is not at the front does not lead', () => {
    assert.equal(leadingVoiceTag('Hello. [voice:af_bella] There.'), '');
    assert.equal(leadingVoiceTag('Hello there.'), '');
    assert.equal(leadingVoiceTag(''), '');
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
        { name: 'am_michael(2)+af_sky(1)', mix: 'am_michael(2)+af_sky(1)' }
    ]);
});

test('a new member invents no name, so there is nothing to define', () => {
    const cast = addToCast([], 'am_michael(2)+af_sky(1)');
    assert.deepEqual(castAliases(cast), {});
});

test('the same voice at a different weight is a different cast member', () => {
    const cast = addToCast(addToCast([], 'af_bella'), 'af_bella(2)');
    assert.deepEqual(cast.map((m) => m.mix), ['af_bella', 'af_bella(2)']);
    assert.deepEqual(cast.map((m) => m.name), ['af_bella', 'af_bella(2)']);
});

test('an empty mix never reaches the cast', () => {
    assert.deepEqual(addToCast([], ''), []);
    assert.deepEqual(addToCast([], '   '), []);
});

test('a member is only spoken while a tag names it', () => {
    const text = '[voice:narrator] One. [voice:af_sky] Two.';
    assert.equal(hasVoiceTagFor(text, 'narrator'), true);
    // the tag pattern is case-insensitive, so what the server would resolve counts as placed
    assert.equal(hasVoiceTagFor(text, 'NARRATOR'), true);
    assert.equal(hasVoiceTagFor(text, 'villain'), false);
    assert.equal(hasVoiceTagFor(text, 'af_bella(2)+af_sky'), false);
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

test('resetting an alias is a rename back to the mix, so nothing is left to define', () => {
    const cast = [
        { name: 'af_bella', mix: 'af_bella' },
        { name: 'narrator', mix: 'am_michael(2)' }
    ];
    const reset = renameCastMember(cast, 'narrator', 'am_michael(2)');
    assert.deepEqual(reset.map((m) => m.name), ['af_bella', 'am_michael(2)']);
    assert.deepEqual(castAliases(reset), {});
});

test('the cast saves as the alias map a request carries', () => {
    const cast = [
        { name: 'af_bella', mix: 'af_bella' },
        { name: 'narrator', mix: 'am_michael(2)+af_sky(1)' }
    ];
    assert.deepEqual(exportCast(cast), {
        voice_aliases: { af_bella: 'af_bella', narrator: 'am_michael(2)+af_sky(1)' }
    });
    assert.deepEqual(exportCast([]), { voice_aliases: {} });
});

test('a saved cast reads back out of a whole request body just as well', () => {
    const saved = exportCast([{ name: 'narrator', mix: 'am_michael(2)' }]);
    const member = [{ name: 'narrator', mix: 'am_michael(2)' }];

    assert.deepEqual(parseCastFile(saved), member);
    assert.deepEqual(parseCastFile({ input: 'Hello.', allow_voice_tags: true, ...saved }), member);
    assert.deepEqual(parseCastFile({ narrator: 'am_michael(2)' }), member);
});

test('an entry that could never be a tag is left out of the import', () => {
    assert.deepEqual(parseCastFile({ voice_aliases: { 'not a name': 'af_bella' } }), []);
    // a leading dash parses as prose in TAG_SOURCE, so the name is refused up front
    assert.deepEqual(parseCastFile({ voice_aliases: { '-bob': 'af_bella' } }), []);
    assert.deepEqual(parseCastFile({ voice_aliases: { narrator: '  ' } }), []);
    assert.deepEqual(parseCastFile({ voice_aliases: { narrator: 12 } }), []);
    assert.deepEqual(parseCastFile(['narrator']), []);
    assert.deepEqual(parseCastFile(null), []);
});

test('only names that stand for something else are sent as aliases', () => {
    const cast = [
        { name: 'af_bella', mix: 'af_bella' },
        { name: 'narrator', mix: 'am_michael(2)+af_sky(1)' }
    ];
    assert.deepEqual(castAliases(cast), { narrator: 'am_michael(2)+af_sky(1)' });
    assert.deepEqual(castAliases([]), {});
});

test('a pace is baked into the text, not carried on the alias', () => {
    const cast = [
        { name: 'af_bella', mix: 'af_bella', rate: 0.8 },
        { name: 'narrator', mix: 'am_michael(2)' }
    ];
    // a self-named member usually stays home, and a pace does not change that
    assert.deepEqual(castAliases(cast), { narrator: 'am_michael(2)' });
    assert.deepEqual(exportCast(cast), {
        voice_aliases: { af_bella: 'af_bella', narrator: 'am_michael(2)' }
    });
    // old-shape files with a {voice, rate} value still read back, rate just falls away
    assert.deepEqual(
        parseCastFile({ voice_aliases: { a: { voice: 'af_sky', rate: 0.8 } } }),
        [{ name: 'a', mix: 'af_sky' }]
    );
});

test('a pace travels beside the voice tag it was inserted with', () => {
    const { text } = insertVoiceTag('Hello there.', 12, 'af_sky', 0.8);
    assert.equal(text, 'Hello there. [voice:af_sky] [rate:0.8] ');
    // no pace set, no rate tag added
    assert.equal(insertVoiceTag('Hello there.', 12, 'af_sky').text, 'Hello there. [voice:af_sky] ');
});

test('a mix with an empty plus-part is unspeakable, even though parsing smooths it over', () => {
    const available = ['af_bella', 'af_sky'];
    assert.equal(isSpeakableMix('af_bella', available), true);
    assert.equal(isSpeakableMix('af_bella(2)+af_sky(1)', available), true);
    assert.equal(isSpeakableMix('af_bella+', available), false);
    assert.equal(isSpeakableMix('af_bella++af_sky', available), false);
    assert.equal(isSpeakableMix('+', available), false);
    assert.equal(isSpeakableMix('af_jane', available), false);
    assert.equal(isSpeakableMix('', available), false);
});

test('tags name their trouble: not in the cast and not a real mix means unspeakable', () => {
    const available = ['af_bella', 'af_sky'];
    const cast = [{ name: 'narrator', mix: 'af_bella(2)+af_sky' }];
    const text = '[voice:narrator] One. [voice:af_bella] Two. [voice:hero] Three. [voice:af_jane] Four.';

    assert.deepEqual(unspeakableTagNames(text, cast, available), ['hero', 'af_jane']);
});

test('unspeakable tag names are counted once, however the case varies', () => {
    const text = '[voice:Hero] One. [voice:hero] Two. [voice:HERO] Three.';

    assert.deepEqual(unspeakableTagNames(text, [], ['af_bella']), ['Hero']);
    assert.deepEqual(unspeakableTagNames(text, [{ name: 'hero', mix: 'af_bella' }], ['af_bella']), []);
    assert.deepEqual(unspeakableTagNames('No tags at all.', [], ['af_bella']), []);
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
