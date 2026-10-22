import assert from 'node:assert/strict';
import test from 'node:test';

const { VoiceService } = await import('../../src/services/VoiceService.js');

function stubVoices(voices, default_voice) {
    globalThis.fetch = async (url) => String(url).includes('/v1/audio/voices')
        ? { ok: true, json: async () => ({ voices, default_voice }) }
        : { ok: false, json: async () => ({}) };
}

test('the server default is preselected over the best graded voice', async () => {
    stubVoices([
        { id: 'af_heart', name: 'af_heart', overall_grade: 'A' },
        { id: 'bf_emma', name: 'bf_emma', overall_grade: 'B-' }
    ], 'bf_emma');
    const service = new VoiceService();

    assert.deepEqual(await service.loadVoices(), ['af_heart', 'bf_emma']);
    assert.deepEqual(service.getSelectedVoices(), ['bf_emma']);
});

test('a default missing from the list falls back to the first voice', async () => {
    stubVoices([{ id: 'af_heart', name: 'af_heart', overall_grade: 'A' }], 'custom');
    const service = new VoiceService();

    await service.loadVoices();
    assert.deepEqual(service.getSelectedVoices(), ['af_heart']);
});

test('graded voices keep their grade, ungraded ones have none', async () => {
    stubVoices([
        { id: 'af_bella', name: 'af_bella', target_quality: 'A', training_duration: 'HH hours', overall_grade: 'A-' },
        { id: 'ef_dora', name: 'ef_dora' }
    ]);
    const service = new VoiceService();

    assert.deepEqual(await service.loadVoices(), ['af_bella', 'ef_dora']);
    assert.equal(service.getGrade('af_bella').overall_grade, 'A-');
    assert.equal(service.getGrade('ef_dora'), undefined);
});

test('voices sort by grade then name, ungraded last, and the best one is preselected', async () => {
    stubVoices([
        { id: 'af_bella', name: 'af_bella', overall_grade: 'A-' },
        { id: 'am_eric', name: 'am_eric', overall_grade: 'D' },
        { id: 'ef_dora', name: 'ef_dora' },
        { id: 'af_heart', name: 'af_heart', overall_grade: 'A' },
        { id: 'am_adam', name: 'am_adam', overall_grade: 'F+' },
        { id: 'am_echo', name: 'am_echo', overall_grade: 'D' }
    ]);
    const service = new VoiceService();

    assert.deepEqual(await service.loadVoices(), ['af_heart', 'af_bella', 'am_echo', 'am_eric', 'am_adam', 'ef_dora']);
    assert.deepEqual(service.getSelectedVoices(), ['af_heart']);
});

test('the legacy string list still loads, with no grades', async () => {
    stubVoices(['af_bella']);
    const service = new VoiceService();

    assert.deepEqual(await service.loadVoices(), ['af_bella']);
    assert.equal(service.getGrade('af_bella'), undefined);
});
