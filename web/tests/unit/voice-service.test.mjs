import assert from 'node:assert/strict';
import test from 'node:test';

const { VoiceService } = await import('../../src/services/VoiceService.js');

function stubVoices(voices) {
    globalThis.fetch = async (url) => String(url).includes('/v1/audio/voices')
        ? { ok: true, json: async () => ({ voices }) }
        : { ok: false, json: async () => ({}) };
}

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

test('the legacy string list still loads, with no grades', async () => {
    stubVoices(['af_bella']);
    const service = new VoiceService();

    assert.deepEqual(await service.loadVoices(), ['af_bella']);
    assert.equal(service.getGrade('af_bella'), undefined);
});
