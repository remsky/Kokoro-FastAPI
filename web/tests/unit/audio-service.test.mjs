import assert from 'node:assert/strict';
import test from 'node:test';

const { AudioService } = await import('../../src/services/AudioService.js');

test('AudioService streams supported MP3 requests with MediaSource regardless of length', () => {
    const service = new AudioService();

    assert.equal(service.shouldUseMseStream('mp3', true), true);
});

test('AudioService does not use MediaSource for unsupported or non-MP3 output', () => {
    const service = new AudioService();

    assert.equal(service.shouldUseMseStream('mp3', false), false);
    assert.equal(service.shouldUseMseStream('wav', true), false);
    assert.equal(service.shouldUseMseStream('pcm', true), false);
});

test('download name is voice + timestamp with unsafe characters replaced', () => {
    const service = new AudioService();

    assert.match(
        service.buildDownloadName('af_bella', 'mp3'),
        /^af_bella_\d{4}-\d{2}-\d{2}T[\d-]+Z\.mp3$/
    );
    assert.match(service.buildDownloadName('af_bella(2)+af_sky(1)', 'wav'), /^af_bella_2_af_sky_1_\d/);
    assert.ok(service.buildDownloadName('', 'mp3').startsWith('speech_'));
});

test('download URL carries the save-as name for the server to echo back', async () => {
    const service = new AudioService();

    service.downloadName = 'af_bella_2026-08-01T12-30-00-000Z.mp3';
    await service.setDownloadPath('/download/tmprloey00i.mp3');
    assert.equal(
        service.getDownloadUrl(),
        '/v1/download/tmprloey00i.mp3?name=af_bella_2026-08-01T12-30-00-000Z.mp3'
    );

    service.downloadName = null;
    await service.setDownloadPath('/download/tmprloey00i.mp3');
    assert.equal(service.getDownloadUrl(), '/v1/download/tmprloey00i.mp3');
});
