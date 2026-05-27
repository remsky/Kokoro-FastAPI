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
