import assert from 'node:assert/strict';
import test from 'node:test';

const { tuneForm, fetchVoicePack, saveVoice, tunedVoiceName, encodeWav, mixToMono } = await import('../../src/services/TuneService.js');

const clip = () => new Blob(['clip'], { type: 'audio/wav' });

function stubForbidden(message) {
    const calls = [];
    globalThis.fetch = async (url, init) => {
        calls.push({ url: String(url), init });
        return {
            ok: false,
            status: 403,
            json: async () => ({ detail: { error: 'forbidden', message } })
        };
    };
    return calls;
}

test('a default tune sends only the clip and the prosody head', () => {
    const form = tuneForm({ file: clip() });

    assert.deepEqual([...form.keys()], ['audio', 'prosody_head']);
    assert.equal(form.get('prosody_head'), 'true');
});

test('every knob that is set rides along, blank ones stay off the form', () => {
    const form = tuneForm(
        { file: clip(), prosodyHead: false, fmax: '320', saveVoice: 'am_me' },
        { input: 'Hello', speed: 1 }
    );

    assert.deepEqual([...form.keys()], ['audio', 'prosody_head', 'fmax', 'request', 'save_voice']);
    assert.equal(form.get('prosody_head'), 'false');
    assert.equal(form.get('fmax'), '320');
    assert.equal(JSON.parse(form.get('request')).input, 'Hello');
    assert.equal(form.get('save_voice'), 'am_me');

    assert.equal(tuneForm({ file: clip(), fmax: '  ' }).has('fmax'), false);
    assert.equal(tuneForm({ file: clip(), fmax: 'auto' }).has('fmax'), false);
});

test('a refused voice pack surfaces the server message', async () => {
    const calls = stubForbidden('Voice tuner is disabled');

    await assert.rejects(
        fetchVoicePack({ file: clip() }),
        /Voice tuner is disabled/
    );
    assert.ok(calls[0].url.endsWith('/dev/tune'));
    assert.equal(calls[0].init.body.get('return_voice_pack'), 'true');
});

test('a refused save surfaces the server message', async () => {
    const calls = stubForbidden('Local voice saving is disabled');

    await assert.rejects(
        saveVoice({ file: clip() }, 'am_me'),
        /Local voice saving is disabled/
    );
    assert.equal(calls[0].init.body.get('save_voice'), 'am_me');
});

test('a recording is written as 16-bit mono pcm wav', async () => {
    const wav = encodeWav(Float32Array.from([0, 1, -1, 0.5, 2]), 16000);
    const view = new DataView(await wav.arrayBuffer());
    const text = (offset, length) => String.fromCharCode(...new Uint8Array(view.buffer, offset, length));

    assert.equal(wav.type, 'audio/wav');
    assert.equal(view.byteLength, 44 + 10);
    assert.equal(text(0, 4), 'RIFF');
    assert.equal(text(8, 4), 'WAVE');
    assert.equal(view.getUint16(22, true), 1);
    assert.equal(view.getUint32(24, true), 16000);
    assert.equal(view.getUint16(34, true), 16);
    assert.equal(view.getUint32(40, true), 10);
    assert.deepEqual(
        [0, 2, 4, 6, 8].map((i) => view.getInt16(44 + i, true)),
        [0, 32767, -32768, 16383, 32767]
    );
});

test('a stereo clip is averaged to mono, a mono clip is passed through', () => {
    const channels = [Float32Array.from([1, 0, -0.5]), Float32Array.from([0, 0, 0.5])];
    const stereo = { numberOfChannels: 2, length: 3, getChannelData: (c) => channels[c] };
    const mono = { numberOfChannels: 1, length: 3, getChannelData: () => channels[0] };

    assert.deepEqual([...mixToMono(stereo)], [0.5, 0, 0]);
    assert.equal(mixToMono(mono), channels[0]);
});

test('a tuned voice name is prefix, cleaned slug, _tuned', () => {
    assert.equal(tunedVoiceName('a', 'Me'), 'a_me_tuned');
    assert.equal(tunedVoiceName('ax', 'Me'), 'ax_me_tuned');
    assert.equal(tunedVoiceName('b', ' my voice! '), 'b_my_voice_tuned');
    assert.equal(tunedVoiceName('a', 'a--b__c'), 'a_a_b_c_tuned');
    assert.equal(tunedVoiceName('a', 'me_tuned'), 'a_me_tuned');
    assert.equal(tunedVoiceName('a', ''), '');
    assert.equal(tunedVoiceName('a', '!!!'), '');
});

test('waveform peaks are the max magnitude per column', async () => {
    const { samplePeaks } = await import('../../src/components/ClipWave.js');

    assert.deepEqual(samplePeaks(Float32Array.from([0.125, -0.5, 0.25, 0.375, -0.75, 0]), 3), [0.5, 0.375, 0.75]);
    assert.deepEqual(samplePeaks(Float32Array.from([0.25]), 100), [0.25]);
    assert.deepEqual(samplePeaks(new Float32Array(0), 10), []);
});
