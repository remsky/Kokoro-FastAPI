import { config } from '../config.js';

export function tunedVoiceName(prefix, name) {
    const slug = String(name ?? '')
        .toLowerCase()
        .replace(/[^a-z0-9_]+/g, '_')
        .replace(/_+/g, '_')
        .replace(/^_|_$/g, '')
        .replace(/_tuned$/, '');
    return slug ? `${prefix}_${slug}_tuned` : '';
}

export function encodeWav(samples, sampleRate) {
    const bytes = samples.length * 2;
    const view = new DataView(new ArrayBuffer(44 + bytes));
    const tag = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
    tag(0, 'RIFF');
    view.setUint32(4, 36 + bytes, true);
    tag(8, 'WAVE');
    tag(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    tag(36, 'data');
    view.setUint32(40, bytes, true);
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([view.buffer], { type: 'audio/wav' });
}

export async function decodeClip(blob) {
    const context = new AudioContext();
    try {
        const audio = await context.decodeAudioData(await blob.arrayBuffer());
        return { samples: mixToMono(audio), sampleRate: audio.sampleRate };
    } finally {
        await context.close();
    }
}

export function mixToMono(audio) {
    if (audio.numberOfChannels === 1) {
        return audio.getChannelData(0);
    }
    const mono = new Float32Array(audio.length);
    for (let c = 0; c < audio.numberOfChannels; c++) {
        const channel = audio.getChannelData(c);
        for (let i = 0; i < mono.length; i++) {
            mono[i] += channel[i] / audio.numberOfChannels;
        }
    }
    return mono;
}

export async function recordClip({ maxSeconds = 30, onTick } = {}) {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
        throw new Error('Recording needs a microphone and HTTPS or localhost');
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = (event) => chunks.push(event.data);
    const stopped = new Promise((resolve) => { recorder.onstop = resolve; });
    const stop = () => {
        if (recorder.state !== 'inactive') {
            recorder.stop();
        }
    };
    const started = Date.now();
    const ticker = setInterval(() => onTick?.(Math.round((Date.now() - started) / 1000)), 1000);
    const limit = setTimeout(stop, maxSeconds * 1000);
    recorder.start();
    const clip = stopped.then(async () => {
        clearInterval(ticker);
        clearTimeout(limit);
        stream.getTracks().forEach((track) => track.stop());
        const { samples, sampleRate } = await decodeClip(new Blob(chunks, { type: recorder.mimeType }));
        return { file: encodeWav(samples, sampleRate), samples, sampleRate };
    });
    return { stop, clip, stream };
}

export function tuneForm({ file, prosodyHead = true, fmax = '', saveVoice = '' }, request = null) {
    const form = new FormData();
    form.append('audio', file);
    form.append('prosody_head', String(prosodyHead));
    if (String(fmax ?? '').trim() !== '' && Number.isFinite(Number(fmax))) {
        form.append('fmax', String(Number(fmax)));
    }
    if (request) {
        form.append('request', JSON.stringify(request));
    }
    if (saveVoice) {
        form.append('save_voice', saveVoice);
    }
    return form;
}

async function postTune(form) {
    const response = await fetch(await config.getApiUrl('/dev/tune'), {
        method: 'POST',
        body: form
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail?.message || 'Voice tuning failed');
    }
    return response;
}

export async function fetchVoicePack(tune) {
    const form = tuneForm(tune);
    form.append('return_voice_pack', 'true');
    return (await postTune(form)).blob();
}

export async function saveVoice(tune, name) {
    return (await postTune(tuneForm({ ...tune, saveVoice: name }))).json();
}
