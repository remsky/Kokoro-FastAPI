import { config } from '../config.js';

export class AudioService {
    constructor() {
        this.mediaSource = null;
        this.sourceBuffer = null;
        this.audio = null;
        this.controller = null;
        this.eventListeners = new Map();
        this.minimumPlaybackSize = 50000;
        this.textLength = 0;
        this.shouldAutoplay = false;
        this.CHARS_PER_CHUNK = 150;
        this.MAX_LEAD_SECONDS = 60;
        this.serverDownloadPath = null;
        this.pendingOperations = [];
        this.objectUrl = null;
        this.chunkQueue = [];
        this.streamFinished = false;
        this.feederWakeup = null;
    }

    supportsMSEMp3() {
        return (
            typeof window !== 'undefined' &&
            'MediaSource' in window &&
            typeof MediaSource.isTypeSupported === 'function' &&
            MediaSource.isTypeSupported('audio/mpeg')
        );
    }

    shouldUseMseStream(responseFormat, canStreamMp3) {
        return responseFormat === 'mp3' && canStreamMp3;
    }

    attachAudioReadinessEvents() {
        if (!this.audio) {
            return;
        }

        const dispatchReady = () => this.dispatchEvent('ready');
        this.audio.addEventListener('loadedmetadata', dispatchReady);
        this.audio.addEventListener('durationchange', dispatchReady);
        this.audio.addEventListener('canplay', dispatchReady);
    }

    async streamAudio(text, voice, speed, onProgress) {
        try {
            const canStreamMp3 = this.supportsMSEMp3();
            console.log('AudioService: Starting stream...', { text, voice, speed, canStreamMp3 });

            if (this.controller) {
                this.controller.abort();
                this.controller = null;
            }

            this.controller = new AbortController();
            this.cleanup();
            onProgress?.(0, 1);
            this.textLength = text.length;
            this.shouldAutoplay = document.getElementById('autoplay-toggle').checked;

            const estimatedChunks = Math.max(1, Math.ceil(this.textLength / this.CHARS_PER_CHUNK));
            const responseFormat = document.getElementById('format-select').value || 'mp3';
            const canUseMseStream = this.shouldUseMseStream(responseFormat, canStreamMp3);

            const apiUrl = await config.getApiUrl('/v1/audio/speech');
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: text,
                    voice: voice,
                    response_format: responseFormat,
                    download_format: responseFormat,
                    stream: true,
                    speed: speed,
                    return_download_link: true,
                    lang_code: document.getElementById('lang-select').value || undefined
                }),
                signal: this.controller.signal
            });

            console.log('AudioService: Got response', {
                status: response.status,
                headers: Object.fromEntries(response.headers.entries())
            });

            const downloadPath = response.headers.get('x-download-path');
            if (downloadPath) {
                this.serverDownloadPath = `/v1${downloadPath}`;
                console.log('Download path received:', this.serverDownloadPath);
            }

            if (!response.ok) {
                const error = await response.json();
                console.error('AudioService: API error', error);
                throw new Error(error.detail?.message || 'Failed to generate speech');
            }

            await this.setupAudioStream(response.body, response, onProgress, estimatedChunks, canUseMseStream);
            return this.audio;
        } catch (error) {
            this.cleanup();
            throw error;
        }
    }

    async setupBlockMode(stream, response, onProgress, estimatedChunks) {
        const reader = stream.getReader();
        const chunks = [];
        let receivedChunks = 0;

        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                chunks.push(value);
                receivedChunks++;
                onProgress?.(receivedChunks, estimatedChunks);
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                return;
            }
            throw error;
        }

        const headers = Object.fromEntries(response.headers.entries());
        const downloadPath = headers['x-download-path'];
        if (downloadPath) {
            this.serverDownloadPath = await config.getApiUrl(`/v1${downloadPath}`);
        }

        onProgress?.(estimatedChunks, estimatedChunks);

        const blobType = response.headers.get('content-type') || 'audio/mpeg';
        const blob = new Blob(chunks, { type: blobType });
        this.audio = new Audio();
        this.attachAudioReadinessEvents();
        this.objectUrl = URL.createObjectURL(blob);
        this.audio.src = this.objectUrl;
        this.audio.load();

        this.audio.addEventListener('error', () => {
            console.error('Audio error (block mode):', this.audio?.error);
            this.dispatchEvent('playbackUnavailable');
        });

        this.audio.addEventListener('ended', () => {
            this.dispatchEvent('ended');
        });

        this.audio.addEventListener('canplay', () => {
            if (this.shouldAutoplay) {
                this.play();
            }
        }, { once: true });

        this.dispatchEvent('complete');

        setTimeout(() => {
            this.dispatchEvent('downloadReady');
        }, 100);
    }

    async setupAudioStream(stream, response, onProgress, estimatedChunks, canUseMseStream) {
        if (!canUseMseStream) {
            console.warn('MSE streaming unavailable for this output. Using block mode (full file then play).');
            await this.setupBlockMode(stream, response, onProgress, estimatedChunks);
            return;
        }

        this.audio = new Audio();
        this.attachAudioReadinessEvents();
        this.mediaSource = new MediaSource();
        this.objectUrl = URL.createObjectURL(this.mediaSource);
        this.audio.src = this.objectUrl;

        this.audio.addEventListener('error', () => {
            console.error('Audio error:', this.audio?.error);
        });

        this.audio.addEventListener('ended', () => {
            this.dispatchEvent('ended');
        });

        return new Promise((resolve, reject) => {
            this.mediaSource.addEventListener('sourceopen', async () => {
                try {
                    this.sourceBuffer = this.mediaSource.addSourceBuffer('audio/mpeg');
                    this.sourceBuffer.mode = 'sequence';

                    this.sourceBuffer.addEventListener('updateend', () => {
                        this.processNextOperation();
                    });

                    await this.processStream(stream, response, onProgress, estimatedChunks);
                    resolve();
                } catch (error) {
                    reject(error);
                }
            }, { once: true });
        });
    }

    async processStream(stream, response, onProgress, estimatedChunks) {
        this.chunkQueue = [];
        this.streamFinished = false;
        this.feederWakeup = null;

        const feederPromise = this.runFeeder().catch((err) => {
            if (err?.name !== 'AbortError') {
                console.warn('Feeder error:', err);
            }
        });

        const reader = stream.getReader();
        let receivedChunks = 0;

        try {
            while (true) {
                const { value, done } = await reader.read();

                if (done) {
                    const headers = Object.fromEntries(response.headers.entries());
                    console.log('Response headers at stream end:', headers);

                    const downloadPath = headers['x-download-path'];
                    if (downloadPath) {
                        this.serverDownloadPath = await config.getApiUrl(`/v1${downloadPath}`);
                        console.log('Download path received:', this.serverDownloadPath);
                    } else {
                        console.warn('No X-Download-Path header found. Available headers:',
                            Object.keys(headers).join(', '));
                    }

                    this.streamFinished = true;
                    this.wakeFeeder();

                    onProgress?.(estimatedChunks, estimatedChunks);
                    this.dispatchEvent('complete');

                    setTimeout(() => {
                        this.dispatchEvent('downloadReady');
                    }, 800);

                    return;
                }

                receivedChunks++;
                onProgress?.(receivedChunks, estimatedChunks);
                this.chunkQueue.push(value);
                this.wakeFeeder();
            }
        } catch (error) {
            this.streamFinished = true;
            this.wakeFeeder();
            if (error.name !== 'AbortError') {
                throw error;
            }
        }
    }

    wakeFeeder() {
        if (this.feederWakeup) {
            const resolve = this.feederWakeup;
            this.feederWakeup = null;
            resolve();
        }
    }

    waitForFeederSignal(timeoutMs) {
        return new Promise((resolve) => {
            this.feederWakeup = resolve;
            if (timeoutMs) {
                setTimeout(() => {
                    if (this.feederWakeup === resolve) {
                        this.feederWakeup = null;
                        resolve();
                    }
                }, timeoutMs);
            }
        });
    }

    async runFeeder() {
        let hasStartedPlaying = false;

        while (true) {
            if (!this.audio || !this.sourceBuffer || !this.mediaSource) {
                return;
            }
            if (this.streamFinished && this.chunkQueue.length === 0) {
                if (this.mediaSource.readyState === 'open') {
                    try {
                        this.mediaSource.endOfStream();
                    } catch (e) {
                        console.warn('endOfStream error:', e);
                    }
                }
                return;
            }
            if (this.chunkQueue.length === 0) {
                await this.waitForFeederSignal();
                continue;
            }

            const currentTime = this.audio.currentTime || 0;
            const buffered = this.sourceBuffer.buffered;

            // Leading-edge backpressure: hold off if we already have plenty queued
            // ahead of currentTime. Keeps MSE buffer bounded so long generations
            // (>10 min) don't hit QuotaExceededError.
            if (buffered.length > 0) {
                const leadingEdge = buffered.end(buffered.length - 1);
                if (leadingEdge - currentTime > this.MAX_LEAD_SECONDS) {
                    await this.waitForFeederSignal(250);
                    continue;
                }
            }

            // Trailing eviction: drop audio more than 30s behind currentTime.
            if (buffered.length > 0) {
                const start = buffered.start(0);
                if (currentTime - start > 30) {
                    const removeEnd = Math.max(start, currentTime - 15);
                    if (removeEnd > start) {
                        await this.removeBufferRange(start, removeEnd);
                    }
                }
            }

            const chunk = this.chunkQueue.shift();
            try {
                if (this.audio?.error) {
                    console.error('Audio error detected:', this.audio.error);
                    continue;
                }

                await this.appendChunk(chunk);
                this.dispatchEvent('ready');

                if (!hasStartedPlaying && this.sourceBuffer?.buffered.length > 0) {
                    hasStartedPlaying = true;
                    if (this.shouldAutoplay) {
                        setTimeout(() => this.play(), 100);
                    }
                }
            } catch (error) {
                if (error.name === 'QuotaExceededError') {
                    this.chunkQueue.unshift(chunk);
                    const buf = this.sourceBuffer?.buffered;
                    if (buf && buf.length > 0) {
                        const start = buf.start(0);
                        const removeEnd = Math.max(start, (this.audio?.currentTime || 0) - 5);
                        if (removeEnd > start) {
                            await this.removeBufferRange(start, removeEnd);
                        }
                    } else {
                        return;
                    }
                } else if (error?.name === 'AbortError') {
                    return;
                } else {
                    console.warn('Buffer error:', error);
                }
            }
        }
    }

    async removeBufferRange(start, end) {
        if (!this.sourceBuffer) {
            return;
        }

        if (end <= start) {
            console.warn('Invalid buffer remove range:', { start, end });
            return;
        }

        return new Promise((resolve) => {
            const doRemove = () => {
                const sourceBuffer = this.sourceBuffer;
                if (!sourceBuffer || !this.mediaSource || this.mediaSource.readyState !== 'open') {
                    resolve();
                    return;
                }

                const onUpdateEnd = () => {
                    sourceBuffer.removeEventListener('updateend', onUpdateEnd);
                    resolve();
                };

                try {
                    sourceBuffer.addEventListener('updateend', onUpdateEnd, { once: true });
                    sourceBuffer.remove(start, end);
                } catch (e) {
                    console.warn('Error removing buffer:', e);
                    sourceBuffer.removeEventListener('updateend', onUpdateEnd);
                    resolve();
                }
            };

            if (this.sourceBuffer.updating) {
                this.sourceBuffer.addEventListener('updateend', () => {
                    doRemove();
                }, { once: true });
            } else {
                doRemove();
            }
        });
    }

    async appendChunk(chunk) {
        if (!this.audio || this.audio.error) {
            console.warn('Skipping chunk append due to audio error');
            return;
        }

        if (!this.sourceBuffer) {
            return;
        }

        return new Promise((resolve, reject) => {
            const operation = { chunk, resolve, reject };
            this.pendingOperations.push(operation);

            if (!this.sourceBuffer.updating) {
                this.processNextOperation();
            }
        });
    }

    processNextOperation() {
        if (!this.sourceBuffer || this.sourceBuffer.updating || this.pendingOperations.length === 0) {
            return;
        }

        if (!this.audio || this.audio.error) {
            console.warn('Skipping operation due to audio error');
            return;
        }

        const operation = this.pendingOperations.shift();

        try {
            this.sourceBuffer.appendBuffer(operation.chunk);

            const onUpdateEnd = () => {
                operation.resolve();
                this.sourceBuffer?.removeEventListener('updateend', onUpdateEnd);
                this.sourceBuffer?.removeEventListener('updateerror', onUpdateError);
                this.processNextOperation();
            };

            const onUpdateError = (event) => {
                operation.reject(event);
                this.sourceBuffer?.removeEventListener('updateend', onUpdateEnd);
                this.sourceBuffer?.removeEventListener('updateerror', onUpdateError);
                if (event.name !== 'InvalidStateError') {
                    this.processNextOperation();
                }
            };

            this.sourceBuffer.addEventListener('updateend', onUpdateEnd);
            this.sourceBuffer.addEventListener('updateerror', onUpdateError);
        } catch (error) {
            operation.reject(error);
            if (error.name !== 'InvalidStateError') {
                this.processNextOperation();
            }
        }
    }

    play() {
        if (this.audio && !this.audio.error) {
            const playPromise = this.audio.play();
            if (playPromise) {
                playPromise.catch(error => {
                    if (error.name !== 'AbortError') {
                        console.error('Playback error:', error);
                    }
                });
            }
            this.dispatchEvent('play');
        }
    }

    pause() {
        if (this.audio) {
            this.audio.pause();
            this.dispatchEvent('pause');
        }
    }

    seek(time) {
        if (this.audio && !this.audio.error) {
            const wasPlaying = !this.audio.paused;
            this.audio.currentTime = time;
            if (wasPlaying) {
                this.play();
            }
        }
    }

    setVolume(volume) {
        if (this.audio) {
            this.audio.volume = Math.max(0, Math.min(1, volume));
        }
    }

    getCurrentTime() {
        return this.audio ? this.audio.currentTime : 0;
    }

    getDuration() {
        return this.audio ? this.audio.duration : 0;
    }

    isPlaying() {
        return this.audio ? !this.audio.paused : false;
    }

    addEventListener(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event).add(callback);

        if (this.audio && ['play', 'pause', 'ended', 'timeupdate'].includes(event)) {
            this.audio.addEventListener(event, callback);
        }
    }

    removeEventListener(event, callback) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.delete(callback);
        }
        if (this.audio) {
            this.audio.removeEventListener(event, callback);
        }
    }

    dispatchEvent(event, data) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.forEach(callback => callback(data));
        }
    }

    revokeObjectUrl() {
        if (this.objectUrl) {
            URL.revokeObjectURL(this.objectUrl);
            this.objectUrl = null;
        }
    }

    rejectPendingOperations(reason) {
        const ops = this.pendingOperations;
        this.pendingOperations = [];
        ops.forEach((op) => {
            try {
                op.reject(reason);
            } catch (e) {
                // ignore
            }
        });
    }

    cancel() {
        if (this.controller) {
            this.controller.abort();
            this.controller = null;
        }

        if (this.audio) {
            this.audio.pause();
            this.audio.src = '';
            this.audio = null;
        }

        if (this.mediaSource && this.mediaSource.readyState === 'open') {
            try {
                this.mediaSource.endOfStream();
            } catch (e) {
            }
        }

        this.mediaSource = null;
        this.sourceBuffer = null;
        this.serverDownloadPath = null;
        this.rejectPendingOperations(new Error('AudioService cancelled'));
        this.chunkQueue = [];
        this.streamFinished = true;
        this.wakeFeeder();
        this.revokeObjectUrl();
    }

    cleanup() {
        if (this.audio) {
            this.eventListeners.forEach((listeners, event) => {
                listeners.forEach((callback) => {
                    this.audio.removeEventListener(event, callback);
                });
            });

            this.audio.pause();
            this.audio.src = '';
            this.audio = null;
        }

        if (this.mediaSource && this.mediaSource.readyState === 'open') {
            try {
                this.mediaSource.endOfStream();
            } catch (e) {
            }
        }

        this.mediaSource = null;
        this.sourceBuffer = null;
        this.serverDownloadPath = null;
        this.rejectPendingOperations(new Error('AudioService cleanup'));
        this.chunkQueue = [];
        this.streamFinished = true;
        this.wakeFeeder();
        this.revokeObjectUrl();
    }

    getDownloadUrl() {
        if (!this.serverDownloadPath) {
            console.warn('No download path available');
            return null;
        }
        return this.serverDownloadPath;
    }
}

export default AudioService;
