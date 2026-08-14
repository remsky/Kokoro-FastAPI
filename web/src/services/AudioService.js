import { config } from '../config.js';
import { BlockLoader } from './audio/BlockLoader.js';
import { MsePipeline } from './audio/MsePipeline.js';

// orchestrator: owns the audio element, event bus, and download paths. picks a playback mode per response and hands the stream to BlockLoader or MsePipeline.
export class AudioService {
    constructor() {
        this.audio = null;
        this.controller = null;
        this.msePipeline = null;
        this.eventListeners = new Map();
        this.minimumPlaybackSize = 50000;
        this.textLength = 0;
        this.shouldAutoplay = false;
        this.CHARS_PER_CHUNK = 150;
        this.serverDownloadPath = null;
        this.downloadName = null;
        this.timingPath = null;
        this.knownDuration = null;
        this.objectUrl = null;
        // once an MSE generation finishes, the full file lives on the server. swapping
        // the <audio> element over to it restores true duration + full seeking, which
        // the bounded streaming buffer can't provide for long generations (#150).
        this.usingFileSource = false;
        this.swapInProgress = false;
        this.fileBlobUrl = null;
        this.preloadPromise = null;
        this.pendingSeek = null;
        this.pendingResume = false;
        this.volume = 1;
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

    attachAudioErrorEvents(mode) {
        this.audio.addEventListener('error', (event) => {
            const audioElement = event.target;
            const errorCode = audioElement?.error?.code;

            // a torn down element errors as its source is released, long after it mattered
            if (audioElement !== this.audio) {
                return;
            }

            console.error(`Audio error (${mode}):`, {
                code: errorCode,
                message: audioElement?.error?.message || 'Unknown audio error',
                src: audioElement?.src,
                networkState: audioElement?.networkState,
                readyState: audioElement?.readyState
            });

            // an abort is user-initiated, not a playback failure
            if (errorCode !== MediaError.MEDIA_ERR_ABORTED) {
                this.dispatchEvent('playbackUnavailable');
            }
        });
    }

    async streamAudio(text, voice, speed, onProgress, options = {}) {
        try {
            const canStreamMp3 = this.supportsMSEMp3();
            console.log('AudioService: Starting stream...', { chars: text.length, voice, speed, canStreamMp3 });

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
            this.downloadName = this.buildDownloadName(voice, responseFormat);

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
                    return_timing: true,
                    lang_code: document.getElementById('lang-select').value || undefined,
                    allow_voice_tags: options.allowVoiceTags || undefined,
                    // short names only mean something alongside the map that defines them
                    voice_aliases: Object.keys(options.voiceAliases || {}).length
                        ? options.voiceAliases
                        : undefined
                }),
                signal: this.controller.signal
            }).catch(error => {
                // Handle abort errors gracefully
                if (error.name === 'AbortError') {
                    console.log('Audio stream request aborted');
                    return null;
                }
                throw error;
            });

            // If request was aborted, return early
            if (!response) {
                return null;
            }

            console.log('AudioService: Got response', {
                status: response.status,
                headers: Object.fromEntries(response.headers.entries())
            });

            const downloadPath = response.headers.get('x-download-path');
            if (downloadPath) {
                await this.setDownloadPath(downloadPath);
                console.log('Download path received:', this.serverDownloadPath);
            }
            this.timingPath = response.headers.get('x-timing-path');

            if (!response.ok) {
                const error = await response.json();
                console.error('AudioService: API error', error);
                const message = (error.detail?.message || 'Failed to generate speech')
                    .replace(/\s*Available voices:.*$/s, '');
                throw new Error(message);
            }

            await this.setupAudioStream(response.body, response, onProgress, estimatedChunks, canUseMseStream);
            return this.audio;
        } catch (error) {
            this.cleanup();
            throw error;
        }
    }

    async setupBlockMode(stream, response, onProgress, estimatedChunks) {
        const loader = new BlockLoader({
            onProgress: (receivedChunks) => onProgress?.(receivedChunks, estimatedChunks)
        });

        const chunks = await loader.load(stream);
        if (!chunks) {
            return;
        }

        const headers = Object.fromEntries(response.headers.entries());
        const downloadPath = headers['x-download-path'];
        if (downloadPath) {
            await this.setDownloadPath(downloadPath);
        }

        onProgress?.(estimatedChunks, estimatedChunks);

        const blobType = response.headers.get('content-type') || 'audio/mpeg';
        const blob = new Blob(chunks, { type: blobType });
        this.audio = new Audio();
        this.audio.volume = this.volume;
        this.attachAudioReadinessEvents();
        this.objectUrl = URL.createObjectURL(blob);
        this.audio.src = this.objectUrl;
        this.audio.load();

        this.attachAudioErrorEvents('block mode');

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
        this.audio.volume = this.volume;
        this.attachAudioReadinessEvents();
        this.attachAudioErrorEvents('stream');

        this.audio.addEventListener('ended', () => {
            this.dispatchEvent('ended');
            // reaching the end is a safe break point too: swap so a replay/scrub uses
            // the full file rather than the evicted streaming buffer.
            if (this.canSwapToFileSource()) {
                this.swapToFileSource();
            }
        });

        this.msePipeline = new MsePipeline(this.audio, {
            onProgress: (receivedChunks) => onProgress?.(receivedChunks, estimatedChunks),
            onReady: () => this.dispatchEvent('ready'),
            onFirstChunk: () => {
                if (this.shouldAutoplay) {
                    setTimeout(() => this.play(), 100);
                }
            },
            onEnd: async () => {
                const headers = Object.fromEntries(response.headers.entries());

                const downloadPath = headers['x-download-path'];
                if (downloadPath) {
                    await this.setDownloadPath(downloadPath);
                } else {
                    console.warn('No X-Download-Path header found. Available headers:',
                        Object.keys(headers).join(', '));
                }

                onProgress?.(estimatedChunks, estimatedChunks);
                await this.loadKnownDuration();
                this.dispatchEvent('complete');

                setTimeout(() => this.dispatchEvent('downloadReady'), 800);

                // hold the finished file for the next seek rather than interrupting playback to install it now
                if (await this.preloadFileSource()) {
                    this.dispatchEvent('ready');
                }
            }
        });

        await this.msePipeline.start(stream);
    }

    play() {
        if (this.audio && !this.audio.error) {
            const duration = this.audio.duration;
            if (this.usingFileSource && Number.isFinite(duration) &&
                duration - this.audio.currentTime <= 0.1) {
                this.audio.currentTime = 0;
            }
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
            this.pendingResume = false;
            this.audio.pause();
            this.dispatchEvent('pause');
            // pausing a finished generation is a safe break point: swap to the full
            // file so the user can scrub the whole track and see the real duration.
            if (this.canSwapToFileSource()) {
                this.swapToFileSource();
            }
        }
    }

    seek(time) {
        if (!this.audio || this.audio.error) {
            return;
        }
        // a seek is already a discontinuity, so spend the swap here instead of interrupting playback
        if (this.swapInProgress || this.canSwapToFileSource()) {
            this.pendingSeek = time;
            this.swapToFileSource(time).then((swapped) => {
                if (!swapped && !this.swapInProgress && !this.usingFileSource &&
                    this.audio && !this.audio.error) {
                    this.audio.currentTime = time;
                }
            });
            return;
        }
        const wasPlaying = !this.audio.paused;
        this.audio.currentTime = time;
        if (wasPlaying) {
            this.play();
        }
    }

    // true only for a finished MSE stream that hasn't been swapped yet. block mode
    // already plays a full-file blob, so it never needs (or gets) a swap.
    canSwapToFileSource() {
        return (
            !this.usingFileSource &&
            !this.swapInProgress &&
            !!this.msePipeline?.canHandoff() &&
            !!this.serverDownloadPath &&
            !!this.audio &&
            !this.audio.error
        );
    }

    // keep the finished file in memory so the switch costs a decode rather than a network seek (#150)
    async preloadFileSource() {
        if (!this.serverDownloadPath) {
            return null;
        }
        if (!this.preloadPromise) {
            this.preloadPromise = (async () => {
                try {
                    const response = await fetch(this.serverDownloadPath, { signal: this.controller?.signal });
                    if (response.ok) {
                        this.fileBlobUrl = URL.createObjectURL(await response.blob());
                    }
                } catch (error) {
                    console.warn('Could not preload the finished file:', error);
                }
                if (!this.fileBlobUrl) {
                    this.preloadPromise = null;
                }
                return this.fileBlobUrl;
            })();
        }
        return await this.preloadPromise;
    }

    // swap the bounded MSE buffer for the finished file. callers pick a moment where the gap is free.
    async swapToFileSource(targetTime = null) {
        if (!this.canSwapToFileSource()) {
            return false;
        }
        this.swapInProgress = true;

        const audio = this.audio;
        const fileUrl = await this.preloadFileSource();

        // without the bytes in hand the stream buffer is worth more than a swap onto a url that just failed
        if (!fileUrl || audio !== this.audio || audio.error || !this.msePipeline) {
            this.swapInProgress = false;
            this.pendingSeek = null;
            return false;
        }

        this.objectUrl = fileUrl;
        this.fileBlobUrl = null;
        const resumePlaying = !audio.paused;
        const resumeTime = targetTime != null ? targetTime : (audio.currentTime || 0);
        const volume = audio.volume;
        const rate = audio.playbackRate;

        // gut the pipeline up front so a late feeder/updateend can't touch MSE state.
        const previousObjectUrl = this.msePipeline.handoff();
        this.msePipeline = null;

        return await new Promise((resolve) => {
            const detach = () => {
                audio.removeEventListener('loadedmetadata', onLoaded);
                audio.removeEventListener('error', onError);
            };

            const onLoaded = () => {
                detach();
                if (audio !== this.audio) { this.swapInProgress = false; resolve(false); return; }
                const duration = audio.duration;
                // a drag that kept moving during the load wins over the position it started from
                const target = this.pendingSeek != null ? this.pendingSeek : resumeTime;
                this.pendingSeek = null;
                if (Number.isFinite(duration) && duration > 0) {
                    // 1:1 timeline (sequence mode appended the same bytes), so the playhead lands where the stream left off
                    audio.currentTime = Math.min(Math.max(target, 0), Math.max(0, duration - 0.05));
                }
                audio.volume = volume;
                audio.playbackRate = rate;
                this.usingFileSource = true;
                this.swapInProgress = false;
                this.dispatchEvent('ready');
                if (resumePlaying) {
                    this.resumeWhenReady(audio);
                }
                resolve(true);
            };

            const onError = () => {
                detach();
                if (audio !== this.audio) { this.swapInProgress = false; resolve(false); return; }
                this.swapInProgress = false;
                this.pendingSeek = null;
                // the stream buffer is gone but the file is still downloadable, so say so rather than leaving a dead player
                console.warn('Failed to switch to file playback:', audio.error);
                this.dispatchEvent('playbackUnavailable');
                resolve(false);
            };

            audio.addEventListener('loadedmetadata', onLoaded, { once: true });
            audio.addEventListener('error', onError, { once: true });

            audio.src = fileUrl;
            audio.load();

            // safe to revoke now that the element no longer references the MSE url.
            if (previousObjectUrl) {
                URL.revokeObjectURL(previousObjectUrl);
            }
        });
    }

    // starting at readyState 1 plays what is decoded and stalls on the rest, which is audible as a stutter
    resumeWhenReady(audio) {
        if (audio.readyState >= 3) {
            this.play();
            return;
        }
        this.pendingResume = true;
        audio.addEventListener('canplay', () => {
            if (this.pendingResume && audio === this.audio) {
                this.pendingResume = false;
                this.play();
            }
        }, { once: true });
    }

    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        if (this.audio) {
            this.audio.volume = this.volume;
        }
    }

    getCurrentTime() {
        return this.audio ? this.audio.currentTime : 0;
    }

    getDuration() {
        const duration = this.audio ? this.audio.duration : 0;
        if (this.msePipeline && this.knownDuration) {
            return this.knownDuration;
        }
        if (Number.isFinite(duration) && duration > 0) {
            return duration;
        }
        return this.knownDuration ?? duration;
    }

    isSeekable() {
        return !!this.audio && !this.audio.error && (!this.msePipeline || !!this.fileBlobUrl);
    }

    async loadKnownDuration() {
        const url = await this.getTimingUrl();
        if (!url) {
            return;
        }
        try {
            const response = await fetch(url);
            if (!response.ok) {
                return;
            }
            const data = await response.json();
            const last = data.chunks?.[data.chunks.length - 1];
            const end = last?.end_time ?? last?.end;
            if (Number.isFinite(end) && end > 0) {
                this.knownDuration = end;
                this.dispatchEvent('ready');
            }
        } catch (error) {
            console.warn('Could not read the generated length:', error);
        }
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
        if (this.fileBlobUrl) {
            URL.revokeObjectURL(this.fileBlobUrl);
            this.fileBlobUrl = null;
        }
        this.preloadPromise = null;
        this.pendingSeek = null;
    }

    cancel() {
        if (this.controller) {
            this.controller.abort();
            this.controller = null;
        }

        this.releaseAudioElement();

        if (this.msePipeline) {
            this.msePipeline.teardown(new Error('AudioService cancelled'));
            this.msePipeline = null;
        }

        this.serverDownloadPath = null;
        this.downloadName = null;
        this.timingPath = null;
        this.knownDuration = null;
        this.usingFileSource = false;
        this.swapInProgress = false;
        this.revokeObjectUrl();
    }

    /** Empties the element by removing src rather than blanking it, which browsers load and report as an error. */
    releaseAudioElement() {
        if (!this.audio) {
            return;
        }

        const audio = this.audio;
        this.audio = null;
        this.eventListeners.forEach((listeners, event) => {
            listeners.forEach((callback) => audio.removeEventListener?.(event, callback));
        });
        audio.pause();
        audio.removeAttribute?.('src');
        audio.load?.();
    }

    cleanup() {
        this.releaseAudioElement();

        if (this.msePipeline) {
            this.msePipeline.teardown(new Error('AudioService cleanup'));
            this.msePipeline = null;
        }

        this.serverDownloadPath = null;
        this.downloadName = null;
        this.timingPath = null;
        this.knownDuration = null;
        this.usingFileSource = false;
        this.swapInProgress = false;
        this.revokeObjectUrl();
    }

    // sent to the server so Content-Disposition carries it, which outranks a.download (#338)
    buildDownloadName(voice, format) {
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        const safeVoice = String(voice || '')
            .replace(/[^A-Za-z0-9._-]+/g, '_')
            .replace(/^[._-]+|[._-]+$/g, '');
        return `${safeVoice || 'speech'}_${stamp}.${format}`;
    }

    async getTimingUrl() {
        return this.timingPath ? config.getApiUrl(`/v1${this.timingPath}`) : null;
    }

    async getTimingDownloadUrl() {
        const url = await this.getTimingUrl();
        if (!url) {
            return null;
        }
        return this.downloadName ? `${url}?name=${encodeURIComponent(this.downloadName)}` : url;
    }

    async setDownloadPath(rawPath) {
        const url = await config.getApiUrl(`/v1${rawPath}`);
        this.serverDownloadPath = this.downloadName
            ? `${url}?name=${encodeURIComponent(this.downloadName)}`
            : url;
    }

    getDownloadUrl() {
        if (!this.serverDownloadPath) {
            console.warn('No download path available');
            return null;
        }
        return this.serverDownloadPath;
    }

    getDownloadName() {
        return this.downloadName;
    }
}

export default AudioService;
