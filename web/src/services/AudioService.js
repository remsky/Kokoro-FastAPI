export class AudioService {
    constructor() {
        this.mediaSource = null;
        this.sourceBuffer = null;
        this.audio = null;
        this.controller = null;
        this.eventListeners = new Map();
        this.minimumPlaybackSize = 50000; // 50KB minimum before playback
        this.textLength = 0;
        this.shouldAutoplay = false;
        this.CHARS_PER_CHUNK = 150; // Estimated chars per chunk
        this.serverDownloadPath = null; // Server-side download path
        this.pendingOperations = []; // Queue for buffer operations
    }

    async streamAudio(text, voice, speed, onProgress) {
        try {
            console.log('AudioService: Starting stream...', { text, voice, speed });
            
            if (this.controller) {
                this.controller.abort();
                this.controller = null;
            }
            
            this.controller = new AbortController();
            this.cleanup();
            onProgress?.(0, 1); // Reset progress to 0
            this.textLength = text.length;
            this.shouldAutoplay = document.getElementById('autoplay-toggle').checked;
            
            // Calculate expected number of chunks based on text length
            const estimatedChunks = Math.max(1, Math.ceil(this.textLength / this.CHARS_PER_CHUNK));
            
            console.log('AudioService: Making API call...', { text, voice, speed });
            
            const response = await fetch('/v1/audio/speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: text,
                    voice: voice,
                    response_format: 'mp3', // Always use mp3 for streaming playback
                    download_format: document.getElementById('format-select').value || 'mp3', // Format for final download
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

            // Check for download path as soon as we get the response
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

            await this.setupAudioStream(response.body, response, onProgress, estimatedChunks);
            return this.audio;
        } catch (error) {
            this.cleanup();
            throw error;
        }
    }

    async setupAudioStream(stream, response, onProgress, estimatedChunks) {
        this.audio = new Audio();
        this.mediaSource = new MediaSource();
        this.audio.src = URL.createObjectURL(this.mediaSource);
        
        // Monitor for audio element errors
        this.audio.addEventListener('error', (e) => {
            console.error('Audio error:', this.audio.error);
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
            });
        });
    }

    async processStream(stream, response, onProgress, estimatedChunks) {
        const reader = stream.getReader();
        let hasStartedPlaying = false;
        let receivedChunks = 0;

        try {
            while (true) {
                const {value, done} = await reader.read();
                
                if (done) {
                    // Get final download path from header after stream is complete
                    const headers = Object.fromEntries(response.headers.entries());
                    console.log('Response headers at stream end:', headers);
                    
                    const downloadPath = headers['x-download-path'];
                    if (downloadPath) {
                        // Prepend /v1 since the router is mounted there
                        this.serverDownloadPath = `/v1${downloadPath}`;
                        console.log('Download path received:', this.serverDownloadPath);
                    } else {
                        console.warn('No X-Download-Path header found. Available headers:',
                            Object.keys(headers).join(', '));
                    }

                    if (this.mediaSource.readyState === 'open') {
                        this.mediaSource.endOfStream();
                    }
                    
                    // Signal completion
                    onProgress?.(estimatedChunks, estimatedChunks);
                    this.dispatchEvent('complete');
                    
                    // Check if we should autoplay for small inputs that didn't trigger during streaming
                    if (this.shouldAutoplay && !hasStartedPlaying && this.sourceBuffer.buffered.length > 0) {
                        setTimeout(() => this.play(), 100);
                    }
                    
                    setTimeout(() => {
                        this.dispatchEvent('downloadReady');
                    }, 800);
                    return;
                }

                receivedChunks++;
                onProgress?.(receivedChunks, estimatedChunks);

                try {
                    // Check for audio errors before proceeding
                    if (this.audio.error) {
                        console.error('Audio error detected:', this.audio.error);
                        continue; // Skip this chunk if audio is in error state
                    }

                    // Only remove old data if we're hitting quota errors
                    if (this.sourceBuffer.buffered.length > 0) {
                        const currentTime = this.audio.currentTime;
                        const start = this.sourceBuffer.buffered.start(0);
                        const end = this.sourceBuffer.buffered.end(0);
                        
                        // Only remove if we have a lot of historical data
                        if (currentTime - start > 30) {
                            const removeEnd = Math.max(start, currentTime - 15);
                            if (removeEnd > start) {
                                await this.removeBufferRange(start, removeEnd);
                            }
                        }
                    }

                    await this.appendChunk(value);

                    if (!hasStartedPlaying && this.sourceBuffer.buffered.length > 0) {
                        hasStartedPlaying = true;
                        if (this.shouldAutoplay) {
                            setTimeout(() => this.play(), 100);
                        }
                    }
                } catch (error) {
                    if (error.name === 'QuotaExceededError') {
                        // If we hit quota, try more aggressive cleanup
                        if (this.sourceBuffer.buffered.length > 0) {
                            const currentTime = this.audio.currentTime;
                            const start = this.sourceBuffer.buffered.start(0);
                            const removeEnd = Math.max(start, currentTime - 5);
                            if (removeEnd > start) {
                                await this.removeBufferRange(start, removeEnd);
                                // Retry append after removing data
                                try {
                                    await this.appendChunk(value);
                                } catch (retryError) {
                                    console.warn('Buffer error after cleanup:', retryError);
                                }
                            }
                        }
                    } else {
                        console.warn('Buffer error:', error);
                    }
                }
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                throw error;
            }
        }
    }

    async removeBufferRange(start, end) {
        // Double check that end is greater than start
        if (end <= start) {
            console.warn('Invalid buffer remove range:', {start, end});
            return;
        }

        return new Promise((resolve) => {
            const doRemove = () => {
                try {
                    this.sourceBuffer.remove(start, end);
                } catch (e) {
                    console.warn('Error removing buffer:', e);
                }
                resolve();
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
        // Don't append if audio is in error state
        if (this.audio.error) {
            console.warn('Skipping chunk append due to audio error');
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
        if (this.sourceBuffer.updating || this.pendingOperations.length === 0) {
            return;
        }

        // Don't process if audio is in error state
        if (this.audio.error) {
            console.warn("Skipping operation due to audio error");
            return;
        }

        const operation = this.pendingOperations.shift();

        try {
            this.sourceBuffer.appendBuffer(operation.chunk);

            // Set up event listeners
            const onUpdateEnd = () => {
                operation.resolve();
                this.sourceBuffer.removeEventListener("updateend", onUpdateEnd);
                this.sourceBuffer.removeEventListener(
                    "updateerror",
                    onUpdateError
                );
                // Process the next operation
                this.processNextOperation();
            };

            const onUpdateError = (event) => {
                operation.reject(event);
                this.sourceBuffer.removeEventListener("updateend", onUpdateEnd);
                this.sourceBuffer.removeEventListener(
                    "updateerror",
                    onUpdateError
                );
                // Decide whether to continue processing
                if (event.name !== "InvalidStateError") {
                    this.processNextOperation();
                }
            };

            this.sourceBuffer.addEventListener("updateend", onUpdateEnd);
            this.sourceBuffer.addEventListener("updateerror", onUpdateError);
        } catch (error) {
            operation.reject(error);
            // Only continue processing if it's not a fatal error
            if (error.name !== "InvalidStateError") {
                this.processNextOperation();
            }
        }
    }

    play() {
        if (this.audio && this.audio.readyState >= 2 && !this.audio.error) {
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

    cancel() {
        if (this.controller) {
            this.controller.abort();
            this.controller = null;
        }

        if (this.audio) {
            this.audio.pause();
            this.audio.src = "";
            this.audio = null;
        }

        if (this.mediaSource && this.mediaSource.readyState === "open") {
            try {
                this.mediaSource.endOfStream();
            } catch (e) {
                // Ignore errors during cleanup
            }
        }

        this.mediaSource = null;
        if (this.sourceBuffer) {
            this.sourceBuffer.removeEventListener("updateend", () => {});
            this.sourceBuffer.removeEventListener("updateerror", () => {});
            this.sourceBuffer = null;
        }
        this.serverDownloadPath = null;
        this.pendingOperations = [];
    }

    cleanup() {
        if (this.audio) {
            this.eventListeners.forEach((listeners, event) => {
                listeners.forEach((callback) => {
                    this.audio.removeEventListener(event, callback);
                });
            });

            this.audio.pause();
            this.audio.src = "";
            this.audio = null;
        }

        if (this.mediaSource && this.mediaSource.readyState === "open") {
            try {
                this.mediaSource.endOfStream();
            } catch (e) {
                // Ignore errors during cleanup
            }
        }

        this.mediaSource = null;
        if (this.sourceBuffer) {
            this.sourceBuffer.removeEventListener("updateend", () => {});
            this.sourceBuffer.removeEventListener("updateerror", () => {});
            this.sourceBuffer = null;
        }
        this.serverDownloadPath = null;
        this.pendingOperations = [];
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
