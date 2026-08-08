// MSE state machine: bounded buffer feeding, quota handling, handoff to file playback. talks back to AudioService only through the callback bag so the event bus stays single-owner.
export class MsePipeline {
    constructor(audio, callbacks = {}) {
        this.audio = audio;
        this.onProgress = callbacks.onProgress;
        this.onReady = callbacks.onReady;
        this.onFirstChunk = callbacks.onFirstChunk;
        this.onEnd = callbacks.onEnd;

        this.mediaSource = null;
        this.sourceBuffer = null;
        this.objectUrl = null;
        this.chunkQueue = [];
        this.pendingOperations = [];
        this.feederWakeup = null;
        this.streamFinished = false;
        this.MAX_LEAD_SECONDS = 60;
    }

    // points the audio element at a fresh MediaSource and consumes the response stream. resolves once the stream is fully read, not when playback ends.
    start(stream) {
        this.mediaSource = new MediaSource();
        this.objectUrl = URL.createObjectURL(this.mediaSource);
        this.audio.src = this.objectUrl;

        return new Promise((resolve, reject) => {
            this.mediaSource.addEventListener('sourceopen', async () => {
                try {
                    this.sourceBuffer = this.mediaSource.addSourceBuffer('audio/mpeg');
                    this.sourceBuffer.mode = 'sequence';

                    this.sourceBuffer.addEventListener('updateend', () => {
                        this.processNextOperation();
                    });

                    await this.processStream(stream);
                    resolve();
                } catch (error) {
                    reject(error);
                }
            }, { once: true });
        });
    }

    async processStream(stream) {
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
                    this.streamFinished = true;
                    this.wakeFeeder();
                    await this.onEnd?.();
                    return;
                }

                receivedChunks++;
                this.onProgress?.(receivedChunks);
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
        let hasDeliveredFirstChunk = false;

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
                this.onReady?.();

                if (!hasDeliveredFirstChunk && this.sourceBuffer?.buffered.length > 0) {
                    hasDeliveredFirstChunk = true;
                    this.onFirstChunk?.();
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

    // true only for a finished stream that still holds MSE state (i.e. not handed off)
    canHandoff() {
        return this.streamFinished && this.mediaSource !== null;
    }

    // stop mid-swap: drop MSE refs so a late feeder/updateend tick can't touch them, and return the object url for the caller to revoke once the element points elsewhere.
    handoff() {
        this.mediaSource = null;
        this.sourceBuffer = null;

        // without a source buffer nothing can settle these, the feeder would await forever
        this.rejectPendingOperations(new Error('AudioService swapped to file source'));
        this.chunkQueue = [];
        this.wakeFeeder();

        const objectUrl = this.objectUrl;
        this.objectUrl = null;
        return objectUrl;
    }

    // full stop for cancel/cleanup: close the MediaSource, settle pending work, revoke the url
    teardown(reason) {
        if (this.mediaSource && this.mediaSource.readyState === 'open') {
            try {
                this.mediaSource.endOfStream();
            } catch (e) {
            }
        }

        this.mediaSource = null;
        this.sourceBuffer = null;
        this.rejectPendingOperations(reason);
        this.chunkQueue = [];
        this.streamFinished = true;
        this.wakeFeeder();

        if (this.objectUrl) {
            URL.revokeObjectURL(this.objectUrl);
            this.objectUrl = null;
        }
    }
}

export default MsePipeline;
