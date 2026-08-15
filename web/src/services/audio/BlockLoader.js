// block-mode fallback reader: buffers the whole response so AudioService can play it from a blob
export class BlockLoader {
    constructor(callbacks = {}) {
        this.onProgress = callbacks.onProgress;
    }

    // resolves with the buffered chunks, or null when the fetch was aborted mid-read
    async load(stream) {
        const reader = stream.getReader();
        const chunks = [];
        let receivedChunks = 0;

        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                chunks.push(value);
                receivedChunks++;
                this.onProgress?.(receivedChunks);
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                return null;
            }
            throw error;
        }

        return chunks;
    }
}

export default BlockLoader;
