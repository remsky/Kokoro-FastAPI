import assert from 'node:assert/strict';
import test from 'node:test';

const { MsePipeline } = await import('../../src/services/audio/MsePipeline.js');

class FakeSourceBuffer extends EventTarget {
    constructor(ranges, { rejectFirstAppend = false } = {}) {
        super();
        this.updating = false;
        this.buffered = {
            length: ranges.length,
            start: (index) => ranges[index][0],
            end: (index) => ranges[index][1],
        };
        this.appends = 0;
        this.removes = [];
        this.rejectFirstAppend = rejectFirstAppend;
    }

    appendBuffer() {
        if (this.rejectFirstAppend) {
            this.rejectFirstAppend = false;
            throw new DOMException('buffer full', 'QuotaExceededError');
        }
        this.appends += 1;
        this.settle();
    }

    remove(start, end) {
        this.removes.push([start, end]);
        this.settle();
    }

    settle() {
        this.updating = true;
        setTimeout(() => {
            this.updating = false;
            this.dispatchEvent(new Event('updateend'));
        }, 0);
    }
}

function feed(sourceBuffer, currentTime) {
    const pipeline = new MsePipeline({ currentTime });
    pipeline.mediaSource = { readyState: 'open', endOfStream() { this.readyState = 'ended'; } };
    pipeline.sourceBuffer = sourceBuffer;
    pipeline.chunkQueue = [new Uint8Array([1])];
    pipeline.streamFinished = true;
    return pipeline;
}

test('audio more than 30s behind the playhead is evicted before the next append', async () => {
    const buffer = new FakeSourceBuffer([[0, 100]]);
    await feed(buffer, 50).runFeeder();

    assert.deepEqual(buffer.removes, [[0, 35]]);
    assert.equal(buffer.appends, 1);
});

test('a quota error frees played audio and retries the same chunk', async () => {
    const buffer = new FakeSourceBuffer([[0, 70]], { rejectFirstAppend: true });
    await feed(buffer, 20).runFeeder();

    assert.deepEqual(buffer.removes, [[0, 15]]);
    assert.equal(buffer.appends, 1);
});

test('appending waits while more than 60s is buffered ahead of the playhead', async () => {
    const buffer = new FakeSourceBuffer([[0, 100]]);
    const pipeline = feed(buffer, 10);
    const done = pipeline.runFeeder();

    await new Promise((resolve) => setTimeout(resolve, 100));
    assert.equal(buffer.appends, 0);

    pipeline.audio.currentTime = 50;
    await done;
    assert.equal(buffer.appends, 1);
});
