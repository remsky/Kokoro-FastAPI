import { expect, test } from '@playwright/test';

// TODO: wire this suite into CI, nothing under .github/workflows runs playwright today

function longText() {
    return Array.from({ length: 2000 }, (_, index) => `word${index}`).join(' ');
}

function mockMediaSource() {
        class MockSourceBuffer extends EventTarget {
            constructor() {
                super();
                this.updating = false;
                this.mode = 'segments';
                this.buffered = {
                    length: 0,
                    start: () => 0,
                    end: () => 0,
                };
            }

            appendBuffer() {
                window.__sourceBufferAppends = (window.__sourceBufferAppends || 0) + 1;
                this.updating = true;
                setTimeout(() => {
                    this.updating = false;
                    this.dispatchEvent(new Event('updateend'));
                }, 0);
            }

            remove() {
                this.updating = true;
                setTimeout(() => {
                    this.updating = false;
                    this.dispatchEvent(new Event('updateend'));
                }, 0);
            }
        }

        class MockMediaSource extends EventTarget {
            constructor() {
                super();
                window.__mediaSourceConstructed = (window.__mediaSourceConstructed || 0) + 1;
                this.readyState = 'closed';
                setTimeout(() => {
                    this.readyState = 'open';
                    this.dispatchEvent(new Event('sourceopen'));
                }, 0);
            }

            static isTypeSupported() {
                return true;
            }

            addSourceBuffer() {
                window.__sourceBufferCreated = (window.__sourceBufferCreated || 0) + 1;
                return new MockSourceBuffer();
            }

            endOfStream() {
                this.readyState = 'ended';
            }
        }

        window.__mediaSourceConstructed = 0;
        window.__sourceBufferCreated = 0;
        window.__sourceBufferAppends = 0;
        Object.defineProperty(window, 'MediaSource', {
            configurable: true,
            value: MockMediaSource,
        });

        // createObjectURL does real WebIDL overload resolution, a look-alike MediaSource throws
        const mockObjectUrl = 'blob:mock-mediasource';
        const realCreateObjectURL = URL.createObjectURL.bind(URL);
        const realRevokeObjectURL = URL.revokeObjectURL.bind(URL);
        URL.createObjectURL = (obj) => (obj instanceof MockMediaSource ? mockObjectUrl : realCreateObjectURL(obj));
        URL.revokeObjectURL = (url) => (url === mockObjectUrl ? undefined : realRevokeObjectURL(url));
}

async function mockServer(page, speechHeaders = {}) {
    const captured = { speechRequestBody: null };

    await page.addInitScript(mockMediaSource);

    await page.route('**/web/config', async (route) => {
        await route.fulfill({
            contentType: 'application/json',
            body: JSON.stringify({ root_path: '', version: 'test' }),
        });
    });

    await page.route('**/v1/audio/voices', async (route) => {
        await route.fulfill({
            contentType: 'application/json',
            body: JSON.stringify({ voices: [{ id: 'af_heart', name: 'af_heart' }] }),
        });
    });

    await page.route('**/v1/audio/speech', async (route) => {
        captured.speechRequestBody = JSON.parse(route.request().postData());
        await route.fulfill({
            contentType: 'audio/mpeg',
            headers: { 'X-Download-Path': '/download/test.mp3', ...speechHeaders },
            body: Buffer.from([0xff, 0xfb, 0x90, 0x64]),
        });
    });

    return captured;
}

test('long MP3 generation uses MediaSource streaming', async ({ page }) => {
    const captured = await mockServer(page);

    await page.goto('/');
    await page.locator('.page-content').fill(longText());
    await page.locator('#generate-btn').click();
    await expect.poll(() => captured.speechRequestBody).not.toBeNull();

    expect(captured.speechRequestBody.response_format).toBe('mp3');
    expect(captured.speechRequestBody.stream).toBe(true);
    await expect.poll(() => page.evaluate(() => window.__mediaSourceConstructed)).toBeGreaterThan(0);
    await expect.poll(() => page.evaluate(() => window.__sourceBufferCreated)).toBeGreaterThan(0);
});

test('the streamed track shows its real length and stays unscrubbable', async ({ page }) => {
    await mockServer(page, { 'X-Timing-Path': '/timing/test.json' });

    await page.route('**/v1/timing/test.json', async (route) => {
        await route.fulfill({
            contentType: 'application/json',
            body: JSON.stringify({ chunks: [{ text: 'one', start_time: 0, end_time: 156.35 }] }),
        });
    });

    await page.goto('/');
    await page.locator('.page-content').fill(longText());
    await page.locator('#generate-btn').click();

    await expect(page.locator('#time-display')).toHaveText('0:00 / 2:36');
    await expect(page.locator('#seek-slider')).toBeDisabled();
    await expect(page.locator('#seek-slider')).not.toHaveClass(/no-duration/);
});
