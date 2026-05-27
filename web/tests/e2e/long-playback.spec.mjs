import { expect, test } from '@playwright/test';

function longText() {
    return Array.from({ length: 2000 }, (_, index) => `word${index}`).join(' ');
}

test('long MP3 generation uses MediaSource streaming', async ({ page }) => {
    await page.addInitScript(() => {
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
    });

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

    let speechRequestBody = null;
    await page.route('**/v1/audio/speech', async (route) => {
        speechRequestBody = JSON.parse(route.request().postData());
        await route.fulfill({
            contentType: 'audio/mpeg',
            headers: { 'X-Download-Path': '/download/test.mp3' },
            body: Buffer.from([0xff, 0xfb, 0x90, 0x64]),
        });
    });

    await page.goto('/');
    await page.locator('.page-content').fill(longText());
    await page.locator('#generate-btn').click();
    await expect.poll(() => speechRequestBody).not.toBeNull();

    expect(speechRequestBody.response_format).toBe('mp3');
    expect(speechRequestBody.stream).toBe(true);
    await expect.poll(() => page.evaluate(() => window.__mediaSourceConstructed)).toBeGreaterThan(0);
    await expect.poll(() => page.evaluate(() => window.__sourceBufferCreated)).toBeGreaterThan(0);
});
