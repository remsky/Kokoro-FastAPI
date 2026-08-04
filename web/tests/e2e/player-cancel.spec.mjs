import { expect, test } from '@playwright/test';

test('cancel mid-generation keeps the controls consistent', async ({ page }) => {
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(error));

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
        await new Promise((resolve) => setTimeout(resolve, 10_000));
        await route.abort().catch(() => {});
    });

    await page.goto('/');
    await page.locator('.page-content').fill('A sentence that never finishes generating.');
    await page.locator('#generate-btn').click();

    await expect(page.locator('#cancel-btn')).toBeVisible();
    await page.locator('#cancel-btn').click();

    await expect(page.locator('#cancel-btn')).toBeHidden();
    await expect(page.locator('#volume-slider')).toHaveValue('100');

    const timeDisplayWrites = await page.evaluate(() => new Promise((resolve) => {
        let count = 0;
        const observer = new MutationObserver(() => { count += 1; });
        observer.observe(document.getElementById('time-display'), {
            childList: true,
            characterData: true,
            subtree: true,
        });
        setTimeout(() => { observer.disconnect(); resolve(count); }, 500);
    }));

    expect(timeDisplayWrites).toBe(0);
    expect(pageErrors).toEqual([]);
});
