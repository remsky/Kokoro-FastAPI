import { expect, test } from './fixtures/app.mjs';

const VOICES = [
    { id: 'am_eric', name: 'am_eric', overall_grade: 'D' },
    { id: 'bf_emma', name: 'bf_emma', overall_grade: 'B-' },
    { id: 'af_sky', name: 'af_sky', overall_grade: 'C-' },
];

const options = (page) => page.locator('.voice-option');

test.beforeEach(async ({ page }) => {
    await page.route('**/v1/audio/voices', (route) => route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ voices: VOICES }),
    }));
    await page.goto('/');
    await expect(page.locator('.selected-voice-tag').first()).toBeVisible({ timeout: 15_000 });
});

test('the sort toggle switches grade order to a to z and remembers it', async ({ page }) => {
    const sort = page.locator('#voice-sort');
    await expect(sort).toHaveAccessibleName('Sort voices A to Z');
    await expect(options(page)).toHaveText(['bf_emma', 'af_sky', 'am_eric']);

    await page.locator('#voice-search').click();
    await sort.click();
    await expect(sort).toHaveAccessibleName('Sort voices by grade');
    await expect(page.locator('#voice-dropdown')).toHaveClass(/show/);
    await expect(options(page)).toHaveText(['af_sky', 'am_eric', 'bf_emma']);

    await page.reload();
    await expect(sort).toHaveAccessibleName('Sort voices by grade');
    await expect(options(page)).toHaveText(['af_sky', 'am_eric', 'bf_emma']);
});
