import { expect, test } from '@playwright/test';

// editor and selector behaviour only, so nothing here asks the server to render
const API = process.env.KOKORO_BASE_URL || 'http://localhost:8880';

test.beforeAll(async ({ request }) => {
    const probe = await request.get(`${API}/web/config`).catch(() => null);
    test.skip(!probe || !probe.ok(), `no Kokoro server at ${API}`);
});

test.beforeEach(async ({ page }) => {
    await page.goto(`${API}/web/`);
    await expect(page.locator('.selected-voice-tag').first()).toBeVisible({ timeout: 15_000 });
});

const editor = (page) => page.locator('.page-content');
const toggle = (page) => page.locator('#voice-tags-toggle');

test('the toggle is off until asked for, and seeds from the mixer', async ({ page }) => {
    await expect(toggle(page)).not.toBeChecked();
    await expect(page.locator('#voice-tag-notice')).toBeHidden();

    await editor(page).fill('Hello there.');
    await toggle(page).check();

    const voice = await page.locator('.selected-voice-tag .voice-name').first().textContent();
    await expect(editor(page)).toHaveValue(`[voice:${voice.trim()}] Hello there.`);
});

test('a weighted mix seeds as a single tag the server will accept', async ({ page }) => {
    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(1).click();
    await expect(page.locator('.selected-voice-tag')).toHaveCount(2);

    // fill() alone leaves the dropdown open over the toggle
    await editor(page).click();
    await editor(page).fill('Hello there.');
    await toggle(page).check();

    // the mixer's own string, which is what the voice parameter would have carried
    await expect(editor(page)).toHaveValue(/^\[voice:\w+\(1\)\+\w+\(1\)\] Hello there\.$/);
});

test('seeding happens once, not on every toggle', async ({ page }) => {
    await editor(page).fill('Hello there.');
    await toggle(page).check();
    const seeded = await editor(page).inputValue();

    await toggle(page).uncheck();
    await toggle(page).check();
    await expect(editor(page)).toHaveValue(seeded);
});

test('clicking a voice inserts it where the caret is', async ({ page }) => {
    await editor(page).fill('First line. Second line.');
    await toggle(page).check();

    // between the two sentences, past the tag the toggle just seeded
    const seeded = await editor(page).inputValue();
    await editor(page).click();
    await page.locator('.page-content').evaluate((el, at) => {
        el.focus();
        el.setSelectionRange(at, at);
    }, seeded.indexOf('Second'));

    await page.locator('#voice-search').click();
    const target = page.locator('.voice-option').nth(3);
    const voice = (await target.textContent()).trim();
    await target.click();

    await expect(editor(page)).toHaveValue(`${seeded.replace('Second', `[voice:${voice}] Second`)}`);
});

test('the mixer stays live in tag mode because untagged text still uses it', async ({ page }) => {
    await toggle(page).check();
    await expect(page.locator('#selected-voices')).toHaveClass(/as-cast/);

    // dimmed, but the weight still decides how anything ahead of the first tag sounds
    const weight = page.locator('.selected-voice-tag input').first();
    await expect(weight).toBeEditable();
});

test('tags left behind with the toggle off can be removed in one click', async ({ page }) => {
    await editor(page).fill('First line.\nSecond line.');
    await toggle(page).check();

    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(2).click();
    await expect(page.locator('#voice-tag-notice')).toBeHidden();

    await toggle(page).uncheck();
    await expect(page.locator('#voice-tag-notice')).toBeVisible();
    await expect(page.locator('#voice-tag-notice-text')).toHaveText('2 voice tags will be read aloud.');

    await page.locator('#remove-voice-tags-btn').click();
    await expect(page.locator('#voice-tag-notice')).toBeHidden();
    await expect(editor(page)).toHaveValue('First line.\nSecond line.');
});

test('pasted tags are noticed without the toggle ever being touched', async ({ page }) => {
    await editor(page).fill('[voice:af_bella] Hello there.');

    await expect(page.locator('#voice-tag-notice')).toBeVisible();
    await expect(page.locator('#voice-tag-notice-text')).toHaveText('1 voice tag will be read aloud.');
});

test('a tag on its own is not enough to generate', async ({ page }) => {
    await editor(page).fill('');
    await toggle(page).check();

    await page.locator('#generate-btn').click();
    await expect(page.locator('#status')).toHaveText('Please enter some text');
});
