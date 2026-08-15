import { expect, test } from '@playwright/test';

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

async function formatIntoPages(page, text) {
    await editor(page).fill(text);
    await page.locator('.chars-input').fill('100');
    await page.locator('.format-btn').click();
    return parseInt(await page.locator('.page-total').textContent(), 10);
}

test('the page number is typed, not clicked forty times', async ({ page }) => {
    const total = await formatIntoPages(page, 'word '.repeat(120).trim());
    expect(total).toBeGreaterThan(2);

    await page.locator('.page-jump').fill(String(total));
    await page.locator('.page-jump').press('Enter');
    await expect(page.locator('.next-btn')).toBeDisabled();
    await expect(page.locator('.prev-btn')).toBeEnabled();

    await page.locator('.page-jump').fill('99');
    await page.locator('.page-jump').press('Enter');
    await expect(page.locator('.page-jump')).toHaveValue(String(total));
    await page.locator('.page-jump').fill('0');
    await page.locator('.page-jump').press('Enter');
    await expect(page.locator('.page-jump')).toHaveValue('1');
    await expect(page.locator('.prev-btn')).toBeDisabled();
});

test('find is there before the text is ever formatted', async ({ page }) => {
    await editor(page).fill('one needle in\nplain text');

    await page.locator('.find-menu summary').click();
    await page.locator('.find-input').fill('needle');
    await expect(page.locator('.find-count')).toHaveText('1 match');

    await page.locator('.find-next-btn').click();
    const selected = await editor(page).evaluate((el) => el.value.slice(el.selectionStart, el.selectionEnd));
    expect(selected).toBe('needle');
});

test('find counts across every page and walks the matches in order', async ({ page }) => {
    // the blank line collapses when formatted, so this catches the highlight drifting off its match
    const filler = 'filler '.repeat(18).trim();
    await formatIntoPages(page, `needle ${filler}\n\nneedle ${filler} needle`);

    await page.locator('.find-menu summary').click();
    await page.locator('.find-input').fill('needle');
    await expect(page.locator('.find-count')).toHaveText('3 matches');

    await page.locator('.find-next-btn').click();
    await expect(page.locator('.find-count')).toHaveText('1 of 3');
    await expect(page.locator('.page-jump')).toHaveValue('1');

    await page.locator('.find-next-btn').click();
    await page.locator('.find-next-btn').click();
    await expect(page.locator('.find-count')).toHaveText('3 of 3');
    await expect(page.locator('.next-btn')).toBeDisabled();
    const selected = await editor(page).evaluate((el) => el.value.slice(el.selectionStart, el.selectionEnd));
    expect(selected).toBe('needle');

    await page.locator('.find-next-btn').click();
    await expect(page.locator('.find-count')).toHaveText('1 of 3');
    await expect(page.locator('.page-jump')).toHaveValue('1');
});

test('replace takes the selected match, replace all takes the document', async ({ page }) => {
    const filler = 'filler '.repeat(18).trim();
    await formatIntoPages(page, `needle ${filler} needle ${filler} needle`);

    await page.locator('.find-menu summary').click();
    await page.locator('.find-input').fill('needle');
    await page.locator('.replace-input').fill('noodle');

    // the first press only finds, the second replaces what it found and moves on
    await page.locator('.replace-one-btn').click();
    await page.locator('.replace-one-btn').click();
    await expect(page.locator('.find-count')).toHaveText('1 of 2');

    await page.locator('.replace-all-btn').click();
    await expect(page.locator('.find-count')).toHaveText('2 replaced');

    // a fresh count reads the whole document, so it proves every page was rewritten
    await page.locator('.find-input').fill('needle');
    await expect(page.locator('.find-count')).toHaveText('0 matches');
    await page.locator('.find-input').fill('noodle');
    await expect(page.locator('.find-count')).toHaveText('3 matches');
});
