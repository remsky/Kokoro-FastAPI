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
const castNames = (page) => page.locator('.cast-member .cast-name');

test('the toggle is off until asked for, and seeds from the mixer', async ({ page }) => {
    await expect(toggle(page)).not.toBeChecked();
    await expect(page.locator('#voice-cast')).toBeHidden();
    await expect(page.locator('#create-tag-btn')).toBeHidden();
    await expect(page.locator('#voice-tag-notice')).toBeHidden();

    const voice = (await page.locator('.selected-voice-tag .voice-name').first().textContent()).trim();
    await editor(page).fill('Hello there.');
    await toggle(page).check();

    await expect(editor(page)).toHaveValue(`[voice:${voice}] Hello there.`);
    // the staged mix moved into the cast, leaving the mixer free for the next voice
    await expect(castNames(page)).toHaveText([voice]);
    await expect(page.locator('.selected-voice-tag')).toHaveCount(0);
});

test('a weighted mix becomes one cast member under a short name', async ({ page }) => {
    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(1).click();
    await expect(page.locator('.selected-voice-tag')).toHaveCount(2);

    // fill() alone leaves the dropdown open over the toggle
    await editor(page).click();
    await editor(page).fill('Hello there.');
    await toggle(page).check();

    // the recipe stays on the chip, the text only carries the name
    const name = (await castNames(page).first().textContent()).trim();
    expect(name).toMatch(/^[a-z]+\d*$/);
    await expect(editor(page)).toHaveValue(`[voice:${name}] Hello there.`);
    await expect(page.locator('.cast-member').first()).toHaveAttribute('data-mix', /^\w+\(1\)\+\w+\(1\)$/);
});

test('seeding happens once, not on every toggle', async ({ page }) => {
    await editor(page).fill('Hello there.');
    await toggle(page).check();
    const seeded = await editor(page).inputValue();

    await toggle(page).uncheck();
    // the default returns to the mixer, so there is still a voice to speak with
    await expect(page.locator('.selected-voice-tag')).toHaveCount(1);

    await toggle(page).check();
    await expect(editor(page)).toHaveValue(seeded);
    await expect(castNames(page)).toHaveCount(1);
});

test('the mixer keeps mixing in tag mode rather than inserting on click', async ({ page }) => {
    await editor(page).fill('Hello there.');
    await toggle(page).check();
    const seeded = await editor(page).inputValue();

    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(1).click();
    await page.locator('.voice-option').nth(2).click();

    // two clicks build one mix, and none of it reaches the text until it is inserted
    await expect(page.locator('.selected-voice-tag')).toHaveCount(2);
    await expect(page.locator('.selected-voice-tag input').first()).toBeEditable();
    await expect(editor(page)).toHaveValue(seeded);
});

test('creating a tag stages the mix without touching the text', async ({ page }) => {
    await editor(page).fill('First line. Second line.');
    await toggle(page).check();
    const seeded = await editor(page).inputValue();

    await page.locator('#voice-search').click();
    const option = page.locator('.voice-option').nth(3);
    const mix = (await option.textContent()).trim();
    await option.click();

    // the dropdown covers the row the button sits in
    await editor(page).click();
    await page.locator('#create-tag-btn').click();

    await expect(castNames(page)).toHaveText([/.+/, mix]);
    await expect(editor(page)).toHaveValue(seeded);
    await expect(page.locator('.selected-voice-tag')).toHaveCount(0);
});

test('a created tag is placed at the caret when its chip is clicked', async ({ page }) => {
    await editor(page).fill('First line. Second line.');
    await toggle(page).check();
    const seeded = await editor(page).inputValue();

    await page.locator('#voice-search').click();
    const option = page.locator('.voice-option').nth(3);
    const mix = (await option.textContent()).trim();
    await option.click();

    // the caret has to survive both presses
    await editor(page).click();
    await page.locator('#create-tag-btn').click();
    await editor(page).evaluate((el, at) => {
        el.focus();
        el.setSelectionRange(at, at);
    }, seeded.indexOf('Second'));
    await page.locator(`.cast-member[data-mix="${mix}"]`).click();

    await expect(editor(page)).toHaveValue(seeded.replace('Second', `[voice:${mix}] Second`));
    await expect(castNames(page)).toHaveCount(2);
});

test('a cast voice is placed again and again from its chip', async ({ page }) => {
    await editor(page).fill('One. Two. Three.');
    await toggle(page).check();

    const chip = page.locator('.cast-member').first();
    await editor(page).click();
    await chip.click();
    await chip.click();

    // re-inserting an existing member never grows the cast
    await expect(castNames(page)).toHaveCount(1);
    expect((await editor(page).inputValue()).match(/\[voice:/g)).toHaveLength(3);
});

test('no tag can be created until something is mixed', async ({ page }) => {
    await toggle(page).check();
    await expect(page.locator('#create-tag-btn')).toBeVisible();
    await expect(page.locator('#create-tag-btn')).toBeDisabled();

    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(1).click();
    await expect(page.locator('#create-tag-btn')).toBeEnabled();
});

test('a cast member can be dropped again', async ({ page }) => {
    await toggle(page).check();
    await expect(castNames(page)).toHaveCount(1);

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="remove"]').click();
    await expect(castNames(page)).toHaveCount(0);
});

test('a cast member can be renamed, and the tags follow', async ({ page }) => {
    await editor(page).fill('First line. Second line.');
    await toggle(page).check();

    const chip = page.locator('.cast-member').first();
    await editor(page).click();
    await chip.click(); // a second tag, so the rename has more than one to follow

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill('narrator');
    await page.locator('.cast-rename-input').press('Enter');

    await expect(castNames(page)).toHaveText(['narrator']);
    expect((await editor(page).inputValue()).match(/\[voice:narrator\]/g)).toHaveLength(2);
});

test('a name that would shadow a real voice is refused', async ({ page }) => {
    await toggle(page).check();
    const taken = (await page.locator('.voice-option').nth(2).textContent()).trim();
    const before = (await castNames(page).first().textContent()).trim();

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill(taken);
    await page.locator('.cast-rename-input').press('Enter');

    await expect(page.locator('#status')).toHaveText(`"${taken}" is already taken`);
    await expect(castNames(page)).toHaveText([before]);
});

test('one speaker can be cleared out of the text from its own menu', async ({ page }) => {
    await editor(page).fill('First line.\nSecond line.');
    await toggle(page).check();

    await page.locator('#voice-search').click();
    const other = page.locator('.voice-option').nth(2);
    const mix = (await other.textContent()).trim();
    await other.click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();
    await page.locator(`.cast-member[data-mix="${mix}"]`).click();
    expect((await editor(page).inputValue()).match(/\[voice:/g)).toHaveLength(2);

    await page.locator(`.cast-member[data-mix="${mix}"] .cast-menu-btn`).click();
    await page.locator('.cast-menu-item[data-action="strip"]').click();

    // its own tags are gone, the other speaker is untouched, and the chip stays
    expect((await editor(page).inputValue()).match(/\[voice:/g)).toHaveLength(1);
    await expect(editor(page)).not.toHaveValue(new RegExp(`voice:${mix}`));
    await expect(castNames(page)).toHaveCount(2);
});

test('editing a mix retunes the member without disturbing its tags', async ({ page }) => {
    await editor(page).fill('First line.');
    await toggle(page).check();
    const name = (await castNames(page).first().textContent()).trim();

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="edit"]').click();

    // the member is back in the mixer, and the button saves rather than adds
    await expect(page.locator('.selected-voice-tag')).toHaveCount(1);
    await expect(page.locator('#create-tag-btn')).toHaveText('Save mix');

    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(2).click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();

    // one member still, now a mix, and the text still points at it
    await expect(castNames(page)).toHaveCount(1);
    await expect(page.locator('.cast-member').first()).toHaveAttribute('data-mix', /\+/);
    const renamed = (await castNames(page).first().textContent()).trim();
    await expect(editor(page)).toHaveValue(`[voice:${renamed}] First line.`);
    expect(renamed).not.toBe(name);
    await expect(page.locator('#create-tag-btn')).toHaveText('Create tag');
});

test('the alias map travels with the request', async ({ page }) => {
    let body = null;
    await page.route('**/v1/audio/speech', async (route) => {
        body = JSON.parse(route.request().postData());
        await route.fulfill({
            contentType: 'audio/mpeg',
            headers: { 'X-Download-Path': '/download/test.mp3' },
            body: Buffer.from([0xff, 0xfb, 0x90, 0x64])
        });
    });

    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(1).click();
    await editor(page).click();
    await editor(page).fill('Hello there.');
    await toggle(page).check();

    const name = (await castNames(page).first().textContent()).trim();
    const mix = await page.locator('.cast-member').first().getAttribute('data-mix');

    await page.locator('#generate-btn').click();
    await expect.poll(() => body).not.toBeNull();

    expect(body.voice).toBe(name);
    expect(body.allow_voice_tags).toBe(true);
    expect(body.voice_aliases).toEqual({ [name]: mix });
});

test('a plain voice needs no alias', async ({ page }) => {
    let body = null;
    await page.route('**/v1/audio/speech', async (route) => {
        body = JSON.parse(route.request().postData());
        await route.fulfill({
            contentType: 'audio/mpeg',
            headers: { 'X-Download-Path': '/download/test.mp3' },
            body: Buffer.from([0xff, 0xfb, 0x90, 0x64])
        });
    });

    await editor(page).fill('Hello there.');
    await toggle(page).check();
    await page.locator('#generate-btn').click();
    await expect.poll(() => body).not.toBeNull();

    // the chip is the voice name itself, so there is nothing to define
    expect(body.voice_aliases).toBeUndefined();
});

test('tags left behind with the toggle off can be removed in one click', async ({ page }) => {
    await editor(page).fill('First line.\nSecond line.');
    await toggle(page).check();

    await page.locator('#voice-search').click();
    const option = page.locator('.voice-option').nth(2);
    const mix = (await option.textContent()).trim();
    await option.click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();
    await page.locator(`.cast-member[data-mix="${mix}"]`).click();
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
