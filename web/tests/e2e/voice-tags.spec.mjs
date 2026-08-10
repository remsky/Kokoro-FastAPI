import { readFile } from 'node:fs/promises';

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
const tagsTab = (page) => page.locator('#voice-tags-tab');
const voicesTab = (page) => page.locator('#voices-tab');
const castNames = (page) => page.locator('.cast-member .cast-name');

test('the tag tab is off until asked for, and seeds from the mixer', async ({ page }) => {
    await expect(voicesTab(page)).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#voice-cast')).toBeHidden();
    await expect(page.locator('#create-tag-btn')).toBeHidden();
    await expect(page.locator('#voice-tag-notice')).toBeHidden();

    const voice = (await page.locator('.selected-voice-tag .voice-name').first().textContent()).trim();
    await editor(page).fill('Hello there.');
    await tagsTab(page).click();

    await expect(editor(page)).toHaveValue(`[voice:${voice}] Hello there.`);
    // the staged mix moved into the cast, leaving the mixer free for the next voice
    await expect(castNames(page)).toHaveText([voice]);
    await expect(page.locator('.selected-voice-tag')).toHaveCount(0);
});

test('a weighted mix becomes one cast member that stands for itself', async ({ page }) => {
    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(1).click();
    await expect(page.locator('.selected-voice-tag')).toHaveCount(2);

    // fill() alone leaves the dropdown open over the tabs
    await editor(page).click();
    await editor(page).fill('Hello there.');
    await tagsTab(page).click();

    // no name is invented, so the tag is the recipe itself
    const name = (await castNames(page).first().textContent()).trim();
    expect(name).toMatch(/^\w+\(1\)\+\w+\(1\)$/);
    await expect(editor(page)).toHaveValue(`[voice:${name}] Hello there.`);
    await expect(page.locator('.cast-member').first()).toHaveAttribute('data-mix', name);
});

test('seeding happens once, not on every switch', async ({ page }) => {
    await editor(page).fill('Hello there.');
    await tagsTab(page).click();
    const seeded = await editor(page).inputValue();

    await voicesTab(page).click();
    // the default returns to the mixer, so there is still a voice to speak with
    await expect(page.locator('.selected-voice-tag')).toHaveCount(1);

    await tagsTab(page).click();
    await expect(editor(page)).toHaveValue(seeded);
    await expect(castNames(page)).toHaveCount(1);
});

test('the voice picked on the voices tab survives a peek at the tags tab', async ({ page }) => {
    // the default joins the cast first, so the restore has somewhere older to point at
    await tagsTab(page).click();
    await voicesTab(page).click();

    await page.locator('.selected-voice-tag .remove-voice').click();
    await page.locator('#voice-search').click();
    const option = page.locator('.voice-option').nth(3);
    const picked = (await option.textContent()).trim();
    await option.click();
    await editor(page).click();

    await tagsTab(page).click();
    await voicesTab(page).click();

    await expect(page.locator('.selected-voice-tag .voice-name')).toHaveText([picked]);
});

test('the mixer keeps mixing in tag mode rather than inserting on click', async ({ page }) => {
    await editor(page).fill('Hello there.');
    await tagsTab(page).click();
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
    await tagsTab(page).click();
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
    await tagsTab(page).click();
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
    await page.locator(`.cast-member[data-mix="${mix}"] .cast-insert-btn`).click();

    await expect(editor(page)).toHaveValue(seeded.replace('Second', `[voice:${mix}] Second`));
    await expect(castNames(page)).toHaveCount(2);
});

test('a cast voice is placed again and again from its chip', async ({ page }) => {
    await editor(page).fill('One. Two. Three.');
    await tagsTab(page).click();

    const chip = page.locator('.cast-member .cast-insert-btn').first();
    await editor(page).click();
    await chip.click();
    await chip.click();

    // re-inserting an existing member never grows the cast
    await expect(castNames(page)).toHaveCount(1);
    expect((await editor(page).inputValue()).match(/\[voice:/g)).toHaveLength(3);
});

test('no tag can be created until something is mixed', async ({ page }) => {
    await tagsTab(page).click();
    await expect(page.locator('#create-tag-btn')).toBeVisible();
    await expect(page.locator('#create-tag-btn')).toBeDisabled();

    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(1).click();
    await expect(page.locator('#create-tag-btn')).toBeEnabled();
});

test('a chip is placed from the keyboard through its own menu', async ({ page }) => {
    await editor(page).fill('First line.');
    await tagsTab(page).click();

    await page.locator('.cast-menu-btn').first().focus();
    await page.keyboard.press('Enter');
    // the menu sits after every chip in the DOM, so opening it has to hand focus over
    await expect(page.locator('.cast-menu-item[data-action="insert"]')).toBeFocused();

    await page.keyboard.press('Enter');
    await expect(page.locator('#cast-menu')).toBeHidden();
    expect((await editor(page).inputValue()).match(/\[voice:/g)).toHaveLength(2);

    await page.locator('.cast-menu-btn').first().focus();
    await page.keyboard.press('Enter');
    await page.keyboard.press('Escape');
    await expect(page.locator('#cast-menu')).toBeHidden();
    await expect(page.locator('.cast-menu-btn').first()).toBeFocused();
});

test('a chip leaves the cast once nothing in the text speaks with it', async ({ page }) => {
    await editor(page).fill('First line.');
    await tagsTab(page).click();
    await expect(castNames(page)).toHaveCount(1);

    // the seeded tag still answers to it, so there is nothing safe to drop yet
    await page.locator('.cast-menu-btn').first().click();
    await expect(page.locator('.cast-menu-item[data-action="remove"]')).toHaveAttribute('aria-disabled', 'true');
    await page.locator('.cast-menu-item[data-action="remove"]').click({ force: true });
    await expect(castNames(page)).toHaveCount(1);

    await page.locator('.cast-menu-item[data-action="strip"]').click();

    await page.locator('.cast-menu-btn').first().click();
    await expect(page.locator('.cast-menu-item[data-action="strip"]')).toHaveAttribute('aria-disabled', 'true');
    await page.locator('.cast-menu-item[data-action="remove"]').click();
    await expect(castNames(page)).toHaveCount(0);
    await expect(editor(page)).toHaveValue('First line.');
});

test('the cast saves as a map the API takes, and joins the cast already there on the way back', async ({ page }) => {
    await editor(page).fill('First line.');
    await tagsTab(page).click();
    const seeded = (await castNames(page).first().textContent()).trim();

    await page.locator('#voice-search').click();
    const other = page.locator('.voice-option').nth(2);
    const mix = (await other.textContent()).trim();
    await other.click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();

    await page.locator(`.cast-member[data-mix="${mix}"] .cast-menu-btn`).click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill('narrator');
    await page.locator('.cast-rename-input').press('Enter');

    await page.locator('#cast-file-menu summary').click();
    const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.locator('#save-cast-btn').click()
    ]);
    const file = await download.path();
    expect(JSON.parse(await readFile(file, 'utf8'))).toEqual({ voice_aliases: { [seeded]: seeded, narrator: mix } });

    await page.reload();
    await expect(page.locator('.selected-voice-tag').first()).toBeVisible({ timeout: 15_000 });
    await tagsTab(page).click();
    await expect(castNames(page)).toHaveText([seeded]);

    await page.locator('#import-cast-input').setInputFiles(file);
    await expect(castNames(page)).toHaveText([seeded, 'narrator']);
    await expect(page.locator('#status')).toHaveText('Added 1 to the cast, skipped 1');
});

test('import (replace) swaps the whole cast for the file, tags in the text and all', async ({ page }) => {
    await editor(page).fill('First line.');
    await tagsTab(page).click();

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill('hero');
    await page.locator('.cast-rename-input').press('Enter');

    // one member tagged in the text and one without, both swapped out alike
    await page.locator('#voice-search').click();
    const other = page.locator('.voice-option').nth(2);
    const mix = (await other.textContent()).trim();
    await other.click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();
    await expect(castNames(page)).toHaveText(['hero', mix]);

    const chooser = page.waitForEvent('filechooser');
    await page.locator('#cast-file-menu summary').click();
    await page.locator('#import-cast-replace-btn').click();
    await (await chooser).setFiles({
        name: 'cast.json',
        mimeType: 'application/json',
        buffer: Buffer.from(JSON.stringify({ voice_aliases: { narrator: mix } }))
    });

    await expect(castNames(page)).toHaveText(['narrator']);
    await expect(page.locator('#status')).toHaveText('Cast replaced with 1, 1 tag in the text cannot speak');
    // the text is not the import's to rewrite
    await expect(editor(page)).toHaveValue('[voice:hero] First line.');
});

test('a cast member can be renamed, and the tags follow', async ({ page }) => {
    await editor(page).fill('First line. Second line.');
    await tagsTab(page).click();

    const chip = page.locator('.cast-member .cast-insert-btn').first();
    await editor(page).click();
    await chip.click(); // a second tag, so the rename has more than one to follow

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill('narrator');
    await page.locator('.cast-rename-input').press('Enter');

    await expect(castNames(page)).toHaveText(['narrator']);
    expect((await editor(page).inputValue()).match(/\[voice:narrator\]/g)).toHaveLength(2);
});

test('resetting an alias hands its tags back to the mix, chip and all', async ({ page }) => {
    await editor(page).fill('First line.');
    await tagsTab(page).click();

    const mix = await page.locator('.cast-member').first().getAttribute('data-mix');
    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill('narrator');
    await page.locator('.cast-rename-input').press('Enter');
    await expect(editor(page)).toHaveValue(`[voice:narrator] First line.`);

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="reset"]').click();

    // the definition goes with the name, so a tag still naming it would reach the server undefined
    await expect(castNames(page)).toHaveText([mix]);
    await expect(editor(page)).toHaveValue(`[voice:${mix}] First line.`);

    await page.locator('.cast-menu-btn').first().click();
    await expect(page.locator('.cast-menu-item[data-action="reset"]')).toHaveAttribute('aria-disabled', 'true');
});

test('a rename opened over another one lands the first rather than replacing it', async ({ page }) => {
    await tagsTab(page).click();
    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(2).click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();
    await expect(castNames(page)).toHaveCount(2);

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill('narrator');

    await page.locator('.cast-menu-btn').nth(1).click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await expect(page.locator('.cast-rename-input')).toBeVisible();
    await page.locator('.cast-rename-input').fill('villain');
    await page.locator('.cast-rename-input').press('Enter');

    await expect(castNames(page)).toHaveText(['narrator', 'villain']);
});

test('a name that would shadow a real voice is refused', async ({ page }) => {
    await tagsTab(page).click();
    const taken = (await page.locator('.voice-option').nth(2).textContent()).trim();
    const before = (await castNames(page).first().textContent()).trim();

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill(taken);
    await page.locator('.cast-rename-input').press('Enter');

    await expect(page.locator('#status')).toHaveText(`"${taken}" is already taken`);
    await expect(castNames(page)).toHaveText([before]);
});

test('a case variant of a taken name is refused, since the server folds aliases', async ({ page }) => {
    await tagsTab(page).click();
    const taken = (await page.locator('.voice-option').nth(2).textContent()).trim();
    const variant = taken.charAt(0).toUpperCase() + taken.slice(1);
    const before = (await castNames(page).first().textContent()).trim();

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill(variant);
    await page.locator('.cast-rename-input').press('Enter');

    await expect(page.locator('#status')).toHaveText(`"${variant}" is already taken`);
    await expect(castNames(page)).toHaveText([before]);
});

test('one speaker can be cleared out of the text from its own menu', async ({ page }) => {
    await editor(page).fill('First line.\nSecond line.');
    await tagsTab(page).click();

    await page.locator('#voice-search').click();
    const other = page.locator('.voice-option').nth(2);
    const mix = (await other.textContent()).trim();
    await other.click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();
    await page.locator(`.cast-member[data-mix="${mix}"] .cast-insert-btn`).click();
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
    await tagsTab(page).click();
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

test('editing a mix onto an existing member merges the chips rather than twinning them', async ({ page }) => {
    await editor(page).fill('First line.');
    await tagsTab(page).click();

    await page.locator('#voice-search').click();
    const option = page.locator('.voice-option').nth(2);
    const other = (await option.textContent()).trim();
    await option.click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();
    await expect(castNames(page)).toHaveCount(2);

    // retune the first member to exactly the second member's mix
    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="edit"]').click();
    await page.locator('.selected-voice-tag .remove-voice').click();
    await page.locator('#voice-search').click();
    await page.locator('.voice-option').nth(2).click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();

    await expect(castNames(page)).toHaveText([other]);
    await expect(editor(page)).toHaveValue(`[voice:${other}] First line.`);
});

test('an import full of malformed mixes reports instead of keeping them', async ({ page }) => {
    await tagsTab(page).click();
    await expect(castNames(page)).toHaveCount(1);

    // empty +-parts survive parsing, so these would 400 at generation if let in
    await page.locator('#import-cast-input').setInputFiles({
        name: 'cast.json',
        mimeType: 'application/json',
        buffer: Buffer.from(JSON.stringify({ voice_aliases: { 'af_bella+': 'af_bella+', villain: 'af_bella++af_sky' } }))
    });

    await expect(page.locator('#status')).toHaveText('Nothing in that file could join the cast');
    await expect(castNames(page)).toHaveCount(1);
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
    await tagsTab(page).click();

    // only a rename makes a name that has to be defined for the server
    const mix = await page.locator('.cast-member').first().getAttribute('data-mix');
    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill('narrator');
    await page.locator('.cast-rename-input').press('Enter');

    await page.locator('#generate-btn').click();
    await expect.poll(() => body).not.toBeNull();

    expect(body.voice).toBe('narrator');
    expect(body.allow_voice_tags).toBe(true);
    expect(body.voice_aliases).toEqual({ narrator: mix });
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
    await tagsTab(page).click();
    await page.locator('#generate-btn').click();
    await expect.poll(() => body).not.toBeNull();

    // the chip is the voice name itself, so there is nothing to define
    expect(body.voice_aliases).toBeUndefined();
});

test('tags left behind on the voices tab can be removed in one click', async ({ page }) => {
    await editor(page).fill('First line.\nSecond line.');
    await tagsTab(page).click();

    await page.locator('#voice-search').click();
    const option = page.locator('.voice-option').nth(2);
    const mix = (await option.textContent()).trim();
    await option.click();
    await editor(page).click();
    await page.locator('#create-tag-btn').click();
    await page.locator(`.cast-member[data-mix="${mix}"] .cast-insert-btn`).click();
    await expect(page.locator('#voice-tag-notice')).toBeHidden();

    await voicesTab(page).click();
    await expect(page.locator('#voice-tag-notice')).toBeVisible();
    await expect(page.locator('#voice-tag-notice-text')).toHaveText('2 voice tags will be read aloud.');

    await page.locator('#remove-voice-tags-btn').click();
    await expect(page.locator('#voice-tag-notice')).toBeHidden();
    await expect(editor(page)).toHaveValue('First line.\nSecond line.');
});

test('pasted tags are noticed without the tabs ever being touched', async ({ page }) => {
    await editor(page).fill('[voice:af_bella] Hello there.');

    await expect(page.locator('#voice-tag-notice')).toBeVisible();
    await expect(page.locator('#voice-tag-notice-text')).toHaveText('1 voice tag will be read aloud.');
});

test('text that does not open with a tag has nothing to speak with', async ({ page }) => {
    await editor(page).fill('Hello there.');
    await tagsTab(page).click();
    await expect(editor(page)).toHaveValue(/^\[voice:/);

    // the seeded tag is taken back out, and the cast member it came from does not stand in for it
    await editor(page).fill('Hello there.');
    await page.locator('#generate-btn').click();
    await expect(page.locator('#status')).toHaveText('Start the text with a voice tag');

    await editor(page).evaluate((el) => {
        el.focus();
        el.setSelectionRange(0, 0);
    });
    await page.locator('.cast-member .cast-insert-btn').first().click();
    await expect(editor(page)).toHaveValue(/^\[voice:/);
});

test('a fat-fingered insert is undone', async ({ page }) => {
    await editor(page).fill('First line.');
    await tagsTab(page).click();

    const before = await editor(page).inputValue();
    await editor(page).evaluate((el) => {
        el.focus();
        el.setSelectionRange(el.value.length, el.value.length);
    });
    await page.locator('.cast-member .cast-insert-btn').first().click();
    expect(await editor(page).inputValue()).not.toBe(before);

    await page.locator('#undo-insert-btn').click();
    await expect(editor(page)).toHaveValue(before);
    await expect(page.locator('#undo-insert-btn')).toBeHidden();
});

test('undo follows a rename', async ({ page }) => {
    await editor(page).fill('First line.');
    await tagsTab(page).click();

    await editor(page).evaluate((el) => {
        el.focus();
        el.setSelectionRange(el.value.length, el.value.length);
    });
    await page.locator('.cast-member .cast-insert-btn').first().click();

    await page.locator('.cast-menu-btn').first().click();
    await page.locator('.cast-menu-item[data-action="rename"]').click();
    await page.locator('.cast-rename-input').fill('narrator');
    await page.locator('.cast-rename-input').press('Enter');

    // the tracked insert followed the rename, so undo lifts the renamed tag back out
    await page.locator('#undo-insert-btn').click();
    await expect(editor(page)).toHaveValue('[voice:narrator] First line.');
});

test('a tag on its own is not enough to generate', async ({ page }) => {
    await editor(page).fill('');
    await tagsTab(page).click();

    await page.locator('#generate-btn').click();
    await expect(page.locator('#status')).toHaveText('Please enter some text');
});
