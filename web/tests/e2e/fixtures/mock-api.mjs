const VOICES = ['af_bella', 'af_heart', 'am_adam', 'am_michael', 'bf_emma', 'bm_george'];

export async function mockApi(page) {
    await page.route('**/web/config', (route) => route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ root_path: '', version: 'test' }),
    }));

    await page.route('**/v1/audio/voices', (route) => route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ voices: VOICES.map((id) => ({ id, name: id })) }),
    }));

    await page.route('https://api.github.com/**', (route) => route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ stargazers_count: 1234 }),
    }));
}
