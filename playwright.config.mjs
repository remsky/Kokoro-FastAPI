import { defineConfig } from '@playwright/test';

const port = Number(process.env.PLAYWRIGHT_STATIC_PORT || 4173);

export default defineConfig({
    testDir: './web/tests/e2e',
    timeout: 30_000,
    use: {
        baseURL: `http://127.0.0.1:${port}`,
    },
    webServer: {
        command: 'node web/tests/e2e/fixtures/static-server.mjs',
        url: `http://127.0.0.1:${port}`,
        reuseExistingServer: true,
        timeout: 10_000,
    },
});
