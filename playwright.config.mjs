import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './web/tests/e2e',
    timeout: 30_000,
    use: {
        baseURL: 'http://127.0.0.1:4173',
    },
    webServer: {
        command: 'node web/tests/e2e/fixtures/static-server.mjs',
        url: 'http://127.0.0.1:4173',
        reuseExistingServer: true,
        timeout: 10_000,
    },
});
