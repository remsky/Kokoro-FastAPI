import { test as base } from '@playwright/test';

import { mockApi } from './mock-api.mjs';

export { expect } from '@playwright/test';

export const test = base.extend({
    page: async ({ page }, use) => {
        await mockApi(page);
        await use(page);
    },
});
