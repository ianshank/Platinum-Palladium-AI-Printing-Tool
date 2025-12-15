/**
 * Playwright configuration for E2E testing.
 */

import { defineConfig, devices } from '@playwright/test';

/**
 * Read environment variables from .env file.
 */
const baseURL = process.env.BASE_URL || 'http://localhost:5173';
const apiURL = process.env.API_URL || 'http://localhost:8000';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { outputFolder: '../../../test-results/e2e/html' }],
    ['json', { outputFile: '../../../test-results/e2e/results.json' }],
    ['list'],
  ],
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'on-first-retry',
    actionTimeout: 10000,
    navigationTimeout: 30000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
  ],
  webServer: [
    {
      command: 'npm run dev',
      cwd: '../../../frontend',
      url: baseURL,
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
    {
      command: 'uvicorn src.ptpd_calibration.api.server:app --reload --port 8000',
      cwd: '../../..',
      url: `${apiURL}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
  ],
  expect: {
    timeout: 5000,
  },
  timeout: 60000,
  outputDir: '../../../test-results/e2e/artifacts',
});
