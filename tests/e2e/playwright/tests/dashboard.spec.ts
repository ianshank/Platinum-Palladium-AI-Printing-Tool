/**
 * Dashboard E2E tests.
 */

import { test, expect } from '@playwright/test';
import { DashboardPage } from '../pages';

test.describe('Dashboard', () => {
  let dashboardPage: DashboardPage;

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page);
    await dashboardPage.goto();
  });

  test('loads successfully', async () => {
    const isLoaded = await dashboardPage.isLoaded();
    expect(isLoaded).toBe(true);
  });

  test('displays page title', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Dashboard');
  });

  test('displays welcome message', async ({ page }) => {
    await expect(page.locator('text=Welcome')).toBeVisible();
  });

  test('displays all stat cards', async ({ page }) => {
    await expect(page.locator('text=Total Prints')).toBeVisible();
    await expect(page.locator('text=Active Curves')).toBeVisible();
    await expect(page.locator('text=This Month')).toBeVisible();
    await expect(page.locator('text=Avg Rating')).toBeVisible();
  });

  test('displays quick actions section', async () => {
    await dashboardPage.verifyQuickActionsVisible();
  });

  test('displays getting started tips', async () => {
    await dashboardPage.verifyGettingStartedVisible();
  });

  test('navigates to calibration from quick actions', async ({ page }) => {
    await dashboardPage.startNewCalibration();
    await expect(page).toHaveURL(/\/calibration/);
  });

  test('navigates to chemistry from quick actions', async ({ page }) => {
    await dashboardPage.openChemistryCalculator();
    await expect(page).toHaveURL(/\/chemistry/);
  });

  test('navigates to AI assistant from quick actions', async ({ page }) => {
    await dashboardPage.openAIAssistant();
    await expect(page).toHaveURL(/\/assistant/);
  });

  test('navigates to session log from quick actions', async ({ page }) => {
    await dashboardPage.openSessionLog();
    await expect(page).toHaveURL(/\/sessions/);
  });

  test('is responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await dashboardPage.goto();

    const isLoaded = await dashboardPage.isLoaded();
    expect(isLoaded).toBe(true);

    // Quick actions should still be visible
    await expect(page.locator('text=New Calibration')).toBeVisible();
  });
});
