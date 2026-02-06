/**
 * Navigation E2E tests.
 */

import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('redirects to dashboard from root', async ({ page }) => {
    await page.waitForURL(/\/(dashboard)?$/);
    // Should either be on root or dashboard
    const url = page.url();
    expect(url.match(/\/(dashboard)?$/)).toBeTruthy();
  });

  test('displays sidebar navigation', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page.locator('nav')).toBeVisible();
  });

  test('sidebar contains all main navigation links', async ({ page }) => {
    await page.goto('/dashboard');

    const navLinks = [
      { text: /Dashboard/i, href: /dashboard/ },
      { text: /Calibration/i, href: /calibration/ },
      { text: /Chemistry/i, href: /chemistry/ },
      { text: /Assistant/i, href: /assistant/ },
      { text: /Sessions/i, href: /sessions/ },
      { text: /Settings/i, href: /settings/ },
    ];

    for (const link of navLinks) {
      const navLink = page.locator('nav a').filter({ hasText: link.text });
      await expect(navLink).toBeVisible();
    }
  });

  test('navigates to calibration page', async ({ page }) => {
    await page.goto('/dashboard');
    await page.locator('nav a').filter({ hasText: /Calibration/i }).click();
    await expect(page).toHaveURL(/\/calibration/);
    await expect(page.locator('h1')).toContainText(/Calibration/);
  });

  test('navigates to chemistry page', async ({ page }) => {
    await page.goto('/dashboard');
    await page.locator('nav a').filter({ hasText: /Chemistry/i }).click();
    await expect(page).toHaveURL(/\/chemistry/);
    await expect(page.locator('h1')).toContainText(/Chemistry/);
  });

  test('navigates to AI assistant page', async ({ page }) => {
    await page.goto('/dashboard');
    await page.locator('nav a').filter({ hasText: /Assistant/i }).click();
    await expect(page).toHaveURL(/\/assistant/);
    await expect(page.locator('h1')).toContainText(/AI Assistant/);
  });

  test('navigates to sessions page', async ({ page }) => {
    await page.goto('/dashboard');
    await page.locator('nav a').filter({ hasText: /Sessions/i }).click();
    await expect(page).toHaveURL(/\/sessions/);
    await expect(page.locator('h1')).toContainText(/Session/);
  });

  test('navigates to settings page', async ({ page }) => {
    await page.goto('/dashboard');
    await page.locator('nav a').filter({ hasText: /Settings/i }).click();
    await expect(page).toHaveURL(/\/settings/);
    await expect(page.locator('h1')).toContainText(/Settings/);
  });

  test('displays 404 page for unknown routes', async ({ page }) => {
    await page.goto('/unknown-page-that-does-not-exist');
    await expect(page.locator('text=404')).toBeVisible();
    await expect(page.locator('text=Page Not Found')).toBeVisible();
  });

  test('404 page has link back to dashboard', async ({ page }) => {
    await page.goto('/unknown-page-that-does-not-exist');
    const dashboardLink = page.locator('a').filter({ hasText: /Dashboard/i });
    await expect(dashboardLink).toBeVisible();

    await dashboardLink.click();
    await expect(page).toHaveURL(/\/dashboard/);
  });

  test('header displays app logo/title', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page.locator('header')).toBeVisible();
    await expect(page.locator('text=PTPD').or(page.locator('text=Calibration'))).toBeVisible();
  });

  test('sidebar can be collapsed', async ({ page }) => {
    await page.goto('/dashboard');

    // Look for collapse button
    const collapseButton = page.locator('button[aria-label*="collapse"], button[aria-label*="toggle"]').first();

    if (await collapseButton.isVisible()) {
      await collapseButton.click();
      await page.waitForTimeout(300);

      // Sidebar should be narrower or icons-only
      const sidebar = page.locator('nav, aside').first();
      const width = await sidebar.evaluate((el) => el.getBoundingClientRect().width);
      expect(width).toBeLessThan(200);
    }
  });

  test('maintains navigation state on page refresh', async ({ page }) => {
    await page.goto('/chemistry');
    await expect(page).toHaveURL(/\/chemistry/);

    await page.reload();
    await expect(page).toHaveURL(/\/chemistry/);
    await expect(page.locator('h1')).toContainText(/Chemistry/);
  });

  test('browser back button works correctly', async ({ page }) => {
    await page.goto('/dashboard');
    await page.locator('nav a').filter({ hasText: /Chemistry/i }).click();
    await expect(page).toHaveURL(/\/chemistry/);

    await page.goBack();
    await expect(page).toHaveURL(/\/dashboard/);
  });
});
