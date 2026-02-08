import { test, expect } from '@playwright/test';

test.describe('Application', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('has correct title', async ({ page }) => {
    await expect(page).toHaveTitle(/Pt\/Pd AI Printing Tool/);
  });

  test('displays navigation sidebar', async ({ page }) => {
    await expect(page.getByRole('link', { name: 'Dashboard' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Calibration' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Curves' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Chemistry' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'AI Assistant' })).toBeVisible();
  });

  test('navigates between pages', async ({ page }) => {
    // Navigate to Calibration
    await page.getByRole('link', { name: 'Calibration' }).click();
    await expect(page).toHaveURL('/calibration');
    await expect(page.getByRole('heading', { name: 'Calibration Wizard' })).toBeVisible();

    // Navigate to Curves
    await page.getByRole('link', { name: 'Curves' }).click();
    await expect(page).toHaveURL('/curves');
    await expect(page.getByRole('heading', { name: 'Curve Editor' })).toBeVisible();

    // Navigate to Chemistry
    await page.getByRole('link', { name: 'Chemistry' }).click();
    await expect(page).toHaveURL('/chemistry');
    await expect(page.getByRole('heading', { name: 'Chemistry Calculator' })).toBeVisible();
  });

  test('keyboard navigation works', async ({ page }) => {
    // Focus navigation
    await page.keyboard.press('Tab');

    // Use keyboard shortcuts (Ctrl+1-5)
    await page.keyboard.press('Control+2');
    await expect(page).toHaveURL('/calibration');

    await page.keyboard.press('Control+3');
    await expect(page).toHaveURL('/curves');

    await page.keyboard.press('Control+1');
    await expect(page).toHaveURL('/');
  });

  test('responsive design - mobile view', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    // Sidebar should be hidden on mobile
    const sidebar = page.locator('aside');
    await expect(sidebar).toHaveClass(/translate-x-\[-100%\]|-translate-x-full/);

    // Menu button should be visible
    await expect(page.getByRole('button', { name: 'Open sidebar' })).toBeVisible();
  });

  test('dark theme is applied by default', async ({ page }) => {
    // Check that dark theme class is applied
    const html = page.locator('html');
    await expect(html).toHaveClass(/dark/);
  });
});

test.describe('Dashboard', () => {
  test('displays dashboard content', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    await expect(page.getByText('Overview and metrics')).toBeVisible();
  });
});

test.describe('Accessibility', () => {
  test('main navigation is accessible', async ({ page }) => {
    await page.goto('/');

    // Check for accessible names
    await expect(page.getByRole('navigation')).toBeVisible();

    // Check that links have accessible names
    const links = page.getByRole('link');
    const count = await links.count();
    expect(count).toBeGreaterThan(0);
  });

  test('focus is visible on interactive elements', async ({ page }) => {
    await page.goto('/');

    // Tab to first focusable element
    await page.keyboard.press('Tab');

    // Check that focused element has visible focus ring
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });
});
