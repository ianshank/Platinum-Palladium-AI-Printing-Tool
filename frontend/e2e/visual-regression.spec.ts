import { test, expect } from '@playwright/test';

test.describe('Visual Regression', () => {
  const pages = [
    { path: '/', name: 'dashboard' },
    { path: '/calibration', name: 'calibration' },
    { path: '/curves', name: 'curves' },
    { path: '/chemistry', name: 'chemistry' },
    { path: '/assistant', name: 'assistant' },
    { path: '/session', name: 'session-log' },
    { path: '/settings', name: 'settings' },
  ];

  for (const { path, name } of pages) {
    test(`${name} page matches visual baseline`, async ({ page }) => {
      await page.goto(path);
      await page.waitForLoadState('networkidle', { timeout: 10000 });

      // Wait for any animations to complete
      await page.waitForTimeout(500);

      await expect(page).toHaveScreenshot(`${name}.png`, {
        fullPage: true,
        maxDiffPixels: 100,
        threshold: 0.2,
      });
    });
  }

  test.describe('Responsive Layouts', () => {
    const viewports = [
      { width: 375, height: 667, name: 'mobile' }, // iPhone SE
      { width: 768, height: 1024, name: 'tablet' }, // iPad
      { width: 1920, height: 1080, name: 'desktop' }, // Full HD
    ];

    for (const viewport of viewports) {
      test(`dashboard matches ${viewport.name} baseline`, async ({ page }) => {
        await page.setViewportSize({ width: viewport.width, height: viewport.height });
        await page.goto('/');
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(500);

        await expect(page).toHaveScreenshot(`dashboard-${viewport.name}.png`, {
          fullPage: true,
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      });
    }

    for (const viewport of viewports) {
      test(`calibration page matches ${viewport.name} baseline`, async ({ page }) => {
        await page.setViewportSize({ width: viewport.width, height: viewport.height });
        await page.goto('/calibration');
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(500);

        await expect(page).toHaveScreenshot(`calibration-${viewport.name}.png`, {
          fullPage: true,
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      });
    }
  });

  test.describe('Component States', () => {
    test('navigation menu expanded state', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      // Try to find and click menu toggle
      const menuToggle = page.locator(
        'button[aria-label*="menu" i], button[aria-label*="navigation" i], button:has-text("Menu")'
      ).first();

      if (await menuToggle.isVisible({ timeout: 2000 }).catch(() => false)) {
        await menuToggle.click();
        await page.waitForTimeout(500);

        await expect(page).toHaveScreenshot('navigation-expanded.png', {
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      }
    });

    test('curve editor with controls visible', async ({ page }) => {
      await page.goto('/curves');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(500);

      // Capture curve editor state
      const curveEditor = page.locator('[data-testid="curve-editor"], [data-testid="curve-display"]').first();

      if (await curveEditor.isVisible({ timeout: 5000 }).catch(() => false)) {
        await expect(curveEditor).toHaveScreenshot('curve-editor-default.png', {
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      }
    });

    test('settings page with all sections visible', async ({ page }) => {
      await page.goto('/settings');
      await page.waitForLoadState('networkidle');

      // Expand all accordions if any
      const accordionTriggers = page.locator('[data-state="closed"]');
      const triggerCount = await accordionTriggers.count();

      for (let i = 0; i < triggerCount; i++) {
        const trigger = accordionTriggers.nth(i);
        if (await trigger.isVisible({ timeout: 1000 }).catch(() => false)) {
          await trigger.click();
          await page.waitForTimeout(300);
        }
      }

      await expect(page).toHaveScreenshot('settings-expanded.png', {
        fullPage: true,
        maxDiffPixels: 100,
        threshold: 0.2,
      });
    });

    test('chat interface empty state', async ({ page }) => {
      await page.goto('/assistant');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(500);

      await expect(page).toHaveScreenshot('chat-empty.png', {
        fullPage: true,
        maxDiffPixels: 100,
        threshold: 0.2,
      });
    });

    test('session log empty or populated state', async ({ page }) => {
      await page.goto('/session');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(500);

      await expect(page).toHaveScreenshot('session-log-state.png', {
        fullPage: true,
        maxDiffPixels: 100,
        threshold: 0.2,
      });
    });
  });

  test.describe('Dark Mode Support', () => {
    test('dashboard in dark mode', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      // Try to enable dark mode
      const darkModeToggle = page.locator(
        'button[aria-label*="dark" i], button[aria-label*="theme" i], [data-testid="theme-toggle"]'
      ).first();

      if (await darkModeToggle.isVisible({ timeout: 2000 }).catch(() => false)) {
        // Check current theme
        const isDarkMode = await page.evaluate(() => {
          return document.documentElement.classList.contains('dark');
        });

        // Toggle to dark mode if not already
        if (!isDarkMode) {
          await darkModeToggle.click();
          await page.waitForTimeout(500);
        }

        await expect(page).toHaveScreenshot('dashboard-dark.png', {
          fullPage: true,
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      } else {
        // Try setting dark mode via system preference
        await page.emulateMedia({ colorScheme: 'dark' });
        await page.waitForTimeout(500);

        await expect(page).toHaveScreenshot('dashboard-dark-system.png', {
          fullPage: true,
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      }
    });
  });

  test.describe('Interaction States', () => {
    test('button hover state', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      const firstButton = page.locator('button').first();

      if (await firstButton.isVisible({ timeout: 5000 }).catch(() => false)) {
        await firstButton.hover();
        await page.waitForTimeout(200);

        await expect(firstButton).toHaveScreenshot('button-hover.png', {
          maxDiffPixels: 50,
          threshold: 0.2,
        });
      }
    });

    test('input focus state', async ({ page }) => {
      await page.goto('/settings');
      await page.waitForLoadState('networkidle');

      const firstInput = page.locator('input[type="text"], input[type="number"]').first();

      if (await firstInput.isVisible({ timeout: 5000 }).catch(() => false)) {
        await firstInput.focus();
        await page.waitForTimeout(200);

        await expect(firstInput).toHaveScreenshot('input-focus.png', {
          maxDiffPixels: 50,
          threshold: 0.2,
        });
      }
    });

    test('dropdown open state', async ({ page }) => {
      await page.goto('/settings');
      await page.waitForLoadState('networkidle');

      // Look for select elements or custom dropdowns
      const dropdown = page.locator('select, [role="combobox"]').first();

      if (await dropdown.isVisible({ timeout: 5000 }).catch(() => false)) {
        await dropdown.click();
        await page.waitForTimeout(300);

        await expect(page).toHaveScreenshot('dropdown-open.png', {
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      }
    });
  });

  test.describe('Loading States', () => {
    test('page loading state', async ({ page }) => {
      // Navigate but don't wait for full load
      await page.goto('/curves', { waitUntil: 'domcontentloaded' });

      // Look for loading indicators
      const loadingIndicator = page.locator(
        '[data-testid="loading"], [role="progressbar"], [aria-busy="true"]'
      ).first();

      if (await loadingIndicator.isVisible({ timeout: 1000 }).catch(() => false)) {
        await expect(loadingIndicator).toHaveScreenshot('loading-indicator.png', {
          maxDiffPixels: 50,
          threshold: 0.2,
        });
      }
    });
  });

  test.describe('Error States', () => {
    test('404 page visual baseline', async ({ page }) => {
      await page.goto('/nonexistent-page');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(500);

      await expect(page).toHaveScreenshot('404-page.png', {
        fullPage: true,
        maxDiffPixels: 100,
        threshold: 0.2,
      });
    });
  });

  test.describe('Print Styles', () => {
    test('dashboard print layout', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      await page.emulateMedia({ media: 'print' });
      await page.waitForTimeout(500);

      await expect(page).toHaveScreenshot('dashboard-print.png', {
        fullPage: true,
        maxDiffPixels: 100,
        threshold: 0.2,
      });
    });
  });

  test.describe('Browser-Specific Rendering', () => {
    test('cross-browser visual consistency check', async ({ page, browserName }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(500);

      // Use browser name in screenshot filename for comparison
      await expect(page).toHaveScreenshot(`dashboard-${browserName}.png`, {
        fullPage: true,
        maxDiffPixels: 150, // Allow slightly more variance for cross-browser
        threshold: 0.25,
      });
    });
  });

  test.describe('Animation Snapshots', () => {
    test('page transition visual check', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      // Navigate to another page
      const navLink = page.locator('a[href="/calibration"]').first();

      if (await navLink.isVisible({ timeout: 5000 }).catch(() => false)) {
        await navLink.click();

        // Wait for transition to complete
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(500);

        await expect(page).toHaveScreenshot('page-transition-complete.png', {
          fullPage: true,
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      }
    });
  });
});
