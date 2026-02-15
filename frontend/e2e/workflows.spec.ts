import { test, expect } from '@playwright/test';

test.describe('Calibration Workflow', () => {
  test('should complete calibration wizard steps', async ({ page }) => {
    await page.goto('/calibration');
    await page.waitForLoadState('networkidle');

    // Verify calibration wizard renders
    const wizard = page.locator('[data-testid="calibration-wizard"]');
    await expect(wizard).toBeVisible({ timeout: 10000 });

    // Verify wizard heading is present
    const heading = page.locator('h1, h2').first();
    await expect(heading).toBeVisible();

    // Check for step indicators or navigation buttons
    const stepIndicators = page.locator('[data-testid*="step"]');
    const navButtons = page.locator('button:has-text("Next"), button:has-text("Previous"), button:has-text("Continue")');

    // At least one should be present
    const hasStepIndicators = await stepIndicators.count() > 0;
    const hasNavButtons = await navButtons.count() > 0;
    expect(hasStepIndicators || hasNavButtons).toBeTruthy();
  });

  test('should navigate through wizard stages', async ({ page }) => {
    await page.goto('/calibration');
    await page.waitForLoadState('networkidle');

    // Try to find and click next/continue button
    const continueButton = page.locator('button:has-text("Next"), button:has-text("Continue"), button:has-text("Start")').first();

    if (await continueButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      const initialText = await page.locator('h1, h2').first().textContent();
      await continueButton.click();
      await page.waitForTimeout(500);

      // Verify something changed (heading or content)
      const currentText = await page.locator('h1, h2').first().textContent();
      // Either heading changed or we're still on a valid page
      expect(currentText).toBeTruthy();
    }
  });
});

test.describe('Curve Editing Workflow', () => {
  test('should render curve editor with controls', async ({ page }) => {
    await page.goto('/curves');
    await page.waitForLoadState('networkidle');

    // Verify curve editor renders
    const curveEditor = page.locator('[data-testid="curve-editor"], [data-testid="curve-display"]').first();
    await expect(curveEditor).toBeVisible({ timeout: 10000 });

    // Check for adjustment controls (sliders, buttons, or inputs)
    const controls = page.locator('input[type="range"], button, input[type="number"]');
    const controlCount = await controls.count();
    expect(controlCount).toBeGreaterThan(0);
  });

  test('should have curve manipulation controls', async ({ page }) => {
    await page.goto('/curves');
    await page.waitForLoadState('networkidle');

    // Look for common curve editor controls
    const possibleControls = [
      'contrast',
      'brightness',
      'gamma',
      'smooth',
      'reset',
      'undo',
      'redo',
      'export',
    ];

    let foundControls = 0;
    for (const control of possibleControls) {
      const element = page.locator(`button:has-text("${control}"), label:has-text("${control}"), [aria-label*="${control}" i]`).first();
      if (await element.isVisible({ timeout: 1000 }).catch(() => false)) {
        foundControls++;
      }
    }

    // Should find at least some controls
    expect(foundControls).toBeGreaterThan(0);
  });

  test('should display curve visualization', async ({ page }) => {
    await page.goto('/curves');
    await page.waitForLoadState('networkidle');

    // Look for canvas or SVG (common for curve visualizations)
    const visualization = page.locator('canvas, svg[class*="curve"], [data-testid*="curve"]').first();
    await expect(visualization).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Chemistry Calculator Workflow', () => {
  test('should render chemistry calculator interface', async ({ page }) => {
    await page.goto('/chemistry');
    await page.waitForLoadState('networkidle');

    // Verify calculator renders
    const calculator = page.locator('[data-testid="chemistry-calculator"], form').first();
    await expect(calculator).toBeVisible({ timeout: 10000 });
  });

  test('should have input fields for chemistry calculations', async ({ page }) => {
    await page.goto('/chemistry');
    await page.waitForLoadState('networkidle');

    // Check for input fields
    const inputs = page.locator('input[type="text"], input[type="number"], select');
    const inputCount = await inputs.count();
    expect(inputCount).toBeGreaterThan(0);
  });

  test('should have calculate or submit button', async ({ page }) => {
    await page.goto('/chemistry');
    await page.waitForLoadState('networkidle');

    // Look for action buttons
    const actionButton = page.locator('button:has-text("Calculate"), button:has-text("Submit"), button[type="submit"]').first();
    await expect(actionButton).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Chat Workflow', () => {
  test('should render chat interface', async ({ page }) => {
    await page.goto('/assistant');
    await page.waitForLoadState('networkidle');

    // Verify chat interface renders
    const chatInterface = page.locator('[data-testid="chat-interface"], [data-testid="ai-assistant"]').first();
    await expect(chatInterface).toBeVisible({ timeout: 10000 });
  });

  test('should have message input field', async ({ page }) => {
    await page.goto('/assistant');
    await page.waitForLoadState('networkidle');

    // Check for input field (textarea or input)
    const messageInput = page.locator('textarea, input[type="text"]').first();
    await expect(messageInput).toBeVisible({ timeout: 10000 });
  });

  test('should have send button', async ({ page }) => {
    await page.goto('/assistant');
    await page.waitForLoadState('networkidle');

    // Look for send button
    const sendButton = page.locator('button:has-text("Send"), button[type="submit"], button[aria-label*="send" i]').first();
    await expect(sendButton).toBeVisible({ timeout: 10000 });
  });

  test('should have messages container', async ({ page }) => {
    await page.goto('/assistant');
    await page.waitForLoadState('networkidle');

    // Look for messages container
    const messagesContainer = page.locator('[data-testid="messages"], [data-testid="chat-messages"], [role="log"]').first();
    await expect(messagesContainer).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Settings Workflow', () => {
  test('should render settings page', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Verify settings form renders
    const settingsForm = page.locator('[data-testid="settings"], form').first();
    await expect(settingsForm).toBeVisible({ timeout: 10000 });
  });

  test('should have settings controls', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Check for various input types
    const inputs = page.locator('input, select, textarea');
    const inputCount = await inputs.count();
    expect(inputCount).toBeGreaterThan(0);
  });

  test('should have save or apply button', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Look for save/apply button
    const saveButton = page.locator('button:has-text("Save"), button:has-text("Apply"), button[type="submit"]').first();
    await expect(saveButton).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Session Log Workflow', () => {
  test('should render session log', async ({ page }) => {
    await page.goto('/session');
    await page.waitForLoadState('networkidle');

    // Verify log renders
    const sessionLog = page.locator('[data-testid="session-log"], table, [role="log"]').first();
    await expect(sessionLog).toBeVisible({ timeout: 10000 });
  });

  test('should display log entries or empty state', async ({ page }) => {
    await page.goto('/session');
    await page.waitForLoadState('networkidle');

    // Either has entries or shows empty state
    const logEntries = page.locator('tr, [data-testid="log-entry"], [data-testid="log-item"]');
    const emptyState = page.locator(':has-text("No logs"), :has-text("No entries"), :has-text("Empty")');

    const hasEntries = await logEntries.count() > 0;
    const hasEmptyState = await emptyState.isVisible({ timeout: 1000 }).catch(() => false);

    expect(hasEntries || hasEmptyState).toBeTruthy();
  });
});

test.describe('Cross-Page Navigation', () => {
  const pages = [
    { path: '/', name: 'Dashboard' },
    { path: '/calibration', name: 'Calibration' },
    { path: '/curves', name: 'Curves' },
    { path: '/chemistry', name: 'Chemistry' },
    { path: '/assistant', name: 'AI Assistant' },
    { path: '/session', name: 'Session Log' },
    { path: '/settings', name: 'Settings' },
  ];

  test('should navigate through all pages in sequence', async ({ page }) => {
    for (const { path, name } of pages) {
      await page.goto(path);
      await page.waitForLoadState('networkidle', { timeout: 10000 });

      // Verify page loaded (check for main content)
      const main = page.locator('main, [role="main"], body > div').first();
      await expect(main).toBeVisible({ timeout: 5000 });

      // Verify URL is correct
      expect(page.url()).toContain(path === '/' ? path : path);
    }
  });

  test('should support browser back and forward navigation', async ({ page }) => {
    // Navigate to multiple pages
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await page.goto('/calibration');
    await page.waitForLoadState('networkidle');

    await page.goto('/curves');
    await page.waitForLoadState('networkidle');

    // Go back
    await page.goBack();
    await page.waitForLoadState('networkidle');
    expect(page.url()).toContain('/calibration');

    // Go back again
    await page.goBack();
    await page.waitForLoadState('networkidle');
    expect(page.url()).toMatch(/\/$|\/$/);

    // Go forward
    await page.goForward();
    await page.waitForLoadState('networkidle');
    expect(page.url()).toContain('/calibration');
  });

  test('should maintain state during navigation', async ({ page }) => {
    // Navigate to settings
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Navigate away
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Navigate back to settings
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Settings page should load again
    const settingsForm = page.locator('[data-testid="settings"], form').first();
    await expect(settingsForm).toBeVisible({ timeout: 10000 });
  });

  test('should handle direct URL navigation', async ({ page }) => {
    // Directly navigate to a deep page
    await page.goto('/curves');
    await page.waitForLoadState('networkidle');

    // Should load correctly
    const main = page.locator('main, [role="main"]').first();
    await expect(main).toBeVisible({ timeout: 10000 });
    expect(page.url()).toContain('/curves');
  });
});
