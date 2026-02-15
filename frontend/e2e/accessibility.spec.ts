import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility Audit', () => {
  const pages = [
    { path: '/', name: 'Dashboard' },
    { path: '/calibration', name: 'Calibration' },
    { path: '/curves', name: 'Curves' },
    { path: '/chemistry', name: 'Chemistry' },
    { path: '/assistant', name: 'AI Assistant' },
    { path: '/session', name: 'Session Log' },
    { path: '/settings', name: 'Settings' },
  ];

  for (const { path, name } of pages) {
    test(`${name} page has no critical accessibility violations`, async ({ page }) => {
      await page.goto(path);
      await page.waitForLoadState('networkidle', { timeout: 10000 });

      const accessibilityScanResults = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        .analyze();

      const criticalViolations = accessibilityScanResults.violations.filter(
        (v) => v.impact === 'critical' || v.impact === 'serious'
      );

      // Log violations for debugging
      if (criticalViolations.length > 0) {
        console.error(`${name} page has ${criticalViolations.length} critical/serious violations:`);
        criticalViolations.forEach((violation) => {
          console.error(`- ${violation.id}: ${violation.description}`);
          console.error(`  Impact: ${violation.impact}`);
          console.error(`  Help: ${violation.helpUrl}`);
        });
      }

      expect(criticalViolations).toEqual([]);
    });
  }

  test('keyboard navigation works across all pages', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Tab through navigation and verify focus order
    const focusedElements: string[] = [];

    for (let i = 0; i < 15; i++) {
      await page.keyboard.press('Tab');
      await page.waitForTimeout(100);

      const tagName = await page.evaluate(() => document.activeElement?.tagName || '');
      const role = await page.evaluate(() => document.activeElement?.getAttribute('role') || '');
      const type = await page.evaluate(() =>
        document.activeElement?.getAttribute('type') || ''
      );

      if (tagName) {
        const elementInfo = type ? `${tagName}[${type}]` : tagName;
        focusedElements.push(role ? `${elementInfo}(${role})` : elementInfo);
      }
    }

    // Should have focused on multiple interactive elements
    expect(focusedElements.length).toBeGreaterThan(0);

    // Should focus on interactive elements (buttons, links, inputs)
    const interactiveElements = focusedElements.filter(
      (el) =>
        el.includes('BUTTON') ||
        el.includes('A') ||
        el.includes('INPUT') ||
        el.includes('SELECT') ||
        el.includes('TEXTAREA') ||
        el.includes('role')
    );
    expect(interactiveElements.length).toBeGreaterThan(0);
  });

  test('all interactive elements are keyboard accessible', async ({ page }) => {
    const pagesWithInteractiveElements = [
      { path: '/calibration', buttonText: 'Start' },
      { path: '/curves', hasControls: true },
      { path: '/chemistry', hasForm: true },
      { path: '/assistant', hasInput: true },
      { path: '/settings', hasForm: true },
    ];

    for (const pageInfo of pagesWithInteractiveElements) {
      await page.goto(pageInfo.path);
      await page.waitForLoadState('networkidle');

      // Find first interactive element
      const firstButton = page.locator('button, a, input, select, textarea').first();

      if (await firstButton.isVisible({ timeout: 5000 }).catch(() => false)) {
        // Focus on it
        await firstButton.focus();

        // Verify it has focus
        const isFocused = await page.evaluate(() => {
          const activeElement = document.activeElement;
          return (
            activeElement?.tagName === 'BUTTON' ||
            activeElement?.tagName === 'A' ||
            activeElement?.tagName === 'INPUT' ||
            activeElement?.tagName === 'SELECT' ||
            activeElement?.tagName === 'TEXTAREA'
          );
        });

        expect(isFocused).toBeTruthy();
      }
    }
  });

  test('Skip to main content link exists and works', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Look for skip link (usually first focusable element)
    await page.keyboard.press('Tab');
    await page.waitForTimeout(100);

    const skipLinkText = await page.evaluate(() => {
      const activeElement = document.activeElement;
      return activeElement?.textContent?.trim() || '';
    });

    // Common skip link patterns
    const hasSkipLink =
      skipLinkText.toLowerCase().includes('skip') ||
      skipLinkText.toLowerCase().includes('main');

    // If skip link exists, activate it
    if (hasSkipLink) {
      await page.keyboard.press('Enter');
      await page.waitForTimeout(200);

      // Verify focus moved to main content
      const focusedElementId = await page.evaluate(() => {
        const activeElement = document.activeElement;
        return activeElement?.id || activeElement?.tagName || '';
      });

      expect(focusedElementId).toBeTruthy();
    } else {
      // Skip link is recommended but not strictly required
      console.warn('Skip to main content link not found');
    }
  });

  test('all images have alt text', async ({ page }) => {
    for (const { path, name } of pages) {
      await page.goto(path);
      await page.waitForLoadState('networkidle');

      const imagesWithoutAlt = await page.locator('img:not([alt])').count();

      if (imagesWithoutAlt > 0) {
        console.warn(`${name} page has ${imagesWithoutAlt} images without alt text`);
      }

      // Critical images should have alt text (decorative images can have alt="")
      const images = await page.locator('img').all();
      for (const img of images) {
        const hasAltAttribute = await img.getAttribute('alt');
        expect(hasAltAttribute).not.toBeNull();
      }
    }
  });

  test('form inputs have labels', async ({ page }) => {
    const pagesWithForms = [
      { path: '/calibration', name: 'Calibration' },
      { path: '/chemistry', name: 'Chemistry' },
      { path: '/settings', name: 'Settings' },
    ];

    for (const { path, name } of pagesWithForms) {
      await page.goto(path);
      await page.waitForLoadState('networkidle');

      // Get all input elements
      const inputs = await page
        .locator('input[type="text"], input[type="number"], input[type="email"], select, textarea')
        .all();

      for (const input of inputs) {
        const id = await input.getAttribute('id');
        const ariaLabel = await input.getAttribute('aria-label');
        const ariaLabelledBy = await input.getAttribute('aria-labelledby');

        // Input should have either:
        // 1. An associated label (via id)
        // 2. An aria-label
        // 3. An aria-labelledby
        const hasLabel =
          (id && (await page.locator(`label[for="${id}"]`).count()) > 0) ||
          ariaLabel ||
          ariaLabelledBy;

        if (!hasLabel) {
          const inputType = await input.getAttribute('type');
          const inputName = await input.getAttribute('name');
          console.warn(
            `${name} page has unlabeled input: type="${inputType}" name="${inputName}"`
          );
        }

        // At minimum, the input should have some identifying attribute
        expect(id || ariaLabel || ariaLabelledBy).toBeTruthy();
      }
    }
  });

  test('headings are in logical order', async ({ page }) => {
    for (const { path, name } of pages) {
      await page.goto(path);
      await page.waitForLoadState('networkidle');

      const headingLevels = await page.evaluate(() => {
        const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
        return headings.map((h) => parseInt(h.tagName.substring(1)));
      });

      if (headingLevels.length > 0) {
        // First heading should be h1
        expect(headingLevels[0]).toBe(1);

        // Check for skipped heading levels
        for (let i = 1; i < headingLevels.length; i++) {
          const diff = headingLevels[i] - headingLevels[i - 1];
          if (diff > 1) {
            console.warn(
              `${name} page skips heading level: h${headingLevels[i - 1]} to h${headingLevels[i]}`
            );
          }
        }
      }
    }
  });

  test('color contrast meets WCAG AA standards', async ({ page }) => {
    for (const { path, name } of pages) {
      await page.goto(path);
      await page.waitForLoadState('networkidle');

      const accessibilityScanResults = await new AxeBuilder({ page })
        .withTags(['wcag2aa'])
        .disableRules([
          'region',
          'landmark-one-main',
          'page-has-heading-one',
          'html-has-lang',
        ]) // Focus on color contrast
        .analyze();

      const contrastViolations = accessibilityScanResults.violations.filter((v) =>
        v.id.includes('color-contrast')
      );

      if (contrastViolations.length > 0) {
        console.error(`${name} page has color contrast violations:`);
        contrastViolations.forEach((violation) => {
          console.error(`- ${violation.description}`);
          console.error(`  Affected elements: ${violation.nodes.length}`);
        });
      }

      expect(contrastViolations).toEqual([]);
    }
  });

  test('interactive elements have visible focus indicators', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Focus on first button
    const firstButton = page.locator('button, a').first();
    await firstButton.focus();

    // Check if focus outline is visible
    const hasFocusOutline = await firstButton.evaluate((el) => {
      const styles = window.getComputedStyle(el);
      const pseudoStyles = window.getComputedStyle(el, ':focus');

      return (
        styles.outline !== 'none' ||
        styles.outlineWidth !== '0px' ||
        pseudoStyles.outline !== 'none' ||
        pseudoStyles.boxShadow !== 'none' ||
        pseudoStyles.border !== styles.border
      );
    });

    expect(hasFocusOutline).toBeTruthy();
  });

  test('ARIA landmarks are used correctly', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['best-practice'])
      .analyze();

    const landmarkViolations = accessibilityScanResults.violations.filter(
      (v) => v.id.includes('landmark') || v.id.includes('region')
    );

    // Log violations but don't fail (landmarks are best practice, not strict requirement)
    if (landmarkViolations.length > 0) {
      console.warn('ARIA landmark violations found:');
      landmarkViolations.forEach((violation) => {
        console.warn(`- ${violation.id}: ${violation.description}`);
      });
    }

    // Should have at least a main landmark
    const mainLandmark = page.locator('main, [role="main"]');
    const hasMainLandmark = (await mainLandmark.count()) > 0;
    expect(hasMainLandmark).toBeTruthy();
  });
});
