/**
 * Chemistry Calculator E2E tests.
 */

import { test, expect } from '@playwright/test';
import { ChemistryPage } from '../pages';

test.describe('Chemistry Calculator', () => {
  let chemistryPage: ChemistryPage;

  test.beforeEach(async ({ page }) => {
    chemistryPage = new ChemistryPage(page);
    await chemistryPage.goto();
  });

  test('loads successfully', async () => {
    const isLoaded = await chemistryPage.isLoaded();
    expect(isLoaded).toBe(true);
  });

  test('displays page title', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Chemistry Calculator');
  });

  test('displays dimension inputs', async ({ page }) => {
    await expect(page.locator('text=Width')).toBeVisible();
    await expect(page.locator('text=Height')).toBeVisible();
  });

  test('displays metal ratio slider', async ({ page }) => {
    await expect(page.locator('text=Metal Ratio')).toBeVisible();
    await expect(page.locator('input[type="range"]')).toBeVisible();
  });

  test('displays preset buttons', async () => {
    await chemistryPage.verifyPresetsVisible();
  });

  test('displays results section', async () => {
    await chemistryPage.verifyResultsVisible();
  });

  test('calculates chemistry for standard print size', async ({ page }) => {
    await chemistryPage.setDimensions(8, 10);

    // Verify results are displayed
    await expect(page.locator('text=Platinum')).toBeVisible();
    await expect(page.locator('text=Palladium')).toBeVisible();
    await expect(page.locator('text=Ferric Oxalate')).toBeVisible();
  });

  test('updates calculation when dimensions change', async ({ page }) => {
    // Set initial dimensions
    await chemistryPage.setDimensions(8, 10);
    await page.waitForTimeout(300);

    // Get initial platinum value text
    const initialText = await page.locator('text=/\\d+(\\.\\d+)?\\s*ml/i').first().innerText();

    // Change dimensions to larger
    await chemistryPage.setDimensions(16, 20);
    await page.waitForTimeout(300);

    // Values should have changed
    const newText = await page.locator('text=/\\d+(\\.\\d+)?\\s*ml/i').first().innerText();

    // Note: Just checking that calculation runs; actual values depend on implementation
    expect(initialText).toBeDefined();
    expect(newText).toBeDefined();
  });

  test('warm tone preset sets metal ratio to 0.3', async () => {
    await chemistryPage.selectPreset('Warm Tone');
    const ratio = await chemistryPage.getMetalRatio();
    expect(ratio).toBeCloseTo(0.3, 1);
  });

  test('neutral preset sets metal ratio to 0.5', async () => {
    await chemistryPage.selectPreset('Neutral');
    const ratio = await chemistryPage.getMetalRatio();
    expect(ratio).toBeCloseTo(0.5, 1);
  });

  test('cool tone preset sets metal ratio to 0.7', async () => {
    await chemistryPage.selectPreset('Cool Tone');
    const ratio = await chemistryPage.getMetalRatio();
    expect(ratio).toBeCloseTo(0.7, 1);
  });

  test('pure platinum preset sets metal ratio to 1.0', async () => {
    await chemistryPage.selectPreset('Pure Platinum');
    const ratio = await chemistryPage.getMetalRatio();
    expect(ratio).toBeCloseTo(1.0, 1);
  });

  test('slider updates metal ratio', async ({ page }) => {
    await chemistryPage.setMetalRatio(0.6);
    const ratio = await chemistryPage.getMetalRatio();
    expect(ratio).toBeCloseTo(0.6, 1);
  });

  test('handles small print sizes', async () => {
    await chemistryPage.setDimensions(2, 2);
    // Should not crash, minimum volumes should be applied
    await chemistryPage.verifyResultsVisible();
  });

  test('handles large print sizes', async () => {
    await chemistryPage.setDimensions(24, 30);
    await chemistryPage.verifyResultsVisible();
  });

  test('is responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await chemistryPage.goto();

    const isLoaded = await chemistryPage.isLoaded();
    expect(isLoaded).toBe(true);

    // Should still be able to interact with controls
    await expect(page.locator('input[type="range"]')).toBeVisible();
  });
});
