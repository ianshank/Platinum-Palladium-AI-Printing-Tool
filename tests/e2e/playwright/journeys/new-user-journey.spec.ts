/**
 * New User Onboarding Journey Test.
 *
 * This test simulates a new user's first experience with the application,
 * exploring all main features.
 */

import { test, expect } from '@playwright/test';
import {
  DashboardPage,
  CalibrationPage,
  ChemistryPage,
  AIAssistantPage,
} from '../pages';

test.describe('User Journey: New User Exploration', () => {
  test('new user explores all main sections of the app', async ({ page }) => {
    // Step 1: New user lands on the app
    await test.step('User arrives at the application', async () => {
      await page.goto('/');
      // Should redirect to dashboard
      await page.waitForURL(/\/(dashboard)?$/);
    });

    // Step 2: User explores the dashboard
    const dashboard = new DashboardPage(page);

    await test.step('User explores the dashboard', async () => {
      await dashboard.goto();
      await expect(page.locator('h1')).toContainText('Dashboard');

      // User sees the welcome message
      await expect(page.locator('text=Welcome')).toBeVisible();

      // User sees stats (even if empty for new user)
      await expect(page.locator('text=Total Prints')).toBeVisible();
    });

    await test.step('User reads getting started tips', async () => {
      await dashboard.verifyGettingStartedVisible();
      await expect(page.locator('text=Start with Calibration')).toBeVisible();
    });

    // Step 3: User explores calibration
    await test.step('User checks out calibration wizard', async () => {
      await dashboard.startNewCalibration();

      await expect(page.locator('h1')).toContainText(/Calibration/);

      // User sees the wizard steps
      await expect(page.locator('text=/Step|Setup|Paper/')).toBeVisible();

      // User goes back to dashboard to continue exploring
      await page.goBack();
    });

    // Step 4: User explores chemistry calculator
    await test.step('User explores chemistry calculator', async () => {
      await dashboard.openChemistryCalculator();

      const chemistry = new ChemistryPage(page);
      await expect(page.locator('h1')).toContainText('Chemistry Calculator');

      // User tries entering some values
      await chemistry.setDimensions(8, 10);
      await chemistry.verifyResultsVisible();

      // User tries a preset
      await chemistry.selectPreset('Neutral');

      await page.goBack();
    });

    // Step 5: User explores AI assistant
    await test.step('User explores AI assistant', async () => {
      await dashboard.openAIAssistant();

      const assistant = new AIAssistantPage(page);
      await expect(page.locator('h1')).toContainText('AI Assistant');

      // User sees the chat interface
      await expect(page.locator('input, textarea')).toBeVisible();

      // User might see quick prompts
      const prompts = await assistant.getQuickPrompts();
      expect(prompts).toBeDefined();

      await page.goBack();
    });

    // Step 6: User checks session log
    await test.step('User checks session log', async () => {
      await dashboard.openSessionLog();

      await expect(page.locator('h1')).toContainText(/Session/);

      // New user sees empty state
      await expect(page.locator('text=/No Sessions|Empty|Start/')).toBeVisible();

      await page.goBack();
    });

    // Step 7: User visits settings
    await test.step('User explores settings', async () => {
      await page.locator('nav a').filter({ hasText: /Settings/i }).click();

      await expect(page.locator('h1')).toContainText('Settings');

      // User sees theme options
      await expect(page.locator('text=Theme')).toBeVisible();

      // User sees about section
      await expect(page.locator('text=About')).toBeVisible();
    });

    // Step 8: User returns to dashboard satisfied
    await test.step('User returns to dashboard', async () => {
      await page.locator('nav a').filter({ hasText: /Dashboard/i }).click();
      await expect(page.locator('h1')).toContainText('Dashboard');
    });
  });

  test('new user completes first chemistry calculation', async ({ page }) => {
    const dashboard = new DashboardPage(page);
    const chemistry = new ChemistryPage(page);

    await dashboard.goto();

    await test.step('User navigates to chemistry calculator', async () => {
      await dashboard.openChemistryCalculator();
    });

    await test.step('User plans their first 8x10 print', async () => {
      await chemistry.setDimensions(8, 10);
      await chemistry.verifyResultsVisible();
    });

    await test.step('User experiments with different presets', async () => {
      // User doesn't know what they want, so they try all presets
      await chemistry.selectPreset('Warm Tone');
      await page.waitForTimeout(500);

      await chemistry.selectPreset('Cool Tone');
      await page.waitForTimeout(500);

      await chemistry.selectPreset('Neutral');
      await page.waitForTimeout(500);
    });

    await test.step('User settles on neutral and notes the recipe', async () => {
      await chemistry.selectPreset('Neutral');
      const recipe = await chemistry.getRecipeValues();

      expect(recipe.platinum).toBeDefined();
      expect(recipe.palladium).toBeDefined();
    });
  });

  test('new user reads about the application', async ({ page }) => {
    await page.goto('/settings');

    await test.step('User reads about PTPD Calibration Studio', async () => {
      await expect(page.locator('text=PTPD Calibration Studio')).toBeVisible();
      await expect(page.locator('text=/AI-powered|platinum|palladium/')).toBeVisible();
    });

    await test.step('User sees technology stack', async () => {
      await expect(page.locator('text=/React|TypeScript|FastAPI/')).toBeVisible();
    });
  });
});

test.describe('User Journey: Mobile User Experience', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('mobile user can navigate all sections', async ({ page }) => {
    await test.step('User loads app on mobile', async () => {
      await page.goto('/dashboard');
      await expect(page.locator('h1')).toContainText('Dashboard');
    });

    await test.step('User can access navigation', async () => {
      // Mobile might have hamburger menu
      const menuButton = page.locator('[aria-label*="menu"], button:has(svg)').first();

      if (await menuButton.isVisible()) {
        await menuButton.click();
      }

      // Navigation should be accessible
      await expect(page.locator('nav a').first()).toBeVisible();
    });

    await test.step('User navigates to chemistry calculator', async () => {
      await page.locator('nav a').filter({ hasText: /Chemistry/i }).click();
      await expect(page.locator('h1')).toContainText('Chemistry');
    });

    await test.step('User can use chemistry calculator on mobile', async () => {
      const chemistry = new ChemistryPage(page);
      await chemistry.setDimensions(5, 7); // Smaller print for mobile
      await chemistry.verifyResultsVisible();
    });
  });
});

test.describe('User Journey: Returning User Quick Task', () => {
  test('returning user quickly calculates chemistry', async ({ page }) => {
    // Simulate a returning user who knows the app
    await page.goto('/chemistry');

    const chemistry = new ChemistryPage(page);

    await test.step('User goes directly to task', async () => {
      await expect(page.locator('h1')).toContainText('Chemistry');
    });

    await test.step('User enters known print size', async () => {
      await chemistry.setDimensions(11, 14);
    });

    await test.step('User selects their preferred preset', async () => {
      await chemistry.selectPreset('Warm Tone');
    });

    await test.step('User verifies recipe and is done', async () => {
      const recipe = await chemistry.getRecipeValues();
      expect(recipe.platinum).toBeDefined();
      // User memorizes values and goes to darkroom
    });
  });
});
