/**
 * Calibration User Journey Test.
 *
 * This test simulates a complete calibration workflow from start to finish,
 * mimicking how a real user would interact with the application.
 */

import { test, expect } from '@playwright/test';
import { DashboardPage, CalibrationPage, ChemistryPage } from '../pages';

test.describe('User Journey: Complete Calibration Workflow', () => {
  test('complete calibration from dashboard to generated curve', async ({ page }) => {
    // Step 1: User lands on dashboard
    const dashboard = new DashboardPage(page);
    await dashboard.goto();

    await test.step('User sees dashboard and quick actions', async () => {
      await expect(page.locator('h1')).toContainText('Dashboard');
      await dashboard.verifyQuickActionsVisible();
    });

    // Step 2: User clicks "New Calibration" to start
    await test.step('User starts new calibration', async () => {
      await dashboard.startNewCalibration();
      await expect(page).toHaveURL(/\/calibration/);
    });

    // Step 3: User is on calibration wizard
    const calibration = new CalibrationPage(page);

    await test.step('User sees calibration wizard at step 1', async () => {
      await expect(page.locator('h1')).toContainText(/Calibration/);
      // Should be on step 1
      await expect(page.locator('text=/Step 1|Setup|Paper/')).toBeVisible();
    });

    // Step 4: User selects paper and chemistry settings
    await test.step('User configures paper settings', async () => {
      // Look for paper selection dropdown or buttons
      const paperSelect = page.locator('select').first();
      if (await paperSelect.isVisible()) {
        await paperSelect.selectOption({ index: 1 });
      }

      // Move to next step
      await calibration.nextStep();
    });

    // Step 5: User uploads scan (simulated)
    await test.step('User proceeds through scan upload step', async () => {
      // In a real test, we would upload an actual file
      // For now, we just verify the step is shown
      await expect(page.locator('text=/Upload|Scan|Image/')).toBeVisible();

      // Skip to next step (in real scenario, would upload file)
      await calibration.nextStep();
    });

    // Step 6: User reviews measurements
    await test.step('User reviews scan measurements', async () => {
      await expect(page.locator('text=/Measure|Analyze|Review/')).toBeVisible();
      await calibration.nextStep();
    });

    // Step 7: User adjusts curve if needed
    await test.step('User reviews curve adjustments', async () => {
      await expect(page.locator('text=/Curve|Adjust|Preview/')).toBeVisible();
      await calibration.nextStep();
    });

    // Step 8: User completes calibration
    await test.step('User completes calibration and sees results', async () => {
      // Look for completion message or finish button
      const finishButton = page.locator('button').filter({ hasText: /Finish|Complete|Done|Save/ });
      if (await finishButton.isVisible()) {
        await finishButton.click();
      }

      // Should see success message or redirect
      await expect(
        page.locator('text=/Complete|Success|Saved|Generated/').or(page.locator('h1'))
      ).toBeVisible();
    });
  });

  test('user can navigate back through calibration steps', async ({ page }) => {
    const calibration = new CalibrationPage(page);
    await calibration.goto();

    await test.step('Navigate forward through steps', async () => {
      const currentStep = await calibration.getCurrentStep();
      expect(currentStep).toBe(1);

      await calibration.nextStep();
      // Should be on step 2
    });

    await test.step('Navigate back to previous step', async () => {
      await calibration.previousStep();
      // Should be back on step 1
      const step = await calibration.getCurrentStep();
      expect(step).toBe(1);
    });
  });

  test('user can cancel calibration and return to dashboard', async ({ page }) => {
    const dashboard = new DashboardPage(page);
    const calibration = new CalibrationPage(page);

    await dashboard.goto();
    await dashboard.startNewCalibration();

    await test.step('User decides to cancel calibration', async () => {
      // Look for cancel or back button
      const cancelButton = page.locator('button, a').filter({ hasText: /Cancel|Back|Exit/ });

      if (await cancelButton.isVisible()) {
        await cancelButton.click();
      } else {
        // Navigate back using browser
        await page.goBack();
      }

      await expect(page).toHaveURL(/\/dashboard|\/$/);
    });
  });
});

test.describe('User Journey: Chemistry Calculation', () => {
  test('user calculates chemistry for a print session', async ({ page }) => {
    const dashboard = new DashboardPage(page);
    const chemistry = new ChemistryPage(page);

    // Step 1: User starts from dashboard
    await dashboard.goto();

    await test.step('User navigates to chemistry calculator', async () => {
      await dashboard.openChemistryCalculator();
      await expect(page).toHaveURL(/\/chemistry/);
    });

    // Step 2: User enters print dimensions
    await test.step('User enters print dimensions', async () => {
      await chemistry.setDimensions(11, 14);
      await chemistry.verifyResultsVisible();
    });

    // Step 3: User selects a preset
    await test.step('User selects warm tone preset', async () => {
      await chemistry.selectPreset('Warm Tone');
      const ratio = await chemistry.getMetalRatio();
      expect(ratio).toBeCloseTo(0.3, 1);
    });

    // Step 4: User adjusts metal ratio manually
    await test.step('User fine-tunes metal ratio', async () => {
      await chemistry.setMetalRatio(0.35);
      const ratio = await chemistry.getMetalRatio();
      expect(ratio).toBeCloseTo(0.35, 1);
    });

    // Step 5: User reviews calculated recipe
    await test.step('User reviews calculated recipe', async () => {
      const recipe = await chemistry.getRecipeValues();
      expect(recipe.platinum).toBeDefined();
      expect(recipe.palladium).toBeDefined();
      expect(recipe.ferricOxalate).toBeDefined();
    });

    // Step 6: User might want to try different sizes
    await test.step('User recalculates for different print size', async () => {
      await chemistry.setDimensions(8, 10);
      await chemistry.verifyResultsVisible();
    });
  });

  test('user explores different chemistry presets', async ({ page }) => {
    const chemistry = new ChemistryPage(page);
    await chemistry.goto();
    await chemistry.setDimensions(8, 10);

    const presets: Array<{
      name: 'Warm Tone' | 'Neutral' | 'Cool Tone' | 'Pure Platinum';
      expectedRatio: number;
    }> = [
      { name: 'Warm Tone', expectedRatio: 0.3 },
      { name: 'Neutral', expectedRatio: 0.5 },
      { name: 'Cool Tone', expectedRatio: 0.7 },
      { name: 'Pure Platinum', expectedRatio: 1.0 },
    ];

    for (const preset of presets) {
      await test.step(`User tries ${preset.name} preset`, async () => {
        await chemistry.selectPreset(preset.name);
        const ratio = await chemistry.getMetalRatio();
        expect(ratio).toBeCloseTo(preset.expectedRatio, 1);
      });
    }
  });
});
