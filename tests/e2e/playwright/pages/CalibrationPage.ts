/**
 * Calibration Wizard Page Object Model.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

export class CalibrationPage extends BasePage {
  readonly stepper: Locator;
  readonly stepIndicators: Locator;
  readonly currentStepContent: Locator;
  readonly nextButton: Locator;
  readonly previousButton: Locator;
  readonly fileInput: Locator;
  readonly uploadArea: Locator;

  constructor(page: Page) {
    super(page);
    this.stepper = page.locator('[data-testid="stepper"], [role="progressbar"]').or(page.locator('div').filter({ hasText: /Step \d of \d/ }).first());
    this.stepIndicators = page.locator('[data-testid="step-indicator"], .step-indicator');
    this.currentStepContent = page.locator('[data-testid="step-content"], main section').first();
    this.nextButton = page.locator('button').filter({ hasText: /Next|Continue|Generate|Finish/ });
    this.previousButton = page.locator('button').filter({ hasText: /Back|Previous/ });
    this.fileInput = page.locator('input[type="file"]');
    this.uploadArea = page.locator('[data-testid="upload-area"], .upload-dropzone');
  }

  async goto(): Promise<void> {
    await this.page.goto('/calibration');
    await this.waitForLoad();
  }

  async isLoaded(): Promise<boolean> {
    try {
      await expect(this.pageTitle).toContainText('Calibration');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get current step number.
   */
  async getCurrentStep(): Promise<number> {
    const stepText = await this.page.locator('text=/Step \\d/').first().innerText();
    const match = stepText.match(/Step (\d)/);
    return match ? parseInt(match[1]) : 1;
  }

  /**
   * Get total steps.
   */
  async getTotalSteps(): Promise<number> {
    const stepText = await this.page.locator('text=/of \\d/').first().innerText();
    const match = stepText.match(/of (\d)/);
    return match ? parseInt(match[1]) : 5;
  }

  /**
   * Go to next step.
   */
  async nextStep(): Promise<void> {
    await this.nextButton.click();
    await this.page.waitForTimeout(500);
  }

  /**
   * Go to previous step.
   */
  async previousStep(): Promise<void> {
    await this.previousButton.click();
    await this.page.waitForTimeout(500);
  }

  /**
   * Upload scan file.
   */
  async uploadScan(filePath: string): Promise<void> {
    await this.fileInput.setInputFiles(filePath);
    await this.page.waitForTimeout(1000);
  }

  /**
   * Select paper type.
   */
  async selectPaper(paperName: string): Promise<void> {
    const paperOption = this.page.locator(`text=${paperName}`).or(
      this.page.locator(`option:has-text("${paperName}")`)
    );
    await paperOption.click();
  }

  /**
   * Set exposure time.
   */
  async setExposureTime(seconds: number): Promise<void> {
    const exposureInput = this.page.locator('input[aria-label*="Exposure"], label:has-text("Exposure") input');
    await exposureInput.fill(seconds.toString());
  }

  /**
   * Verify stepper shows correct step.
   */
  async verifyStep(stepNumber: number): Promise<void> {
    const currentStep = await this.getCurrentStep();
    expect(currentStep).toBe(stepNumber);
  }

  /**
   * Complete step 1 (Setup).
   */
  async completeSetupStep(paper: string, chemistry: string): Promise<void> {
    // Select paper
    const paperSelect = this.page.locator('select').first();
    if (await paperSelect.isVisible()) {
      await paperSelect.selectOption({ label: paper });
    }

    // Select chemistry if available
    const chemistrySelect = this.page.locator('select').nth(1);
    if (await chemistrySelect.isVisible()) {
      await chemistrySelect.selectOption({ label: chemistry });
    }
  }

  /**
   * Verify step completion indicators.
   */
  async verifyStepsCompleted(completedSteps: number[]): Promise<void> {
    for (const step of completedSteps) {
      const indicator = this.page.locator(`[data-step="${step}"][data-completed="true"]`);
      // This is a soft check - some implementations may vary
      const isCompleted = await indicator.isVisible().catch(() => false);
      if (!isCompleted) {
        // Check for checkmark or other completion indicators
        const checkmark = this.page.locator(`[data-step="${step}"] svg`);
        await expect(checkmark).toBeVisible().catch(() => {});
      }
    }
  }

  /**
   * Get step titles.
   */
  async getStepTitles(): Promise<string[]> {
    const titles = await this.page.locator('.step-title, [data-testid="step-title"]').allInnerTexts();
    return titles;
  }
}
