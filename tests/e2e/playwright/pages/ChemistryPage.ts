/**
 * Chemistry Calculator Page Object Model.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

export class ChemistryPage extends BasePage {
  readonly widthInput: Locator;
  readonly heightInput: Locator;
  readonly metalRatioSlider: Locator;
  readonly presetButtons: Locator;
  readonly resultsSection: Locator;
  readonly platinumValue: Locator;
  readonly palladiumValue: Locator;
  readonly ferricOxalateValue: Locator;
  readonly totalVolumeValue: Locator;

  constructor(page: Page) {
    super(page);
    this.widthInput = page.locator('input[aria-label*="Width"], label:has-text("Width") + input, label:has-text("Width") input');
    this.heightInput = page.locator('input[aria-label*="Height"], label:has-text("Height") + input, label:has-text("Height") input');
    this.metalRatioSlider = page.locator('input[type="range"]');
    this.presetButtons = page.locator('button').filter({ hasText: /Warm Tone|Neutral|Cool Tone|Pure Platinum/ });
    this.resultsSection = page.locator('h2, h3').filter({ hasText: 'Calculated Recipe' }).locator('..');
    this.platinumValue = page.locator('text=Platinum').locator('..').locator('span, p').filter({ hasText: /ml/ });
    this.palladiumValue = page.locator('text=Palladium').locator('..').locator('span, p').filter({ hasText: /ml/ });
    this.ferricOxalateValue = page.locator('text=Ferric Oxalate').locator('..').locator('span, p').filter({ hasText: /ml/ });
    this.totalVolumeValue = page.locator('text=Total Volume').locator('..').locator('span, p').filter({ hasText: /ml/ });
  }

  async goto(): Promise<void> {
    await this.page.goto('/chemistry');
    await this.waitForLoad();
  }

  async isLoaded(): Promise<boolean> {
    try {
      await expect(this.pageTitle).toContainText('Chemistry Calculator');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Set print dimensions.
   */
  async setDimensions(width: number, height: number): Promise<void> {
    await this.widthInput.fill(width.toString());
    await this.heightInput.fill(height.toString());
    // Wait for calculation to complete
    await this.page.waitForTimeout(300);
  }

  /**
   * Set metal ratio using slider.
   */
  async setMetalRatio(ratio: number): Promise<void> {
    await this.metalRatioSlider.fill(ratio.toString());
    await this.page.waitForTimeout(200);
  }

  /**
   * Click a preset button.
   */
  async selectPreset(preset: 'Warm Tone' | 'Neutral' | 'Cool Tone' | 'Pure Platinum'): Promise<void> {
    await this.page.locator('button').filter({ hasText: preset }).click();
    await this.page.waitForTimeout(200);
  }

  /**
   * Get calculated recipe values.
   */
  async getRecipeValues(): Promise<{
    platinum: string;
    palladium: string;
    ferricOxalate: string;
    totalVolume: string;
  }> {
    return {
      platinum: await this.getValueText('Platinum'),
      palladium: await this.getValueText('Palladium'),
      ferricOxalate: await this.getValueText('Ferric Oxalate'),
      totalVolume: await this.getValueText('Total Volume'),
    };
  }

  /**
   * Helper to get value text next to a label.
   */
  private async getValueText(label: string): Promise<string> {
    const container = this.page.locator(`text=${label}`).locator('xpath=ancestor::div[1]');
    const valueElement = container.locator('span, p').filter({ hasText: /\d/ }).last();

    if (await valueElement.isVisible()) {
      return valueElement.innerText();
    }
    return '0';
  }

  /**
   * Verify results section is visible.
   */
  async verifyResultsVisible(): Promise<void> {
    await expect(this.page.locator('text=Calculated Recipe')).toBeVisible();
    await expect(this.page.locator('text=Platinum')).toBeVisible();
    await expect(this.page.locator('text=Palladium')).toBeVisible();
  }

  /**
   * Get the current metal ratio from slider.
   */
  async getMetalRatio(): Promise<number> {
    const value = await this.metalRatioSlider.inputValue();
    return parseFloat(value);
  }

  /**
   * Verify preset buttons are visible.
   */
  async verifyPresetsVisible(): Promise<void> {
    await expect(this.page.locator('button').filter({ hasText: 'Warm Tone' })).toBeVisible();
    await expect(this.page.locator('button').filter({ hasText: 'Neutral' })).toBeVisible();
    await expect(this.page.locator('button').filter({ hasText: 'Cool Tone' })).toBeVisible();
    await expect(this.page.locator('button').filter({ hasText: 'Pure Platinum' })).toBeVisible();
  }
}
