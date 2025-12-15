/**
 * Dashboard Page Object Model.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

export class DashboardPage extends BasePage {
  readonly statsGrid: Locator;
  readonly quickActions: Locator;
  readonly gettingStarted: Locator;
  readonly newCalibrationButton: Locator;
  readonly chemistryButton: Locator;
  readonly aiAssistantButton: Locator;
  readonly sessionButton: Locator;

  constructor(page: Page) {
    super(page);
    this.statsGrid = page.locator('[data-testid="stats-grid"]').or(page.locator('div').filter({ hasText: 'Total Prints' }).first());
    this.quickActions = page.locator('h2').filter({ hasText: 'Quick Actions' }).locator('..');
    this.gettingStarted = page.locator('h2').filter({ hasText: 'Getting Started' }).locator('..');
    this.newCalibrationButton = page.locator('a').filter({ hasText: 'New Calibration' });
    this.chemistryButton = page.locator('a').filter({ hasText: 'Calculate Chemistry' });
    this.aiAssistantButton = page.locator('a').filter({ hasText: 'Ask AI Assistant' });
    this.sessionButton = page.locator('a').filter({ hasText: 'Log Session' });
  }

  async goto(): Promise<void> {
    await this.page.goto('/dashboard');
    await this.waitForLoad();
  }

  async isLoaded(): Promise<boolean> {
    try {
      await expect(this.pageTitle).toContainText('Dashboard');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get all stat card values.
   */
  async getStatValues(): Promise<Record<string, string>> {
    const stats: Record<string, string> = {};

    const totalPrints = this.page.locator('text=Total Prints').locator('..').locator('p, span').first();
    const activeCurves = this.page.locator('text=Active Curves').locator('..').locator('p, span').first();
    const thisMonth = this.page.locator('text=This Month').locator('..').locator('p, span').first();
    const avgRating = this.page.locator('text=Avg Rating').locator('..').locator('p, span').first();

    if (await totalPrints.isVisible()) stats.totalPrints = await totalPrints.innerText();
    if (await activeCurves.isVisible()) stats.activeCurves = await activeCurves.innerText();
    if (await thisMonth.isVisible()) stats.thisMonth = await thisMonth.innerText();
    if (await avgRating.isVisible()) stats.avgRating = await avgRating.innerText();

    return stats;
  }

  /**
   * Click New Calibration button.
   */
  async startNewCalibration(): Promise<void> {
    await this.newCalibrationButton.click();
    await this.page.waitForURL('**/calibration**');
  }

  /**
   * Click Chemistry Calculator button.
   */
  async openChemistryCalculator(): Promise<void> {
    await this.chemistryButton.click();
    await this.page.waitForURL('**/chemistry**');
  }

  /**
   * Click AI Assistant button.
   */
  async openAIAssistant(): Promise<void> {
    await this.aiAssistantButton.click();
    await this.page.waitForURL('**/assistant**');
  }

  /**
   * Click Log Session button.
   */
  async openSessionLog(): Promise<void> {
    await this.sessionButton.click();
    await this.page.waitForURL('**/sessions**');
  }

  /**
   * Verify quick actions are visible.
   */
  async verifyQuickActionsVisible(): Promise<void> {
    await expect(this.newCalibrationButton).toBeVisible();
    await expect(this.chemistryButton).toBeVisible();
    await expect(this.aiAssistantButton).toBeVisible();
    await expect(this.sessionButton).toBeVisible();
  }

  /**
   * Verify getting started tips are visible.
   */
  async verifyGettingStartedVisible(): Promise<void> {
    await expect(this.page.locator('text=Start with Calibration')).toBeVisible();
    await expect(this.page.locator('text=Log Your Sessions')).toBeVisible();
    await expect(this.page.locator('text=Ask the AI')).toBeVisible();
  }
}
