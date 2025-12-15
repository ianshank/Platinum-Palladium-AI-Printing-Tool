/**
 * Base Page Object Model class.
 * Provides common functionality for all page objects.
 */

import { Page, Locator, expect } from '@playwright/test';

export abstract class BasePage {
  readonly page: Page;
  readonly pageTitle: Locator;

  constructor(page: Page) {
    this.page = page;
    this.pageTitle = page.locator('h1').first();
  }

  /**
   * Navigate to this page.
   */
  abstract goto(): Promise<void>;

  /**
   * Verify the page has loaded correctly.
   */
  abstract isLoaded(): Promise<boolean>;

  /**
   * Get the page heading text.
   */
  async getTitle(): Promise<string> {
    return this.pageTitle.innerText();
  }

  /**
   * Wait for the page to fully load.
   */
  async waitForLoad(): Promise<void> {
    await this.page.waitForLoadState('networkidle');
  }

  /**
   * Take a screenshot of the current page.
   */
  async screenshot(name: string): Promise<void> {
    await this.page.screenshot({ path: `screenshots/${name}.png`, fullPage: true });
  }

  /**
   * Get all navigation links.
   */
  async getNavLinks(): Promise<string[]> {
    const links = await this.page.locator('nav a').all();
    return Promise.all(links.map((link) => link.getAttribute('href').then((href) => href || '')));
  }

  /**
   * Navigate using sidebar.
   */
  async navigateTo(path: string): Promise<void> {
    await this.page.locator(`nav a[href="${path}"]`).click();
    await this.waitForLoad();
  }

  /**
   * Verify toast notification appears.
   */
  async expectToast(message: string): Promise<void> {
    const toast = this.page.locator('[role="alert"]').filter({ hasText: message });
    await expect(toast).toBeVisible({ timeout: 5000 });
  }

  /**
   * Close any open modal.
   */
  async closeModal(): Promise<void> {
    const closeButton = this.page.locator('[aria-label="Close modal"], [data-testid="modal-close"]');
    if (await closeButton.isVisible()) {
      await closeButton.click();
    }
  }

  /**
   * Check if element is visible.
   */
  async isVisible(selector: string): Promise<boolean> {
    return this.page.locator(selector).isVisible();
  }

  /**
   * Wait for element to be visible.
   */
  async waitForElement(selector: string, timeout = 10000): Promise<void> {
    await this.page.locator(selector).waitFor({ state: 'visible', timeout });
  }
}
