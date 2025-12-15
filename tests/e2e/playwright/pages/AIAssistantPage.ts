/**
 * AI Assistant Page Object Model.
 */

import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

export class AIAssistantPage extends BasePage {
  readonly chatContainer: Locator;
  readonly messageInput: Locator;
  readonly sendButton: Locator;
  readonly quickPrompts: Locator;
  readonly messages: Locator;
  readonly loadingIndicator: Locator;
  readonly clearButton: Locator;

  constructor(page: Page) {
    super(page);
    this.chatContainer = page.locator('[data-testid="chat-container"], .chat-container, main');
    this.messageInput = page.locator('input[placeholder*="message"], textarea[placeholder*="message"], input[type="text"]').last();
    this.sendButton = page.locator('button[type="submit"], button:has-text("Send"), button[aria-label*="Send"]');
    this.quickPrompts = page.locator('[data-testid="quick-prompts"], .quick-prompts').or(
      page.locator('button').filter({ hasText: /How do I|What|Help me/ })
    );
    this.messages = page.locator('[data-testid="message"], .message, [role="log"] > div');
    this.loadingIndicator = page.locator('[data-testid="loading"], .loading, [aria-busy="true"]');
    this.clearButton = page.locator('button').filter({ hasText: /Clear|New Chat/ });
  }

  async goto(): Promise<void> {
    await this.page.goto('/assistant');
    await this.waitForLoad();
  }

  async isLoaded(): Promise<boolean> {
    try {
      await expect(this.pageTitle).toContainText('AI Assistant');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Send a message to the AI assistant.
   */
  async sendMessage(message: string): Promise<void> {
    await this.messageInput.fill(message);
    await this.sendButton.click();
    // Wait for response
    await this.waitForResponse();
  }

  /**
   * Wait for AI response to appear.
   */
  async waitForResponse(timeout = 30000): Promise<void> {
    // Wait for loading to start
    await this.page.waitForTimeout(500);

    // Wait for loading to finish
    try {
      await this.loadingIndicator.waitFor({ state: 'hidden', timeout });
    } catch {
      // Loading indicator may not be visible if response is fast
    }

    // Ensure a response message appeared
    await this.page.waitForTimeout(500);
  }

  /**
   * Get all messages in the chat.
   */
  async getMessages(): Promise<{ role: string; content: string }[]> {
    const messageElements = await this.messages.all();
    const messages: { role: string; content: string }[] = [];

    for (const element of messageElements) {
      const text = await element.innerText();
      const isUser = await element.getAttribute('data-role') === 'user' ||
        (await element.locator('.user-message').isVisible().catch(() => false));

      messages.push({
        role: isUser ? 'user' : 'assistant',
        content: text,
      });
    }

    return messages;
  }

  /**
   * Get the last assistant response.
   */
  async getLastResponse(): Promise<string> {
    const assistantMessages = this.page.locator('[data-role="assistant"], .assistant-message').last();
    if (await assistantMessages.isVisible()) {
      return assistantMessages.innerText();
    }

    // Fallback: get the last message
    const lastMessage = this.messages.last();
    return lastMessage.innerText();
  }

  /**
   * Click a quick prompt button.
   */
  async useQuickPrompt(promptText: string): Promise<void> {
    const promptButton = this.page.locator('button').filter({ hasText: promptText });
    await promptButton.click();
    await this.waitForResponse();
  }

  /**
   * Get available quick prompts.
   */
  async getQuickPrompts(): Promise<string[]> {
    const buttons = await this.page.locator('button').filter({ hasText: /\?$|How|What|Help/ }).all();
    return Promise.all(buttons.map((btn) => btn.innerText()));
  }

  /**
   * Clear chat history.
   */
  async clearChat(): Promise<void> {
    if (await this.clearButton.isVisible()) {
      await this.clearButton.click();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Verify message appears in chat.
   */
  async verifyMessageSent(message: string): Promise<void> {
    await expect(this.page.locator(`text=${message}`).first()).toBeVisible();
  }

  /**
   * Verify AI response received.
   */
  async verifyResponseReceived(): Promise<void> {
    const messages = await this.getMessages();
    const hasResponse = messages.some((m) => m.role === 'assistant' && m.content.length > 0);
    expect(hasResponse).toBe(true);
  }

  /**
   * Check if chat is empty.
   */
  async isChatEmpty(): Promise<boolean> {
    const messageCount = await this.messages.count();
    return messageCount === 0;
  }
}
