/**
 * AI Assistant User Journey Test.
 *
 * This test simulates a user interacting with the AI assistant
 * to get help with platinum/palladium printing.
 */

import { test, expect } from '@playwright/test';
import { DashboardPage, AIAssistantPage } from '../pages';

test.describe('User Journey: Getting Help from AI Assistant', () => {
  test('user asks AI for printing advice', async ({ page }) => {
    const dashboard = new DashboardPage(page);
    const assistant = new AIAssistantPage(page);

    // Step 1: User starts from dashboard
    await dashboard.goto();

    await test.step('User navigates to AI Assistant', async () => {
      await dashboard.openAIAssistant();
      await expect(page).toHaveURL(/\/assistant/);
    });

    // Step 2: User sees the chat interface
    await test.step('User sees empty chat interface', async () => {
      await expect(page.locator('h1')).toContainText(/AI Assistant/);
      // Should see input area
      await expect(page.locator('input, textarea')).toBeVisible();
    });

    // Step 3: User sees quick prompts
    await test.step('User sees quick prompt suggestions', async () => {
      const prompts = await assistant.getQuickPrompts();
      expect(prompts.length).toBeGreaterThan(0);
    });

    // Step 4: User types a question
    await test.step('User asks about improving blacks', async () => {
      await assistant.sendMessage('How can I get deeper blacks in my prints?');
      await assistant.verifyMessageSent('deeper blacks');
    });

    // Step 5: User waits for and reads response
    await test.step('User receives AI response', async () => {
      // Note: In actual E2E with real backend, this would verify actual response
      await assistant.waitForResponse();
      // The response verification depends on whether mock or real backend is used
    });

    // Step 6: User asks a follow-up question
    await test.step('User asks follow-up question', async () => {
      await assistant.sendMessage('What exposure time would you recommend?');
      await assistant.verifyMessageSent('exposure time');
    });
  });

  test('user uses quick prompts', async ({ page }) => {
    const assistant = new AIAssistantPage(page);
    await assistant.goto();

    await test.step('User clicks on a quick prompt', async () => {
      const prompts = await assistant.getQuickPrompts();

      if (prompts.length > 0) {
        // Click the first quick prompt
        const firstPrompt = prompts[0];
        await page.locator('button').filter({ hasText: firstPrompt.substring(0, 20) }).click();

        // Should trigger sending that prompt
        await assistant.waitForResponse();
      }
    });
  });

  test('user can have multi-turn conversation', async ({ page }) => {
    const assistant = new AIAssistantPage(page);
    await assistant.goto();

    const questions = [
      'What paper should I use for my first platinum print?',
      'How do I mix the sensitizer solution?',
      'What is the typical development time?',
    ];

    for (let i = 0; i < questions.length; i++) {
      await test.step(`User asks question ${i + 1}`, async () => {
        await assistant.sendMessage(questions[i]);
        await assistant.verifyMessageSent(questions[i].substring(0, 20));
        await assistant.waitForResponse();
      });
    }

    await test.step('User verifies conversation history', async () => {
      const messages = await assistant.getMessages();
      // Should have user messages and assistant responses
      expect(messages.length).toBeGreaterThan(0);
    });
  });
});

test.describe('User Journey: Troubleshooting with AI', () => {
  test('user troubleshoots uneven coating issue', async ({ page }) => {
    const assistant = new AIAssistantPage(page);
    await assistant.goto();

    await test.step('User describes their problem', async () => {
      await assistant.sendMessage(
        "I'm having trouble with uneven coating. The sensitizer seems to pool in some areas and leave other areas thin. What am I doing wrong?"
      );
      await assistant.verifyMessageSent('uneven coating');
    });

    await test.step('User waits for troubleshooting advice', async () => {
      await assistant.waitForResponse();
      // In real scenario, would verify response contains relevant advice
    });

    await test.step('User asks for clarification', async () => {
      await assistant.sendMessage('Should I change my brush technique?');
      await assistant.verifyMessageSent('brush technique');
      await assistant.waitForResponse();
    });

    await test.step('User thanks the assistant', async () => {
      await assistant.sendMessage('Thanks, that helps!');
      await assistant.verifyMessageSent('Thanks');
    });
  });

  test('user asks about specific chemistry adjustments', async ({ page }) => {
    const assistant = new AIAssistantPage(page);
    await assistant.goto();

    await test.step('User asks about warm tones', async () => {
      await assistant.sendMessage(
        'How do I achieve warmer tones in my prints? Currently getting neutral but want more warmth.'
      );
      await assistant.verifyMessageSent('warmer tones');
      await assistant.waitForResponse();
    });

    await test.step('User asks about specific ratio', async () => {
      await assistant.sendMessage('What palladium to platinum ratio should I use for warm tones?');
      await assistant.verifyMessageSent('palladium to platinum');
      await assistant.waitForResponse();
    });
  });
});

test.describe('User Journey: Learning from AI', () => {
  test('new user learns about platinum printing basics', async ({ page }) => {
    const assistant = new AIAssistantPage(page);
    await assistant.goto();

    const learningQuestions = [
      "I'm new to platinum printing. What are the essential supplies I need?",
      'What safety precautions should I take?',
      'How long does the whole process typically take?',
    ];

    for (const question of learningQuestions) {
      await test.step(`User asks: ${question.substring(0, 40)}...`, async () => {
        await assistant.sendMessage(question);
        await assistant.waitForResponse();
      });
    }

    await test.step('User has learned from the conversation', async () => {
      const messages = await assistant.getMessages();
      expect(messages.length).toBeGreaterThanOrEqual(learningQuestions.length);
    });
  });
});
