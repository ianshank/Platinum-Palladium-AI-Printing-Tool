/**
 * AIAssistantPage Tests
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { AIAssistantPage } from './AIAssistantPage';

expect.extend(toHaveNoViolations);

vi.mock('@/components/assistant/AIAssistant', () => ({
  AIAssistant: ({ className }: { className?: string }) => (
    <div data-testid="mock-ai-assistant" className={className}>
      AIAssistant
    </div>
  ),
}));

describe('AIAssistantPage', () => {
  it('renders the page container', () => {
    render(<AIAssistantPage />);
    expect(screen.getByTestId('assistant-page')).toBeInTheDocument();
  });

  it('renders the AIAssistant component', () => {
    render(<AIAssistantPage />);
    expect(screen.getByTestId('mock-ai-assistant')).toBeInTheDocument();
  });

  it('passes h-full className to AIAssistant', () => {
    render(<AIAssistantPage />);
    expect(screen.getByTestId('mock-ai-assistant')).toHaveClass('h-full');
  });

  describe('Accessibility', () => {
    it('has no accessibility violations', async () => {
      const { container } = render(<AIAssistantPage />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
});
