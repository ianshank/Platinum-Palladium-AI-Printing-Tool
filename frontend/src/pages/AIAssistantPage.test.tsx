/**
 * AIAssistantPage Tests
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AIAssistantPage } from './AIAssistantPage';

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
});
