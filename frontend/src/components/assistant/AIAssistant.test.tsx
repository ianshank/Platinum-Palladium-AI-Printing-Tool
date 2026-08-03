/**
 * AIAssistant Component Tests
 *
 * Covers:
 * - Empty state with suggestions
 * - Message rendering (user and assistant bubbles)
 * - Input form and send behaviour
 * - Loading and streaming indicators
 * - Error display
 * - New conversation and clear actions
 * - Accessibility
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, screen } from '@testing-library/react';
import { renderWithProviders, userEvent } from '@/test-utils';
import { AIAssistant } from './AIAssistant';

// Mock logger
vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
  },
}));

// --- Mock useChat hook ---
const mockUseChat = {
  messages: [] as Array<{
    id: string;
    role: string;
    content: string;
    timestamp: string;
  }>,
  isLoading: false,
  isStreaming: false,
  streamContent: '',
  error: null as string | null,
  isBusy: false,
  sendSuggestion: vi.fn(),
  clear: vi.fn(),
  newConversation: vi.fn(),
};

vi.mock('@/hooks/useChat', () => ({
  useChat: () => mockUseChat,
}));

// --- Mock API hooks (useRecipeSuggestion, useTroubleshootRequest) ---
const mockRequestRecipe = vi.fn();
const mockRequestTroubleshoot = vi.fn();

vi.mock('@/api/hooks', () => ({
  useRecipeSuggestion: () => ({ mutate: mockRequestRecipe, isPending: false }),
  useTroubleshootRequest: () => ({ mutate: mockRequestTroubleshoot, isPending: false }),
}));

describe('AIAssistant', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseChat.messages = [];
    mockUseChat.isLoading = false;
    mockUseChat.isStreaming = false;
    mockUseChat.streamContent = '';
    mockUseChat.error = null;
    mockUseChat.isBusy = false;
  });

  // Helper: use renderWithProviders so Zustand store and QueryClient are available
  const renderAssistant = (props?: Partial<{ className: string }>) =>
    renderWithProviders(<AIAssistant {...props} />);

  describe('Empty State', () => {
    it('renders the assistant container', () => {
      renderAssistant();
      expect(screen.getByTestId('ai-assistant')).toBeInTheDocument();
    });

    it('shows empty state when no messages', () => {
      renderAssistant();
      expect(screen.getByTestId('empty-chat')).toBeInTheDocument();
    });

    it('displays suggestion buttons', () => {
      renderAssistant();
      const suggestions = screen.getAllByTestId('suggestion-btn');
      expect(suggestions.length).toBe(4);
    });

    it('shows greeting text', () => {
      renderAssistant();
      expect(screen.getByText('How can I help?')).toBeInTheDocument();
    });
  });

  describe('Messages', () => {
    it('renders user messages', () => {
      mockUseChat.messages = [
        {
          id: 'msg-1',
          role: 'user',
          content: 'Hello',
          timestamp: '2026-02-07T10:00:00Z',
        },
      ];

      renderAssistant();
      expect(screen.getByTestId('message-msg-1')).toBeInTheDocument();
      expect(screen.getByText('Hello')).toBeInTheDocument();
    });

    it('renders assistant messages', () => {
      mockUseChat.messages = [
        {
          id: 'msg-2',
          role: 'assistant',
          content: 'Hi! How can I help?',
          timestamp: '2026-02-07T10:00:05Z',
        },
      ];

      renderAssistant();
      expect(screen.getByTestId('message-msg-2')).toBeInTheDocument();
      expect(screen.getByText('Hi! How can I help?')).toBeInTheDocument();
    });

    it('hides empty state when messages exist', () => {
      mockUseChat.messages = [
        {
          id: 'msg-1',
          role: 'user',
          content: 'Test',
          timestamp: '2026-02-07T10:00:00Z',
        },
      ];

      renderAssistant();
      expect(screen.queryByTestId('empty-chat')).not.toBeInTheDocument();
    });

    it('renders multiple messages', () => {
      mockUseChat.messages = [
        {
          id: 'msg-1',
          role: 'user',
          content: 'Q1',
          timestamp: '2026-02-07T10:00:00Z',
        },
        {
          id: 'msg-2',
          role: 'assistant',
          content: 'A1',
          timestamp: '2026-02-07T10:00:01Z',
        },
        {
          id: 'msg-3',
          role: 'user',
          content: 'Q2',
          timestamp: '2026-02-07T10:00:02Z',
        },
      ];

      renderAssistant();
      expect(screen.getByTestId('message-msg-1')).toBeInTheDocument();
      expect(screen.getByTestId('message-msg-2')).toBeInTheDocument();
      expect(screen.getByTestId('message-msg-3')).toBeInTheDocument();
    });
  });

  describe('Input & Send', () => {
    it('renders chat input', () => {
      renderAssistant();
      expect(screen.getByTestId('chat-input')).toBeInTheDocument();
    });

    it('renders send button', () => {
      renderAssistant();
      expect(screen.getByTestId('send-btn')).toBeInTheDocument();
    });

    it('send button is disabled when input is empty', () => {
      renderAssistant();
      expect(screen.getByTestId('send-btn')).toBeDisabled();
    });

    it('send button is enabled when input has text', () => {
      renderAssistant();
      fireEvent.change(screen.getByTestId('chat-input'), {
        target: { value: 'Test message' },
      });
      expect(screen.getByTestId('send-btn')).not.toBeDisabled();
    });

    it('calls sendSuggestion on form submit', () => {
      renderAssistant();
      fireEvent.change(screen.getByTestId('chat-input'), {
        target: { value: 'Test message' },
      });
      fireEvent.submit(screen.getByTestId('chat-form'));
      expect(mockUseChat.sendSuggestion).toHaveBeenCalledWith('Test message');
    });

    it('clears input after send', () => {
      renderAssistant();
      const input = screen.getByTestId('chat-input');
      fireEvent.change(input, { target: { value: 'Test message' } });
      fireEvent.submit(screen.getByTestId('chat-form'));
      expect(input).toHaveValue('');
    });

    it('disables input while busy', () => {
      mockUseChat.isBusy = true;
      renderAssistant();
      expect(screen.getByTestId('chat-input')).toBeDisabled();
    });

    it('disables send button while busy', () => {
      mockUseChat.isBusy = true;
      renderAssistant();
      expect(screen.getByTestId('send-btn')).toBeDisabled();
    });

    it('sends on Enter key', () => {
      renderAssistant();
      const input = screen.getByTestId('chat-input');
      fireEvent.change(input, { target: { value: 'Enter test' } });
      fireEvent.keyDown(input, { key: 'Enter', shiftKey: false });
      expect(mockUseChat.sendSuggestion).toHaveBeenCalledWith('Enter test');
    });

    it('does not send on Shift+Enter', () => {
      renderAssistant();
      const input = screen.getByTestId('chat-input');
      fireEvent.change(input, { target: { value: 'Shift test' } });
      fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
      expect(mockUseChat.sendSuggestion).not.toHaveBeenCalled();
    });

    it('does not send empty input', () => {
      renderAssistant();
      fireEvent.submit(screen.getByTestId('chat-form'));
      expect(mockUseChat.sendSuggestion).not.toHaveBeenCalled();
    });

    it('does not send while busy', () => {
      mockUseChat.isBusy = true;
      renderAssistant();
      fireEvent.change(screen.getByTestId('chat-input'), {
        target: { value: 'Busy test' },
      });
      fireEvent.submit(screen.getByTestId('chat-form'));
      expect(mockUseChat.sendSuggestion).not.toHaveBeenCalled();
    });
  });

  describe('Suggestions', () => {
    it('calls sendSuggestion when clicking a suggestion', () => {
      renderAssistant();
      const suggestions = screen.getAllByTestId('suggestion-btn');
      fireEvent.click(suggestions[0]!);
      expect(mockUseChat.sendSuggestion).toHaveBeenCalledWith(
        'What is the ideal Pt/Pd metal ratio for a warm-tone print?'
      );
    });
  });

  describe('Loading & Streaming', () => {
    it('shows loading indicator when busy without stream content', () => {
      mockUseChat.isBusy = true;
      renderAssistant();
      expect(screen.getByTestId('loading-indicator')).toBeInTheDocument();
    });

    it('shows streaming message with content', () => {
      mockUseChat.isStreaming = true;
      mockUseChat.streamContent = 'Typing...';
      renderAssistant();
      expect(screen.getByTestId('streaming-message')).toBeInTheDocument();
      expect(screen.getByText('Typing...')).toBeInTheDocument();
    });

    it('does not show loading indicator when stream has content', () => {
      mockUseChat.isStreaming = true;
      mockUseChat.isBusy = true;
      mockUseChat.streamContent = 'Typing...';
      renderAssistant();
      expect(screen.queryByTestId('loading-indicator')).not.toBeInTheDocument();
    });

    it('hides empty state when busy', () => {
      mockUseChat.isBusy = true;
      renderAssistant();
      expect(screen.queryByTestId('empty-chat')).not.toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('shows error message', () => {
      mockUseChat.error = 'Something went wrong';
      renderAssistant();
      expect(screen.getByTestId('chat-error')).toBeInTheDocument();
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('error has alert role for accessibility', () => {
      mockUseChat.error = 'Error!';
      renderAssistant();
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('does not show error when null', () => {
      renderAssistant();
      expect(screen.queryByTestId('chat-error')).not.toBeInTheDocument();
    });
  });

  describe('Actions', () => {
    it('shows clear button when messages exist', () => {
      mockUseChat.messages = [
        {
          id: 'msg-1',
          role: 'user',
          content: 'Hi',
          timestamp: '2026-02-07T10:00:00Z',
        },
      ];
      renderAssistant();
      expect(screen.getByTestId('clear-chat-btn')).toBeInTheDocument();
    });

    it('hides clear button when no messages', () => {
      renderAssistant();
      expect(screen.queryByTestId('clear-chat-btn')).not.toBeInTheDocument();
    });

    it('calls clear on clear click', () => {
      mockUseChat.messages = [
        {
          id: 'msg-1',
          role: 'user',
          content: 'Hi',
          timestamp: '2026-02-07T10:00:00Z',
        },
      ];
      renderAssistant();
      fireEvent.click(screen.getByTestId('clear-chat-btn'));
      expect(mockUseChat.clear).toHaveBeenCalledTimes(1);
    });

    it('shows new conversation button', () => {
      renderAssistant();
      expect(screen.getByTestId('new-conversation-btn')).toBeInTheDocument();
    });

    it('calls newConversation on new chat click', () => {
      renderAssistant();
      fireEvent.click(screen.getByTestId('new-conversation-btn'));
      expect(mockUseChat.newConversation).toHaveBeenCalledTimes(1);
    });
  });

  describe('Accessibility', () => {
    it('messages area has log role', () => {
      renderAssistant();
      expect(screen.getByRole('log')).toBeInTheDocument();
    });

    it('messages area has aria-label', () => {
      renderAssistant();
      expect(screen.getByTestId('messages-area')).toHaveAttribute(
        'aria-label',
        'Chat messages'
      );
    });

    it('chat input has aria-label', () => {
      renderAssistant();
      expect(screen.getByTestId('chat-input')).toHaveAttribute(
        'aria-label',
        'Chat message'
      );
    });

    it('send button has aria-label', () => {
      renderAssistant();
      expect(screen.getByTestId('send-btn')).toHaveAttribute(
        'aria-label',
        'Send message'
      );
    });
  });

  describe('Customization', () => {
    it('applies custom className', () => {
      renderAssistant({ className: 'my-custom-class' });
      expect(screen.getByTestId('ai-assistant')).toHaveClass('my-custom-class');
    });
  });

  describe('AI Quick Actions', () => {
    it('renders context panel with chemistry state values', () => {
      renderAssistant();
      expect(screen.getByTestId('context-panel')).toBeInTheDocument();
      // Default paper size from store (first standard size: 4x5) and default metalRatio (0.5)
      expect(screen.getByTestId('context-paper')).toBeInTheDocument();
      expect(screen.getByTestId('context-ratio')).toBeInTheDocument();
      expect(screen.getByTestId('context-calibrations')).toBeInTheDocument();
    });

    it('renders Get Recipe and Troubleshoot buttons', () => {
      renderAssistant();
      expect(screen.getByTestId('recipe-btn')).toBeInTheDocument();
      expect(screen.getByTestId('troubleshoot-btn')).toBeInTheDocument();
    });

    it('calls useRecipeSuggestion when Get Recipe is clicked', async () => {
      renderAssistant();
      await userEvent.click(screen.getByTestId('recipe-btn'));
      expect(mockRequestRecipe).toHaveBeenCalledTimes(1);
      const callArg = mockRequestRecipe.mock.calls[0]?.[0] as {
        paper_type: string;
        characteristics: string;
      };
      expect(typeof callArg.paper_type).toBe('string');
      expect(callArg.characteristics).toMatch(/Pt \/ \d+% Pd/);
    });

    it('calls useTroubleshootRequest when Troubleshoot is clicked with input text', async () => {
      renderAssistant();
      const input = screen.getByTestId('chat-input');
      await userEvent.type(input, 'Highlights are too bright');
      await userEvent.click(screen.getByTestId('troubleshoot-btn'));
      expect(mockRequestTroubleshoot).toHaveBeenCalledWith({
        problem: 'Highlights are too bright',
      });
    });
  });
});
