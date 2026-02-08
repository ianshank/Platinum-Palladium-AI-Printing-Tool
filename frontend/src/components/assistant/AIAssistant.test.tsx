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
import { fireEvent, render, screen } from '@testing-library/react';
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

  describe('Empty State', () => {
    it('renders the assistant container', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('ai-assistant')).toBeInTheDocument();
    });

    it('shows empty state when no messages', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('empty-chat')).toBeInTheDocument();
    });

    it('displays suggestion buttons', () => {
      render(<AIAssistant />);
      const suggestions = screen.getAllByTestId('suggestion-btn');
      expect(suggestions.length).toBe(4);
    });

    it('shows greeting text', () => {
      render(<AIAssistant />);
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

      render(<AIAssistant />);
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

      render(<AIAssistant />);
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

      render(<AIAssistant />);
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

      render(<AIAssistant />);
      expect(screen.getByTestId('message-msg-1')).toBeInTheDocument();
      expect(screen.getByTestId('message-msg-2')).toBeInTheDocument();
      expect(screen.getByTestId('message-msg-3')).toBeInTheDocument();
    });
  });

  describe('Input & Send', () => {
    it('renders chat input', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('chat-input')).toBeInTheDocument();
    });

    it('renders send button', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('send-btn')).toBeInTheDocument();
    });

    it('send button is disabled when input is empty', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('send-btn')).toBeDisabled();
    });

    it('send button is enabled when input has text', () => {
      render(<AIAssistant />);
      fireEvent.change(screen.getByTestId('chat-input'), {
        target: { value: 'Test message' },
      });
      expect(screen.getByTestId('send-btn')).not.toBeDisabled();
    });

    it('calls sendSuggestion on form submit', () => {
      render(<AIAssistant />);
      fireEvent.change(screen.getByTestId('chat-input'), {
        target: { value: 'Test message' },
      });
      fireEvent.submit(screen.getByTestId('chat-form'));
      expect(mockUseChat.sendSuggestion).toHaveBeenCalledWith('Test message');
    });

    it('clears input after send', () => {
      render(<AIAssistant />);
      const input = screen.getByTestId('chat-input');
      fireEvent.change(input, { target: { value: 'Test message' } });
      fireEvent.submit(screen.getByTestId('chat-form'));
      expect(input).toHaveValue('');
    });

    it('disables input while busy', () => {
      mockUseChat.isBusy = true;
      render(<AIAssistant />);
      expect(screen.getByTestId('chat-input')).toBeDisabled();
    });

    it('disables send button while busy', () => {
      mockUseChat.isBusy = true;
      render(<AIAssistant />);
      expect(screen.getByTestId('send-btn')).toBeDisabled();
    });

    it('sends on Enter key', () => {
      render(<AIAssistant />);
      const input = screen.getByTestId('chat-input');
      fireEvent.change(input, { target: { value: 'Enter test' } });
      fireEvent.keyDown(input, { key: 'Enter', shiftKey: false });
      expect(mockUseChat.sendSuggestion).toHaveBeenCalledWith('Enter test');
    });

    it('does not send on Shift+Enter', () => {
      render(<AIAssistant />);
      const input = screen.getByTestId('chat-input');
      fireEvent.change(input, { target: { value: 'Shift test' } });
      fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
      expect(mockUseChat.sendSuggestion).not.toHaveBeenCalled();
    });

    it('does not send empty input', () => {
      render(<AIAssistant />);
      fireEvent.submit(screen.getByTestId('chat-form'));
      expect(mockUseChat.sendSuggestion).not.toHaveBeenCalled();
    });

    it('does not send while busy', () => {
      mockUseChat.isBusy = true;
      render(<AIAssistant />);
      fireEvent.change(screen.getByTestId('chat-input'), {
        target: { value: 'Busy test' },
      });
      fireEvent.submit(screen.getByTestId('chat-form'));
      expect(mockUseChat.sendSuggestion).not.toHaveBeenCalled();
    });
  });

  describe('Suggestions', () => {
    it('calls sendSuggestion when clicking a suggestion', () => {
      render(<AIAssistant />);
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
      render(<AIAssistant />);
      expect(screen.getByTestId('loading-indicator')).toBeInTheDocument();
    });

    it('shows streaming message with content', () => {
      mockUseChat.isStreaming = true;
      mockUseChat.streamContent = 'Typing...';
      render(<AIAssistant />);
      expect(screen.getByTestId('streaming-message')).toBeInTheDocument();
      expect(screen.getByText('Typing...')).toBeInTheDocument();
    });

    it('does not show loading indicator when stream has content', () => {
      mockUseChat.isStreaming = true;
      mockUseChat.isBusy = true;
      mockUseChat.streamContent = 'Typing...';
      render(<AIAssistant />);
      expect(screen.queryByTestId('loading-indicator')).not.toBeInTheDocument();
    });

    it('hides empty state when busy', () => {
      mockUseChat.isBusy = true;
      render(<AIAssistant />);
      expect(screen.queryByTestId('empty-chat')).not.toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('shows error message', () => {
      mockUseChat.error = 'Something went wrong';
      render(<AIAssistant />);
      expect(screen.getByTestId('chat-error')).toBeInTheDocument();
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('error has alert role for accessibility', () => {
      mockUseChat.error = 'Error!';
      render(<AIAssistant />);
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('does not show error when null', () => {
      render(<AIAssistant />);
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
      render(<AIAssistant />);
      expect(screen.getByTestId('clear-chat-btn')).toBeInTheDocument();
    });

    it('hides clear button when no messages', () => {
      render(<AIAssistant />);
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
      render(<AIAssistant />);
      fireEvent.click(screen.getByTestId('clear-chat-btn'));
      expect(mockUseChat.clear).toHaveBeenCalledTimes(1);
    });

    it('shows new conversation button', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('new-conversation-btn')).toBeInTheDocument();
    });

    it('calls newConversation on new chat click', () => {
      render(<AIAssistant />);
      fireEvent.click(screen.getByTestId('new-conversation-btn'));
      expect(mockUseChat.newConversation).toHaveBeenCalledTimes(1);
    });
  });

  describe('Accessibility', () => {
    it('messages area has log role', () => {
      render(<AIAssistant />);
      expect(screen.getByRole('log')).toBeInTheDocument();
    });

    it('messages area has aria-label', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('messages-area')).toHaveAttribute(
        'aria-label',
        'Chat messages'
      );
    });

    it('chat input has aria-label', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('chat-input')).toHaveAttribute(
        'aria-label',
        'Chat message'
      );
    });

    it('send button has aria-label', () => {
      render(<AIAssistant />);
      expect(screen.getByTestId('send-btn')).toHaveAttribute(
        'aria-label',
        'Send message'
      );
    });
  });

  describe('Customization', () => {
    it('applies custom className', () => {
      render(<AIAssistant className="my-custom-class" />);
      expect(screen.getByTestId('ai-assistant')).toHaveClass('my-custom-class');
    });
  });
});
