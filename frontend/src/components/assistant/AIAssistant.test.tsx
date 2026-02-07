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

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
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

// Mock API
vi.mock('@/api/client', () => ({
    api: {
        chat: {
            send: vi.fn().mockResolvedValue({ response: 'Mock AI response' }),
            recipeHelp: vi.fn(),
            troubleshoot: vi.fn(),
        },
    },
}));

// Store mock
const mockChatState = {
    messages: [] as Array<{ id: string; role: string; content: string; timestamp: string }>,
    contexts: [],
    selectedContextIds: [],
    isLoading: false,
    isStreaming: false,
    streamContent: '',
    error: null as string | null,
    conversationId: null,
    addMessage: vi.fn(),
    updateMessage: vi.fn(),
    deleteMessage: vi.fn(),
    clearMessages: vi.fn(),
    addContext: vi.fn(),
    removeContext: vi.fn(),
    toggleContextSelection: vi.fn(),
    clearContexts: vi.fn(),
    startStreaming: vi.fn(),
    appendStreamContent: vi.fn(),
    finishStreaming: vi.fn(),
    cancelStreaming: vi.fn(),
    setLoading: vi.fn(),
    setError: vi.fn(),
    startNewConversation: vi.fn(),
    resetChat: vi.fn(),
};

vi.mock('@/stores', () => ({
    useStore: (selector: (state: any) => any) =>
        selector({ chat: mockChatState }),
}));

describe('AIAssistant', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockChatState.messages = [];
        mockChatState.isLoading = false;
        mockChatState.isStreaming = false;
        mockChatState.streamContent = '';
        mockChatState.error = null;
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
            mockChatState.messages = [
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
            mockChatState.messages = [
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
            mockChatState.messages = [
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
            mockChatState.messages = [
                { id: 'msg-1', role: 'user', content: 'Q1', timestamp: '2026-02-07T10:00:00Z' },
                { id: 'msg-2', role: 'assistant', content: 'A1', timestamp: '2026-02-07T10:00:01Z' },
                { id: 'msg-3', role: 'user', content: 'Q2', timestamp: '2026-02-07T10:00:02Z' },
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

        it('calls addMessage on form submit', () => {
            render(<AIAssistant />);
            fireEvent.change(screen.getByTestId('chat-input'), {
                target: { value: 'Test message' },
            });
            fireEvent.submit(screen.getByTestId('chat-form'));
            expect(mockChatState.addMessage).toHaveBeenCalledWith({
                role: 'user',
                content: 'Test message',
            });
        });

        it('disables input while loading', () => {
            mockChatState.isLoading = true;
            render(<AIAssistant />);
            expect(screen.getByTestId('chat-input')).toBeDisabled();
        });

        it('disables send button while loading', () => {
            mockChatState.isLoading = true;
            render(<AIAssistant />);
            expect(screen.getByTestId('send-btn')).toBeDisabled();
        });
    });

    describe('Loading & Streaming', () => {
        it('shows loading indicator when loading without stream content', () => {
            mockChatState.isLoading = true;
            render(<AIAssistant />);
            expect(screen.getByTestId('loading-indicator')).toBeInTheDocument();
        });

        it('shows streaming message with content', () => {
            mockChatState.isStreaming = true;
            mockChatState.streamContent = 'Typing...';
            render(<AIAssistant />);
            expect(screen.getByTestId('streaming-message')).toBeInTheDocument();
            expect(screen.getByText('Typing...')).toBeInTheDocument();
        });

        it('does not show loading indicator when stream has content', () => {
            mockChatState.isStreaming = true;
            mockChatState.streamContent = 'Typing...';
            render(<AIAssistant />);
            expect(screen.queryByTestId('loading-indicator')).not.toBeInTheDocument();
        });
    });

    describe('Error State', () => {
        it('shows error message', () => {
            mockChatState.error = 'Something went wrong';
            render(<AIAssistant />);
            expect(screen.getByTestId('chat-error')).toBeInTheDocument();
            expect(screen.getByText('Something went wrong')).toBeInTheDocument();
        });

        it('error has alert role for accessibility', () => {
            mockChatState.error = 'Error!';
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
            mockChatState.messages = [
                { id: 'msg-1', role: 'user', content: 'Hi', timestamp: '2026-02-07T10:00:00Z' },
            ];
            render(<AIAssistant />);
            expect(screen.getByTestId('clear-chat-btn')).toBeInTheDocument();
        });

        it('hides clear button when no messages', () => {
            render(<AIAssistant />);
            expect(screen.queryByTestId('clear-chat-btn')).not.toBeInTheDocument();
        });

        it('calls clearMessages on clear click', () => {
            mockChatState.messages = [
                { id: 'msg-1', role: 'user', content: 'Hi', timestamp: '2026-02-07T10:00:00Z' },
            ];
            render(<AIAssistant />);
            fireEvent.click(screen.getByTestId('clear-chat-btn'));
            expect(mockChatState.clearMessages).toHaveBeenCalledTimes(1);
        });

        it('shows new conversation button', () => {
            render(<AIAssistant />);
            expect(screen.getByTestId('new-conversation-btn')).toBeInTheDocument();
        });

        it('calls startNewConversation on new chat click', () => {
            render(<AIAssistant />);
            fireEvent.click(screen.getByTestId('new-conversation-btn'));
            expect(mockChatState.startNewConversation).toHaveBeenCalledTimes(1);
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
