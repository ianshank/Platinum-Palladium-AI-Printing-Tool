/**
 * useChat Hook Tests
 *
 * Covers:
 * - Initial state (empty messages, not busy)
 * - sendSuggestion flow (calls useSendMessage mutation)
 * - Guard: does not send empty or whitespace-only text
 * - Guard: does not send while loading/streaming
 * - clear delegates to store.clearMessages
 * - newConversation delegates to store.startNewConversation
 * - isBusy computed property
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useChat } from './useChat';

// --- Mock dependencies ---

vi.mock('@/lib/logger', () => ({
    logger: {
        debug: vi.fn(),
        info: vi.fn(),
        error: vi.fn(),
        warn: vi.fn(),
    },
}));

const mockMutate = vi.fn();

vi.mock('@/api/hooks', () => ({
    useSendMessage: (opts: Record<string, unknown>) => {
        // Store options for testing callbacks
        (globalThis as any).__sendMessageOpts = opts;
        return { mutate: mockMutate };
    },
}));

// Store mock state
const mockChatState = {
    messages: [] as Array<{ id: string; role: string; content: string; timestamp: string }>,
    isLoading: false,
    isStreaming: false,
    streamContent: '',
    error: null as string | null,
    conversationId: null as string | null,
    startStreaming: vi.fn(),
    appendStreamContent: vi.fn(),
    finishStreaming: vi.fn(),
    clearMessages: vi.fn(),
    startNewConversation: vi.fn(),
};

const mockUIState = {
    addToast: vi.fn(),
};

vi.mock('@/stores', () => ({
    useStore: (selector: (state: any) => any) =>
        selector({ chat: mockChatState, ui: mockUIState }),
}));

describe('useChat', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockChatState.messages = [];
        mockChatState.isLoading = false;
        mockChatState.isStreaming = false;
        mockChatState.streamContent = '';
        mockChatState.error = null;
    });

    it('returns initial state correctly', () => {
        const { result } = renderHook(() => useChat());

        expect(result.current.messages).toEqual([]);
        expect(result.current.isLoading).toBe(false);
        expect(result.current.isStreaming).toBe(false);
        expect(result.current.streamContent).toBe('');
        expect(result.current.error).toBeNull();
        expect(result.current.isBusy).toBe(false);
    });

    it('sendSuggestion calls mutate with trimmed text', () => {
        const { result } = renderHook(() => useChat());

        act(() => {
            result.current.sendSuggestion('  Hello world  ');
        });

        expect(mockMutate).toHaveBeenCalledWith({ message: 'Hello world' });
    });

    it('sendSuggestion does not send empty string', () => {
        const { result } = renderHook(() => useChat());

        act(() => {
            result.current.sendSuggestion('');
        });

        expect(mockMutate).not.toHaveBeenCalled();
    });

    it('sendSuggestion does not send whitespace-only string', () => {
        const { result } = renderHook(() => useChat());

        act(() => {
            result.current.sendSuggestion('   ');
        });

        expect(mockMutate).not.toHaveBeenCalled();
    });

    it('sendSuggestion does not send while loading', () => {
        mockChatState.isLoading = true;
        const { result } = renderHook(() => useChat());

        act(() => {
            result.current.sendSuggestion('Hello');
        });

        expect(mockMutate).not.toHaveBeenCalled();
    });

    it('sendSuggestion does not send while streaming', () => {
        mockChatState.isStreaming = true;
        const { result } = renderHook(() => useChat());

        act(() => {
            result.current.sendSuggestion('Hello');
        });

        expect(mockMutate).not.toHaveBeenCalled();
    });

    it('clear delegates to store clearMessages', () => {
        const { result } = renderHook(() => useChat());

        act(() => {
            result.current.clear();
        });

        expect(mockChatState.clearMessages).toHaveBeenCalledTimes(1);
    });

    it('newConversation delegates to store startNewConversation', () => {
        const { result } = renderHook(() => useChat());

        act(() => {
            result.current.newConversation();
        });

        expect(mockChatState.startNewConversation).toHaveBeenCalledTimes(1);
    });

    it('isBusy is true when loading', () => {
        mockChatState.isLoading = true;
        const { result } = renderHook(() => useChat());
        expect(result.current.isBusy).toBe(true);
    });

    it('isBusy is true when streaming', () => {
        mockChatState.isStreaming = true;
        const { result } = renderHook(() => useChat());
        expect(result.current.isBusy).toBe(true);
    });

    it('isBusy is false when neither loading nor streaming', () => {
        const { result } = renderHook(() => useChat());
        expect(result.current.isBusy).toBe(false);
    });

    it('reflects messages from store', () => {
        const msgs = [
            { id: 'msg-1', role: 'user', content: 'Hi', timestamp: '2026-02-07T10:00:00Z' },
        ];
        mockChatState.messages = msgs;

        const { result } = renderHook(() => useChat());
        expect(result.current.messages).toBe(msgs);
    });

    it('reflects error from store', () => {
        mockChatState.error = 'Something broke';
        const { result } = renderHook(() => useChat());
        expect(result.current.error).toBe('Something broke');
    });

    it('reflects streamContent from store', () => {
        mockChatState.streamContent = 'Partial response...';
        const { result } = renderHook(() => useChat());
        expect(result.current.streamContent).toBe('Partial response...');
    });
});
