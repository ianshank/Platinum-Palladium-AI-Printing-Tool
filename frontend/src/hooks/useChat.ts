/**
 * useChat hook
 *
 * Orchestrates AI chat interactions by composing:
 *  - `useSendMessage` (TanStack mutation → api.chat.send)
 *  - Zustand chat slice (messages, streaming, error state)
 *
 * The hook owns the send logic and exposes a minimal surface for UI components:
 *  { messages, isLoading, isStreaming, streamContent, error, isBusy,
 *    sendSuggestion, clear, newConversation }
 *
 * No hardcoded values — retry/stale logic comes from TanStack defaults.
 */

import { useCallback } from 'react';
import { useStore } from '@/stores';
import { useSendMessage } from '@/api/hooks';
import { logger } from '@/lib/logger';
import type { ChatMessage } from '@/stores/slices/chatSlice';

/**
 * Return type of the useChat hook — kept explicit so consumers
 * can type-narrow without importing internals.
 */
export interface UseChatReturn {
    /** Full message history from the store */
    messages: ChatMessage[];
    /** True while the mutation is in flight */
    isLoading: boolean;
    /** True while streaming content is being appended */
    isStreaming: boolean;
    /** Partial content received so far during streaming */
    streamContent: string;
    /** Latest error message, if any */
    error: string | null;
    /** Convenience: isLoading || isStreaming */
    isBusy: boolean;
    /** Send an arbitrary text (user input or suggestion chip) */
    sendSuggestion: (text: string) => void;
    /** Clear all messages */
    clear: () => void;
    /** Start a brand-new conversation (clears + new ID) */
    newConversation: () => void;
}

/**
 * Custom hook for AI chat orchestration.
 */
export function useChat(): UseChatReturn {
    // --- Zustand selectors (stable references) ---
    const messages = useStore((s) => s.chat.messages);
    const isLoading = useStore((s) => s.chat.isLoading);
    const isStreaming = useStore((s) => s.chat.isStreaming);
    const streamContent = useStore((s) => s.chat.streamContent);
    const error = useStore((s) => s.chat.error);

    const startStreaming = useStore((s) => s.chat.startStreaming);
    const appendStreamContent = useStore((s) => s.chat.appendStreamContent);
    const finishStreaming = useStore((s) => s.chat.finishStreaming);
    const clearMessages = useStore((s) => s.chat.clearMessages);
    const startNewConversation = useStore((s) => s.chat.startNewConversation);
    const addToast = useStore((s) => s.ui.addToast);

    // --- TanStack mutation ---
    const sendMutation = useSendMessage({
        onSuccess: (data) => {
            // Simulate streaming for non-streaming endpoint:
            // append full response, then finish.
            startStreaming();
            if (data.response) {
                appendStreamContent(data.response);
            }
            finishStreaming();
            logger.debug('useChat: Response received');
        },
        onError: (err) => {
            addToast({
                title: 'AI Error',
                description: err.response?.data?.message ?? err.message,
                variant: 'error',
            });
            logger.error('useChat: Send failed', { error: err.message });
        },
    });

    const sendSuggestion = useCallback(
        (text: string) => {
            const trimmed = text.trim();
            if (!trimmed || isLoading || isStreaming) return;

            logger.debug('useChat: Sending message', { length: trimmed.length });
            sendMutation.mutate({ message: trimmed });
        },
        [isLoading, isStreaming, sendMutation]
    );

    const clear = useCallback(() => {
        logger.debug('useChat: Clearing messages');
        clearMessages();
    }, [clearMessages]);

    const newConversation = useCallback(() => {
        logger.debug('useChat: Starting new conversation');
        startNewConversation();
    }, [startNewConversation]);

    return {
        messages,
        isLoading,
        isStreaming,
        streamContent,
        error,
        isBusy: isLoading || isStreaming,
        sendSuggestion,
        clear,
        newConversation,
    };
}
