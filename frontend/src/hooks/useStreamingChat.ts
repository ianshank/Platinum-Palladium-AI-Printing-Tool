/**
 * useStreamingChat hook
 *
 * Provides real-time SSE streaming for AI chat responses.
 * Falls back to non-streaming API if the streaming endpoint fails.
 *
 * Uses the existing Zustand chat slice for state management.
 */

import { useCallback, useRef } from 'react';
import { useStore } from '@/stores';
import { useSendMessage } from '@/api/hooks';
import { streamChatResponse } from '@/api/streaming';
import { logger } from '@/lib/logger';
import type { ChatMessage } from '@/stores/slices/chatSlice';

export interface UseStreamingChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  isStreaming: boolean;
  streamContent: string;
  error: string | null;
  isBusy: boolean;
  sendMessage: (text: string) => void;
  cancelStream: () => void;
  clear: () => void;
  newConversation: () => void;
}

export function useStreamingChat(): UseStreamingChatReturn {
  const abortControllerRef = useRef<AbortController | null>(null);

  // Zustand selectors
  const messages = useStore((s) => s.chat.messages);
  const isLoading = useStore((s) => s.chat.isLoading);
  const isStreaming = useStore((s) => s.chat.isStreaming);
  const streamContent = useStore((s) => s.chat.streamContent);
  const error = useStore((s) => s.chat.error);

  const addMessage = useStore((s) => s.chat.addMessage);
  const startStreaming = useStore((s) => s.chat.startStreaming);
  const appendStreamContent = useStore((s) => s.chat.appendStreamContent);
  const finishStreaming = useStore((s) => s.chat.finishStreaming);
  const cancelStreaming = useStore((s) => s.chat.cancelStreaming);
  const setLoading = useStore((s) => s.chat.setLoading);
  const setError = useStore((s) => s.chat.setError);
  const clearMessages = useStore((s) => s.chat.clearMessages);
  const startNewConversation = useStore((s) => s.chat.startNewConversation);
  const addToast = useStore((s) => s.ui.addToast);

  // Fallback non-streaming mutation
  const sendMutation = useSendMessage({
    onSuccess: (data) => {
      startStreaming();
      if (data.response) {
        appendStreamContent(data.response);
      }
      finishStreaming();
    },
    onError: (err) => {
      addToast({
        title: 'AI Error',
        description: err.response?.data?.message ?? err.message,
        variant: 'error',
      });
    },
  });

  const sendMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isLoading || isStreaming) return;

      // Add user message to store
      addMessage({ role: 'user', content: trimmed });
      setLoading(true);

      // Create abort controller for cancellation
      const controller = new AbortController();
      abortControllerRef.current = controller;

      logger.debug('useStreamingChat: Starting SSE stream', { length: trimmed.length });

      // Start streaming
      startStreaming();
      setLoading(false);

      streamChatResponse(trimmed, true, {
        onChunk: (content) => {
          appendStreamContent(content);
        },
        onDone: () => {
          finishStreaming();
          abortControllerRef.current = null;
          logger.debug('useStreamingChat: Stream completed');
        },
        onError: (errorMsg) => {
          cancelStreaming();
          setError(errorMsg);
          abortControllerRef.current = null;
          logger.error('useStreamingChat: Stream error', { error: errorMsg });

          // Fallback to non-streaming
          logger.info('useStreamingChat: Falling back to non-streaming API');
          sendMutation.mutate({ message: trimmed });
        },
        signal: controller.signal,
      }).catch((err: Error) => {
        if (err.name === 'AbortError') return;
        cancelStreaming();
        setError(err.message);
        abortControllerRef.current = null;

        // Fallback to non-streaming
        logger.info('useStreamingChat: Falling back to non-streaming API');
        sendMutation.mutate({ message: trimmed });
      });
    },
    [isLoading, isStreaming, addMessage, setLoading, startStreaming, appendStreamContent, finishStreaming, cancelStreaming, setError, sendMutation]
  );

  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      cancelStreaming();
      logger.debug('useStreamingChat: Stream cancelled by user');
    }
  }, [cancelStreaming]);

  const clear = useCallback(() => {
    logger.debug('useStreamingChat: Clearing messages');
    clearMessages();
  }, [clearMessages]);

  const newConversation = useCallback(() => {
    logger.debug('useStreamingChat: Starting new conversation');
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    startNewConversation();
  }, [startNewConversation]);

  return {
    messages,
    isLoading,
    isStreaming,
    streamContent,
    error,
    isBusy: isLoading || isStreaming,
    sendMessage,
    cancelStream,
    clear,
    newConversation,
  };
}
