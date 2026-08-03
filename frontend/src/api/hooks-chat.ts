/**
 * Chat and LLM API hooks
 * Handles messaging, recipe suggestions, and troubleshooting
 */

import {
  useMutation,
  type UseMutationOptions,
} from '@tanstack/react-query';
import { api, type ApiError, type AxiosError } from './client';
import type { ChatResponse } from '@/types/models';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';

// ============================================================================
// Chat Messages
// ============================================================================

/**
 * Send a message to the LLM assistant
 */
export function useSendMessage(
  options?: UseMutationOptions<
    ChatResponse,
    AxiosError<ApiError>,
    { message: string; context?: string[] }
  >
) {
  const addMessage = useStore((state) => state.chat.addMessage);
  const setLoading = useStore((state) => state.chat.setLoading);
  const setError = useStore((state) => state.chat.setError);

  return useMutation({
    mutationFn: (data) => {
      addMessage({ role: 'user', content: data.message });
      return api.chat.send(data);
    },
    onMutate: () => {
      setLoading(true);
    },
    onSuccess: (data) => {
      logger.info('Chat message sent', {
        responseLength: data.response.length,
      });
      addMessage({
        role: 'assistant',
        content: data.response,
        metadata: { context: data.context_used ?? [] },
      });
    },
    onError: (error) => {
      logger.error('Chat message failed', { error: error.message });
      setError(error.response?.data?.message ?? error.message);
    },
    onSettled: () => {
      setLoading(false);
    },
    ...options,
  });
}

// ============================================================================
// Recipe Suggestions
// ============================================================================

/**
 * Get chemistry recipe suggestions based on paper type and characteristics
 */
export function useRecipeSuggestion(
  options?: UseMutationOptions<
    ChatResponse,
    AxiosError<ApiError>,
    { paper_type: string; characteristics: string }
  >
) {
  const addMessage = useStore((state) => state.chat.addMessage);
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: (data) => {
      addMessage({
        role: 'user',
        content: `Recipe for ${data.paper_type}: ${data.characteristics}`,
      });
      return api.chat.recipe(data);
    },
    onSuccess: (data) => {
      logger.info('Recipe suggestion received');
      addMessage({ role: 'assistant', content: data.response });
    },
    onError: (error) => {
      logger.error('Recipe suggestion failed', { error: error.message });
      addToast({
        title: 'Recipe Request Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

// ============================================================================
// Troubleshooting
// ============================================================================

/**
 * Get troubleshooting suggestions for common printing issues
 */
export function useTroubleshootRequest(
  options?: UseMutationOptions<
    ChatResponse,
    AxiosError<ApiError>,
    { problem: string }
  >
) {
  const addMessage = useStore((state) => state.chat.addMessage);
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: (data) => {
      addMessage({ role: 'user', content: `Troubleshoot: ${data.problem}` });
      return api.chat.troubleshoot(data);
    },
    onSuccess: (data) => {
      logger.info('Troubleshooting advice received');
      addMessage({ role: 'assistant', content: data.response });
    },
    onError: (error) => {
      logger.error('Troubleshoot request failed', { error: error.message });
      addToast({
        title: 'Troubleshoot Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}
