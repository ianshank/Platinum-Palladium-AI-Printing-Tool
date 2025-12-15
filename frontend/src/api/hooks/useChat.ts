/**
 * TanStack Query hooks for AI chat operations.
 */

import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../client';
import { apiConfig } from '@/config/api.config';
import type { ChatResponse } from '@/types';

/**
 * Send a chat message
 */
export const useChat = () => {
  return useMutation({
    mutationFn: async ({
      message,
      includeHistory = true,
    }: {
      message: string;
      includeHistory?: boolean;
    }): Promise<string> => {
      const { data } = await apiClient.post<ChatResponse>(
        apiConfig.endpoints.chat.message,
        {
          message,
          include_history: includeHistory,
        }
      );
      return data.response;
    },
  });
};

/**
 * Get recipe suggestion from AI
 */
export const useSuggestRecipe = () => {
  return useMutation({
    mutationFn: async ({
      paperType,
      characteristics,
    }: {
      paperType: string;
      characteristics: string;
    }): Promise<string> => {
      const { data } = await apiClient.post<ChatResponse>(
        apiConfig.endpoints.chat.recipe,
        {
          paper_type: paperType,
          characteristics,
        }
      );
      return data.response;
    },
  });
};

/**
 * Get troubleshooting help from AI
 */
export const useTroubleshoot = () => {
  return useMutation({
    mutationFn: async (problem: string): Promise<string> => {
      const { data } = await apiClient.post<ChatResponse>(
        apiConfig.endpoints.chat.troubleshoot,
        { problem }
      );
      return data.response;
    },
  });
};

/**
 * Chat message type
 */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    curveId?: string;
    paperType?: string;
    confidence?: number;
  };
}

/**
 * Quick prompts for common questions
 */
export const quickPrompts = [
  {
    id: 'exposure',
    label: 'Exposure Help',
    prompt: 'What exposure time should I use for an 8x10 print with Arches Platine?',
  },
  {
    id: 'contrast',
    label: 'Contrast Issues',
    prompt: 'My prints are too flat. How can I increase contrast?',
  },
  {
    id: 'highlights',
    label: 'Highlight Detail',
    prompt: 'I am losing highlight detail. What adjustments should I make?',
  },
  {
    id: 'chemistry',
    label: 'Chemistry Mixing',
    prompt: 'What is the recommended metal ratio for warm tones?',
  },
  {
    id: 'linearization',
    label: 'Linearization',
    prompt: 'How do I create a linearization curve for my new paper?',
  },
  {
    id: 'troubleshoot',
    label: 'General Troubleshoot',
    prompt: 'My prints have uneven tones. What could be causing this?',
  },
] as const;
