/**
 * TanStack Query client configuration.
 */

import { QueryClient } from '@tanstack/react-query';
import { ApiError } from './client';

/**
 * Query client with default options
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 30 * 60 * 1000, // 30 minutes (garbage collection time, formerly cacheTime)
      retry: (failureCount, error) => {
        // Don't retry on 4xx errors (client errors)
        if (error instanceof ApiError && error.statusCode >= 400 && error.statusCode < 500) {
          return false;
        }
        // Retry up to 3 times for server errors
        return failureCount < 3;
      },
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
      onError: (error) => {
        // Global error handling for mutations
        console.error('Mutation error:', error);
      },
    },
  },
});

/**
 * Query key factory for consistent key management
 */
export const queryKeys = {
  // Health
  health: ['health'] as const,

  // Curves
  curves: {
    all: ['curves'] as const,
    lists: () => [...queryKeys.curves.all, 'list'] as const,
    list: (filters?: Record<string, unknown>) =>
      [...queryKeys.curves.lists(), filters] as const,
    details: () => [...queryKeys.curves.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.curves.details(), id] as const,
  },

  // Calibrations
  calibrations: {
    all: ['calibrations'] as const,
    lists: () => [...queryKeys.calibrations.all, 'list'] as const,
    list: (filters?: Record<string, unknown>) =>
      [...queryKeys.calibrations.lists(), filters] as const,
    details: () => [...queryKeys.calibrations.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.calibrations.details(), id] as const,
  },

  // Scan
  scan: {
    all: ['scan'] as const,
    analysis: (uploadId: string) => [...queryKeys.scan.all, 'analysis', uploadId] as const,
  },

  // Statistics
  statistics: {
    all: ['statistics'] as const,
    dashboard: () => [...queryKeys.statistics.all, 'dashboard'] as const,
    sessions: () => [...queryKeys.statistics.all, 'sessions'] as const,
  },

  // Chat
  chat: {
    all: ['chat'] as const,
    history: () => [...queryKeys.chat.all, 'history'] as const,
  },

  // Chemistry
  chemistry: {
    all: ['chemistry'] as const,
    presets: () => [...queryKeys.chemistry.all, 'presets'] as const,
    recipes: () => [...queryKeys.chemistry.all, 'recipes'] as const,
    recipe: (id: string) => [...queryKeys.chemistry.recipes(), id] as const,
  },
} as const;

/**
 * Helper to invalidate related queries
 */
export const invalidateQueries = {
  curves: () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.curves.all });
  },
  calibrations: () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.calibrations.all });
  },
  statistics: () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.statistics.all });
  },
  all: () => {
    queryClient.invalidateQueries();
  },
};
