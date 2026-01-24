/**
 * API Client
 * Configured Axios instance with interceptors for authentication, error handling, and logging
 */

import axios, { type AxiosError, type AxiosInstance, type AxiosRequestConfig, type AxiosResponse } from 'axios';
import { config } from '@/config';
import { logger } from '@/lib/logger';

/**
 * API Error response structure
 */
export interface ApiError {
  message: string;
  code?: string;
  details?: Record<string, unknown>;
}

/**
 * Create configured Axios instance
 */
function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: config.api.baseUrl,
    timeout: config.api.timeout,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor for logging
  client.interceptors.request.use(
    (requestConfig) => {
      const { method, url, data } = requestConfig;
      logger.debug('API Request', {
        method: method?.toUpperCase(),
        url,
        hasData: !!data,
      });
      return requestConfig;
    },
    (error: AxiosError) => {
      logger.error('API Request Error', { error: error.message });
      return Promise.reject(error);
    }
  );

  // Response interceptor for logging and error handling
  client.interceptors.response.use(
    (response: AxiosResponse) => {
      const { config: reqConfig, status } = response;
      logger.debug('API Response', {
        method: reqConfig.method?.toUpperCase(),
        url: reqConfig.url,
        status,
      });
      return response;
    },
    async (error: AxiosError<ApiError>) => {
      const { config: reqConfig, response } = error;

      logger.error('API Response Error', {
        method: reqConfig?.method?.toUpperCase(),
        url: reqConfig?.url,
        status: response?.status,
        message: response?.data?.message ?? error.message,
      });

      // Handle specific error codes
      if (response?.status === 401) {
        // Handle unauthorized - could trigger logout
        logger.warn('Unauthorized request');
      }

      if (response?.status === 429) {
        // Handle rate limiting
        logger.warn('Rate limited');
      }

      return Promise.reject(error);
    }
  );

  return client;
}

/**
 * Singleton API client instance
 */
export const apiClient = createApiClient();

/**
 * Type-safe API request helper
 */
export async function apiRequest<T>(
  config: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.request<T>(config);
  return response.data;
}

/**
 * API endpoints namespace
 */
export const api = {
  // Health check
  health: {
    check: () => apiRequest<{ status: string }>({ method: 'GET', url: '/api/health' }),
  },

  // Curves
  curves: {
    generate: (data: { measurements: unknown[]; type?: string }) =>
      apiRequest<{ id: string; points: { x: number; y: number }[] }>({
        method: 'POST',
        url: '/api/curves/generate',
        data,
      }),

    get: (id: string) =>
      apiRequest<{ id: string; points: { x: number; y: number }[]; metadata: unknown }>({
        method: 'GET',
        url: `/api/curves/${id}`,
      }),

    modify: (data: { curveId: string; modification: string; value: number }) =>
      apiRequest<{ points: { x: number; y: number }[] }>({
        method: 'POST',
        url: '/api/curves/modify',
        data,
      }),

    smooth: (data: { curveId: string; method: string; amount: number }) =>
      apiRequest<{ points: { x: number; y: number }[] }>({
        method: 'POST',
        url: '/api/curves/smooth',
        data,
      }),

    export: (data: { curveId: string; format: string }) =>
      apiRequest<Blob>({
        method: 'POST',
        url: '/api/curves/export',
        data,
        responseType: 'blob',
      }),
  },

  // Scan / Step Tablet
  scan: {
    upload: (file: File, onProgress?: (progress: number) => void) => {
      const formData = new FormData();
      formData.append('file', file);

      return apiRequest<{ id: string; measurements: unknown[]; preview: string }>({
        method: 'POST',
        url: '/api/scan/upload',
        data: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
          if (onProgress && e.total) {
            onProgress(Math.round((e.loaded * 100) / e.total));
          }
        },
      });
    },
  },

  // Calibrations
  calibrations: {
    list: () =>
      apiRequest<{ calibrations: unknown[] }>({
        method: 'GET',
        url: '/api/calibrations',
      }),

    create: (data: unknown) =>
      apiRequest<{ id: string }>({
        method: 'POST',
        url: '/api/calibrations',
        data,
      }),

    get: (id: string) =>
      apiRequest<unknown>({
        method: 'GET',
        url: `/api/calibrations/${id}`,
      }),
  },

  // Chat
  chat: {
    send: (data: { message: string; context?: string[] }) =>
      apiRequest<{ response: string; context_used: string[] }>({
        method: 'POST',
        url: '/api/chat',
        data,
      }),

    recipe: (data: { query: string }) =>
      apiRequest<{ suggestions: unknown[] }>({
        method: 'POST',
        url: '/api/chat/recipe',
        data,
      }),

    troubleshoot: (data: { problem: string }) =>
      apiRequest<{ suggestions: unknown[] }>({
        method: 'POST',
        url: '/api/chat/troubleshoot',
        data,
      }),
  },

  // Statistics
  statistics: {
    get: () =>
      apiRequest<{ stats: unknown }>({
        method: 'GET',
        url: '/api/statistics',
      }),
  },

  // Analyze
  analyze: {
    densities: (data: { measurements: unknown[] }) =>
      apiRequest<{ analysis: unknown }>({
        method: 'POST',
        url: '/api/analyze',
        data,
      }),
  },
};

export type { AxiosError };
