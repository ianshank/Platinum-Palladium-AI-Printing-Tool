/**
 * API client configuration.
 * Provides a configured Axios instance with interceptors.
 */

import axios, {
  AxiosError,
  AxiosInstance,
  InternalAxiosRequestConfig,
} from 'axios';
import { apiConfig } from '@/config/api.config';
import { env } from '@/config/env';

/**
 * Error response from the API
 */
interface ApiErrorResponse {
  detail?: string;
  message?: string;
  errors?: Record<string, string[]>;
}

/**
 * Custom error class for API errors
 */
export class ApiError extends Error {
  public statusCode: number;
  public errors?: Record<string, string[]>;

  constructor(message: string, statusCode: number, errors?: Record<string, string[]>) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.errors = errors;
  }
}

/**
 * Create the API client instance
 */
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: apiConfig.baseUrl,
    timeout: apiConfig.timeout,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor
  client.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      // Add auth token if available (for future use)
      const token = localStorage.getItem('auth_token');
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }

      // Log requests in debug mode
      if (env.VITE_DEBUG_MODE) {
        console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data);
      }

      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor
  client.interceptors.response.use(
    (response) => {
      // Log responses in debug mode
      if (env.VITE_DEBUG_MODE) {
        console.debug(`[API] Response ${response.status}`, response.data);
      }
      return response;
    },
    (error: AxiosError<ApiErrorResponse>) => {
      // Extract error message
      const message =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        'An unexpected error occurred';

      // Create structured error
      const apiError = new ApiError(
        message,
        error.response?.status || 500,
        error.response?.data?.errors
      );

      // Log errors
      if (env.VITE_DEBUG_MODE) {
        console.error('[API Error]', {
          url: error.config?.url,
          status: error.response?.status,
          message,
        });
      }

      return Promise.reject(apiError);
    }
  );

  return client;
};

/**
 * Singleton API client instance
 */
export const apiClient = createApiClient();

/**
 * Helper function for file uploads with multipart/form-data
 */
export const uploadFile = async (
  url: string,
  file: File,
  additionalData?: Record<string, string>
) => {
  const formData = new FormData();
  formData.append('file', file);

  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      formData.append(key, value);
    });
  }

  return apiClient.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

/**
 * Helper function for downloading files
 */
export const downloadFile = async (url: string, filename: string) => {
  const response = await apiClient.get(url, {
    responseType: 'blob',
  });

  const blob = new Blob([response.data]);
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;

  // Temporarily add to DOM for cross-browser reliability
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(link.href);
};
