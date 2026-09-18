/**
 * API Client
 * Configured Axios instance with interceptors for authentication, error handling, and logging
 */

import axios, {
  type AxiosError,
  type AxiosInstance,
  type AxiosRequestConfig,
  type AxiosResponse,
} from 'axios';
import { config, isDev } from '@/config';
import {
  DEFAULT_NEGATIVE_COLOR_MODE,
  DEFAULT_NEGATIVE_FORMAT,
  DEFAULT_NEGATIVE_NAME,
} from '@/config/curves';
import type { CurveRequestBody } from '@/api/generated';
import { logger } from '@/lib/logger';
import type {
  AnalysisResponse,
  CalibrationCreateResponse,
  CalibrationListResponse,
  CalibrationRecord,
  ChatResponse,
  CurveData,
  CurveEnhanceResponse,
  CurveGenerationResponse,
  CurveModificationRequest,
  CurveModificationResponse,
  CurveSmoothingResponse,
  CurveSmoothRequest,
  EnforceMonotonicityResponse,
  QuadParseResponse,
  QuadUploadResponse,
  ScanUploadResponse,
  StatisticsResponse,
} from '@/types/models';

/** Formats `POST /api/export/negative` offers, mirroring the server's map. */
export type NegativeFormat =
  | 'tiff'
  | 'tiff_16bit'
  | 'png'
  | 'png_16bit'
  | 'jpeg'
  | 'jpeg_high';

/** Colour handling for the negative, mirroring the server's `ColorMode`. */
export type NegativeColorMode = 'grayscale' | 'rgb' | 'preserve';

/** Arguments for {@link api.negative.export}. */
export interface NegativeExportOptions {
  file: File;
  /** A stored curve to linearise with. Takes precedence over `densities`. */
  curveId?: string;
  /** Measured densities to generate a curve from, when no curve is stored. */
  densities?: number[];
  format?: NegativeFormat;
  colorMode?: NegativeColorMode;
  invert?: boolean;
  /** Download filename stem; the server sanitises it. */
  name?: string;
  onProgress?: (progress: number) => void;
}

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
    baseURL: isDev ? '' : config.api.baseUrl,
    timeout: config.api.timeout,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor for logging
  client.interceptors.request.use(
    (requestConfig) => {
      const { method, url } = requestConfig;
      const data = requestConfig.data as unknown;
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
export async function apiRequest<T>(config: AxiosRequestConfig): Promise<T> {
  const response = await apiClient.request<T>(config);
  return response.data;
}

/**
 * API endpoints namespace
 */
export const api = {
  // Health check
  health: {
    check: () =>
      apiRequest<{ status: string }>({ method: 'GET', url: '/api/health' }),
  },

  // Curves
  curves: {
    // The body type comes from the generated schema, so a field the server
    // does not read is a compile error rather than a silently ignored value.
    generate: (data: CurveRequestBody) =>
      apiRequest<CurveGenerationResponse>({
        method: 'POST',
        url: '/api/curves/generate',
        data,
      }),

    get: (id: string) =>
      apiRequest<CurveData>({
        method: 'GET',
        url: `/api/curves/${id}`,
      }),

    modify: (data: CurveModificationRequest) =>
      apiRequest<CurveModificationResponse>({
        method: 'POST',
        url: '/api/curves/modify',
        data,
      }),

    smooth: (data: CurveSmoothRequest) =>
      apiRequest<CurveSmoothingResponse>({
        method: 'POST',
        url: '/api/curves/smooth',
        data,
      }),

    enhance: (data: {
      name: string;
      input_values: number[];
      output_values: number[];
      goal: string;
      additional_context?: string;
      paper_type?: string;
    }) =>
      apiRequest<CurveEnhanceResponse>({
        method: 'POST',
        url: '/api/curves/enhance',
        data,
      }),

    enforceMonotonicity: (curveId: string, direction: string = 'increasing') =>
      apiRequest<EnforceMonotonicityResponse>({
        method: 'POST',
        url: `/api/curves/${curveId}/enforce-monotonicity`,
        params: { direction },
      }),

    export: (data: { curveId: string; format: string }) =>
      apiRequest<Blob>({
        method: 'POST',
        url: `/api/curves/${data.curveId}/export`,
        params: { format: data.format },
        responseType: 'blob',
      }),

    uploadQuad: (file: File, channel: string) => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('channel', channel);

      return apiRequest<QuadUploadResponse>({
        method: 'POST',
        url: '/api/curves/upload-quad',
        data: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    },

    // The endpoint declares Form fields, not a JSON body. Sending JSON made
    // every call fail validation with "Field required: content".
    parseQuad: (content: string, name: string, channel: string) => {
      const form = new FormData();
      form.append('content', content);
      form.append('name', name);
      form.append('channel', channel);
      return apiRequest<QuadParseResponse>({
        method: 'POST',
        url: '/api/curves/parse-quad',
        data: form,
      });
    },
  },

  // Digital negative export
  negative: {
    /**
     * Render an uploaded image as a digital negative and download the file.
     *
     * The endpoint declares Form fields and streams a file back, so the body
     * is multipart and the response is a Blob. Supplying neither `curveId`
     * nor `densities` inverts the image without linearising it, which is what
     * you want for a file that is already linearised.
     */
    export: (options: NegativeExportOptions) => {
      const form = new FormData();
      form.append('file', options.file);
      form.append('format', options.format ?? DEFAULT_NEGATIVE_FORMAT);
      form.append('invert', String(options.invert ?? true));
      form.append(
        'color_mode',
        options.colorMode ?? DEFAULT_NEGATIVE_COLOR_MODE
      );
      form.append('name', options.name ?? DEFAULT_NEGATIVE_NAME);
      if (options.curveId) {
        form.append('curve_id', options.curveId);
      }
      options.densities?.forEach((density) =>
        form.append('densities', String(density))
      );

      return apiRequest<Blob>({
        method: 'POST',
        url: '/api/export/negative',
        data: form,
        responseType: 'blob',
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
          if (options.onProgress && e.total) {
            options.onProgress(Math.round((e.loaded * 100) / e.total));
          }
        },
      });
    },
  },

  // Scan / Step Tablet
  scan: {
    upload: (
      file: File,
      tabletType: string = 'stouffer_21',
      onProgress?: (progress: number) => void
    ) => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('tablet_type', tabletType);

      return apiRequest<ScanUploadResponse>({
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
    list: (paperType?: string, limit: number = 50) =>
      apiRequest<CalibrationListResponse>({
        method: 'GET',
        url: '/api/calibrations',
        params: { paper_type: paperType, limit },
      }),

    create: (data: Omit<CalibrationRecord, 'id' | 'timestamp'>) =>
      apiRequest<CalibrationCreateResponse>({
        method: 'POST',
        url: '/api/calibrations',
        data,
      }),

    get: (id: string) =>
      apiRequest<CalibrationRecord>({
        method: 'GET',
        url: `/api/calibrations/${id}`,
      }),
  },

  // Chat
  chat: {
    send: (data: { message: string; include_history?: boolean }) =>
      apiRequest<ChatResponse>({
        method: 'POST',
        url: '/api/chat',
        data,
      }),

    recipe: (data: { paper_type: string; characteristics: string }) =>
      apiRequest<ChatResponse>({
        method: 'POST',
        url: '/api/chat/recipe',
        data,
      }),

    troubleshoot: (data: { problem: string }) =>
      apiRequest<ChatResponse>({
        method: 'POST',
        url: '/api/chat/troubleshoot',
        data,
      }),
  },

  // Statistics
  statistics: {
    get: () =>
      apiRequest<StatisticsResponse>({
        method: 'GET',
        url: '/api/statistics',
      }),
  },

  // Analyze
  analyze: {
    densities: (data: { measurements: number[] }) =>
      apiRequest<AnalysisResponse>({
        method: 'POST',
        url: '/api/analyze',
        data,
      }),
  },
};

export type { AxiosError };
