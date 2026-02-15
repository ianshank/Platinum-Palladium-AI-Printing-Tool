/**
 * Tests for API client
 * Verifies Axios configuration, interceptors, and all endpoint methods
 */

/* eslint-disable @typescript-eslint/unbound-method */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { InternalAxiosRequestConfig } from 'axios';

// Use vi.hoisted so mock references are available before vi.mock factories run
const { mocks, interceptors } = vi.hoisted(() => {
  // Will hold captured interceptor handlers - using any to avoid strict optional property issues
  const captured = {} as {
    requestInterceptor: (config: InternalAxiosRequestConfig) => InternalAxiosRequestConfig;
    requestErrorInterceptor: (error: unknown) => Promise<never>;
    responseInterceptor: (response: unknown) => unknown;
    responseErrorInterceptor: (error: unknown) => Promise<never>;
  };

  const mockFns = {
    axiosCreate: vi.fn(),
    request: vi.fn(),
    requestInterceptorUse: vi.fn((success: unknown, error: unknown) => {
      captured.requestInterceptor = success as typeof captured.requestInterceptor;
      captured.requestErrorInterceptor = error as typeof captured.requestErrorInterceptor;
    }),
    responseInterceptorUse: vi.fn((success: unknown, error: unknown) => {
      captured.responseInterceptor = success as typeof captured.responseInterceptor;
      captured.responseErrorInterceptor = error as typeof captured.responseErrorInterceptor;
    }),
  };

  // Create the instance that axios.create will return
  const axiosInstance = {
    request: mockFns.request,
    interceptors: {
      request: {
        use: mockFns.requestInterceptorUse,
      },
      response: {
        use: mockFns.responseInterceptorUse,
      },
    },
  };

  // Setup axios.create to return the instance
  mockFns.axiosCreate.mockReturnValue(axiosInstance);

  return {
    mocks: mockFns,
    interceptors: captured,
  };
});

// Mock dependencies BEFORE importing client module
vi.mock('axios', () => ({
  default: {
    create: mocks.axiosCreate,
  },
}));

vi.mock('@/config', () => ({
  config: {
    api: {
      baseUrl: 'http://test-api.example.com',
      timeout: 15000,
    },
  },
}));

vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
  },
}));

// Import after mocks are set up
import { config } from '@/config';
import { logger } from '@/lib/logger';
import { api, apiRequest } from './client';
import type { ApiError } from './client';
import { ChemistryType, ContrastAgent, DeveloperType, PaperSizing } from '@/types/models';

describe('API Client', () => {
  // Note: axios.create is called once during module import, before any tests run
  // Configuration tests check that initial setup, other tests use mocks.request

  describe('apiClient configuration', () => {
    it('creates axios instance with correct configuration', () => {
      // Verify axios.create was called with correct config
      expect(mocks.axiosCreate).toHaveBeenCalledWith({
        baseURL: config.api.baseUrl,
        timeout: config.api.timeout,
        headers: {
          'Content-Type': 'application/json',
        },
      });

      // Verify interceptors were registered
      expect(mocks.requestInterceptorUse).toHaveBeenCalledTimes(1);
      expect(mocks.requestInterceptorUse).toHaveBeenCalledWith(
        expect.any(Function),
        expect.any(Function)
      );

      expect(mocks.responseInterceptorUse).toHaveBeenCalledTimes(1);
      expect(mocks.responseInterceptorUse).toHaveBeenCalledWith(
        expect.any(Function),
        expect.any(Function)
      );
    });
  });

  describe('request interceptor', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('logs debug message with method and URL', () => {
      const requestConfig: InternalAxiosRequestConfig = {
        method: 'post',
        url: '/api/test',
        headers: {} as any,
      };

      interceptors.requestInterceptor(requestConfig);

      expect(logger.debug).toHaveBeenCalledWith('API Request', {
        method: 'POST',
        url: '/api/test',
        hasData: false,
      });
    });

    it('logs debug message with hasData=true when data present', () => {
      const requestConfig: InternalAxiosRequestConfig = {
        method: 'post',
        url: '/api/test',
        data: { test: 'data' },
        headers: {} as any,
      };

      interceptors.requestInterceptor(requestConfig);

      expect(logger.debug).toHaveBeenCalledWith('API Request', {
        method: 'POST',
        url: '/api/test',
        hasData: true,
      });
    });

    it('returns config unchanged', () => {
      const requestConfig: InternalAxiosRequestConfig = {
        method: 'get',
        url: '/api/test',
        headers: {} as any,
      };

      const result = interceptors.requestInterceptor(requestConfig);

      expect(result).toBe(requestConfig);
    });

    it('logs error on request error', async () => {
      const error = new Error('Request failed');

      await expect(interceptors.requestErrorInterceptor(error)).rejects.toThrow(
        'Request failed'
      );

      expect(logger.error).toHaveBeenCalledWith('API Request Error', {
        error: 'Request failed',
      });
    });
  });

  describe('response interceptor', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('logs debug message with method, URL, and status', () => {
      const response = {
        config: {
          method: 'get',
          url: '/api/test',
        },
        status: 200,
        data: { success: true },
      };

      interceptors.responseInterceptor(response);

      expect(logger.debug).toHaveBeenCalledWith('API Response', {
        method: 'GET',
        url: '/api/test',
        status: 200,
      });
    });

    it('returns response unchanged', () => {
      const response = {
        config: { method: 'get', url: '/api/test' },
        status: 200,
        data: { success: true },
      };

      const result = interceptors.responseInterceptor(response);

      expect(result).toBe(response);
    });

    it('logs error on response error', async () => {
      const error = {
        config: {
          method: 'post',
          url: '/api/test',
        },
        response: {
          status: 500,
          data: {
            message: 'Internal server error',
          } as ApiError,
        },
      };

      await expect(interceptors.responseErrorInterceptor(error)).rejects.toEqual(error);

      expect(logger.error).toHaveBeenCalledWith('API Response Error', {
        method: 'POST',
        url: '/api/test',
        status: 500,
        message: 'Internal server error',
      });
    });

    it('uses error.message when response.data.message is unavailable', async () => {
      const error = {
        message: 'Network error',
        config: {
          method: 'get',
          url: '/api/test',
        },
        response: {
          status: 0,
          data: {} as ApiError,
        },
      };

      await expect(interceptors.responseErrorInterceptor(error)).rejects.toEqual(error);

      expect(logger.error).toHaveBeenCalledWith(
        'API Response Error',
        expect.objectContaining({
          message: 'Network error',
        })
      );
    });

    it('logs warning on 401 unauthorized error', async () => {
      const error = {
        config: { method: 'get', url: '/api/protected' },
        response: {
          status: 401,
          data: { message: 'Unauthorized' } as ApiError,
        },
      };

      await expect(interceptors.responseErrorInterceptor(error)).rejects.toEqual(error);

      expect(logger.warn).toHaveBeenCalledWith('Unauthorized request');
    });

    it('logs warning on 429 rate limit error', async () => {
      const error = {
        config: { method: 'post', url: '/api/test' },
        response: {
          status: 429,
          data: { message: 'Too many requests' } as ApiError,
        },
      };

      await expect(interceptors.responseErrorInterceptor(error)).rejects.toEqual(error);

      expect(logger.warn).toHaveBeenCalledWith('Rate limited');
    });
  });

  describe('apiRequest', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('returns response.data from successful request', async () => {
      const mockData = { id: 123, name: 'Test' };
      const mockResponse = { data: mockData };

      mocks.request.mockResolvedValue(mockResponse);

      const result = await apiRequest({ method: 'GET', url: '/api/test' });

      expect(result).toEqual(mockData);
    });

    it('passes config to axios.request', async () => {
      const mockResponse = { data: { success: true } };
      mocks.request.mockResolvedValue(mockResponse);

      const config = { method: 'POST', url: '/api/test', data: { test: true } };
      await apiRequest(config);

      expect(mocks.request).toHaveBeenCalledWith(config);
    });

    it('throws error on failed request', async () => {
      const mockError = new Error('Request failed');
      mocks.request.mockRejectedValue(mockError);

      await expect(
        apiRequest({ method: 'GET', url: '/api/test' })
      ).rejects.toThrow('Request failed');
    });
  });

  describe('api.health', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('check() calls GET /api/health', async () => {
      const mockResponse = { data: { status: 'ok' } };
      mocks.request.mockResolvedValue(mockResponse);

      const result = await api.health.check();

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/api/health',
      });
      expect(result).toEqual({ status: 'ok' });
    });
  });

  describe('api.curves', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('generate() calls POST /api/curves/generate', async () => {
      const mockResponse = {
        data: { id: '123', curve_type: 'calibration' },
      };
      mocks.request.mockResolvedValue(mockResponse);

      const data = {
        measurements: [0.1, 0.5, 1.2],
        type: 'calibration',
        name: 'Test Curve',
      };

      await api.curves.generate(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/curves/generate',
        data,
      });
    });

    it('get() calls GET /api/curves/:id', async () => {
      const mockResponse = { data: { id: '123', name: 'Test' } };
      mocks.request.mockResolvedValue(mockResponse);

      await api.curves.get('123');

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/api/curves/123',
      });
    });

    it('modify() calls POST /api/curves/modify', async () => {
      const mockResponse = { data: { success: true } };
      mocks.request.mockResolvedValue(mockResponse);

      const data = {
        name: 'Test',
        input_values: [0, 128, 255],
        output_values: [0, 128, 255],
        adjustment_type: 'contrast',
        amount: 0.5,
      };

      await api.curves.modify(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/curves/modify',
        data,
      });
    });

    it('smooth() calls POST /api/curves/smooth', async () => {
      const mockResponse = { data: { smoothed: true } };
      mocks.request.mockResolvedValue(mockResponse);

      const data = {
        name: 'Test',
        input_values: [0, 128, 255],
        output_values: [0, 128, 255],
        method: 'gaussian',
      };

      await api.curves.smooth(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/curves/smooth',
        data,
      });
    });

    it('enhance() calls POST /api/curves/enhance', async () => {
      const mockResponse = { data: { enhanced: true } };
      mocks.request.mockResolvedValue(mockResponse);

      const data = {
        name: 'Test',
        input_values: [0, 128, 255],
        output_values: [0, 128, 255],
        goal: 'improve contrast',
      };

      await api.curves.enhance(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/curves/enhance',
        data,
      });
    });

    it('enforceMonotonicity() calls POST /api/curves/:id/enforce-monotonicity', async () => {
      const mockResponse = { data: { monotonic: true } };
      mocks.request.mockResolvedValue(mockResponse);

      await api.curves.enforceMonotonicity('123', 'increasing');

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/curves/123/enforce-monotonicity',
        params: { direction: 'increasing' },
      });
    });

    it('enforceMonotonicity() uses default direction', async () => {
      const mockResponse = { data: { monotonic: true } };
      mocks.request.mockResolvedValue(mockResponse);

      await api.curves.enforceMonotonicity('123');

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/curves/123/enforce-monotonicity',
        params: { direction: 'increasing' },
      });
    });

    it('export() calls POST /api/curves/export with blob responseType', async () => {
      const mockBlob = new Blob(['test']);
      const mockResponse = { data: mockBlob };
      mocks.request.mockResolvedValue(mockResponse);

      const data = { curveId: '123', format: 'qtr' };

      await api.curves.export(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/curves/export',
        data,
        responseType: 'blob',
      });
    });
  });

  describe('api.scan', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('upload() creates FormData and calls POST /api/scan/upload', async () => {
      const mockResponse = { data: { uploadId: '123' } };
      mocks.request.mockResolvedValue(mockResponse);

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      await api.scan.upload(file, 'stouffer_21');

      expect(mocks.request).toHaveBeenCalledWith(
        expect.objectContaining({
          method: 'POST',
          url: '/api/scan/upload',
          headers: { 'Content-Type': 'multipart/form-data' },
        })
      );

      // Verify FormData contains correct entries
      const callArgs = mocks.request.mock.calls[0]?.[0];
      expect(callArgs?.data).toBeInstanceOf(FormData);

      const formData = callArgs?.data as FormData;
      expect(formData.get('file')).toBe(file);
      expect(formData.get('tablet_type')).toBe('stouffer_21');
    });

    it('upload() uses default tablet type', async () => {
      const mockResponse = { data: { uploadId: '123' } };
      mocks.request.mockResolvedValue(mockResponse);

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      await api.scan.upload(file);

      const callArgs = mocks.request.mock.calls[0]?.[0];
      const formData = callArgs?.data as FormData;
      expect(formData.get('tablet_type')).toBe('stouffer_21');
    });

    it('upload() calls onProgress callback with upload progress', async () => {
      const mockResponse = { data: { uploadId: '123' } };
      mocks.request.mockResolvedValue(mockResponse);

      const onProgress = vi.fn();
      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      await api.scan.upload(file, 'stouffer_21', onProgress);

      // Get the onUploadProgress callback
      const callArgs = mocks.request.mock.calls[0]?.[0];
      const onUploadProgress = callArgs?.onUploadProgress as (event: {
        loaded: number;
        total?: number;
      }) => void;

      expect(onUploadProgress).toBeDefined();

      // Simulate upload progress
      onUploadProgress({ loaded: 50, total: 100 });
      expect(onProgress).toHaveBeenCalledWith(50);

      onUploadProgress({ loaded: 100, total: 100 });
      expect(onProgress).toHaveBeenCalledWith(100);
    });

    it('upload() does not call onProgress when total is undefined', async () => {
      const mockResponse = { data: { uploadId: '123' } };
      mocks.request.mockResolvedValue(mockResponse);

      const onProgress = vi.fn();
      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      await api.scan.upload(file, 'stouffer_21', onProgress);

      const callArgs = mocks.request.mock.calls[0]?.[0];
      const onUploadProgress = callArgs?.onUploadProgress as (event: {
        loaded: number;
        total?: number;
      }) => void;

      // Simulate upload progress without total
      onUploadProgress({ loaded: 50 });
      expect(onProgress).not.toHaveBeenCalled();
    });

    it('upload() does not error when onProgress is not provided', async () => {
      const mockResponse = { data: { uploadId: '123' } };
      mocks.request.mockResolvedValue(mockResponse);

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      await api.scan.upload(file, 'stouffer_21');

      const callArgs = mocks.request.mock.calls[0]?.[0];
      const onUploadProgress = callArgs?.onUploadProgress as (event: {
        loaded: number;
        total?: number;
      }) => void;

      // Should not throw
      expect(() => onUploadProgress({ loaded: 50, total: 100 })).not.toThrow();
    });
  });

  describe('api.calibrations', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('list() calls GET /api/calibrations with params', async () => {
      const mockResponse = { data: { calibrations: [] } };
      mocks.request.mockResolvedValue(mockResponse);

      await api.calibrations.list('arches', 25);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/api/calibrations',
        params: { paper_type: 'arches', limit: 25 },
      });
    });

    it('list() uses default limit', async () => {
      const mockResponse = { data: { calibrations: [] } };
      mocks.request.mockResolvedValue(mockResponse);

      await api.calibrations.list('arches');

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/api/calibrations',
        params: { paper_type: 'arches', limit: 50 },
      });
    });

    it('list() handles undefined paper type', async () => {
      const mockResponse = { data: { calibrations: [] } };
      mocks.request.mockResolvedValue(mockResponse);

      await api.calibrations.list(undefined, 25);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/api/calibrations',
        params: { paper_type: undefined, limit: 25 },
      });
    });

    it('create() calls POST /api/calibrations', async () => {
      const mockResponse = { data: { id: '123' } };
      mocks.request.mockResolvedValue(mockResponse);

      const data = {
        name: 'Test',
        paper_type: 'arches',
        paper_sizing: PaperSizing.GELATIN,
        chemistry_type: ChemistryType.PLATINUM_PALLADIUM,
        metal_ratio: 0.5,
        contrast_agent: ContrastAgent.NA2,
        contrast_amount: 0.1,
        developer: DeveloperType.POTASSIUM_OXALATE,
        exposure_time: 300,
        measured_densities: [0.1, 0.5, 1.0],
        tags: ['test'],
      };

      await api.calibrations.create(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/calibrations',
        data,
      });
    });

    it('get() calls GET /api/calibrations/:id', async () => {
      const mockResponse = { data: { id: '123', name: 'Test' } };
      mocks.request.mockResolvedValue(mockResponse);

      await api.calibrations.get('123');

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/api/calibrations/123',
      });
    });
  });

  describe('api.chat', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('send() calls POST /api/chat', async () => {
      const mockResponse = { data: { response: 'Hello' } };
      mocks.request.mockResolvedValue(mockResponse);

      const data = { message: 'Test message', include_history: true };

      await api.chat.send(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/chat',
        data,
      });
    });

    it('recipe() calls POST /api/chat/recipe', async () => {
      const mockResponse = { data: { response: 'Recipe suggestion' } };
      mocks.request.mockResolvedValue(mockResponse);

      const data = {
        paper_type: 'arches',
        characteristics: ['warm', 'high contrast'],
      };

      await api.chat.recipe(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/chat/recipe',
        data,
      });
    });

    it('troubleshoot() calls POST /api/chat/troubleshoot', async () => {
      const mockResponse = { data: { response: 'Try this fix' } };
      mocks.request.mockResolvedValue(mockResponse);

      const data = { problem: 'Low contrast in shadows' };

      await api.chat.troubleshoot(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/chat/troubleshoot',
        data,
      });
    });
  });

  describe('api.statistics', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('get() calls GET /api/statistics', async () => {
      const mockResponse = { data: { total: 100 } };
      mocks.request.mockResolvedValue(mockResponse);

      await api.statistics.get();

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/api/statistics',
      });
    });
  });

  describe('api.analyze', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('densities() calls POST /api/analyze', async () => {
      const mockResponse = { data: { analysis: {} } };
      mocks.request.mockResolvedValue(mockResponse);

      const data = { measurements: [0.1, 0.5, 1.2] };

      await api.analyze.densities(data);

      expect(mocks.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/api/analyze',
        data,
      });
    });
  });
});
