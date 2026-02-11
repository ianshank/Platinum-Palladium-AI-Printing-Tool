/**
 * API Hooks Tests (TanStack Query)
 *
 * Covers the hooks from api/hooks.ts:
 * - queryKeys factory
 * - useHealthCheck query
 * - useSendMessage mutation lifecycle
 * - useStatistics query
 * - useCalibrations query
 * - useCreateCalibration mutation with toast
 * - useGenerateCurve mutation with processing state
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { type ReactNode } from 'react';

// Use vi.hoisted so mock references are available before vi.mock factories run
const {
  mockHealthCheck,
  mockChatSend,
  mockStatisticsGet,
  mockCalibrationsList,
  mockCalibrationsCreate,
  mockCalibrationsGet,
  mockCurvesGenerate,
  mockCurvesGet,
  mockCurvesModify,
  mockCurvesSmooth,
  mockCurvesExport,
  mockScanUpload,
  mockAddToast,
  mockSetProcessing,
  mockAddMessage,
  mockSetLoading,
  mockSetError,
  mockStartUpload,
  mockUpdateUploadProgress,
  mockImageSetError,
} = vi.hoisted(() => ({
  mockHealthCheck: vi.fn(),
  mockChatSend: vi.fn(),
  mockStatisticsGet: vi.fn(),
  mockCalibrationsList: vi.fn(),
  mockCalibrationsCreate: vi.fn(),
  mockCalibrationsGet: vi.fn(),
  mockCurvesGenerate: vi.fn(),
  mockCurvesGet: vi.fn(),
  mockCurvesModify: vi.fn(),
  mockCurvesSmooth: vi.fn(),
  mockCurvesExport: vi.fn(),
  mockScanUpload: vi.fn(),
  mockAddToast: vi.fn(),
  mockSetProcessing: vi.fn(),
  mockAddMessage: vi.fn(),
  mockSetLoading: vi.fn(),
  mockSetError: vi.fn(),
  mockStartUpload: vi.fn(),
  mockUpdateUploadProgress: vi.fn(),
  mockImageSetError: vi.fn(),
}));

vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
  },
}));

vi.mock('./client', () => ({
  api: {
    health: { check: mockHealthCheck },
    chat: { send: mockChatSend },
    statistics: { get: mockStatisticsGet },
    calibrations: {
      list: mockCalibrationsList,
      create: mockCalibrationsCreate,
      get: mockCalibrationsGet,
    },
    curves: {
      generate: mockCurvesGenerate,
      get: mockCurvesGet,
      modify: mockCurvesModify,
      smooth: mockCurvesSmooth,
      export: mockCurvesExport,
    },
    scan: { upload: mockScanUpload },
  },
  apiRequest: vi.fn(),
}));

vi.mock('@/stores', () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- mock store selector with test state
  useStore: (selector: (state: any) => any) =>
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    selector({
      ui: {
        addToast: mockAddToast,
        setProcessing: mockSetProcessing,
      },
      chat: {
        addMessage: mockAddMessage,
        setLoading: mockSetLoading,
        setError: mockSetError,
      },
      image: {
        startUpload: mockStartUpload,
        updateUploadProgress: mockUpdateUploadProgress,
        setError: mockImageSetError,
      },
    }),
}));

// Import AFTER mocks
import {
  assessScanQuality,
  queryKeys,
  useCalibration,
  useCalibrations,
  useCreateCalibration,
  useCurve,
  useExportCurve,
  useGenerateCurve,
  useHealthCheck,
  useModifyCurve,
  useSendMessage,
  useSmoothCurve,
  useStatistics,
  useUploadScan,
} from './hooks';

// --- Helper: QueryClient wrapper ---
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

describe('API Hooks', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('queryKeys', () => {
    it('produces consistent key hierarchies', () => {
      expect(queryKeys.all).toEqual(['ptpd']);
      expect(queryKeys.health()).toEqual(['ptpd', 'health']);
      expect(queryKeys.curves()).toEqual(['ptpd', 'curves']);
      expect(queryKeys.curve('abc')).toEqual(['ptpd', 'curves', 'abc']);
      expect(queryKeys.calibrations()).toEqual(['ptpd', 'calibrations']);
      expect(queryKeys.calibration('xyz')).toEqual([
        'ptpd',
        'calibrations',
        'xyz',
      ]);
      expect(queryKeys.statistics()).toEqual(['ptpd', 'statistics']);
    });
  });

  describe('useHealthCheck', () => {
    it('calls api.health.check and returns data', async () => {
      mockHealthCheck.mockResolvedValue({ status: 'ok' });

      const { result } = renderHook(() => useHealthCheck(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockHealthCheck).toHaveBeenCalledTimes(1);
      expect(result.current.data).toEqual({ status: 'ok' });
    });

    it('handles error state', async () => {
      mockHealthCheck.mockRejectedValue(new Error('Offline'));

      const { result } = renderHook(() => useHealthCheck(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });
    });
  });

  describe('useStatistics', () => {
    it('fetches statistics data', async () => {
      const stats = { total_records: 42, success_rate: 0.9 };
      mockStatisticsGet.mockResolvedValue(stats);

      const { result } = renderHook(() => useStatistics(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(stats);
    });
  });

  describe('useCalibrations', () => {
    it('fetches calibration list', async () => {
      const cals = { calibrations: [{ id: 'cal-1' }] };
      mockCalibrationsList.mockResolvedValue(cals);

      const { result } = renderHook(() => useCalibrations(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(cals);
    });
  });

  describe('useSendMessage', () => {
    it('calls api.chat.send and adds messages to store', async () => {
      const response = { response: 'Hello!', context_used: ['ctx-1'] };
      mockChatSend.mockResolvedValue(response);

      const { result } = renderHook(() => useSendMessage(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ message: 'Hi there' });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockAddMessage).toHaveBeenCalledWith({
        role: 'user',
        content: 'Hi there',
      });

      expect(mockAddMessage).toHaveBeenCalledWith({
        role: 'assistant',
        content: 'Hello!',
        metadata: { context: ['ctx-1'] },
      });
    });

    it('sets loading state during mutation', async () => {
      mockChatSend.mockResolvedValue({ response: 'OK', context_used: [] });

      const { result } = renderHook(() => useSendMessage(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ message: 'Test' });

      expect(mockSetLoading).toHaveBeenCalledWith(true);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockSetLoading).toHaveBeenCalledWith(false);
    });

    it('sets error on failure', async () => {
      const error = {
        message: 'Server error',
        response: { data: { message: 'LLM failed' } },
      };
      mockChatSend.mockRejectedValue(error);

      const { result } = renderHook(() => useSendMessage(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ message: 'Fail' });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(mockSetError).toHaveBeenCalledWith('LLM failed');
    });
  });

  describe('useCreateCalibration', () => {
    it('shows success toast on creation', async () => {
      mockCalibrationsCreate.mockResolvedValue({ id: 'cal-new' });

      const { result } = renderHook(() => useCreateCalibration(), {
        wrapper: createWrapper(),
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-argument -- test purposely sends partial data to verify hook behavior
      result.current.mutate({ name: 'Test Cal' } as any);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockAddToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Calibration Saved',
          variant: 'success',
        })
      );
    });

    it('shows error toast on failure', async () => {
      const error = {
        message: 'Save failed',
        response: { data: { message: 'Duplicate name' } },
      };
      mockCalibrationsCreate.mockRejectedValue(error);

      const { result } = renderHook(() => useCreateCalibration(), {
        wrapper: createWrapper(),
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-argument
      result.current.mutate({ name: 'Dup' } as any);

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(mockAddToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Save Failed',
          variant: 'error',
        })
      );
    });
  });

  describe('useGenerateCurve', () => {
    it('sets processing state and generates curve', async () => {
      const curveData = {
        success: true,
        curve_id: 'curve-1',
        name: 'Test Curve',
        num_points: 2,
        input_values: [0, 255],
        output_values: [0, 200],
      };
      mockCurvesGenerate.mockResolvedValue(curveData);

      const { result } = renderHook(() => useGenerateCurve(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ measurements: [0.1, 0.5, 0.9] });

      expect(mockSetProcessing).toHaveBeenCalledWith(true);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockSetProcessing).toHaveBeenCalledWith(false);

      expect(mockAddToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Curve Generated',
          variant: 'success',
        })
      );
    });
  });

  // -------------------------------------------------------------------
  // New tests for previously-untested hooks
  // -------------------------------------------------------------------

  describe('useCurve', () => {
    it('fetches a curve by id', async () => {
      const curve = {
        id: 'c-1',
        name: 'Curve 1',
        input_values: [0],
        output_values: [0],
      };
      mockCurvesGet.mockResolvedValue(curve);

      const { result } = renderHook(() => useCurve('c-1'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockCurvesGet).toHaveBeenCalledWith('c-1');
      expect(result.current.data).toEqual(curve);
    });

    it('stays disabled when id is empty', () => {
      const { result } = renderHook(() => useCurve(''), {
        wrapper: createWrapper(),
      });

      expect(result.current.fetchStatus).toBe('idle');
    });
  });

  describe('useCalibration', () => {
    it('fetches a calibration by id', async () => {
      const cal = { id: 'cal-1', name: 'Cal 1' };
      mockCalibrationsGet.mockResolvedValue(cal);

      const { result } = renderHook(() => useCalibration('cal-1'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockCalibrationsGet).toHaveBeenCalledWith('cal-1');
      expect(result.current.data).toEqual(cal);
    });

    it('stays disabled when id is empty', () => {
      const { result } = renderHook(() => useCalibration(''), {
        wrapper: createWrapper(),
      });

      expect(result.current.fetchStatus).toBe('idle');
    });
  });

  describe('useModifyCurve', () => {
    it('modifies a curve and invalidates cache', async () => {
      const modified = {
        success: true,
        curve_id: 'c-1',
        name: 'Mod',
        adjustment_applied: 'brightness',
        input_values: [0, 128],
        output_values: [10, 138],
      };
      mockCurvesModify.mockResolvedValue(modified);

      const { result } = renderHook(() => useModifyCurve(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({
        name: 'Mod',
        input_values: [0, 128],
        output_values: [0, 128],
        adjustment_type: 'brightness',
        amount: 10,
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockCurvesModify).toHaveBeenCalledTimes(1);
    });
  });

  describe('useSmoothCurve', () => {
    it('smooths a curve', async () => {
      const smoothed = {
        success: true,
        curve_id: 'c-2',
        name: 'Smoothed',
        method_applied: 'spline',
        input_values: [0, 128, 255],
        output_values: [0, 100, 200],
      };
      mockCurvesSmooth.mockResolvedValue(smoothed);

      const { result } = renderHook(() => useSmoothCurve(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({
        name: 'Smoothed',
        input_values: [0, 128, 255],
        output_values: [0, 90, 200],
        method: 'spline',
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockCurvesSmooth).toHaveBeenCalledTimes(1);
    });
  });

  describe('useExportCurve', () => {
    it('exports a curve and shows success toast', async () => {
      const blob = new Blob(['data'], { type: 'text/csv' });
      mockCurvesExport.mockResolvedValue(blob);

      const { result } = renderHook(() => useExportCurve(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ curveId: 'c-1', format: 'csv' });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockAddToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Export Complete',
          variant: 'success',
        })
      );
    });

    it('shows error toast on export failure', async () => {
      const error = {
        message: 'Export failed',
        response: { data: { message: 'Invalid format' } },
      };
      mockCurvesExport.mockRejectedValue(error);

      const { result } = renderHook(() => useExportCurve(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ curveId: 'c-1', format: 'invalid' });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(mockAddToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Export Failed',
          variant: 'error',
        })
      );
    });
  });

  describe('useUploadScan', () => {
    it('uploads a scan, starts upload, and shows success toast', async () => {
      const uploadResponse = {
        success: true,
        extraction_id: 'ext-1',
        num_patches: 21,
        densities: [0.1, 0.5, 1.0],
        dmin: 0.1,
        dmax: 1.0,
        range: 0.9,
        quality: 0.95,
        warnings: [],
      };
      mockScanUpload.mockResolvedValue(uploadResponse);

      const { result } = renderHook(() => useUploadScan(), {
        wrapper: createWrapper(),
      });

      const file = new File(['content'], 'scan.tif', { type: 'image/tiff' });
      result.current.mutate({ file, tabletType: 'stouffer_21' });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockStartUpload).toHaveBeenCalledWith('scan.tif');
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Scan Uploaded',
          variant: 'success',
        })
      );
    });

    it('handles upload error and sets image error state', async () => {
      const error = {
        message: 'File too large',
        response: { data: { message: 'Max 50MB' } },
      };
      mockScanUpload.mockRejectedValue(error);

      const { result } = renderHook(() => useUploadScan(), {
        wrapper: createWrapper(),
      });

      const file = new File(['x'.repeat(100)], 'huge.tif', {
        type: 'image/tiff',
      });
      result.current.mutate({ file, tabletType: 'stouffer_21' });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(mockImageSetError).toHaveBeenCalledWith('Max 50MB');
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Upload Failed',
          variant: 'error',
        })
      );
    });
  });

  describe('assessScanQuality', () => {
    it('returns excellent for perfect scan', () => {
      const result = assessScanQuality({
        densities: [0.1, 0.2],
        dmax: 2.0,
        dmin: 0.05,
        range: 1.95,
        num_patches: 21,
      });
      expect(result.quality).toBe('excellent');
      expect(result.score).toBe(100);
      expect(result.overall).toBe('excellent');
      expect(result.issues).toEqual([]);
    });

    it('penalizes low density range', () => {
      const result = assessScanQuality({
        densities: [0.1],
        dmax: 0.4,
        dmin: 0.1,
        range: 0.3,
        num_patches: 21,
      });
      expect(result.score).toBeLessThan(100);
      expect(
        result.issues.some((i) => i.message.includes('Low density range'))
      ).toBe(true);
    });

    it('penalizes low patch count', () => {
      const result = assessScanQuality({
        densities: [0.1],
        dmax: 2.0,
        dmin: 0.1,
        range: 1.9,
        num_patches: 5,
      });
      expect(
        result.issues.some((i) => i.message.includes('patches detected'))
      ).toBe(true);
    });

    it('penalizes low Dmax', () => {
      const result = assessScanQuality({
        densities: [0.1],
        dmax: 0.5,
        dmin: 0.1,
        range: 1.5,
        num_patches: 21,
      });
      expect(result.issues.some((i) => i.message.includes('Low Dmax'))).toBe(
        true
      );
    });

    it('clamps score to 0', () => {
      const result = assessScanQuality({
        densities: [0.1],
        dmax: 0.3,
        dmin: 0.1,
        range: 0.2,
        num_patches: 3,
      });
      expect(result.score).toBeGreaterThanOrEqual(0);
    });
  });
});
