/**
 * TanStack Query hooks for API operations
 * Provides caching, loading states, and error handling
 */

import { useMutation, type UseMutationOptions, useQuery, useQueryClient, type UseQueryOptions } from '@tanstack/react-query';
import { api, type ApiError, type AxiosError } from './client';
import type {
  CalibrationCreateResponse,
  CalibrationListResponse,
  CalibrationRecord,
  ChatResponse,
  CurveGenerationResponse,
  CurveModificationRequest,
  CurveModificationResponse,
  CurveSmoothingResponse,
  CurveSmoothRequest,
  ScanUploadResponse,
  StatisticsResponse,
} from '@/types/models';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';

// Query key factory for consistent cache invalidation
export const queryKeys = {
  all: ['ptpd'] as const,
  health: () => [...queryKeys.all, 'health'] as const,
  curves: () => [...queryKeys.all, 'curves'] as const,
  curve: (id: string) => [...queryKeys.curves(), id] as const,
  calibrations: () => [...queryKeys.all, 'calibrations'] as const,
  calibration: (id: string) => [...queryKeys.calibrations(), id] as const,
  statistics: () => [...queryKeys.all, 'statistics'] as const,
};

// ============================================================================
// Health Check
// ============================================================================

export function useHealthCheck(
  options?: Omit<UseQueryOptions<{ status: string }, AxiosError<ApiError>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.health(),
    queryFn: () => api.health.check(),
    staleTime: 30000, // 30 seconds
    ...options,
  });
}

// ============================================================================
// Curves
// ============================================================================

export function useCurve(
  id: string,
  options?: Omit<UseQueryOptions<unknown, AxiosError<ApiError>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.curve(id),
    queryFn: () => api.curves.get(id),
    enabled: !!id,
    ...options,
  });
}

export function useGenerateCurve(
  options?: UseMutationOptions<
    CurveGenerationResponse,
    AxiosError<ApiError>,
    { measurements: number[]; type?: string; name?: string; curve_type?: string }
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);
  const setProcessing = useStore((state) => state.ui.setProcessing);

  return useMutation({
    mutationFn: (data: { measurements: number[]; type?: string; name?: string; curve_type?: string }) => api.curves.generate(data),
    onMutate: () => {
      setProcessing(true);
      logger.info('Generating curve...');
    },
    onSuccess: (data) => {
      logger.info('Curve generated', { id: data.curve_id, pointCount: data.input_values.length });
      addToast({
        title: 'Curve Generated',
        description: `Created curve with ${data.input_values.length} points`,
        variant: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.curves() });
    },
    onError: (error) => {
      logger.error('Curve generation failed', { error: error.message });
      addToast({
        title: 'Generation Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    onSettled: () => {
      setProcessing(false);
    },
    ...options,
  });
}

export function useModifyCurve(
  options?: UseMutationOptions<
    CurveModificationResponse,
    AxiosError<ApiError>,
    CurveModificationRequest
  >
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CurveModificationRequest) => api.curves.modify(data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.curves() });
    },
    ...options,
  });
}

export function useSmoothCurve(
  options?: UseMutationOptions<
    CurveSmoothingResponse,
    AxiosError<ApiError>,
    CurveSmoothRequest
  >
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CurveSmoothRequest) => api.curves.smooth(data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.curves() });
    },
    ...options,
  });
}

export function useExportCurve(
  options?: UseMutationOptions<
    Blob,
    AxiosError<ApiError>,
    { curveId: string; format: string }
  >
) {
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: (data) => api.curves.export(data),
    onSuccess: () => {
      addToast({
        title: 'Export Complete',
        description: 'Curve exported successfully',
        variant: 'success',
      });
    },
    onError: (error) => {
      addToast({
        title: 'Export Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

// ============================================================================
// Scan / Upload
// ============================================================================

export function useUploadScan(
  options?: UseMutationOptions<
    ScanUploadResponse,
    AxiosError<ApiError>,
    File
  >
) {
  const addToast = useStore((state) => state.ui.addToast);
  const startUpload = useStore((state) => state.image.startUpload);
  const updateUploadProgress = useStore((state) => state.image.updateUploadProgress);
  const setError = useStore((state) => state.image.setError);

  return useMutation({
    mutationFn: (file: File) => {
      startUpload(file.name);
      return api.scan.upload(file, 'stouffer_21', updateUploadProgress);
    },
    onSuccess: (data) => {
      logger.info('Scan uploaded', { extraction_id: data.extraction_id });
      addToast({
        title: 'Scan Uploaded',
        description: `Detected ${data.num_patches} measurements`,
        variant: 'success',
      });
    },
    onError: (error) => {
      setError(error.response?.data?.message ?? error.message);
      addToast({
        title: 'Upload Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

// ============================================================================
// Calibrations
// ============================================================================

export function useCalibrations(
  options?: Omit<UseQueryOptions<CalibrationListResponse, AxiosError<ApiError>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.calibrations(),
    queryFn: () => api.calibrations.list(),
    ...options,
  });
}

export function useCalibration(
  id: string,
  options?: Omit<UseQueryOptions<unknown, AxiosError<ApiError>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.calibration(id),
    queryFn: () => api.calibrations.get(id),
    enabled: !!id,
    ...options,
  });
}

export function useCreateCalibration(
  options?: UseMutationOptions<CalibrationCreateResponse, AxiosError<ApiError>, Omit<CalibrationRecord, 'id' | 'timestamp'>>
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: (data: Omit<CalibrationRecord, 'id' | 'timestamp'>) => api.calibrations.create(data),
    onSuccess: () => {
      addToast({
        title: 'Calibration Saved',
        variant: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.calibrations() });
    },
    onError: (error) => {
      addToast({
        title: 'Save Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

// ============================================================================
// Chat
// ============================================================================

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
      addMessage({
        role: 'assistant',
        content: data.response,
        metadata: { context: data.context_used ?? [] },
      });
    },
    onError: (error) => {
      setError(error.response?.data?.message ?? error.message);
    },
    onSettled: () => {
      setLoading(false);
    },
    ...options,
  });
}

// ============================================================================
// Statistics
// ============================================================================

export function useStatistics(
  options?: Omit<UseQueryOptions<StatisticsResponse, AxiosError<ApiError>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.statistics(),
    queryFn: () => api.statistics.get(),
    ...options,
  });
}
