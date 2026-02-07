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
    { file: File; tabletType: string }
  >
) {
  const addToast = useStore((state) => state.ui.addToast);
  const startUpload = useStore((state) => state.image.startUpload);
  const updateUploadProgress = useStore((state) => state.image.updateUploadProgress);
  const setError = useStore((state) => state.image.setError);

  return useMutation({
    mutationFn: ({ file, tabletType }: { file: File; tabletType: string }) => {
      startUpload(file.name);
      return api.scan.upload(file, tabletType, updateUploadProgress);
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

// ============================================================================
// Scan Quality Assessment
// ============================================================================

/** Input shape for scan quality assessment (matches Step2Analyze usage) */
export interface ScanAnalysisInput {
  densities: number[];
  dmax: number;
  dmin: number;
  range: number;
  num_patches: number;
}

export interface QualityIssue {
  type: 'warning' | 'error';
  message: string;
  suggestion?: string;
}

export interface QualityAssessment {
  quality: string;
  score: number;
  overall: 'excellent' | 'good' | 'acceptable' | 'poor';
  issues: QualityIssue[];
}

// ── Quality assessment thresholds ───────────────────────────
/** Density range below this is considered low — likely under-exposed. */
const DENSITY_RANGE_LOW = 0.5;
/** Density range below this is moderate — exposure may need adjustment. */
const DENSITY_RANGE_MODERATE = 1.0;
/** Minimum patches needed for a reliable calibration curve. */
const MIN_PATCH_COUNT = 11;
/** Dmax below this means insufficient shadow detail. */
const DMAX_LOW_THRESHOLD = 0.8;

// ── Score penalty weights ───────────────────────────────────
const PENALTY_LOW_RANGE = 20;
const PENALTY_MODERATE_RANGE = 10;
const PENALTY_LOW_PATCHES = 30;
const PENALTY_LOW_DMAX = 15;

// ── Overall grade boundaries ────────────────────────────────
const GRADE_EXCELLENT = 90;
const GRADE_GOOD = 70;
const GRADE_ACCEPTABLE = 50;

/**
 * Assess scan quality from extracted density data.
 * Returns quality grade, score (0–100), and any issues found.
 */
export const assessScanQuality = (input: ScanAnalysisInput): QualityAssessment => {
  const issues: QualityIssue[] = [];
  let score = 100;

  // Check density range
  if (input.range < DENSITY_RANGE_LOW) {
    issues.push({ type: 'warning', message: 'Low density range', suggestion: 'Increase exposure time or check chemistry' });
    score -= PENALTY_LOW_RANGE;
  } else if (input.range < DENSITY_RANGE_MODERATE) {
    issues.push({ type: 'warning', message: 'Moderate density range', suggestion: 'Consider adjusting exposure' });
    score -= PENALTY_MODERATE_RANGE;
  }

  // Check patch count
  if (input.num_patches < MIN_PATCH_COUNT) {
    issues.push({ type: 'error', message: `Only ${input.num_patches} patches detected`, suggestion: 'Rescan with better alignment' });
    score -= PENALTY_LOW_PATCHES;
  }

  // Check Dmax
  if (input.dmax < DMAX_LOW_THRESHOLD) {
    issues.push({ type: 'warning', message: 'Low Dmax — print may lack shadow detail', suggestion: 'Check chemistry concentration' });
    score -= PENALTY_LOW_DMAX;
  }

  const clampedScore = Math.max(0, Math.min(100, score));
  const overall = clampedScore >= GRADE_EXCELLENT ? 'excellent' : clampedScore >= GRADE_GOOD ? 'good' : clampedScore >= GRADE_ACCEPTABLE ? 'acceptable' : 'poor';

  return {
    quality: overall,
    score: clampedScore,
    overall,
    issues,
  };
};
