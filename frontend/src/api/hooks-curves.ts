/**
 * Curve management API hooks
 * Handles curve generation, modification, export, and enhancement
 */

import {
  useMutation,
  type UseMutationOptions,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from '@tanstack/react-query';
import { api, type ApiError, type AxiosError } from './client';
import type {
  CurveEnhanceResponse,
  CurveGenerationResponse,
  CurveModificationRequest,
  CurveModificationResponse,
  CurveSmoothingResponse,
  CurveSmoothRequest,
  QuadParseResponse,
  QuadUploadResponse,
} from '@/types/models';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';
import { queryKeys } from './hooks';

// ============================================================================
// Curves Query Hooks
// ============================================================================

/**
 * Fetch a specific curve by ID
 */
export function useCurve(
  id: string,
  options?: Omit<
    UseQueryOptions<unknown, AxiosError<ApiError>>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: queryKeys.curve(id),
    queryFn: () => api.curves.get(id),
    enabled: !!id,
    ...options,
  });
}

// ============================================================================
// Curve Generation & Modification
// ============================================================================

/**
 * Generate a new calibration curve from density measurements
 */
export function useGenerateCurve(
  options?: UseMutationOptions<
    CurveGenerationResponse,
    AxiosError<ApiError>,
    {
      measurements: number[];
      type?: string;
      name?: string;
      curve_type?: string;
    }
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);
  const setProcessing = useStore((state) => state.ui.setProcessing);

  return useMutation({
    mutationFn: (data: {
      measurements: number[];
      type?: string;
      name?: string;
      curve_type?: string;
    }) => api.curves.generate(data),
    onMutate: () => {
      setProcessing(true);
      logger.info('Generating curve...');
    },
    onSuccess: (data) => {
      logger.info('Curve generated', {
        id: data.curve_id,
        pointCount: data.input_values.length,
      });
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

/**
 * Modify an existing curve with transformations
 */
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

/**
 * Apply smoothing to a curve
 */
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

/**
 * Save a curve with optional updates
 */
export function useSaveCurve(
  options?: UseMutationOptions<
    CurveModificationResponse,
    AxiosError<ApiError>,
    CurveModificationRequest
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui?.addToast);

  return useMutation({
    mutationFn: (data: CurveModificationRequest) => api.curves.modify(data),
    onSuccess: (data) => {
      logger.info('Curve saved', { curveId: data.curve_id, name: data.name });
      addToast?.({
        title: 'Curve Saved',
        description: `"${data.name}" saved successfully`,
        variant: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.curves() });
    },
    onError: (error) => {
      logger.error('Curve save failed', { error: error.message });
      addToast?.({
        title: 'Save Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

// ============================================================================
// AI Enhancement
// ============================================================================

/**
 * Apply AI-powered enhancements to a curve
 */
export function useEnhanceCurve(
  options?: UseMutationOptions<
    CurveEnhanceResponse,
    AxiosError<ApiError>,
    {
      name: string;
      input_values: number[];
      output_values: number[];
      goal: string;
      additional_context?: string;
      paper_type?: string;
    }
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: (data) => api.curves.enhance(data),
    onSuccess: (data) => {
      logger.info('Curve enhanced', { curveId: data.curve_id, goal: data.goal });
      addToast({
        title: 'AI Enhancement Applied',
        description: `Goal: ${data.goal} — Confidence: ${Math.round(data.confidence * 100)}%`,
        variant: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.curves() });
    },
    onError: (error) => {
      logger.error('Curve enhance failed', { error: error.message });
      addToast({
        title: 'Enhancement Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

// ============================================================================
// Export
// ============================================================================

/**
 * Export a curve in various formats
 */
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
// Quad File Operations
// ============================================================================

/**
 * Upload a QuadTone RIP file
 */
export function useUploadQuadFile(
  options?: UseMutationOptions<
    QuadUploadResponse,
    AxiosError<ApiError>,
    { file: File; channel: string }
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: ({ file, channel }: { file: File; channel: string }) =>
      api.curves.uploadQuad(file, channel),
    onSuccess: (data) => {
      logger.info('Quad file uploaded', {
        profile: data.profile_name,
        channels: data.active_channels,
      });
      addToast({
        title: 'Profile Loaded',
        description: `Loaded "${data.profile_name}" with ${data.active_channels.length} channels`,
        variant: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.curves() });
    },
    onError: (error) => {
      logger.error('Quad file upload failed', { error: error.message });
      addToast({
        title: 'Upload Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

/**
 * Parse QuadTone RIP file content
 */
export function useParseQuadContent(
  options?: UseMutationOptions<
    QuadParseResponse,
    AxiosError<ApiError>,
    { content: string; name?: string; channel: string }
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: ({
      content,
      name,
      channel,
    }: {
      content: string;
      name?: string;
      channel: string;
    }) => api.curves.parseQuad(content, name ?? 'Uploaded Profile', channel),
    onSuccess: (data) => {
      logger.info('Quad content parsed', {
        profile: data.profile_name,
        channels: data.active_channels,
      });
      addToast({
        title: 'Content Parsed',
        description: `Parsed "${data.profile_name}" with ${data.active_channels.length} channels`,
        variant: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.curves() });
    },
    onError: (error) => {
      logger.error('Quad content parse failed', { error: error.message });
      addToast({
        title: 'Parse Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}
