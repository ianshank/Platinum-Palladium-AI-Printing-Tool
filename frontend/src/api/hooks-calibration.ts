/**
 * Calibration API hooks
 * Handles calibration record management and scan uploads
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
  CalibrationCreateResponse,
  CalibrationListResponse,
  CalibrationRecord,
  ScanUploadResponse,
} from '@/types/models';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';
import { queryKeys } from './hooks';

// ============================================================================
// Calibrations Query Hooks
// ============================================================================

/**
 * Fetch all calibration records
 */
export function useCalibrations(
  options?: Omit<
    UseQueryOptions<CalibrationListResponse, AxiosError<ApiError>>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: queryKeys.calibrations(),
    queryFn: () => api.calibrations.list(),
    ...options,
  });
}

/**
 * Fetch a specific calibration record
 */
export function useCalibration(
  id: string,
  options?: Omit<
    UseQueryOptions<unknown, AxiosError<ApiError>>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: queryKeys.calibration(id),
    queryFn: () => api.calibrations.get(id),
    enabled: !!id,
    ...options,
  });
}

// ============================================================================
// Calibration Creation
// ============================================================================

/**
 * Create a new calibration record
 */
export function useCreateCalibration(
  options?: UseMutationOptions<
    CalibrationCreateResponse,
    AxiosError<ApiError>,
    Omit<CalibrationRecord, 'id' | 'timestamp'>
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: (data: Omit<CalibrationRecord, 'id' | 'timestamp'>) =>
      api.calibrations.create(data),
    onSuccess: () => {
      logger.info('Calibration created');
      addToast({
        title: 'Calibration Saved',
        variant: 'success',
      });
      void queryClient.invalidateQueries({
        queryKey: queryKeys.calibrations(),
      });
    },
    onError: (error) => {
      logger.error('Calibration creation failed', { error: error.message });
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
// Scan Upload
// ============================================================================

/**
 * Upload a step tablet scan for analysis
 */
export function useUploadScan(
  options?: UseMutationOptions<
    ScanUploadResponse,
    AxiosError<ApiError>,
    { file: File; tabletType: string }
  >
) {
  const addToast = useStore((state) => state.ui.addToast);
  const startUpload = useStore((state) => state.image.startUpload);
  const updateUploadProgress = useStore(
    (state) => state.image.updateUploadProgress
  );
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
      logger.error('Scan upload failed', { error: error.message });
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
