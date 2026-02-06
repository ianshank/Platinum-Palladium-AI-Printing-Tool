/**
 * TanStack Query hooks for curve operations.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient, uploadFile } from '../client';
import { queryKeys } from '../queryClient';
import { apiConfig } from '@/config/api.config';
import type {
  CurveData,
  CurveGenerateRequest,
  CurveGenerateResponse,
  CurveModifyRequest,
  CurveModifyResponse,
  CurveEnhanceRequest,
  CurveEnhanceResponse,
} from '@/types';

/**
 * Fetch a stored curve by ID
 */
export const useCurve = (curveId: string | undefined) => {
  return useQuery({
    queryKey: queryKeys.curves.detail(curveId || ''),
    queryFn: async () => {
      const { data } = await apiClient.get<CurveData>(
        apiConfig.endpoints.curves.byId(curveId!)
      );
      return data;
    },
    enabled: !!curveId,
  });
};

/**
 * Generate a new curve from densities
 */
export const useGenerateCurve = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (request: CurveGenerateRequest) => {
      const { data } = await apiClient.post<CurveGenerateResponse>(
        apiConfig.endpoints.curves.generate,
        request
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.curves.all });
    },
  });
};

/**
 * Modify an existing curve
 */
export const useModifyCurve = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (request: CurveModifyRequest) => {
      const { data } = await apiClient.post<CurveModifyResponse>(
        apiConfig.endpoints.curves.modify,
        request
      );
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.curves.detail(data.curve_id),
      });
    },
  });
};

/**
 * Smooth a curve
 */
export const useSmoothCurve = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (request: {
      input_values: number[];
      output_values: number[];
      name?: string;
      method: string;
      strength: number;
      preserve_endpoints?: boolean;
    }) => {
      const { data } = await apiClient.post<CurveModifyResponse>(
        apiConfig.endpoints.curves.smooth,
        request
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.curves.all });
    },
  });
};

/**
 * Blend two curves
 */
export const useBlendCurves = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (request: {
      curve1_inputs: number[];
      curve1_outputs: number[];
      curve2_inputs: number[];
      curve2_outputs: number[];
      name?: string;
      mode: string;
      weight?: number;
    }) => {
      const { data } = await apiClient.post<CurveModifyResponse>(
        apiConfig.endpoints.curves.blend,
        request
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.curves.all });
    },
  });
};

/**
 * AI-enhance a curve
 */
export const useEnhanceCurve = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (request: CurveEnhanceRequest) => {
      const { data } = await apiClient.post<CurveEnhanceResponse>(
        apiConfig.endpoints.curves.enhance,
        request
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.curves.all });
    },
  });
};

/**
 * Upload a .quad file
 */
export const useUploadQuadFile = () => {
  return useMutation({
    mutationFn: async ({ file, channel }: { file: File; channel?: string }) => {
      const additionalData = channel ? { channel } : undefined;
      const { data } = await uploadFile(
        apiConfig.endpoints.curves.uploadQuad,
        file,
        additionalData
      );
      return data;
    },
  });
};

/**
 * Parse .quad content from string
 */
export const useParseQuadContent = () => {
  return useMutation({
    mutationFn: async ({
      content,
      name,
      channel,
    }: {
      content: string;
      name?: string;
      channel?: string;
    }) => {
      const formData = new FormData();
      formData.append('content', content);
      if (name) formData.append('name', name);
      if (channel) formData.append('channel', channel);

      const { data } = await apiClient.post(
        apiConfig.endpoints.curves.parseQuad,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );
      return data;
    },
  });
};

/**
 * Export a curve to file format
 */
export const useExportCurve = () => {
  return useMutation({
    mutationFn: async ({
      densities,
      name,
      format,
    }: {
      densities: number[];
      name: string;
      format: string;
    }) => {
      const formData = new FormData();
      formData.append('densities', JSON.stringify(densities));
      formData.append('name', name);
      formData.append('format', format);

      const { data } = await apiClient.post(
        apiConfig.endpoints.curves.export,
        formData,
        {
          responseType: 'blob',
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );
      return data;
    },
  });
};
