/**
 * TanStack Query hooks for MCTS API operations
 * Provides caching, loading states, and error handling for MCTS endpoints
 */

import {
  useMutation,
  type UseMutationOptions,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from '@tanstack/react-query';
import { mctsApi } from './mcts';
import type { AxiosError } from './client';
import type { ApiError } from './client';
import type {
  MCTSEvaluateRequest,
  MCTSEvaluateResponse,
  MCTSExportFormat,
  MCTSFeedbackRequest,
  MCTSRecommendation,
  MCTSSearchRequest,
  MCTSSearchResponse,
  MCTSStatusResponse,
  MCTSTrainingStatus,
  MCTSTrainRequest,
  MCTSTrainResponse,
} from '@/types/mcts';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';

/**
 * Query key factory for MCTS endpoints
 */
export const mctsQueryKeys = {
  all: ['mcts'] as const,
  status: () => [...mctsQueryKeys.all, 'status'] as const,
  recommendations: (paperType?: string, limit?: number) =>
    [...mctsQueryKeys.all, 'recommendations', paperType, limit] as const,
  search: (id: string) => [...mctsQueryKeys.all, 'search', id] as const,
  trainingStatus: (sessionId: string) =>
    [...mctsQueryKeys.all, 'training', sessionId] as const,
};

/**
 * Get MCTS engine status
 */
export function useMCTSStatus(
  options?: Omit<
    UseQueryOptions<MCTSStatusResponse, AxiosError<ApiError>>,
    'queryKey' | 'queryFn'
  >
): ReturnType<typeof useQuery<MCTSStatusResponse, AxiosError<ApiError>>> {
  return useQuery({
    queryKey: mctsQueryKeys.status(),
    queryFn: () => mctsApi.getStatus(),
    staleTime: 60000, // 1 minute
    ...options,
  });
}

/**
 * Run MCTS search mutation
 */
export function useMCTSSearch(
  options?: UseMutationOptions<
    MCTSSearchResponse,
    AxiosError<ApiError>,
    MCTSSearchRequest
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);
  const setSearching = useStore((state) => state.mcts.setSearching);
  const setCurrentResult = useStore((state) => state.mcts.setCurrentResult);
  const setError = useStore((state) => state.mcts.setError);

  return useMutation({
    mutationFn: (request: MCTSSearchRequest) => mctsApi.search(request),
    onMutate: () => {
      setSearching(true);
      setError(null);
      logger.info('Starting MCTS search...');
    },
    onSuccess: (data) => {
      logger.info('MCTS search completed', {
        searchId: data.searchId,
        qualityScore: data.qualityScore,
        searchTime: data.searchTimeSeconds,
      });
      setCurrentResult(data);
      addToast({
        title: 'Search Complete',
        description: `Found optimal parameters (quality: ${(data.qualityScore * 100).toFixed(1)}%)`,
        variant: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: mctsQueryKeys.all });
    },
    onError: (error) => {
      logger.error('MCTS search failed', { error: error.message });
      setError(error.response?.data?.message ?? error.message);
      addToast({
        title: 'Search Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    onSettled: () => {
      setSearching(false);
    },
    ...options,
  });
}

/**
 * Evaluate parameters mutation
 */
export function useMCTSEvaluate(
  options?: UseMutationOptions<
    MCTSEvaluateResponse,
    AxiosError<ApiError>,
    MCTSEvaluateRequest
  >
) {
  const setEvaluating = useStore((state) => state.mcts.setEvaluating);
  const setEvaluateResult = useStore((state) => state.mcts.setEvaluateResult);
  const setError = useStore((state) => state.mcts.setError);
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: (request: MCTSEvaluateRequest) => mctsApi.evaluate(request),
    onMutate: () => {
      setEvaluating(true);
      setError(null);
      logger.debug('Evaluating parameters...');
    },
    onSuccess: (data) => {
      logger.info('Parameter evaluation complete', {
        qualityScore: data.qualityScore,
        gamma: data.gamma,
      });
      setEvaluateResult(data);
      addToast({
        title: 'Evaluation Complete',
        description: `Quality score: ${(data.qualityScore * 100).toFixed(1)}%`,
        variant: 'success',
      });
    },
    onError: (error) => {
      logger.error('Parameter evaluation failed', { error: error.message });
      setError(error.response?.data?.message ?? error.message);
      addToast({
        title: 'Evaluation Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    onSettled: () => {
      setEvaluating(false);
    },
    ...options,
  });
}

/**
 * Get parameter recommendations
 */
export function useMCTSRecommendations(
  paperType?: string,
  limit: number = 5,
  options?: Omit<
    UseQueryOptions<MCTSRecommendation[], AxiosError<ApiError>>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: mctsQueryKeys.recommendations(paperType, limit),
    queryFn: () => mctsApi.getRecommendations(paperType, limit),
    staleTime: 300000, // 5 minutes
    ...options,
  });
}

/**
 * Start training mutation
 */
export function useMCTSTraining(
  options?: UseMutationOptions<
    MCTSTrainResponse,
    AxiosError<ApiError>,
    MCTSTrainRequest
  >
) {
  const queryClient = useQueryClient();
  const addToast = useStore((state) => state.ui.addToast);
  const setTrainingStatus = useStore((state) => state.mcts.setTrainingStatus);

  return useMutation({
    mutationFn: (request: MCTSTrainRequest) => mctsApi.startTraining(request),
    onMutate: () => {
      setTrainingStatus('training');
      logger.info('Starting MCTS training...');
    },
    onSuccess: (data) => {
      logger.info('Training session started', {
        sessionId: data.sessionId,
      });
      addToast({
        title: 'Training Started',
        description: data.message,
        variant: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: mctsQueryKeys.all });
    },
    onError: (error) => {
      logger.error('Failed to start training', { error: error.message });
      setTrainingStatus('failed');
      addToast({
        title: 'Training Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

/**
 * Get training status query
 */
export function useMCTSTrainingStatus(
  sessionId: string,
  options?: Omit<
    UseQueryOptions<MCTSTrainingStatus, AxiosError<ApiError>>,
    'queryKey' | 'queryFn'
  >
) {
  const setTrainingStatus = useStore((state) => state.mcts.setTrainingStatus);

  return useQuery({
    queryKey: mctsQueryKeys.trainingStatus(sessionId),
    queryFn: async () => {
      const data = await mctsApi.getTrainingStatus(sessionId);
      if (data.status === 'completed') {
        setTrainingStatus('completed');
      } else if (data.status === 'failed') {
        setTrainingStatus('failed');
      }
      return data;
    },
    enabled: !!sessionId,
    refetchInterval: (query) => {
      const data = query.state.data;
      // Poll every 2 seconds while training
      if (data?.status === 'training' || data?.status === 'starting') {
        return 2000;
      }
      return false;
    },
    ...options,
  });
}

/**
 * Submit feedback mutation
 */
export function useMCTSFeedback(
  options?: UseMutationOptions<
    { success: boolean; message: string },
    AxiosError<ApiError>,
    MCTSFeedbackRequest
  >
) {
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: (feedback: MCTSFeedbackRequest) =>
      mctsApi.submitFeedback(feedback),
    onSuccess: (data) => {
      logger.info('Feedback submitted successfully');
      addToast({
        title: 'Feedback Submitted',
        description: data.message,
        variant: 'success',
      });
    },
    onError: (error) => {
      logger.error('Failed to submit feedback', { error: error.message });
      addToast({
        title: 'Feedback Failed',
        description: error.response?.data?.message ?? error.message,
        variant: 'error',
      });
    },
    ...options,
  });
}

/**
 * Export result mutation
 */
export function useMCTSExport(
  options?: UseMutationOptions<
    unknown,
    AxiosError<ApiError>,
    { parameters: Record<string, number>; format: MCTSExportFormat }
  >
) {
  const addToast = useStore((state) => state.ui.addToast);

  return useMutation({
    mutationFn: ({ parameters, format }) => mctsApi.export(parameters, format),
    onSuccess: () => {
      addToast({
        title: 'Export Complete',
        description: 'Calibration result exported successfully',
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
