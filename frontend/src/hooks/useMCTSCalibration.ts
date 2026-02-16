/**
 * MCTS Calibration orchestration hook
 * Combines store state with API operations for easy component integration
 */

import { useStore } from '@/stores';
import {
  useMCTSEvaluate,
  useMCTSRecommendations,
  useMCTSSearch,
  useMCTSStatus,
  useMCTSTraining,
} from '@/api/mctsHooks';
import type { MCTSSearchRequest, MCTSTrainRequest } from '@/types/mcts';
import { logger } from '@/lib/logger';

/**
 * Main MCTS calibration hook
 * Provides unified interface for MCTS search, evaluation, and training
 */
export function useMCTSCalibration() {
  // Store state
  const store = useStore((state) => state.mcts);
  const setSearchConfig = useStore((state) => state.mcts.setSearchConfig);
  const resetSearch = useStore((state) => state.mcts.resetSearch);

  // API hooks
  const searchMutation = useMCTSSearch();
  const evaluateMutation = useMCTSEvaluate();
  const trainingMutation = useMCTSTraining();
  const statusQuery = useMCTSStatus();
  const recommendationsQuery = useMCTSRecommendations(
    store.searchConfig.paperType,
    5,
    { enabled: false } // Only fetch when explicitly requested
  );

  /**
   * Run MCTS search with current or provided configuration
   */
  const runSearch = async (config?: Partial<MCTSSearchRequest>): Promise<void> => {
    try {
      const searchConfig = config
        ? { ...store.searchConfig, ...config }
        : store.searchConfig;

      logger.info('Running MCTS search', { config: searchConfig });

      await searchMutation.mutateAsync(searchConfig);
    } catch (error) {
      logger.error('MCTS search error', { error });
      throw error;
    }
  };

  /**
   * Evaluate a parameter set
   */
  const evaluateParameters = async (
    parameters: Record<string, number>
  ): Promise<void> => {
    try {
      logger.info('Evaluating parameters', { parameters });
      await evaluateMutation.mutateAsync({ parameters });
    } catch (error) {
      logger.error('Parameter evaluation error', { error });
      throw error;
    }
  };

  /**
   * Start training session
   */
  const startTraining = async (numEpisodes?: number): Promise<void> => {
    try {
      const request: MCTSTrainRequest = numEpisodes ? { numEpisodes } : {};
      logger.info('Starting training', { numEpisodes });
      await trainingMutation.mutateAsync(request);
    } catch (error) {
      logger.error('Training start error', { error });
      throw error;
    }
  };

  /**
   * Load recommendations
   */
  const loadRecommendations = async (): Promise<void> => {
    try {
      logger.info('Loading recommendations');
      await recommendationsQuery.refetch();
    } catch (error) {
      logger.error('Recommendations load error', { error });
      throw error;
    }
  };

  return {
    // State
    ...store,
    engineStatus: statusQuery.data ?? null,
    isEngineReady: statusQuery.data?.engineReady ?? false,
    isTorchAvailable: statusQuery.data?.torchAvailable ?? false,
    parameterRanges: statusQuery.data?.parameterRanges ?? {},

    // Query states
    isLoadingStatus: statusQuery.isLoading,
    isLoadingRecommendations: recommendationsQuery.isLoading,

    // Actions
    setSearchConfig,
    runSearch,
    evaluateParameters,
    startTraining,
    loadRecommendations,
    resetSearch,

    // Raw mutation objects (for advanced usage)
    searchMutation,
    evaluateMutation,
    trainingMutation,
  };
}
