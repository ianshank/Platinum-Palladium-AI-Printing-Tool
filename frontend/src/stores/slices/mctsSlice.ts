/**
 * MCTS calibration slice
 * Manages MCTS search state, results, and training status
 */

import type { StateCreator } from 'zustand';
import { logger } from '@/lib/logger';
import type {
  MCTSEvaluateResponse,
  MCTSRecommendation,
  MCTSSearchProgress,
  MCTSSearchRequest,
  MCTSSearchResponse,
  MCTSStatusResponse,
} from '@/types/mcts';

export interface MCTSSlice {
  // State
  searchConfig: MCTSSearchRequest;
  currentResult: MCTSSearchResponse | null;
  evaluateResult: MCTSEvaluateResponse | null;
  isSearching: boolean;
  isEvaluating: boolean;
  searchProgress: MCTSSearchProgress;
  trainingStatus: 'idle' | 'training' | 'completed' | 'failed';
  recommendations: MCTSRecommendation[];
  status: MCTSStatusResponse | null;
  error: string | null;

  // Actions
  setSearchConfig: (config: Partial<MCTSSearchRequest>) => void;
  setFixedParameter: (name: string, value: number) => void;
  removeFixedParameter: (name: string) => void;
  setSearching: (searching: boolean) => void;
  setSearchProgress: (progress: MCTSSearchProgress) => void;
  setCurrentResult: (result: MCTSSearchResponse | null) => void;
  setEvaluateResult: (result: MCTSEvaluateResponse | null) => void;
  setEvaluating: (evaluating: boolean) => void;
  setTrainingStatus: (status: MCTSSlice['trainingStatus']) => void;
  setRecommendations: (recs: MCTSRecommendation[]) => void;
  setStatus: (status: MCTSStatusResponse) => void;
  setError: (error: string | null) => void;
  resetSearch: () => void;
}

const initialState = {
  searchConfig: {
    fixedParameters: {},
    targetAesthetics: {},
  } as MCTSSearchRequest,
  currentResult: null as MCTSSearchResponse | null,
  evaluateResult: null as MCTSEvaluateResponse | null,
  isSearching: false,
  isEvaluating: false,
  searchProgress: {
    iteration: 0,
    total: 0,
    bestScore: 0,
  },
  trainingStatus: 'idle' as const,
  recommendations: [] as MCTSRecommendation[],
  status: null as MCTSStatusResponse | null,
  error: null as string | null,
};

export const createMCTSSlice: StateCreator<
  { mcts: MCTSSlice },
  [['zustand/immer', never]],
  [],
  MCTSSlice
> = (set, _get) => ({
  ...initialState,

  setSearchConfig: (config) => {
    logger.debug('MCTS: setSearchConfig', { config });
    set((state) => {
      state.mcts.searchConfig = {
        ...state.mcts.searchConfig,
        ...config,
      };
    });
  },

  setFixedParameter: (name, value) => {
    logger.debug('MCTS: setFixedParameter', { name, value });
    set((state) => {
      if (!state.mcts.searchConfig.fixedParameters) {
        state.mcts.searchConfig.fixedParameters = {};
      }
      state.mcts.searchConfig.fixedParameters[name] = value;
    });
  },

  removeFixedParameter: (name) => {
    logger.debug('MCTS: removeFixedParameter', { name });
    set((state) => {
      if (state.mcts.searchConfig.fixedParameters) {
        // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
        delete state.mcts.searchConfig.fixedParameters[name];
      }
    });
  },

  setSearching: (searching) => {
    logger.debug('MCTS: setSearching', { searching });
    set((state) => {
      state.mcts.isSearching = searching;
      if (!searching) {
        state.mcts.searchProgress = initialState.searchProgress;
      }
    });
  },

  setSearchProgress: (progress) => {
    set((state) => {
      state.mcts.searchProgress = progress;
    });
  },

  setCurrentResult: (result) => {
    logger.debug('MCTS: setCurrentResult', {
      hasResult: !!result,
      qualityScore: result?.qualityScore,
    });
    set((state) => {
      state.mcts.currentResult = result;
    });
  },

  setEvaluateResult: (result) => {
    logger.debug('MCTS: setEvaluateResult', {
      hasResult: !!result,
      qualityScore: result?.qualityScore,
    });
    set((state) => {
      state.mcts.evaluateResult = result;
    });
  },

  setEvaluating: (evaluating) => {
    logger.debug('MCTS: setEvaluating', { evaluating });
    set((state) => {
      state.mcts.isEvaluating = evaluating;
    });
  },

  setTrainingStatus: (status) => {
    logger.debug('MCTS: setTrainingStatus', { status });
    set((state) => {
      state.mcts.trainingStatus = status;
    });
  },

  setRecommendations: (recs) => {
    logger.debug('MCTS: setRecommendations', { count: recs.length });
    set((state) => {
      state.mcts.recommendations = recs;
    });
  },

  setStatus: (status) => {
    logger.debug('MCTS: setStatus', {
      engineReady: status.engineReady,
      torchAvailable: status.torchAvailable,
    });
    set((state) => {
      state.mcts.status = status;
    });
  },

  setError: (error) => {
    if (error) {
      logger.error('MCTS: error', { error });
    }
    set((state) => {
      state.mcts.error = error;
    });
  },

  resetSearch: () => {
    logger.debug('MCTS: resetSearch');
    set((state) => {
      state.mcts.searchConfig = initialState.searchConfig;
      state.mcts.currentResult = initialState.currentResult;
      state.mcts.evaluateResult = initialState.evaluateResult;
      state.mcts.isSearching = initialState.isSearching;
      state.mcts.isEvaluating = initialState.isEvaluating;
      state.mcts.searchProgress = initialState.searchProgress;
      state.mcts.error = initialState.error;
    });
  },
});
