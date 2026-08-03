/**
 * MCTS slice tests
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { createStore } from '@/stores';
import type {
  MCTSEvaluateResponse,
  MCTSRecommendation,
  MCTSSearchResponse,
  MCTSStatusResponse,
} from '@/types/mcts';

describe('mctsSlice', () => {
  let store: ReturnType<typeof createStore>;

  beforeEach(() => {
    store = createStore();
  });

  describe('initial state', () => {
    it('should have correct initial values', () => {
      const state = store.getState().mcts;

      expect(state.searchConfig).toEqual({
        fixedParameters: {},
        targetAesthetics: {},
      });
      expect(state.currentResult).toBeNull();
      expect(state.evaluateResult).toBeNull();
      expect(state.isSearching).toBe(false);
      expect(state.isEvaluating).toBe(false);
      expect(state.searchProgress).toEqual({
        iteration: 0,
        total: 0,
        bestScore: 0,
      });
      expect(state.trainingStatus).toBe('idle');
      expect(state.recommendations).toEqual([]);
      expect(state.status).toBeNull();
      expect(state.error).toBeNull();
    });
  });

  describe('setSearchConfig', () => {
    it('should update search configuration', () => {
      const config = {
        paperType: 'arches_platine',
        uvSource: 'sun',
        numSimulations: 1000,
      };

      store.getState().mcts.setSearchConfig(config);

      const state = store.getState().mcts;
      expect(state.searchConfig.paperType).toBe('arches_platine');
      expect(state.searchConfig.uvSource).toBe('sun');
      expect(state.searchConfig.numSimulations).toBe(1000);
    });

    it('should merge partial config with existing', () => {
      store.getState().mcts.setSearchConfig({ paperType: 'arches_platine' });
      store.getState().mcts.setSearchConfig({ uvSource: 'sun' });

      const state = store.getState().mcts;
      expect(state.searchConfig.paperType).toBe('arches_platine');
      expect(state.searchConfig.uvSource).toBe('sun');
    });
  });

  describe('setFixedParameter and removeFixedParameter', () => {
    it('should add fixed parameter', () => {
      store.getState().mcts.setFixedParameter('exposure_time', 120);

      const state = store.getState().mcts;
      expect(state.searchConfig.fixedParameters?.['exposure_time']).toBe(120);
    });

    it('should update existing fixed parameter', () => {
      store.getState().mcts.setFixedParameter('exposure_time', 120);
      store.getState().mcts.setFixedParameter('exposure_time', 150);

      const state = store.getState().mcts;
      expect(state.searchConfig.fixedParameters?.['exposure_time']).toBe(150);
    });

    it('should remove fixed parameter', () => {
      store.getState().mcts.setFixedParameter('exposure_time', 120);
      store.getState().mcts.removeFixedParameter('exposure_time');

      const state = store.getState().mcts;
      expect(state.searchConfig.fixedParameters?.['exposure_time']).toBeUndefined();
    });
  });

  describe('setSearching', () => {
    it('should set searching state', () => {
      store.getState().mcts.setSearching(true);
      expect(store.getState().mcts.isSearching).toBe(true);

      store.getState().mcts.setSearching(false);
      expect(store.getState().mcts.isSearching).toBe(false);
    });

    it('should reset progress when stopping search', () => {
      store.getState().mcts.setSearchProgress({ iteration: 50, total: 100, bestScore: 0.8 });
      store.getState().mcts.setSearching(false);

      const state = store.getState().mcts;
      expect(state.searchProgress).toEqual({ iteration: 0, total: 0, bestScore: 0 });
    });
  });

  describe('setCurrentResult', () => {
    it('should set search result', () => {
      const result: MCTSSearchResponse = {
        searchId: 'test-123',
        bestParameters: { exposure_time: 120 },
        predictedCurve: [0, 0.5, 1.0],
        qualityScore: 0.85,
        alternatives: [],
        searchTimeSeconds: 12.5,
        numSimulations: 500,
      };

      store.getState().mcts.setCurrentResult(result);

      expect(store.getState().mcts.currentResult).toEqual(result);
    });

    it('should clear result when null', () => {
      const result: MCTSSearchResponse = {
        searchId: 'test-123',
        bestParameters: {},
        predictedCurve: [],
        qualityScore: 0.85,
        alternatives: [],
        searchTimeSeconds: 12.5,
        numSimulations: 500,
      };

      store.getState().mcts.setCurrentResult(result);
      store.getState().mcts.setCurrentResult(null);

      expect(store.getState().mcts.currentResult).toBeNull();
    });
  });

  describe('setEvaluateResult', () => {
    it('should set evaluation result', () => {
      const result: MCTSEvaluateResponse = {
        densityCurve: [0, 0.5, 1.0],
        dmin: 0.05,
        dmax: 1.85,
        densityRange: 1.80,
        gamma: 2.2,
        qualityScore: 0.9,
      };

      store.getState().mcts.setEvaluateResult(result);

      expect(store.getState().mcts.evaluateResult).toEqual(result);
    });
  });

  describe('setTrainingStatus', () => {
    it('should update training status', () => {
      store.getState().mcts.setTrainingStatus('training');
      expect(store.getState().mcts.trainingStatus).toBe('training');

      store.getState().mcts.setTrainingStatus('completed');
      expect(store.getState().mcts.trainingStatus).toBe('completed');

      store.getState().mcts.setTrainingStatus('failed');
      expect(store.getState().mcts.trainingStatus).toBe('failed');
    });
  });

  describe('setRecommendations', () => {
    it('should set recommendations', () => {
      const recommendations: MCTSRecommendation[] = [
        {
          parameters: { exposure_time: 120 },
          predictedQuality: 0.85,
          rationale: 'Test recommendation',
        },
      ];

      store.getState().mcts.setRecommendations(recommendations);

      expect(store.getState().mcts.recommendations).toEqual(recommendations);
    });
  });

  describe('setStatus', () => {
    it('should set engine status', () => {
      const status: MCTSStatusResponse = {
        engineReady: true,
        networksLoaded: true,
        torchAvailable: true,
        parameterRanges: {
          exposure_time: { min: 60, max: 300, default: 120, unit: 'seconds' },
        },
      };

      store.getState().mcts.setStatus(status);

      expect(store.getState().mcts.status).toEqual(status);
    });
  });

  describe('setError', () => {
    it('should set error message', () => {
      store.getState().mcts.setError('Test error');
      expect(store.getState().mcts.error).toBe('Test error');
    });

    it('should clear error when null', () => {
      store.getState().mcts.setError('Test error');
      store.getState().mcts.setError(null);
      expect(store.getState().mcts.error).toBeNull();
    });
  });

  describe('resetSearch', () => {
    it('should reset all search state to initial values', () => {
      // Set some state
      store.getState().mcts.setSearchConfig({ paperType: 'test', numSimulations: 1000 });
      store.getState().mcts.setSearching(true);
      store.getState().mcts.setCurrentResult({
        searchId: 'test',
        bestParameters: {},
        predictedCurve: [],
        qualityScore: 0.8,
        alternatives: [],
        searchTimeSeconds: 10,
        numSimulations: 500,
      });
      store.getState().mcts.setError('Some error');

      // Reset
      store.getState().mcts.resetSearch();

      // Verify reset
      const state = store.getState().mcts;
      expect(state.searchConfig).toEqual({
        fixedParameters: {},
        targetAesthetics: {},
      });
      expect(state.currentResult).toBeNull();
      expect(state.evaluateResult).toBeNull();
      expect(state.isSearching).toBe(false);
      expect(state.isEvaluating).toBe(false);
      expect(state.searchProgress).toEqual({ iteration: 0, total: 0, bestScore: 0 });
      expect(state.error).toBeNull();
    });
  });
});
