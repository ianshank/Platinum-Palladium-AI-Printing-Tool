/**
 * MCTS API client
 * Provides typed API calls for MCTS calibration endpoints
 */

import { apiClient } from './client';
import type {
  MCTSEvaluateRequest,
  MCTSEvaluateResponse,
  MCTSExportFormat,
  MCTSFeedbackRequest,
  MCTSRecommendation,
  MCTSRecommendationsResponse,
  MCTSSearchRequest,
  MCTSSearchResponse,
  MCTSStatusResponse,
  MCTSTrainingStatus,
  MCTSTrainRequest,
  MCTSTrainResponse,
} from '@/types/mcts';

/**
 * MCTS API endpoints
 */
export const mctsApi = {
  /**
   * Run MCTS search for optimal calibration parameters
   */
  search: async (request: MCTSSearchRequest): Promise<MCTSSearchResponse> => {
    // Convert camelCase to snake_case for API
    const apiRequest = {
      paper_type: request.paperType,
      uv_source: request.uvSource,
      target_curve: request.targetCurve,
      fixed_parameters: request.fixedParameters,
      target_aesthetics: request.targetAesthetics,
      num_simulations: request.numSimulations,
    };

    const response = await apiClient.post<{
      search_id: string;
      best_parameters: Record<string, number>;
      predicted_curve: number[];
      quality_score: number;
      alternatives: Record<string, number>[];
      search_time_seconds: number;
      num_simulations: number;
    }>('/api/mcts/search', apiRequest);

    // Convert snake_case to camelCase for frontend
    return {
      searchId: response.data.search_id,
      bestParameters: response.data.best_parameters,
      predictedCurve: response.data.predicted_curve,
      qualityScore: response.data.quality_score,
      alternatives: response.data.alternatives,
      searchTimeSeconds: response.data.search_time_seconds,
      numSimulations: response.data.num_simulations,
    };
  },

  /**
   * Evaluate a parameter set against the simulator
   */
  evaluate: async (
    request: MCTSEvaluateRequest
  ): Promise<MCTSEvaluateResponse> => {
    const response = await apiClient.post<{
      density_curve: number[];
      dmin: number;
      dmax: number;
      density_range: number;
      gamma: number;
      quality_score: number;
    }>('/api/mcts/evaluate', request);

    return {
      densityCurve: response.data.density_curve,
      dmin: response.data.dmin,
      dmax: response.data.dmax,
      densityRange: response.data.density_range,
      gamma: response.data.gamma,
      qualityScore: response.data.quality_score,
    };
  },

  /**
   * Get MCTS engine status
   */
  getStatus: async (): Promise<MCTSStatusResponse> => {
    const response = await apiClient.get<{
      engine_ready: boolean;
      networks_loaded: boolean;
      torch_available: boolean;
      parameter_ranges: Record<
        string,
        { min: number; max: number; default: number; unit: string }
      >;
    }>('/api/mcts/status');

    return {
      engineReady: response.data.engine_ready,
      networksLoaded: response.data.networks_loaded,
      torchAvailable: response.data.torch_available,
      parameterRanges: response.data.parameter_ranges,
    };
  },

  /**
   * Get parameter recommendations
   */
  getRecommendations: async (
    paperType?: string,
    limit: number = 5
  ): Promise<MCTSRecommendation[]> => {
    const response = await apiClient.get<MCTSRecommendationsResponse>(
      '/api/mcts/recommendations',
      {
        params: {
          paper_type: paperType,
          limit,
        },
      }
    );

    return response.data.recommendations;
  },

  /**
   * Start training session
   */
  startTraining: async (
    request: MCTSTrainRequest = {}
  ): Promise<MCTSTrainResponse> => {
    const response = await apiClient.post<{
      session_id: string;
      status: string;
      message: string;
    }>('/api/mcts/train', {
      num_episodes: request.numEpisodes,
    });

    return {
      sessionId: response.data.session_id,
      status: response.data.status,
      message: response.data.message,
    };
  },

  /**
   * Get training status
   */
  getTrainingStatus: async (sessionId: string): Promise<MCTSTrainingStatus> => {
    const response = await apiClient.get<{
      session_id: string;
      status: 'starting' | 'training' | 'completed' | 'failed';
      episodes_completed: number;
      num_episodes: number;
      error: string | null;
    }>(`/api/mcts/train/${sessionId}/status`);

    return {
      sessionId: response.data.session_id,
      status: response.data.status,
      episodesCompleted: response.data.episodes_completed,
      numEpisodes: response.data.num_episodes,
      error: response.data.error,
    };
  },

  /**
   * Submit real measurement feedback
   */
  submitFeedback: async (
    feedback: MCTSFeedbackRequest
  ): Promise<{ success: boolean; message: string }> => {
    const response = await apiClient.post<{
      success: boolean;
      message: string;
    }>('/api/mcts/feedback', {
      parameters: feedback.parameters,
      measured_curve: feedback.measuredCurve,
      quality_rating: feedback.qualityRating,
    });

    return response.data;
  },

  /**
   * Export calibration result
   */
  export: async (
    parameters: Record<string, number>,
    format: MCTSExportFormat = 'json'
  ): Promise<unknown> => {
    const response = await apiClient.post<unknown>(
      '/api/mcts/export',
      parameters,
      {
        params: { format },
      }
    );

    return response.data;
  },
};
