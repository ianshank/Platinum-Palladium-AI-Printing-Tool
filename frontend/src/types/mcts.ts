/**
 * MCTS (Monte Carlo Tree Search) calibration types
 * Matches Python API response shapes from src/ptpd_calibration/api/mcts_router.py
 */

/**
 * Parameter range definition
 */
export interface ParameterRange {
  name: string;
  minValue: number;
  maxValue: number;
  defaultValue: number;
  step?: number;
  unit: string;
}

/**
 * MCTS search request
 */
export interface MCTSSearchRequest {
  paperType?: string | undefined;
  uvSource?: string | undefined;
  targetCurve?: number[];
  fixedParameters?: Record<string, number>;
  targetAesthetics?: Record<string, number>;
  numSimulations?: number;
}

/**
 * MCTS search response
 */
export interface MCTSSearchResponse {
  searchId: string;
  bestParameters: Record<string, number>;
  predictedCurve: number[];
  qualityScore: number;
  alternatives: Record<string, number>[];
  searchTimeSeconds: number;
  numSimulations: number;
}

/**
 * MCTS evaluate request
 */
export interface MCTSEvaluateRequest {
  parameters: Record<string, number>;
}

/**
 * MCTS evaluate response
 */
export interface MCTSEvaluateResponse {
  densityCurve: number[];
  dmin: number;
  dmax: number;
  densityRange: number;
  gamma: number;
  qualityScore: number;
}

/**
 * MCTS status response
 */
export interface MCTSStatusResponse {
  engineReady: boolean;
  networksLoaded: boolean;
  torchAvailable: boolean;
  parameterRanges: Record<string, { min: number; max: number; default: number; unit: string }>;
}

/**
 * MCTS recommendation
 */
export interface MCTSRecommendation {
  parameters: Record<string, number>;
  predictedQuality: number;
  rationale: string;
}

/**
 * MCTS feedback request
 */
export interface MCTSFeedbackRequest {
  parameters: Record<string, number>;
  measuredCurve: number[];
  qualityRating: number;
}

/**
 * Search progress info
 */
export interface MCTSSearchProgress {
  iteration: number;
  total: number;
  bestScore: number;
}

/**
 * Training request
 */
export interface MCTSTrainRequest {
  numEpisodes?: number;
}

/**
 * Training response
 */
export interface MCTSTrainResponse {
  sessionId: string;
  status: string;
  message: string;
}

/**
 * Training status
 */
export interface MCTSTrainingStatus {
  sessionId: string;
  status: 'starting' | 'training' | 'completed' | 'failed';
  episodesCompleted: number;
  numEpisodes: number;
  error: string | null;
}

/**
 * Export format options
 */
export type MCTSExportFormat = 'json' | 'csv' | 'recipe';

/**
 * Recommendations response wrapper
 */
export interface MCTSRecommendationsResponse {
  recommendations: MCTSRecommendation[];
}
