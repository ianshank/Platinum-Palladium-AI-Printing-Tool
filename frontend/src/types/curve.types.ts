/**
 * Curve-related types.
 * These types are used throughout the application for curve data and operations.
 */

/**
 * Curve type enum (mirrors backend CurveType)
 */
export type CurveType =
  | 'linear'
  | 'spline'
  | 'polynomial'
  | 'paper_white'
  | 'aesthetic'
  | 'gamma'
  | 'perceptual';

/**
 * Core curve data structure
 */
export interface CurveData {
  id: string;
  name: string;
  created_at: string;
  curve_type: CurveType;
  paper_type?: string;
  chemistry?: string;
  notes?: string;
  input_values: number[];
  output_values: number[];
  source_extraction_id?: string;
}

/**
 * Curve statistics
 */
export interface CurveStatistics {
  dmin: number;
  dmax: number;
  range: number;
  midpoint: number;
  gamma: number;
  isMonotonic: boolean;
  maxError: number;
  rmsError: number;
}

/**
 * Curve modification options
 */
export interface CurveModification {
  type:
    | 'brightness'
    | 'contrast'
    | 'gamma'
    | 'levels'
    | 'highlights'
    | 'shadows'
    | 'midtones';
  amount: number;
  pivot?: number;
  blackPoint?: number;
  whitePoint?: number;
}

/**
 * Curve smoothing options
 */
export interface CurveSmoothingOptions {
  method: 'gaussian' | 'savgol' | 'moving_average' | 'spline';
  strength: number;
  preserveEndpoints: boolean;
}

/**
 * Curve blend options
 */
export interface CurveBlendOptions {
  mode: 'average' | 'weighted' | 'multiply' | 'screen' | 'overlay' | 'min' | 'max';
  weight: number;
}

/**
 * Curve enhancement goals
 */
export type EnhancementGoal =
  | 'linearization'
  | 'maximize_range'
  | 'smooth_gradation'
  | 'highlight_detail'
  | 'shadow_detail'
  | 'neutral_midtones'
  | 'print_stability';

/**
 * Curve comparison result
 */
export interface CurveComparisonResult {
  curves: CurveData[];
  differences: {
    inputValue: number;
    values: number[];
    maxDifference: number;
  }[];
  overallSimilarity: number;
}

/**
 * QuadTone RIP profile data
 */
export interface QuadProfile {
  profile_name: string;
  resolution: number;
  ink_limit: number;
  media_type?: string;
  all_channels: string[];
  active_channels: string[];
  curve_data?: {
    input_values: number[];
    output_values: number[];
  };
}

/**
 * Curve point for editing
 */
export interface CurvePoint {
  x: number;
  y: number;
  selected?: boolean;
  locked?: boolean;
}

/**
 * Curve editor state
 */
export interface CurveEditorState {
  curve: CurveData | null;
  originalCurve: CurveData | null;
  selectedPoints: number[];
  zoomLevel: number;
  showGrid: boolean;
  showLinear: boolean;
  modifications: CurveModification[];
  undoStack: CurveData[];
  redoStack: CurveData[];
}
