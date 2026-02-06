/**
 * API request and response types.
 * These types mirror the backend Pydantic models.
 */

// Generic API response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Health check
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version?: string;
}

// Curve generation
export interface CurveGenerateRequest {
  densities: number[];
  name?: string;
  curve_type?: string;
  paper_type?: string;
  chemistry?: string;
}

export interface CurveGenerateResponse {
  success: boolean;
  curve_id: string;
  name: string;
  num_points: number;
  input_values: number[];
  output_values: number[];
}

// Curve modification
export interface CurveModifyRequest {
  input_values: number[];
  output_values: number[];
  name?: string;
  adjustment_type: string;
  amount: number;
  pivot?: number;
  black_point?: number;
  white_point?: number;
}

export interface CurveModifyResponse {
  success: boolean;
  curve_id: string;
  name: string;
  adjustment_applied: string;
  input_values: number[];
  output_values: number[];
}

// Curve smoothing
export interface CurveSmoothRequest {
  input_values: number[];
  output_values: number[];
  name?: string;
  method: string;
  strength: number;
  preserve_endpoints?: boolean;
}

// Curve blending
export interface CurveBlendRequest {
  curve1_inputs: number[];
  curve1_outputs: number[];
  curve2_inputs: number[];
  curve2_outputs: number[];
  name?: string;
  mode: string;
  weight?: number;
}

// Curve enhancement
export interface CurveEnhanceRequest {
  input_values: number[];
  output_values: number[];
  name?: string;
  goal: string;
  paper_type?: string;
  additional_context?: string;
}

export interface CurveEnhanceResponse {
  success: boolean;
  curve_id: string;
  name: string;
  goal: string;
  confidence: number;
  analysis: string;
  changes_made: string[];
  input_values: number[];
  output_values: number[];
}

// Scan upload
export interface ScanUploadResponse {
  success: boolean;
  extraction_id: string;
  num_patches: number;
  densities: number[];
  dmin: number;
  dmax: number;
  range: number;
  quality: number;
  warnings: string[];
}

// Density analysis
export interface AnalyzeRequest {
  densities: number[];
}

export interface AnalyzeResponse {
  dmin: number;
  dmax: number;
  range: number;
  is_monotonic: boolean;
  max_error: number;
  rms_error: number;
  suggestions: string[];
}

// Calibration
export interface CalibrationRequest {
  paper_type: string;
  exposure_time: number;
  metal_ratio?: number;
  contrast_agent?: string;
  contrast_amount?: number;
  developer?: string;
  chemistry_type?: string;
  densities?: number[];
  notes?: string;
}

export interface CalibrationRecord {
  id: string;
  paper_type: string;
  exposure_time: number;
  metal_ratio: number;
  contrast_agent: string;
  contrast_amount: number;
  developer: string;
  chemistry_type: string;
  measured_densities: number[];
  timestamp: string;
  notes?: string;
  dmax?: number;
}

export interface CalibrationListResponse {
  count: number;
  records: CalibrationRecord[];
}

// Chat
export interface ChatRequest {
  message: string;
  include_history?: boolean;
}

export interface ChatResponse {
  response: string;
}

export interface RecipeRequest {
  paper_type: string;
  characteristics: string;
}

export interface TroubleshootRequest {
  problem: string;
}

// Statistics
export interface StatisticsResponse {
  total_calibrations: number;
  total_curves: number;
  paper_types: string[];
  recent_activity: {
    date: string;
    count: number;
  }[];
}

// Chemistry calculation
export interface ChemistryCalculateRequest {
  width_inches: number;
  height_inches: number;
  platinum_ratio?: number;
  paper_absorbency?: string;
  coating_method?: string;
  contrast_boost?: number;
  na2_ratio?: number;
  margin_inches?: number;
  include_cost?: boolean;
}

export interface ChemistryRecipeResponse {
  print_dimensions: {
    width_inches: number;
    height_inches: number;
    coating_width_inches: number;
    coating_height_inches: number;
    coating_area_sq_inches: number;
  };
  drops: {
    ferric_oxalate_1: number;
    ferric_oxalate_2_contrast: number;
    palladium: number;
    platinum: number;
    na2: number;
    total: number;
  };
  milliliters: {
    ferric_oxalate_1: number;
    ferric_oxalate_2_contrast: number;
    palladium: number;
    platinum: number;
    na2: number;
    total: number;
  };
  metal_ratios: {
    platinum_percent: number;
    palladium_percent: number;
  };
  settings: {
    paper_absorbency: string;
    coating_method: string;
    contrast_boost: number;
  };
  estimated_cost_usd: number | null;
  notes: string[];
}
