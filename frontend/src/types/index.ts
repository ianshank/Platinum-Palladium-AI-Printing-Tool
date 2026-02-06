/**
 * Type exports.
 * Re-exports all types from individual modules.
 */

// API types
export type {
  ApiResponse,
  HealthResponse,
  CurveGenerateRequest,
  CurveGenerateResponse,
  CurveModifyRequest,
  CurveModifyResponse,
  CurveSmoothRequest,
  CurveBlendRequest,
  CurveEnhanceRequest,
  CurveEnhanceResponse,
  ScanUploadResponse,
  AnalyzeRequest,
  AnalyzeResponse,
  CalibrationRequest,
  CalibrationRecord,
  CalibrationListResponse,
  ChatRequest,
  ChatResponse,
  RecipeRequest,
  TroubleshootRequest,
  StatisticsResponse,
  ChemistryCalculateRequest,
  ChemistryRecipeResponse,
} from './api.types';

// Curve types
export type {
  CurveType,
  CurveData,
  CurveStatistics,
  CurveModification,
  CurveSmoothingOptions,
  CurveBlendOptions,
  EnhancementGoal,
  CurveComparisonResult,
  QuadProfile,
  CurvePoint,
  CurveEditorState,
} from './curve.types';

// Chemistry types
export type {
  PaperAbsorbency,
  CoatingMethod,
  ContrastAgent,
  DeveloperType,
  ChemistryType,
  PrintDimensions,
  CoatingArea,
  SolutionAmounts,
  MetalRatio,
  ChemistryRecipe,
  ChemistryPreset,
  PaperProfile,
  ChemistryInventory,
  ChemistryCalculationInput,
  ExposureCalculationInput,
  ExposureCalculationResult,
} from './chemistry.types';

// Session types
export type {
  PrintResult,
  PrintRecord,
  PrintSession,
  SessionStatistics,
  DashboardMetrics,
  UserPreferences,
  Notification,
} from './session.types';

/**
 * Common utility types
 */

// Nullable type helper
export type Nullable<T> = T | null;

// Optional type helper
export type Optional<T> = T | undefined;

// Result type for operations that can fail
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

// Async operation state
export interface AsyncState<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
}

// Pagination params
export interface PaginationParams {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

// Paginated response
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

// Filter params
export interface FilterParams {
  search?: string;
  dateFrom?: string;
  dateTo?: string;
  [key: string]: string | undefined;
}
