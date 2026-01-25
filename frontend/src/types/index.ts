/**
 * Shared TypeScript type definitions
 */

// ============================================================================
// Common Types
// ============================================================================

/**
 * Generic API response wrapper
 */
export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: 'success' | 'error';
}

/**
 * Pagination parameters
 */
export interface PaginationParams {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

/**
 * Paginated response
 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

// ============================================================================
// Calibration Types
// ============================================================================

/**
 * Step tablet types
 */
export type TabletType = '21-step' | '31-step' | '41-step' | 'custom';

/**
 * Curve interpolation methods
 */
export type InterpolationType = 'linear' | 'cubic' | 'monotonic' | 'pchip';

/**
 * Export formats
 */
export type ExportFormat = 'qtr' | 'piezography' | 'csv' | 'json' | 'acv' | 'cube';

/**
 * Density measurement from a step tablet
 */
export interface DensityMeasurement {
  step: number;
  targetDensity: number;
  measuredDensity: number;
  deltaE?: number;
}

/**
 * Point on a calibration curve
 */
export interface CurvePoint {
  x: number;
  y: number;
  isControlPoint?: boolean;
}

/**
 * Calibration curve data
 */
export interface CurveData {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  type: InterpolationType;
  points: CurvePoint[];
  calibrationId?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Calibration record
 */
export interface CalibrationRecord {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  tabletType: TabletType;
  measurements: DensityMeasurement[];
  curveId?: string;
  notes?: string;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Chemistry Types
// ============================================================================

/**
 * Coating methods
 */
export type CoatingMethod = 'brush' | 'rod' | 'puddle';

/**
 * Developer types
 */
export type DeveloperType = 'potassium_oxalate' | 'ammonium_citrate';

/**
 * Contrast agent types
 */
export type ContrastAgentType = 'na2' | 'dichromate' | 'none';

/**
 * Paper size definition
 */
export interface PaperSize {
  name: string;
  widthInches: number;
  heightInches: number;
  custom?: boolean;
}

/**
 * Chemistry recipe
 */
export interface ChemistryRecipe {
  totalVolume: number;
  platinumMl: number;
  palladiumMl: number;
  ferricOxalateMl: number;
  contrastAgent?: {
    type: ContrastAgentType;
    amount: number;
  };
  developer: {
    type: DeveloperType;
    concentration: number;
    temperatureC: number;
  };
}

// ============================================================================
// Image Types
// ============================================================================

/**
 * Image metadata
 */
export interface ImageMetadata {
  id: string;
  name: string;
  type: string;
  size: number;
  width: number;
  height: number;
  url: string;
  dataUrl?: string;
  thumbnail?: string;
  uploadedAt: string;
}

/**
 * Image histogram data
 */
export interface ImageHistogram {
  red: number[];
  green: number[];
  blue: number[];
  luminance: number[];
}

// ============================================================================
// Session Types
// ============================================================================

/**
 * Print result status
 */
export type PrintResult = 'success' | 'partial' | 'failure';

/**
 * Print session record
 */
export interface PrintRecord {
  id: string;
  timestamp: string;
  paperType: string;
  paperSize: string;
  chemistry: {
    platinumMl: number;
    palladiumMl: number;
    metalRatio: number;
  };
  exposure: {
    timeSeconds: number;
    uvSource: string;
  };
  result: PrintResult;
  notes?: string;
  imageUrl?: string;
  calibrationId?: string;
  curveId?: string;
}

/**
 * Session statistics
 */
export interface SessionStats {
  totalPrints: number;
  successRate: number;
  averageExposure: number;
  mostUsedPaper: string | null;
  recentActivity: Array<{
    date: string;
    count: number;
  }>;
}

// ============================================================================
// Chat Types
// ============================================================================

/**
 * Chat message role
 */
export type ChatRole = 'user' | 'assistant' | 'system';

/**
 * Chat message
 */
export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  timestamp: string;
  metadata?: {
    model?: string;
    tokens?: number;
    context?: string[];
  };
}

/**
 * Chat context item
 */
export interface ChatContext {
  id: string;
  label: string;
  content: string;
}

// ============================================================================
// UI Types
// ============================================================================

/**
 * Toast variants
 */
export type ToastVariant = 'default' | 'success' | 'warning' | 'error';

/**
 * Toast notification
 */
export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant?: ToastVariant;
  duration?: number;
}

/**
 * Theme mode
 */
export type ThemeMode = 'light' | 'dark';

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Make all properties optional recursively
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Extract value type from object
 */
export type ValueOf<T> = T[keyof T];

/**
 * Make specific properties required
 */
export type RequireKeys<T, K extends keyof T> = T & Required<Pick<T, K>>;

/**
 * Async function return type
 */
export type AsyncReturnType<T extends (...args: unknown[]) => Promise<unknown>> =
  T extends (...args: unknown[]) => Promise<infer R> ? R : never;
