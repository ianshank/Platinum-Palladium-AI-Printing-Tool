/**
 * TanStack Query hooks for API operations
 * Provides caching, loading states, and error handling
 *
 * This file serves as the main entry point for all API hooks.
 * Domain-specific hooks are organized in:
 * - hooks-curves.ts: Curve generation, modification, enhancement
 * - hooks-calibration.ts: Calibration and scan upload
 * - hooks-chat.ts: Chat messages, recipes, troubleshooting
 * - hooks-mcts.ts: Machine learning recommendations
 */

import {
  useQuery,
  type UseQueryOptions,
} from '@tanstack/react-query';
import { api, type ApiError, type AxiosError } from './client';
import type { StatisticsResponse } from '@/types/models';

// ============================================================================
// Query Key Factory
// ============================================================================

/**
 * Centralized query key factory for consistent cache invalidation
 * Used across all API hook files
 */
export const queryKeys = {
  all: ['ptpd'] as const,
  health: () => [...queryKeys.all, 'health'] as const,
  curves: () => [...queryKeys.all, 'curves'] as const,
  curve: (id: string) => [...queryKeys.curves(), id] as const,
  calibrations: () => [...queryKeys.all, 'calibrations'] as const,
  calibration: (id: string) => [...queryKeys.calibrations(), id] as const,
  statistics: () => [...queryKeys.all, 'statistics'] as const,
};

// ============================================================================
// Health Check
// ============================================================================

/**
 * Check backend health status
 */
export function useHealthCheck(
  options?: Omit<
    UseQueryOptions<{ status: string }, AxiosError<ApiError>>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: queryKeys.health(),
    queryFn: () => api.health.check(),
    staleTime: 30000, // 30 seconds
    ...options,
  });
}

// ============================================================================
// Statistics
// ============================================================================

/**
 * Fetch system statistics (calibration count, curve count, etc.)
 */
export function useStatistics(
  options?: Omit<
    UseQueryOptions<StatisticsResponse, AxiosError<ApiError>>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: queryKeys.statistics(),
    queryFn: () => api.statistics.get(),
    ...options,
  });
}

// ============================================================================
// Backward Compatibility Exports
// ============================================================================
// Re-export all domain-specific hooks for backward compatibility
// This ensures existing imports like `import { useCurve } from '@/api/hooks'` still work

/**
 * Curve hooks - see hooks-curves.ts for details
 */
export {
  useCurve,
  useGenerateCurve,
  useModifyCurve,
  useSmoothCurve,
  useSaveCurve,
  useEnhanceCurve,
  useExportCurve,
  useUploadQuadFile,
  useParseQuadContent,
} from './hooks-curves';

/**
 * Calibration hooks - see hooks-calibration.ts for details
 */
export {
  useCalibrations,
  useCalibration,
  useCreateCalibration,
  useUploadScan,
} from './hooks-calibration';

/**
 * Chat and LLM hooks - see hooks-chat.ts for details
 */
export {
  useSendMessage,
  useRecipeSuggestion,
  useTroubleshootRequest,
} from './hooks-chat';

/**
 * MCTS hooks - see hooks-mcts.ts for details
 */
export {
  useMCTSStatus,
  useMCTSSearch,
  useMCTSEvaluate,
  useMCTSRecommendations,
  useMCTSTraining,
  useMCTSTrainingStatus,
  useMCTSFeedback,
  useMCTSExport,
  mctsQueryKeys,
} from './hooks-mcts';

// ============================================================================
// Scan Quality Assessment
// ============================================================================

/** Input shape for scan quality assessment (matches Step2Analyze usage) */
export interface ScanAnalysisInput {
  densities: number[];
  dmax: number;
  dmin: number;
  range: number;
  num_patches: number;
}

export interface QualityIssue {
  type: 'warning' | 'error';
  message: string;
  suggestion?: string;
}

export interface QualityAssessment {
  quality: string;
  score: number;
  overall: 'excellent' | 'good' | 'acceptable' | 'poor';
  issues: QualityIssue[];
}

// ── Quality assessment thresholds ───────────────────────────
/** Density range below this is considered low — likely under-exposed. */
const DENSITY_RANGE_LOW = 0.5;
/** Density range below this is moderate — exposure may need adjustment. */
const DENSITY_RANGE_MODERATE = 1.0;
/** Minimum patches needed for a reliable calibration curve. */
const MIN_PATCH_COUNT = 11;
/** Dmax below this means insufficient shadow detail. */
const DMAX_LOW_THRESHOLD = 0.8;

// ── Score penalty weights ───────────────────────────────────
const PENALTY_LOW_RANGE = 20;
const PENALTY_MODERATE_RANGE = 10;
const PENALTY_LOW_PATCHES = 30;
const PENALTY_LOW_DMAX = 15;

// ── Overall grade boundaries ────────────────────────────────
const GRADE_EXCELLENT = 90;
const GRADE_GOOD = 70;
const GRADE_ACCEPTABLE = 50;

/**
 * Assess scan quality from extracted density data.
 * Returns quality grade, score (0–100), and any issues found.
 */
export const assessScanQuality = (
  input: ScanAnalysisInput
): QualityAssessment => {
  const issues: QualityIssue[] = [];
  let score = 100;

  // Check density range
  if (input.range < DENSITY_RANGE_LOW) {
    issues.push({
      type: 'warning',
      message: 'Low density range',
      suggestion: 'Increase exposure time or check chemistry',
    });
    score -= PENALTY_LOW_RANGE;
  } else if (input.range < DENSITY_RANGE_MODERATE) {
    issues.push({
      type: 'warning',
      message: 'Moderate density range',
      suggestion: 'Consider adjusting exposure',
    });
    score -= PENALTY_MODERATE_RANGE;
  }

  // Check patch count
  if (input.num_patches < MIN_PATCH_COUNT) {
    issues.push({
      type: 'error',
      message: `Only ${input.num_patches} patches detected`,
      suggestion: 'Rescan with better alignment',
    });
    score -= PENALTY_LOW_PATCHES;
  }

  // Check Dmax
  if (input.dmax < DMAX_LOW_THRESHOLD) {
    issues.push({
      type: 'warning',
      message: 'Low Dmax — print may lack shadow detail',
      suggestion: 'Check chemistry concentration',
    });
    score -= PENALTY_LOW_DMAX;
  }

  const clampedScore = Math.max(0, Math.min(100, score));
  const overall =
    clampedScore >= GRADE_EXCELLENT
      ? 'excellent'
      : clampedScore >= GRADE_GOOD
        ? 'good'
        : clampedScore >= GRADE_ACCEPTABLE
          ? 'acceptable'
          : 'poor';

  return {
    quality: overall,
    score: clampedScore,
    overall,
    issues,
  };
};
