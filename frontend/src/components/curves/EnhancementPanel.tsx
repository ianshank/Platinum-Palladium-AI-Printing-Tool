/**
 * AI curve enhancement panel
 * Provides interface for AI-powered curve optimization
 */

import React, { useCallback, useMemo } from 'react';
import { Button } from '@/components/ui/Button';
import type { CurveEnhanceResponse } from '@/types/models';
import { logger } from '@/lib/logger';
import { cn } from '@/lib/utils';

export type EnhancementGoal =
  | 'linearization'
  | 'maximize_range'
  | 'smooth_gradation'
  | 'highlight_detail'
  | 'shadow_detail'
  | 'neutral_midtones'
  | 'print_stability';

const ENHANCEMENT_GOALS: readonly {
  value: EnhancementGoal;
  label: string;
}[] = [
  { value: 'linearization', label: 'Linearization' },
  { value: 'maximize_range', label: 'Maximize Range' },
  { value: 'smooth_gradation', label: 'Smooth Gradation' },
  { value: 'highlight_detail', label: 'Highlight Detail' },
  { value: 'shadow_detail', label: 'Shadow Detail' },
  { value: 'neutral_midtones', label: 'Neutral Midtones' },
  { value: 'print_stability', label: 'Print Stability' },
] as const;

export interface EnhancementPanelProps {
  /** Currently selected enhancement goal */
  enhancementGoal: EnhancementGoal;
  /** Callback when enhancement goal changes */
  onGoalChange: (goal: EnhancementGoal) => void;
  /** Whether enhancement request is in progress */
  isEnhancing: boolean;
  /** Callback when Enhance button is clicked */
  onEnhance: () => void;
  /** Result of enhancement request (if completed) */
  enhanceResult?: CurveEnhanceResponse | null;
  /** Callback to dismiss enhancement result panel */
  onDismissResult: () => void;
  /** Whether result panel is shown */
  showResult: boolean;
  /** Optional CSS class */
  className?: string;
}

/**
 * Panel for AI-powered curve enhancement
 * Allows selection of optimization goals and displays results
 *
 * @example
 * ```tsx
 * <EnhancementPanel
 *   enhancementGoal="linearization"
 *   onGoalChange={setGoal}
 *   isEnhancing={false}
 *   onEnhance={handleEnhance}
 *   enhanceResult={result}
 *   onDismissResult={dismissResult}
 *   showResult={showPanel}
 * />
 * ```
 */
export const EnhancementPanel = React.forwardRef<
  HTMLDivElement,
  EnhancementPanelProps
>(
  (
    {
      enhancementGoal,
      onGoalChange,
      isEnhancing,
      onEnhance,
      enhanceResult,
      onDismissResult,
      showResult,
      className,
    },
    ref
  ) => {
    const handleGoalChange = useCallback(
      (e: React.ChangeEvent<HTMLSelectElement>) => {
        const goal = e.target.value as EnhancementGoal;
        logger.debug('EnhancementPanel: goal changed', { goal });
        onGoalChange(goal);
      },
      [onGoalChange]
    );

    const handleEnhanceClick = useCallback(() => {
      logger.info('EnhancementPanel: enhance clicked', {
        goal: enhancementGoal,
      });
      onEnhance();
    }, [enhancementGoal, onEnhance]);

    const handleDismissClick = useCallback(() => {
      logger.debug('EnhancementPanel: result dismissed');
      onDismissResult();
    }, [onDismissResult]);

    const goals = useMemo(() => ENHANCEMENT_GOALS, []);

    return (
      <div
        ref={ref}
        className={cn('mt-2 border-t pt-4', className)}
      >
        <p className="mb-2 text-sm font-medium text-foreground">
          AI Enhancement
        </p>
        <div className="flex gap-2">
          <select
            value={enhancementGoal}
            onChange={handleGoalChange}
            aria-label="Enhancement goal"
            data-testid="enhancement-goal-select"
            className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {goals.map((g) => (
              <option key={g.value} value={g.value}>
                {g.label}
              </option>
            ))}
          </select>
          <Button
            onClick={handleEnhanceClick}
            isLoading={isEnhancing}
            loadingText="Enhancing..."
            disabled={isEnhancing}
            data-testid="ai-enhance-btn"
            className="shrink-0"
          >
            AI Enhance
          </Button>
        </div>

        {showResult && enhanceResult && (
          <div
            className="mt-3 rounded-md border bg-muted/30 p-3 text-sm"
            data-testid="enhance-result"
            role="alert"
            aria-live="polite"
          >
            <div className="flex items-center justify-between">
              <span className="font-medium">
                Confidence: {Math.round(enhanceResult.confidence * 100)}%
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDismissClick}
                data-testid="enhance-dismiss-btn"
              >
                Dismiss
              </Button>
            </div>
            <p className="mt-1 text-muted-foreground">
              {enhanceResult.analysis}
            </p>
            {enhanceResult.changes_made.length > 0 && (
              <ul className="mt-1 list-inside list-disc text-xs text-muted-foreground">
                {enhanceResult.changes_made.map((change, i) => (
                  <li key={i}>{change}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    );
  }
);

EnhancementPanel.displayName = 'EnhancementPanel';
