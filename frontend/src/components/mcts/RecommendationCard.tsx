/**
 * MCTS Recommendation Card
 * Displays a single parameter recommendation with rationale
 */

import { type FC } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import type { MCTSRecommendation } from '@/types/mcts';
import { logger } from '@/lib/logger';

export interface RecommendationCardProps {
  recommendation: MCTSRecommendation;
  onUse?: (parameters: Record<string, number>) => void;
  className?: string;
}

/**
 * Card displaying a single MCTS parameter recommendation
 */
export const RecommendationCard: FC<RecommendationCardProps> = ({
  recommendation,
  onUse,
  className,
}) => {
  const { parameters, predictedQuality, rationale } = recommendation;

  const handleUse = (): void => {
    logger.info('Using recommendation parameters', { parameters });
    onUse?.(parameters);
  };

  return (
    <div
      className={cn(
        'flex flex-col gap-4 rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/50',
        className
      )}
      data-testid="recommendation-card"
    >
      {/* Quality Score Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              'h-2 w-2 rounded-full',
              predictedQuality >= 0.8
                ? 'bg-success'
                : predictedQuality >= 0.6
                  ? 'bg-warning'
                  : 'bg-muted-foreground'
            )}
          />
          <span className="text-sm font-medium text-foreground">
            Quality: {(predictedQuality * 100).toFixed(0)}%
          </span>
        </div>
        <div className="rounded-full bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">
          Recommended
        </div>
      </div>

      {/* Rationale */}
      <p className="text-sm text-muted-foreground">{rationale}</p>

      {/* Parameters Table */}
      <div className="flex flex-col gap-1">
        <h4 className="text-xs font-medium text-muted-foreground">Parameters</h4>
        <div className="space-y-1">
          {Object.entries(parameters)
            .slice(0, 5)
            .map(([name, value]) => (
              <div key={name} className="flex justify-between text-xs">
                <span className="text-muted-foreground">{name.replace(/_/g, ' ')}</span>
                <span className="font-mono text-foreground">
                  {typeof value === 'number' ? value.toFixed(3) : value}
                </span>
              </div>
            ))}
          {Object.keys(parameters).length > 5 && (
            <p className="text-xs text-muted-foreground">
              +{Object.keys(parameters).length - 5} more...
            </p>
          )}
        </div>
      </div>

      {/* Use Button */}
      <Button onClick={handleUse} size="sm" variant="outline" className="w-full">
        Use These Parameters
      </Button>
    </div>
  );
};

RecommendationCard.displayName = 'RecommendationCard';
