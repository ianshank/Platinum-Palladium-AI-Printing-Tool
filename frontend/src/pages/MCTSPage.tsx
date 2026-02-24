/**
 * MCTS Calibration Search Page
 * Main page for MCTS-based parameter optimization
 */

import { type FC, useEffect } from 'react';
import { useMCTSCalibration } from '@/hooks/useMCTSCalibration';
import { SearchConfigPanel } from '@/components/mcts/SearchConfigPanel';
import { ResultsDashboard } from '@/components/mcts/ResultsDashboard';
import { RecommendationCard } from '@/components/mcts/RecommendationCard';
import { logger } from '@/lib/logger';

/**
 * Generate a stable key from recommendation parameters
 */
function generateRecommendationKey(parameters: Record<string, number>): string {
  const sortedEntries = Object.entries(parameters).sort(([a], [b]) =>
    a.localeCompare(b)
  );
  return sortedEntries.map(([key, value]) => `${key}:${value}`).join('|');
}

/**
 * MCTS calibration search page
 */
export const MCTSPage: FC = () => {
  const {
    currentResult,
    recommendations,
    loadRecommendations,
    isLoadingRecommendations,
    evaluateParameters,
  } = useMCTSCalibration();

  useEffect(() => {
    logger.debug('MCTSPage: mounted');
  }, []);

  const handleUseRecommendation = async (
    parameters: Record<string, number>
  ): Promise<void> => {
    try {
      await evaluateParameters(parameters);
    } catch (error) {
      logger.error('Failed to evaluate recommendation', { error });
    }
  };

  return (
    <div className="flex flex-col gap-6 p-6" data-testid="mcts-page">
      {/* Page Header */}
      <div>
        <p className="mt-0 text-sm text-muted-foreground">
          Use Monte Carlo Tree Search to find optimal calibration parameters
          based on your paper type, UV source, and target aesthetics.
        </p>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Configuration Panel */}
        <div className="lg:col-span-1">
          <SearchConfigPanel />
        </div>

        {/* Results Dashboard */}
        <div className="lg:col-span-2">
          <ResultsDashboard />
        </div>
      </div>

      {/* Recommendations Section */}
      {recommendations.length > 0 && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">
                Recommendations
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Pre-computed parameter sets for common scenarios
              </p>
            </div>
            <button
              onClick={() => void loadRecommendations()}
              disabled={isLoadingRecommendations}
              className="text-sm text-primary hover:underline disabled:opacity-50"
            >
              {isLoadingRecommendations ? 'Loading...' : 'Refresh'}
            </button>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {recommendations.map((rec) => (
              <RecommendationCard
                key={generateRecommendationKey(rec.parameters)}
                recommendation={rec}
                onUse={handleUseRecommendation}
              />
            ))}
          </div>
        </div>
      )}

      {/* Info Section */}
      {!currentResult && recommendations.length === 0 && (
        <div className="rounded-lg border border-border bg-muted/30 p-6">
          <h3 className="font-medium text-foreground">How MCTS Search Works</h3>
          <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
            <li>• Configure your paper type and UV source (optional)</li>
            <li>• Fix specific parameters if you have known constraints</li>
            <li>• Run the search to find optimal parameter combinations</li>
            <li>• Review predicted density curves and quality scores</li>
            <li>• Export results or use them directly in calibration</li>
          </ul>
        </div>
      )}
    </div>
  );
};

MCTSPage.displayName = 'MCTSPage';
