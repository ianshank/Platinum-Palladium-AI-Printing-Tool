/**
 * MCTS Results Dashboard
 * Displays search results with predicted curve and optimal parameters
 */

import { type FC } from 'react';
import Plot from 'react-plotly.js';
import { cn } from '@/lib/utils';
import { useMCTSCalibration } from '@/hooks/useMCTSCalibration';
import { useMCTSExport } from '@/api/mctsHooks';
import { Button } from '@/components/ui/Button';

export interface ResultsDashboardProps {
  className?: string;
}

/**
 * Results dashboard showing predicted density curve and parameters
 */
export const ResultsDashboard: FC<ResultsDashboardProps> = ({ className }) => {
  const { currentResult, evaluateResult } = useMCTSCalibration();
  const exportMutation = useMCTSExport();

  const result = currentResult ?? null;
  const evaluation = evaluateResult ?? null;

  if (!result && !evaluation) {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center rounded-lg border border-border bg-card p-12',
          className
        )}
        data-testid="results-placeholder"
      >
        <div className="text-center">
          <h3 className="text-lg font-semibold text-muted-foreground">
            No Results Yet
          </h3>
          <p className="mt-2 text-sm text-muted-foreground">
            Configure parameters and run a search to see results
          </p>
        </div>
      </div>
    );
  }

  const curve = result?.predictedCurve ?? evaluation?.densityCurve ?? [];
  const qualityScore = result?.qualityScore ?? evaluation?.qualityScore ?? 0;
  const parameters = result?.bestParameters ?? {};

  // Generate x-axis values (0-1 range)
  const xValues = curve.map((_, i) => i / (curve.length - 1));

  return (
    <div
      className={cn(
        'flex flex-col gap-6 rounded-lg border border-border bg-card p-6',
        className
      )}
      data-testid="results-dashboard"
    >
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-foreground">
            Search Results
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Optimal parameters and predicted density curve
          </p>
        </div>
        {result && (
          <div className="text-right">
            <p className="text-xs text-muted-foreground">Search Time</p>
            <p className="text-sm font-medium text-foreground">
              {result.searchTimeSeconds.toFixed(2)}s
            </p>
          </div>
        )}
      </div>

      {/* Quality Score Gauge */}
      <div className="flex flex-col gap-2 rounded-md bg-muted/50 p-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-foreground">
            Quality Score
          </span>
          <span className="text-2xl font-bold text-foreground">
            {(qualityScore * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-success transition-all"
            style={{ width: `${qualityScore * 100}%` }}
          />
        </div>
      </div>

      {/* Predicted Density Curve */}
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium text-foreground">
          Predicted Density Curve
        </h3>
        <div className="rounded-md border border-border">
          <Plot
            data={[
              {
                x: xValues,
                y: curve,
                type: 'scatter',
                mode: 'lines+markers',
                marker: { color: 'rgb(99, 102, 241)', size: 4 },
                line: { color: 'rgb(99, 102, 241)', width: 2 },
                name: 'Density',
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 50, r: 20, t: 20, b: 40 },
              xaxis: {
                title: { text: 'Input (0-1)' },
                gridcolor: 'rgba(128, 128, 128, 0.1)',
              },
              yaxis: {
                title: { text: 'Density' },
                gridcolor: 'rgba(128, 128, 128, 0.1)',
              },
              plot_bgcolor: 'rgba(0, 0, 0, 0)',
              paper_bgcolor: 'rgba(0, 0, 0, 0)',
              font: {
                color: 'currentColor',
                size: 12,
              },
            }}
            config={{
              responsive: true,
              displayModeBar: false,
            }}
            style={{ width: '100%', height: '300px' }}
          />
        </div>
      </div>

      {/* Optimal Parameters Table */}
      {Object.keys(parameters).length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-medium text-foreground">
            Optimal Parameters
          </h3>
          <div className="overflow-hidden rounded-md border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="px-4 py-2 text-left font-medium text-foreground">
                    Parameter
                  </th>
                  <th className="px-4 py-2 text-right font-medium text-foreground">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(parameters).map(([name, value], index) => (
                  <tr
                    key={name}
                    className={cn(
                      'border-t border-border',
                      index % 2 === 0 ? 'bg-card' : 'bg-muted/30'
                    )}
                  >
                    <td className="px-4 py-2 text-foreground">
                      {name.replace(/_/g, ' ')}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-foreground">
                      {typeof value === 'number' ? value.toFixed(4) : value}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Metrics Grid */}
      {evaluation && (
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-md bg-muted/50 p-3">
            <p className="text-xs text-muted-foreground">Dmin</p>
            <p className="mt-1 text-lg font-semibold text-foreground">
              {evaluation.dmin.toFixed(3)}
            </p>
          </div>
          <div className="rounded-md bg-muted/50 p-3">
            <p className="text-xs text-muted-foreground">Dmax</p>
            <p className="mt-1 text-lg font-semibold text-foreground">
              {evaluation.dmax.toFixed(3)}
            </p>
          </div>
          <div className="rounded-md bg-muted/50 p-3">
            <p className="text-xs text-muted-foreground">Density Range</p>
            <p className="mt-1 text-lg font-semibold text-foreground">
              {evaluation.densityRange.toFixed(3)}
            </p>
          </div>
          <div className="rounded-md bg-muted/50 p-3">
            <p className="text-xs text-muted-foreground">Gamma</p>
            <p className="mt-1 text-lg font-semibold text-foreground">
              {evaluation.gamma.toFixed(3)}
            </p>
          </div>
        </div>
      )}

      {/* Alternative Parameters */}
      {result && result.alternatives.length > 1 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-medium text-foreground">
            Alternative Parameter Sets ({result.alternatives.length - 1})
          </h3>
          <p className="text-xs text-muted-foreground">
            Other high-quality parameter combinations found during search
          </p>
        </div>
      )}

      {/* Export Button */}
      <Button
        variant="outline"
        className="w-full"
        onClick={() => {
          if (parameters && Object.keys(parameters).length > 0) {
            exportMutation.mutate({ parameters, format: 'json' });
          }
        }}
        disabled={
          !parameters ||
          Object.keys(parameters).length === 0 ||
          exportMutation.isPending
        }
      >
        {exportMutation.isPending ? 'Exporting...' : 'Export Results'}
      </Button>
    </div>
  );
};

ResultsDashboard.displayName = 'ResultsDashboard';
