/**
 * MCTS Search Configuration Panel
 * Allows user to configure search parameters and constraints
 */

import { type FC, useState } from 'react';
import { cn } from '@/lib/utils';
import { useMCTSCalibration } from '@/hooks/useMCTSCalibration';
import { Button } from '@/components/ui/Button';
import { logger } from '@/lib/logger';

export interface SearchConfigPanelProps {
  className?: string;
}

/**
 * Search configuration panel with parameter sliders and search controls
 */
export const SearchConfigPanel: FC<SearchConfigPanelProps> = ({ className }) => {
  const {
    searchConfig,
    setSearchConfig,
    setFixedParameter,
    removeFixedParameter,
    runSearch,
    isSearching,
    isEngineReady,
    isTorchAvailable,
    parameterRanges,
  } = useMCTSCalibration();

  const [numSimulations, setNumSimulations] = useState(500);

  const handleRunSearch = async (): Promise<void> => {
    try {
      await runSearch({
        ...searchConfig,
        numSimulations,
      });
    } catch (error) {
      logger.error('Search failed', { error });
    }
  };

  const handleToggleParameter = (paramName: string, isFixed: boolean): void => {
    if (isFixed) {
      const range = parameterRanges[paramName];
      if (range) {
        setFixedParameter(paramName, range.default);
      }
    } else {
      removeFixedParameter(paramName);
    }
  };

  const handleParameterChange = (paramName: string, value: number): void => {
    setFixedParameter(paramName, value);
  };

  return (
    <div
      className={cn(
        'flex flex-col gap-6 rounded-lg border border-border bg-card p-6',
        className
      )}
      data-testid="search-config-panel"
    >
      <div>
        <h2 className="text-lg font-semibold text-foreground">Search Configuration</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure parameters and constraints for MCTS search
        </p>
      </div>

      {/* Engine Status */}
      <div className="flex items-center gap-2 rounded-md bg-muted/50 p-3">
        <div
          className={cn(
            'h-2 w-2 rounded-full',
            isEngineReady ? 'bg-success' : 'bg-destructive'
          )}
        />
        <span className="text-sm text-foreground">
          {isEngineReady ? 'Engine Ready' : 'Engine Not Ready'}
        </span>
        {!isTorchAvailable && (
          <span className="text-xs text-muted-foreground">(PyTorch not available)</span>
        )}
      </div>

      {/* Paper Type */}
      <div className="flex flex-col gap-2">
        <label htmlFor="paper-type" className="text-sm font-medium text-foreground">
          Paper Type
        </label>
        <select
          id="paper-type"
          className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          value={searchConfig.paperType ?? ''}
          onChange={(e) => setSearchConfig({ paperType: e.target.value || undefined })}
        >
          <option value="">Any</option>
          <option value="arches_platine">Arches Platine</option>
          <option value="hahnemuhle_platinum">Hahnemühle Platinum</option>
          <option value="bergger_cot320">Bergger COT 320</option>
          <option value="fabriano_artistico">Fabriano Artistico</option>
        </select>
      </div>

      {/* UV Source */}
      <div className="flex flex-col gap-2">
        <label htmlFor="uv-source" className="text-sm font-medium text-foreground">
          UV Source
        </label>
        <select
          id="uv-source"
          className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          value={searchConfig.uvSource ?? ''}
          onChange={(e) => setSearchConfig({ uvSource: e.target.value || undefined })}
        >
          <option value="">Any</option>
          <option value="sun">Natural Sunlight</option>
          <option value="uv_led">UV LED</option>
          <option value="metal_halide">Metal Halide</option>
          <option value="uv_tubes">UV Tubes</option>
        </select>
      </div>

      {/* Number of Simulations */}
      <div className="flex flex-col gap-2">
        <label htmlFor="num-simulations" className="text-sm font-medium text-foreground">
          Simulations: {numSimulations}
        </label>
        <input
          type="range"
          id="num-simulations"
          min="50"
          max="2000"
          step="50"
          value={numSimulations}
          onChange={(e) => setNumSimulations(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>50 (fast)</span>
          <span>2000 (thorough)</span>
        </div>
      </div>

      {/* Fixed Parameters */}
      {Object.keys(parameterRanges).length > 0 && (
        <div className="flex flex-col gap-3">
          <h3 className="text-sm font-medium text-foreground">Fixed Parameters</h3>
          <div className="flex flex-col gap-3">
            {Object.entries(parameterRanges).map(([name, range]) => {
              const isFixed = name in (searchConfig.fixedParameters ?? {});
              const value = searchConfig.fixedParameters?.[name] ?? range.default;

              return (
                <div key={name} className="flex flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <label className="text-sm text-foreground">
                      {name.replace(/_/g, ' ')}
                    </label>
                    <input
                      type="checkbox"
                      checked={isFixed}
                      onChange={(e) => handleToggleParameter(name, e.target.checked)}
                      className="h-4 w-4"
                    />
                  </div>
                  {isFixed && (
                    <div className="flex flex-col gap-1">
                      <input
                        type="range"
                        min={range.min}
                        max={range.max}
                        step={(range.max - range.min) / 100}
                        value={value}
                        onChange={(e) =>
                          handleParameterChange(name, Number(e.target.value))
                        }
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>
                          {value.toFixed(2)} {range.unit}
                        </span>
                        <span>
                          {range.min}–{range.max}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Search Button */}
      <Button
        onClick={handleRunSearch}
        disabled={!isEngineReady || isSearching}
        isLoading={isSearching}
        loadingText="Searching..."
        className="w-full"
      >
        Run MCTS Search
      </Button>

      {!isEngineReady && (
        <p className="text-xs text-muted-foreground">
          MCTS engine is not ready. Check that PyTorch is installed.
        </p>
      )}
    </div>
  );
};

SearchConfigPanel.displayName = 'SearchConfigPanel';
