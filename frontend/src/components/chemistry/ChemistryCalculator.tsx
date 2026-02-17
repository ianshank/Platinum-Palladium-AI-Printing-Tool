/**
 * ChemistryCalculator Component
 *
 * Interactive coating recipe calculator for Pt/Pd printing.
 * Reads from and writes to the chemistry Zustand store slice.
 *
 * Features:
 * - Paper size selection (standard + custom)
 * - Metal ratio slider (Pt/Pd blend)
 * - Coating method picker
 * - Contrast level control
 * - Developer settings
 * - Live recipe calculation with formatted output
 * - Reset and clear actions
 *
 * No hardcoded values — all options come from the store constants.
 */

import { type FC, useCallback, useMemo, useState } from 'react';
import { cn } from '@/lib/utils';
import { logger } from '@/lib/logger';
import { useStore } from '@/stores';
import {
  type ChemistryRecipe,
  type PaperSize,
  STANDARD_PAPER_SIZES,
  TEMPERATURE_MAX,
  TEMPERATURE_MIN,
} from '@/stores/slices/chemistrySlice';

export interface ChemistryCalculatorProps {
  className?: string;
}

/** Mapping of coating method keys to display labels */
const COATING_METHOD_LABELS: Record<string, string> = {
  brush: 'Brush',
  rod: 'Glass Rod',
  puddle: 'Puddle Push',
};

/** Mapping of developer type keys to display labels */
const DEVELOPER_LABELS: Record<string, string> = {
  potassium_oxalate: 'Potassium Oxalate',
  ammonium_citrate: 'Ammonium Citrate',
};

/** Format a number to fixed decimal places, stripping trailing zeros */
function formatMl(value: number, decimals = 2): string {
  return value.toFixed(decimals).replace(/\.?0+$/, '');
}

/** Recipe display sub-component */
const RecipeOutput: FC<{ recipe: ChemistryRecipe }> = ({ recipe }) => (
  <div
    className="space-y-4 rounded-lg border bg-card p-5"
    data-testid="recipe-output"
  >
    <div className="flex items-start justify-between gap-3">
      <h3 className="text-lg font-semibold">Calculated Recipe</h3>
      {recipe.isStale && (
        <div
          className="flex items-center gap-1.5 rounded-md bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-800 dark:bg-amber-900/20 dark:text-amber-300"
          data-testid="recipe-stale-badge"
        >
          <svg
            className="h-3.5 w-3.5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          Outdated
        </div>
      )}
    </div>
    {recipe.isStale && (
      <div
        className="rounded-md border-l-4 border-amber-400 bg-amber-50 p-3 dark:bg-amber-900/20"
        data-testid="recipe-stale-warning"
      >
        <p className="text-sm text-amber-800 dark:text-amber-300">
          Recipe is outdated. Input parameters have changed — click Calculate to
          update.
        </p>
      </div>
    )}

    <div className="grid gap-3 sm:grid-cols-2">
      <RecipeRow
        label="Total Volume"
        value={`${formatMl(recipe.totalVolume)} ml`}
        testId="recipe-total"
      />
      <RecipeRow
        label="Platinum (Pt)"
        value={`${formatMl(recipe.platinumMl)} ml`}
        testId="recipe-platinum"
      />
      <RecipeRow
        label="Palladium (Pd)"
        value={`${formatMl(recipe.palladiumMl)} ml`}
        testId="recipe-palladium"
      />
      <RecipeRow
        label="Ferric Oxalate"
        value={`${formatMl(recipe.ferricOxalateMl)} ml`}
        testId="recipe-ferric"
      />
    </div>

    {recipe.contrastAgent && recipe.contrastAgent.type !== 'none' && (
      <div className="mt-3 border-t pt-3">
        <RecipeRow
          label="Contrast Agent"
          value={`${recipe.contrastAgent.type.toUpperCase()} — ${formatMl(recipe.contrastAgent.amount)} drops per 10ml`}
          testId="recipe-contrast"
        />
      </div>
    )}

    <div className="mt-3 border-t pt-3">
      <h4 className="mb-2 text-sm font-medium text-muted-foreground">
        Developer
      </h4>
      <div className="grid gap-2 sm:grid-cols-2">
        <RecipeRow
          label="Type"
          value={
            DEVELOPER_LABELS[recipe.developer.type] ?? recipe.developer.type
          }
          testId="recipe-dev-type"
        />
        <RecipeRow
          label="Temperature"
          value={`${recipe.developer.temperatureC}°C`}
          testId="recipe-dev-temp"
        />
      </div>
    </div>
  </div>
);
RecipeOutput.displayName = 'RecipeOutput';

/** Label + value row */
const RecipeRow: FC<{ label: string; value: string; testId: string }> = ({
  label,
  value,
  testId,
}) => (
  <div
    className="flex justify-between rounded-md bg-muted/40 px-3 py-2"
    data-testid={testId}
  >
    <span className="text-sm text-muted-foreground">{label}</span>
    <span className="text-sm font-medium">{value}</span>
  </div>
);
RecipeRow.displayName = 'RecipeRow';

export const ChemistryCalculator: FC<ChemistryCalculatorProps> = ({
  className,
}) => {
  // --- Local state for validation ---
  const [temperatureError, setTemperatureError] = useState<string | null>(null);

  // --- Store selectors ---
  const paperSize = useStore((s) => s.chemistry.paperSize);
  const customSizes = useStore((s) => s.chemistry.customSizes);
  const metalRatio = useStore((s) => s.chemistry.metalRatio);
  const coatingMethod = useStore((s) => s.chemistry.coatingMethod);
  const contrastLevel = useStore((s) => s.chemistry.contrastLevel);
  const developer = useStore((s) => s.chemistry.developer);
  const recipe = useStore((s) => s.chemistry.recipe);

  // --- Store actions ---
  const setPaperSize = useStore((s) => s.chemistry.setPaperSize);
  const setMetalRatio = useStore((s) => s.chemistry.setMetalRatio);
  const setCoatingMethod = useStore((s) => s.chemistry.setCoatingMethod);
  const setContrastLevel = useStore((s) => s.chemistry.setContrastLevel);
  const setDeveloper = useStore((s) => s.chemistry.setDeveloper);
  const calculateRecipe = useStore((s) => s.chemistry.calculateRecipe);
  const clearRecipe = useStore((s) => s.chemistry.clearRecipe);
  const resetChemistry = useStore((s) => s.chemistry.resetChemistry);

  const allPaperSizes = useMemo<PaperSize[]>(
    () => [...STANDARD_PAPER_SIZES, ...customSizes],
    [customSizes]
  );

  const handlePaperChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const selected = allPaperSizes.find((s) => s.name === e.target.value);
      if (selected) {
        logger.debug('ChemistryCalculator: Paper size changed', {
          name: selected.name,
        });
        setPaperSize(selected);
      }
    },
    [allPaperSizes, setPaperSize]
  );

  const handleCalculate = useCallback(() => {
    logger.debug('ChemistryCalculator: Calculate pressed');
    calculateRecipe();
  }, [calculateRecipe]);

  const handleReset = useCallback(() => {
    logger.debug('ChemistryCalculator: Reset pressed');
    resetChemistry();
    setTemperatureError(null);
  }, [resetChemistry]);

  const handleTemperatureChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(e.target.value);

      // Validate temperature range
      if (value < TEMPERATURE_MIN || value > TEMPERATURE_MAX) {
        setTemperatureError(
          `Temperature must be between ${TEMPERATURE_MIN}°C and ${TEMPERATURE_MAX}°C`
        );
      } else {
        setTemperatureError(null);
      }

      // Update store (store will clamp the value)
      setDeveloper({ temperatureC: value });
    },
    [setDeveloper]
  );

  const ptPercent = Math.round(metalRatio * 100);
  const pdPercent = 100 - ptPercent;

  return (
    <div
      className={cn('space-y-6', className)}
      data-testid="chemistry-calculator"
    >
      {/* --- Paper Size --- */}
      <fieldset className="space-y-2">
        <legend className="text-sm font-medium">Paper Size</legend>
        <select
          value={paperSize.name}
          onChange={handlePaperChange}
          className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          data-testid="paper-size-select"
          aria-label="Paper size"
        >
          {allPaperSizes.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} ({s.widthInches}&quot; × {s.heightInches}&quot;)
            </option>
          ))}
        </select>
        <p className="text-xs text-muted-foreground">
          Area: {(paperSize.widthInches * paperSize.heightInches).toFixed(1)} sq
          in
        </p>
      </fieldset>

      {/* --- Metal Ratio --- */}
      <fieldset className="space-y-2">
        <legend className="text-sm font-medium">
          Metal Ratio — {ptPercent}% Pt / {pdPercent}% Pd
        </legend>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={metalRatio}
          onChange={(e) => setMetalRatio(Number(e.target.value))}
          className="w-full accent-primary"
          data-testid="metal-ratio-slider"
          aria-label="Metal ratio"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>100% Pd</span>
          <span>50/50</span>
          <span>100% Pt</span>
        </div>
      </fieldset>

      {/* --- Coating Method --- */}
      <fieldset className="space-y-2">
        <legend className="text-sm font-medium">Coating Method</legend>
        <div
          className="flex gap-2"
          role="radiogroup"
          aria-label="Coating method"
        >
          {Object.entries(COATING_METHOD_LABELS).map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="radio"
              aria-checked={coatingMethod === key}
              onClick={() =>
                setCoatingMethod(key as 'brush' | 'rod' | 'puddle')
              }
              className={cn(
                'flex-1 rounded-md border px-3 py-2 text-sm transition-colors',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                coatingMethod === key
                  ? 'border-primary bg-primary/10 font-medium text-primary'
                  : 'hover:bg-accent'
              )}
              data-testid={`coating-${key}`}
            >
              {label}
            </button>
          ))}
        </div>
      </fieldset>

      {/* --- Contrast Level --- */}
      <fieldset className="space-y-2">
        <legend className="text-sm font-medium">
          Contrast Level — Grade {contrastLevel}
        </legend>
        <input
          type="range"
          min={0}
          max={5}
          step={1}
          value={contrastLevel}
          onChange={(e) => setContrastLevel(Number(e.target.value))}
          className="w-full accent-primary"
          data-testid="contrast-slider"
          aria-label="Contrast level"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>0 (None)</span>
          <span>5 (Max)</span>
        </div>
      </fieldset>

      {/* --- Developer --- */}
      <fieldset className="space-y-3">
        <legend className="text-sm font-medium">Developer</legend>
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor="dev-type" className="text-xs text-muted-foreground">
              Type
            </label>
            <select
              id="dev-type"
              value={developer.type}
              onChange={(e) =>
                setDeveloper({
                  type: e.target.value as
                    | 'potassium_oxalate'
                    | 'ammonium_citrate',
                })
              }
              className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              data-testid="developer-type-select"
            >
              {Object.entries(DEVELOPER_LABELS).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="dev-temp" className="text-xs text-muted-foreground">
              Temperature (°C)
            </label>
            <input
              id="dev-temp"
              type="number"
              min={TEMPERATURE_MIN}
              max={TEMPERATURE_MAX}
              value={developer.temperatureC}
              onChange={handleTemperatureChange}
              className={cn(
                'mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2',
                temperatureError
                  ? 'border-destructive focus:ring-destructive'
                  : 'focus:ring-primary'
              )}
              data-testid="developer-temp-input"
              aria-invalid={temperatureError ? true : false}
              aria-describedby={temperatureError ? 'temp-error' : undefined}
            />
            {temperatureError && (
              <p
                id="temp-error"
                className="mt-1 text-xs text-destructive"
                data-testid="temp-error-message"
              >
                {temperatureError}
              </p>
            )}
          </div>
        </div>
      </fieldset>

      {/* --- Action Buttons --- */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={handleCalculate}
          className={cn(
            'flex-1 rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground',
            'hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
            'transition-colors'
          )}
          data-testid="calculate-btn"
        >
          Calculate Recipe
        </button>
        {recipe && (
          <button
            type="button"
            onClick={clearRecipe}
            className="rounded-md border px-4 py-2.5 text-sm hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            data-testid="clear-btn"
          >
            Clear
          </button>
        )}
        <button
          type="button"
          onClick={handleReset}
          className="rounded-md border px-4 py-2.5 text-sm text-muted-foreground hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          data-testid="reset-btn"
        >
          Reset All
        </button>
      </div>

      {/* --- Recipe Output --- */}
      {recipe && <RecipeOutput recipe={recipe} />}
    </div>
  );
};

ChemistryCalculator.displayName = 'ChemistryCalculator';
