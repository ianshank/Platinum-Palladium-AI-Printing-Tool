/**
 * Curve adjustment controls panel
 * Provides sliders and dropdowns for curve transformations
 */

import React, { useCallback, useMemo } from 'react';
import { Button } from '@/components/ui/Button';
import * as SliderPrimitive from '@radix-ui/react-slider';
import { cn } from '@/lib/utils';
import { logger } from '@/lib/logger';

export type AdjustmentType = 'contrast' | 'brightness' | 'gamma' | 'sigmoid';

const ADJUSTMENT_OPTIONS: readonly {
  value: AdjustmentType;
  label: string;
}[] = [
  { value: 'contrast', label: 'Contrast' },
  { value: 'brightness', label: 'Brightness' },
  { value: 'gamma', label: 'Gamma' },
  { value: 'sigmoid', label: 'Sigmoid' },
];

// Slider component from CurveEditor (extracted for reuse)
const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      'relative flex w-full touch-none select-none items-center',
      className
    )}
    {...props}
  >
    <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary/20">
      <SliderPrimitive.Range className="absolute h-full bg-primary" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

export interface AdjustmentPanelProps {
  /** Current adjustment type */
  adjustmentType: AdjustmentType;
  /** Callback when adjustment type changes */
  onAdjustmentTypeChange: (type: AdjustmentType) => void;
  /** Current adjustment amount (-100 to 100) */
  amount: number;
  /** Callback when adjustment amount changes */
  onAmountChange: (amount: number) => void;
  /** Whether adjustment is being applied */
  isApplying: boolean;
  /** Callback when Apply Adjustment button is clicked */
  onApply: () => void;
  /** Error message to display (if any) */
  error?: string | null;
  /** Optional CSS class */
  className?: string;
}

/**
 * Panel for adjusting curve transformations
 * Provides controls for contrast, brightness, gamma, and sigmoid adjustments
 *
 * @example
 * ```tsx
 * <AdjustmentPanel
 *   adjustmentType="contrast"
 *   onAdjustmentTypeChange={setType}
 *   amount={25}
 *   onAmountChange={setAmount}
 *   isApplying={false}
 *   onApply={handleApply}
 * />
 * ```
 */
export const AdjustmentPanel = React.forwardRef<
  HTMLDivElement,
  AdjustmentPanelProps
>(
  (
    {
      adjustmentType,
      onAdjustmentTypeChange,
      amount,
      onAmountChange,
      isApplying,
      onApply,
      error,
      className,
    },
    ref
  ) => {
    const handleTypeChange = useCallback(
      (e: React.ChangeEvent<HTMLSelectElement>) => {
        const type = e.target.value as AdjustmentType;
        logger.debug('AdjustmentPanel: adjustment type changed', { type });
        onAdjustmentTypeChange(type);
      },
      [onAdjustmentTypeChange]
    );

    const handleAmountChange = useCallback(
      (vals: number[]) => {
        const newAmount = vals[0] ?? 0;
        logger.debug('AdjustmentPanel: adjustment amount changed', {
          amount: newAmount,
        });
        onAmountChange(newAmount);
      },
      [onAmountChange]
    );

    const handleApplyClick = useCallback(() => {
      logger.info('AdjustmentPanel: apply clicked', {
        adjustmentType,
        amount,
      });
      onApply();
    }, [adjustmentType, amount, onApply]);

    const adjustmentOptions = useMemo(() => ADJUSTMENT_OPTIONS, []);

    return (
      <div
        ref={ref}
        className={cn(
          'grid grid-cols-1 gap-6 rounded-md bg-muted/50 p-4 md:grid-cols-2',
          className
        )}
      >
        <div className="space-y-4">
          <label
            htmlFor="adjustment-type-select"
            className="text-sm font-medium"
          >
            Adjustment Type
          </label>
          <select
            id="adjustment-type-select"
            value={adjustmentType}
            onChange={handleTypeChange}
            className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {adjustmentOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-4">
          <div className="flex justify-between">
            <label
              htmlFor="adjustment-amount-slider"
              className="text-sm font-medium"
            >
              Amount
            </label>
            <span className="text-sm text-muted-foreground">{amount}</span>
          </div>
          <Slider
            id="adjustment-amount-slider"
            value={[amount]}
            min={-100}
            max={100}
            step={1}
            onValueChange={handleAmountChange}
            aria-label="Adjustment amount"
          />
        </div>

        <div className="md:col-span-2">
          <Button
            onClick={handleApplyClick}
            disabled={isApplying}
            isLoading={isApplying}
            className="w-full md:w-auto"
          >
            Apply Adjustment
          </Button>
          {error && (
            <p className="mt-2 text-sm text-destructive" role="alert">
              {error}
            </p>
          )}
        </div>
      </div>
    );
  }
);

AdjustmentPanel.displayName = 'AdjustmentPanel';
