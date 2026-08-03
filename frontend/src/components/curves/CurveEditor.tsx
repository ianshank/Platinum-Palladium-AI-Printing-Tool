import React, { useCallback, useState } from 'react';
import { type CurveData, type CurveEnhanceResponse } from '@/types/models';
import { api } from '@/api/client';
import { useEnhanceCurve, useSaveCurve } from '@/api/hooks';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Save } from 'lucide-react';
import { logger } from '@/lib/logger';
import { useUndoRedo } from '@/hooks/useUndoRedo';
import { config } from '@/config';
import { cn } from '@/lib/utils';
import { CurveChart } from './CurveChart';
import { AdjustmentPanel, type AdjustmentType } from './AdjustmentPanel';
import { type EnhancementGoal, EnhancementPanel } from './EnhancementPanel';
import { UndoRedoControls } from './UndoRedoControls';

export interface CurveEditorProps {
  /** Initial curve data to load */
  initialCurve?: CurveData;
  /** Callback when curve is saved */
  onSave?: (curve: CurveData) => void;
  /** Optional CSS class */
  className?: string;
}

function makeLinearCurve(length: number): number[] {
  return Array.from({ length }, (_, i) => i);
}

/**
 * Orchestrates the curve editing interface
 * Composes CurveChart, AdjustmentPanel, and EnhancementPanel
 *
 * @example
 * ```tsx
 * <CurveEditor
 *   initialCurve={curve}
 *   onSave={handleSave}
 * />
 * ```
 */
export function CurveEditor({
  initialCurve,
  onSave,
  className,
}: CurveEditorProps): React.ReactElement {
  const curveLength = config.calibration.maxCurvePoints;
  const maxValue = curveLength - 1;

  // Local state for curve data
  const [name, setName] = useState(initialCurve?.name || 'New Curve');
  const [inputValues, setInputValues] = useState<number[]>(
    initialCurve?.input_values || makeLinearCurve(curveLength)
  );

  // Output values with undo/redo support
  const {
    state: outputValues,
    setState: setOutputValues,
    undo,
    redo,
    canUndo,
    canRedo,
    reset: resetOutputValues,
  } = useUndoRedo<number[]>(
    initialCurve?.output_values || makeLinearCurve(curveLength)
  );

  // Adjustment state
  const [adjustmentType, setAdjustmentType] =
    useState<AdjustmentType>('contrast');
  const [amount, setAmount] = useState<number>(0);
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // AI Enhancement state
  const [enhancementGoal, setEnhancementGoal] = useState<EnhancementGoal>('linearization');
  const [enhanceResult, setEnhanceResult] = useState<CurveEnhanceResponse | null>(null);
  const [showEnhancePanel, setShowEnhancePanel] = useState(false);

  // Save mutation
  const { mutate: saveCurve, isPending: isSaving } = useSaveCurve();

  // AI enhance mutation
  const { mutate: enhanceCurve, isPending: isEnhancing } = useEnhanceCurve();

  // Ref for the chart container to calculate coordinates
  const chartContainerRef = React.useRef<HTMLDivElement>(null);

  // Handle chart container click to add control points
  const handleChartClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>): void => {
      const container = chartContainerRef.current;
      if (!container) return;

      // Get click position relative to container
      const rect = container.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      // Account for chart margins (from LineChart margin prop)
      const marginLeft = 0;
      const marginTop = 5;
      const marginRight = 20;
      const marginBottom = 5;

      // Calculate the actual chart area dimensions
      const chartWidth = rect.width - marginLeft - marginRight;
      const chartHeight = rect.height - marginTop - marginBottom;

      // Adjust click position for margins
      const adjustedX = clickX - marginLeft;
      const adjustedY = clickY - marginTop;

      // Check if click is within chart area
      if (
        adjustedX < 0 ||
        adjustedX > chartWidth ||
        adjustedY < 0 ||
        adjustedY > chartHeight
      ) {
        logger.debug('CurveEditor: click outside chart area');
        return;
      }

      // Convert pixel coordinates to data coordinates
      // X-axis: linear mapping from 0 to maxValue
      const inputValue = (adjustedX / chartWidth) * maxValue;

      // Y-axis: inverted because SVG/canvas Y goes top-to-bottom
      // Top of chart (adjustedY = 0) should be maxValue (255)
      // Bottom of chart (adjustedY = chartHeight) should be 0
      const outputValue = maxValue - (adjustedY / chartHeight) * maxValue;

      // Clamp values to valid range
      const clampedInput = Math.max(
        0,
        Math.min(maxValue, Math.round(inputValue))
      );
      const clampedOutput = Math.max(
        0,
        Math.min(maxValue, Math.round(outputValue))
      );

      logger.info('CurveEditor: adding control point', {
        input: clampedInput,
        output: clampedOutput,
        clickPos: { x: clickX, y: clickY },
        chartArea: { width: chartWidth, height: chartHeight },
      });

      // Modify the output value at the clicked input position
      // This creates a "control point" effect by setting outputValues[clampedInput] = clampedOutput
      const newOutputValues = [...outputValues];
      newOutputValues[clampedInput] = clampedOutput;

      // Apply the change through undo/redo system
      setOutputValues(newOutputValues);

      logger.debug('CurveEditor: control point added', {
        index: clampedInput,
        oldValue: outputValues[clampedInput],
        newValue: clampedOutput,
      });
    },
    [outputValues, maxValue, setOutputValues]
  );

  const handleApplyAdjustment = useCallback(async (): Promise<void> => {
    logger.info('CurveEditor: applying adjustment', { adjustmentType, amount });
    setIsApplying(true);
    setError(null);
    try {
      const response = await api.curves.modify({
        name,
        input_values: inputValues,
        output_values: outputValues,
        adjustment_type: adjustmentType,
        amount: amount,
      });

      if (response.success) {
        logger.info('CurveEditor: adjustment applied', { adjustmentType });
        setOutputValues(response.output_values);
      } else {
        logger.warn('CurveEditor: adjustment returned success=false');
        setError('Failed to apply adjustment');
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Error applying adjustment';
      logger.error('CurveEditor: adjustment failed', { error: message });
      setError(message);
    } finally {
      setIsApplying(false);
    }
  }, [adjustmentType, amount, name, inputValues, outputValues, setOutputValues]);

  const handleReset = useCallback((): void => {
    logger.info('CurveEditor: resetting curve');
    if (initialCurve) {
      setInputValues(initialCurve.input_values);
      resetOutputValues(initialCurve.output_values);
    } else {
      const linear = makeLinearCurve(curveLength);
      setInputValues(linear);
      resetOutputValues(linear);
    }
    setAmount(0);
  }, [initialCurve, curveLength, resetOutputValues]);

  const handleSave = useCallback((): void => {
    logger.info('CurveEditor: saving curve', {
      name,
      pointCount: inputValues.length,
    });

    saveCurve(
      {
        name,
        input_values: inputValues,
        output_values: outputValues,
        adjustment_type: 'brightness',
        amount: 0,
      },
      {
        onSuccess: (response) => {
          logger.info('CurveEditor: curve saved', {
            curveId: response.curve_id,
          });
          if (onSave) {
            onSave({
              id: response.curve_id,
              name: response.name,
              created_at: new Date().toISOString(),
              curve_type: initialCurve?.curve_type || 'custom',
              input_values: response.input_values,
              output_values: response.output_values,
            } as CurveData);
          }
        },
        onError: (err) => {
          const rawMessage = err.response?.data?.message ?? err.message;
          const message =
            typeof rawMessage === 'string' && rawMessage.trim().length > 0
              ? rawMessage
              : 'Failed to save curve. Please try again.';
          logger.error('CurveEditor: save failed', { error: message });
          setError(message);
        },
      }
    );
  }, [name, inputValues, outputValues, initialCurve, onSave, saveCurve]);

  const handleAIEnhance = useCallback((): void => {
    logger.info('CurveEditor: requesting AI enhancement', {
      goal: enhancementGoal,
      name,
    });
    setError(null);
    enhanceCurve(
      {
        name,
        input_values: inputValues,
        output_values: outputValues,
        goal: enhancementGoal,
      },
      {
        onSuccess: (data) => {
          if (data.success) {
            setOutputValues(data.output_values);
            setEnhanceResult(data);
            setShowEnhancePanel(true);
          }
        },
        onError: (err) => {
          const rawMessage = err.response?.data?.message ?? err.message;
          setError(rawMessage || 'AI enhancement failed');
        },
      }
    );
  }, [enhancementGoal, name, inputValues, outputValues, setOutputValues, enhanceCurve]);

  return (
    <div
      className={cn(
        'space-y-6 rounded-lg border bg-card p-6 text-card-foreground shadow-sm',
        className
      )}
    >
      {/* Header: Curve Name and Controls */}
      <div className="flex items-center justify-between">
        <div className="max-w-sm flex-1">
          <label
            htmlFor="curve-name-input"
            className="mb-1 block text-sm font-medium"
          >
            Curve Name
          </label>
          <Input
            id="curve-name-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Curve Name"
          />
        </div>
        <div className="flex gap-2">
          <UndoRedoControls
            canUndo={canUndo}
            canRedo={canRedo}
            onUndo={undo}
            onRedo={redo}
            onReset={handleReset}
          />
          <Button
            onClick={handleSave}
            isLoading={isSaving}
            disabled={isSaving}
          >
            <Save className="mr-2 h-4 w-4" />
            {isSaving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>

      {/* Curve Chart */}
      <CurveChart
        ref={chartContainerRef}
        inputValues={inputValues}
        outputValues={outputValues}
        curveName={name}
        maxValue={maxValue}
        onChartClick={handleChartClick}
      />

      {/* Adjustment Panel and Enhancement Panel */}
      <div className="space-y-4">
        <AdjustmentPanel
          adjustmentType={adjustmentType}
          onAdjustmentTypeChange={setAdjustmentType}
          amount={amount}
          onAmountChange={setAmount}
          isApplying={isApplying}
          onApply={handleApplyAdjustment}
          error={error}
        />

        <EnhancementPanel
          enhancementGoal={enhancementGoal}
          onGoalChange={setEnhancementGoal}
          isEnhancing={isEnhancing}
          onEnhance={handleAIEnhance}
          enhanceResult={enhanceResult}
          onDismissResult={() => setShowEnhancePanel(false)}
          showResult={showEnhancePanel}
        />
      </div>
    </div>
  );
}
