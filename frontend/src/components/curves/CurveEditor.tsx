import React, { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { type CurveData } from '@/types/models';
import { api } from '@/api/client';
import { useSaveCurve } from '@/api/hooks';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { RefreshCw, Redo2, Save, Undo2 } from 'lucide-react';
import * as SliderPrimitive from '@radix-ui/react-slider';
import { logger } from '@/lib/logger';
import { useUndoRedo } from '@/hooks/useUndoRedo';

import { cn } from '@/lib/utils';

// --- UI Components (Inline for speed, move to ui/ later) ---

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
    <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-gray-200 bg-secondary/20">
      <SliderPrimitive.Range className="absolute h-full bg-blue-600 bg-primary" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-primary bg-background bg-white ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

// Simple Select wrapper
const AdjustmentSelect = ({
  value,
  onChange,
  id,
}: {
  value: string;
  onChange: (val: string) => void;
  id?: string;
}) => (
  <select
    id={id}
    value={value}
    onChange={(e) => onChange(e.target.value)}
    className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
  >
    <option value="contrast">Contrast</option>
    <option value="brightness">Brightness</option>
    <option value="gamma">Gamma</option>
    <option value="sigmoid">Sigmoid</option>
  </select>
);

interface CurveEditorProps {
  initialCurve?: CurveData;
  onSave?: (curve: CurveData) => void;
  className?: string;
}

export function CurveEditor({
  initialCurve,
  onSave,
  className,
}: CurveEditorProps) {
  // Local state for curve data
  const [name, setName] = useState(initialCurve?.name || 'New Curve');
  const [inputValues, setInputValues] = useState<number[]>(
    initialCurve?.input_values || Array.from({ length: 256 }, (_, i) => i)
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
    initialCurve?.output_values || Array.from({ length: 256 }, (_, i) => i)
  );

  // Adjustment state
  const [adjustmentType, setAdjustmentType] = useState<string>('contrast');
  const [amount, setAmount] = useState<number>(0);
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Save mutation
  const { mutate: saveCurve, isPending: isSaving } = useSaveCurve();

  // Prepare data for Recharts
  const chartData = useMemo(() => {
    return inputValues.map((input, index) => ({
      input,
      output: outputValues[index],
      reference: input, // Linear reference
    }));
  }, [inputValues, outputValues]);

  const handleApplyAdjustment = async () => {
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
        setOutputValues(response.output_values);
      } else {
        setError('Failed to apply adjustment');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error applying adjustment');
    } finally {
      setIsApplying(false);
    }
  };

  const handleReset = () => {
    if (initialCurve) {
      setInputValues(initialCurve.input_values);
      resetOutputValues(initialCurve.output_values);
    } else {
      // Reset to linear
      const linear = Array.from({ length: 256 }, (_, i) => i);
      setInputValues(linear);
      resetOutputValues(linear);
    }
    setAmount(0);
  };

  const handleSave = () => {
    logger.info('CurveEditor: saving curve', { name, pointCount: inputValues.length });

    saveCurve(
      {
        name,
        input_values: inputValues,
        output_values: outputValues,
        adjustment_type: 'none',
        amount: 0,
      },
      {
        onSuccess: (response) => {
          logger.info('CurveEditor: curve saved', { curveId: response.curve_id });
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
          setError(err.response?.data?.message ?? err.message ?? 'Save failed');
        },
      }
    );
  };

  return (
    <div
      className={cn(
        'space-y-6 rounded-lg border bg-card p-6 text-card-foreground shadow-sm',
        className
      )}
    >
      <div className="flex items-center justify-between">
        <div className="max-w-sm flex-1">
          <label htmlFor="curve-name-input" className="mb-1 block text-sm font-medium">Curve Name</label>
          <Input
            id="curve-name-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Curve Name"
          />
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={undo}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            aria-label="Undo"
          >
            <Undo2 className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            onClick={redo}
            disabled={!canRedo}
            title="Redo (Ctrl+Y / Ctrl+Shift+Z)"
            aria-label="Redo"
          >
            <Redo2 className="h-4 w-4" />
          </Button>
          <Button variant="outline" onClick={handleReset} title="Reset">
            <RefreshCw className="mr-2 h-4 w-4" />
            Reset
          </Button>
          <Button onClick={handleSave} isLoading={isSaving} disabled={isSaving}>
            <Save className="mr-2 h-4 w-4" />
            {isSaving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>

      <div className="h-[400px] w-full rounded-md border bg-white/5 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="input"
              type="number"
              domain={[0, 255]}
              tick={{ fontSize: 12 }}
              label={{
                value: 'Input Density',
                position: 'insideBottom',
                offset: -5,
              }}
            />
            <YAxis
              type="number"
              domain={[0, 255]}
              tick={{ fontSize: 12 }}
              label={{
                value: 'Output Density',
                angle: -90,
                position: 'insideLeft',
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                borderRadius: '4px',
                border: '1px solid #ccc',
              }}
            />
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 255, y: 255 },
              ]}
              stroke="#ccc"
              strokeDasharray="3 3"
            />
            <Line
              type="monotone"
              dataKey="output"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 gap-6 rounded-md bg-muted/50 p-4 md:grid-cols-2">
        <div className="space-y-4">
          <label htmlFor="adjustment-type-select" className="text-sm font-medium">Adjustment Type</label>
          <AdjustmentSelect
            id="adjustment-type-select"
            value={adjustmentType}
            onChange={setAdjustmentType}
          />
        </div>

        <div className="space-y-4">
          <div className="flex justify-between">
            <label htmlFor="adjustment-amount-slider" className="text-sm font-medium">Amount</label>
            <span className="text-sm text-gray-500">{amount}</span>
          </div>
          <Slider
            id="adjustment-amount-slider"
            value={[amount]}
            min={-100}
            max={100}
            step={1}
            onValueChange={(vals) => setAmount(vals[0] ?? 0)}
          />
        </div>

        <div className="md:col-span-2">
          <Button
            onClick={handleApplyAdjustment}
            disabled={isApplying}
            className="w-full md:w-auto"
          >
            {isApplying ? (
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
            ) : null}
            Apply Adjustment
          </Button>
          {error && <p className="mt-2 text-sm text-destructive">{error}</p>}
        </div>
      </div>
    </div>
  );
}
