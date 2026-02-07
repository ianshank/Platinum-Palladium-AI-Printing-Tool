import React, { useState, useMemo } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine
} from 'recharts';
import { CurveData, CurveModificationResponse } from '@/types/models';
import { api } from '@/api/client';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Save, RefreshCw, ChevronRight } from 'lucide-react';
import * as SliderPrimitive from '@radix-ui/react-slider';
import * as SelectPrimitive from '@radix-ui/react-select';
import { cn } from '@/lib/utils';

// --- UI Components (Inline for speed, move to ui/ later) ---

const Slider = React.forwardRef<
    React.ElementRef<typeof SliderPrimitive.Root>,
    React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
    <SliderPrimitive.Root
        ref={ref}
        className={cn(
            "relative flex w-full touch-none select-none items-center",
            className
        )}
        {...props}
    >
        <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary/20 bg-gray-200">
            <SliderPrimitive.Range className="absolute h-full bg-primary bg-blue-600" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-primary bg-background bg-white ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
    </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

// Simple Select wrapper
const AdjustmentSelect = ({ value, onChange }: { value: string, onChange: (val: string) => void }) => (
    <select
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

export function CurveEditor({ initialCurve, onSave, className }: CurveEditorProps) {
    // Local state for curve data
    const [name, setName] = useState(initialCurve?.name || 'New Curve');
    const [inputValues, setInputValues] = useState<number[]>(initialCurve?.input_values || Array.from({ length: 256 }, (_, i) => i));
    const [outputValues, setOutputValues] = useState<number[]>(initialCurve?.output_values || Array.from({ length: 256 }, (_, i) => i));

    // Adjustment state
    const [adjustmentType, setAdjustmentType] = useState<string>('contrast');
    const [amount, setAmount] = useState<number>(0);
    const [isApplying, setIsApplying] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Prepare data for Recharts
    const chartData = useMemo(() => {
        return inputValues.map((input, index) => ({
            input,
            output: outputValues[index],
            reference: input // Linear reference
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
        } catch (err: any) {
            setError(err.message || 'Error applying adjustment');
        } finally {
            setIsApplying(false);
        }
    };

    const handleReset = () => {
        if (initialCurve) {
            setInputValues(initialCurve.input_values);
            setOutputValues(initialCurve.output_values);
        } else {
            // Reset to linear
            const linear = Array.from({ length: 256 }, (_, i) => i);
            setInputValues(linear);
            setOutputValues(linear);
        }
        setAmount(0);
    };

    const handleSave = () => {
        // TODO: Implement save via API or callback
        // For now just call callback
        if (onSave) {
            onSave({
                id: initialCurve?.id || 'new', // placeholder
                name,
                created_at: new Date().toISOString(),
                curve_type: initialCurve?.curve_type || 'custom',
                input_values: inputValues,
                output_values: outputValues,
            } as CurveData);
        }
    };

    return (
        <div className={cn("space-y-6 p-6 border rounded-lg bg-card text-card-foreground shadow-sm", className)}>
            <div className="flex items-center justify-between">
                <div className="flex-1 max-w-sm">
                    <label className="text-sm font-medium mb-1 block">Curve Name</label>
                    <Input
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Curve Name"
                    />
                </div>
                <div className="flex gap-2">
                    <Button variant="outline" onClick={handleReset} title="Reset">
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Reset
                    </Button>
                    <Button onClick={handleSave}>
                        <Save className="w-4 h-4 mr-2" />
                        Save
                    </Button>
                </div>
            </div>

            <div className="h-[400px] w-full bg-white/5 rounded-md border p-4">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis
                            dataKey="input"
                            type="number"
                            domain={[0, 255]}
                            tick={{ fontSize: 12 }}
                            label={{ value: 'Input Density', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                            type="number"
                            domain={[0, 255]}
                            tick={{ fontSize: 12 }}
                            label={{ value: 'Output Density', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.9)', borderRadius: '4px', border: '1px solid #ccc' }}
                        />
                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 255, y: 255 }]} stroke="#ccc" strokeDasharray="3 3" />
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

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4 bg-muted/50 rounded-md">
                <div className="space-y-4">
                    <label className="text-sm font-medium">Adjustment Type</label>
                    <AdjustmentSelect
                        value={adjustmentType}
                        onChange={setAdjustmentType}
                    />
                </div>

                <div className="space-y-4">
                    <div className="flex justify-between">
                        <label className="text-sm font-medium">Amount</label>
                        <span className="text-sm text-gray-500">{amount}</span>
                    </div>
                    <Slider
                        value={[amount]}
                        min={-100}
                        max={100}
                        step={1}
                        onValueChange={(vals) => setAmount(vals[0])}
                    />
                </div>

                <div className="md:col-span-2">
                    <Button
                        onClick={handleApplyAdjustment}
                        disabled={isApplying}
                        className="w-full md:w-auto"
                    >
                        {isApplying ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : null}
                        Apply Adjustment
                    </Button>
                    {error && <p className="text-sm text-destructive mt-2">{error}</p>}
                </div>
            </div>
        </div>
    );
}
