/**
 * Curve visualization chart component
 * Renders the input/output density mapping with Recharts
 */

import React, { useMemo } from 'react';
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
import { logger } from '@/lib/logger';
import { cn } from '@/lib/utils';

export interface CurveChartProps {
  /** Input density values (x-axis) */
  inputValues: number[];
  /** Output density values (y-axis) */
  outputValues: number[];
  /** Curve name for accessibility label */
  curveName: string;
  /** Maximum value for chart axes (typically 255) */
  maxValue: number;
  /** Callback when chart area is clicked to add control points */
  onChartClick?: (e: React.MouseEvent<HTMLDivElement>) => void;
  /** Optional CSS class */
  className?: string;
  /** Chart height in pixels */
  height?: number;
}

interface ChartDataPoint {
  input: number;
  output: number;
}

/**
 * Renders curve as a chart with input/output density mapping
 *
 * @example
 * ```tsx
 * <CurveChart
 *   inputValues={[0, 64, 128, 192, 255]}
 *   outputValues={[0, 70, 128, 185, 255]}
 *   curveName="My Calibration"
 *   maxValue={255}
 *   onChartClick={handleClick}
 * />
 * ```
 */
export const CurveChart = React.forwardRef<HTMLDivElement, CurveChartProps>(
  (
    {
      inputValues,
      outputValues,
      curveName,
      maxValue,
      onChartClick,
      className,
      height = 400,
    },
    ref
  ) => {
    // Prepare data for Recharts
    const chartData = useMemo<ChartDataPoint[]>(() => {
      logger.debug('CurveChart: preparing chart data', {
        inputCount: inputValues.length,
        outputCount: outputValues.length,
      });
      return inputValues.map((input, index) => ({
        input,
        output: outputValues[index] ?? 0,
      }));
    }, [inputValues, outputValues]);

    const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
      if (onChartClick) {
        logger.debug('CurveChart: chart clicked');
        onChartClick(e);
      }
    };

    return (
      <div
        ref={ref}
        onClick={handleClick}
        className={cn(
          'h-[400px] w-full cursor-crosshair rounded-md border bg-white/5 p-4',
          className
        )}
        style={{ height }}
        role="img"
        aria-label={`Curve chart for ${curveName} showing input vs output density mapping. Click to add control points (experimental).`}
        title="Click on the chart to add a control point"
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="input"
              type="number"
              domain={[0, maxValue]}
              tick={{ fontSize: 12 }}
              label={{
                value: 'Input Density',
                position: 'insideBottom',
                offset: -5,
              }}
            />
            <YAxis
              type="number"
              domain={[0, maxValue]}
              tick={{ fontSize: 12 }}
              label={{
                value: 'Output Density',
                angle: -90,
                position: 'insideLeft',
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                borderRadius: '4px',
                border: '1px solid hsl(var(--border))',
                color: 'hsl(var(--card-foreground))',
              }}
            />
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: maxValue, y: maxValue },
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
    );
  }
);

CurveChart.displayName = 'CurveChart';
