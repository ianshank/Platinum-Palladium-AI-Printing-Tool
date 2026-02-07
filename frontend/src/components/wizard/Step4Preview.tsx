import { useEffect, useMemo, useState } from 'react';
import styled, { useTheme } from 'styled-components';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import { useStore } from '@/stores';
import { useGenerateCurve } from '@/api/hooks/useCurves';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[8]};
`;

const ChartContainer = styled.div`
  height: 400px;
  width: 100%;
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const MetricsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${({ theme }) => theme.spacing[4]};
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const MetricItem = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const MetricLabel = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const MetricValue = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const ErrorMessage = styled.div`
  color: ${({ theme }) => theme.colors.semantic.error};
  background-color: ${({ theme }) => `rgba(239, 68, 68, 0.1)`};
  padding: ${({ theme }) => theme.spacing[4]};
  border-radius: ${({ theme }) => theme.radii.md};
  text-align: center;
`;

const StatusMessage = styled.div`
  color: ${({ theme }) => theme.colors.text.secondary};
  text-align: center;
  padding: ${({ theme }) => theme.spacing[8]};
`;

export function Step4Preview() {
    const theme = useTheme();

    // Store state
    const densities = useStore((state) => state.calibration.current?.measurements || []);
    const curveName = useStore((state) => state.calibration.current?.name || '');
    // Read from metadata
    const curveStrategy = useStore((state) => (state.calibration.current?.metadata?.curveStrategy as string) || 'monotonic');

    const generatedCurve = useStore((state) => state.curve.current);
    const setGeneratedCurve = useStore((state) => state.curve.setCurve);

    const nextStep = useStore((state) => state.calibration.nextStep);
    const prevStep = useStore((state) => state.calibration.previousStep);

    // API Hooks
    const { mutate: generateCurve, isPending, error } = useGenerateCurve();
    const [hasTriedGeneration, setHasTriedGeneration] = useState(false);

    // Auto-generate curve on mount if not already generated
    useEffect(() => {
        if (!generatedCurve && !isPending && !hasTriedGeneration && !error) {
            setHasTriedGeneration(true);
            generateCurve(
                {
                    measurements: densities.map(d => d.measuredDensity),
                    name: curveName,
                    curve_type: curveStrategy,
                },
                {
                    onSuccess: (data) => {
                        const points = data.input_values.map((x: number, i: number) => ({
                            x,
                            y: data.output_values[i] ?? 0,
                        }));

                        setGeneratedCurve({
                            id: data.curve_id,
                            name: data.name,
                            type: 'monotonic', // Default or from response if available
                            points,
                            createdAt: new Date().toISOString(),
                            updatedAt: new Date().toISOString(),
                            metadata: {
                                curve_type: 'calibration',
                            },
                        });
                    },
                }
            );
        }
    }, [
        densities,
        curveName,
        curveStrategy,
        generatedCurve,
        isPending,
        hasTriedGeneration,
        error,
        generateCurve,
        setGeneratedCurve,
    ]);

    const chartData = useMemo(() => {
        if (!generatedCurve) return [];

        return generatedCurve.points.map((point) => ({
            input: point.x,
            output: point.y,
        }));
    }, [generatedCurve]);

    const handleRegenerate = () => {
        setHasTriedGeneration(true);
        generateCurve(
            {
                measurements: densities.map(d => d.measuredDensity),
                name: curveName,
                curve_type: curveStrategy,
            },
            {
                onSuccess: (data) => {
                    const points = data.input_values.map((x: number, i: number) => ({
                        x,
                        y: data.output_values[i] ?? 0,
                    }));

                    setGeneratedCurve({
                        id: data.curve_id,
                        name: data.name,
                        type: 'monotonic',
                        points,
                        createdAt: new Date().toISOString(),
                        updatedAt: new Date().toISOString(),
                        metadata: {
                            curve_type: 'calibration',
                        },
                    });
                },
            }
        );
    };

    return (
        <Container>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3>Curve Preview</h3>
                <button
                    onClick={handleRegenerate}
                    disabled={isPending}
                    style={{
                        padding: '8px 16px',
                        backgroundColor: 'transparent',
                        color: theme.colors.accent.primary,
                        border: `1px solid ${theme.colors.accent.primary}`,
                        borderRadius: theme.radii.md,
                        cursor: isPending ? 'wait' : 'pointer',
                        fontSize: theme.typography.fontSize.sm,
                    }}
                >
                    {isPending ? 'Generating...' : 'Regenerate'}
                </button>
            </div>

            {error ? (
                <ErrorMessage>
                    <p>Failed to generate curve</p>
                    <p style={{ fontSize: '0.8em', marginTop: '0.5rem' }}>
                        {error instanceof Error ? error.message : 'Unknown error occurred'}
                    </p>
                    <button
                        onClick={handleRegenerate}
                        style={{ marginTop: '1rem', cursor: 'pointer', textDecoration: 'underline' }}
                    >
                        Try Again
                    </button>
                </ErrorMessage>
            ) : !generatedCurve ? (
                <StatusMessage>
                    Generating your calibration curve...
                </StatusMessage>
            ) : (
                <>
                    <MetricsContainer>
                        <MetricItem>
                            <MetricLabel>Points</MetricLabel>
                            <MetricValue>{generatedCurve.points.length}</MetricValue>
                        </MetricItem>
                        <MetricItem>
                            <MetricLabel>Mode</MetricLabel>
                            <MetricValue style={{ textTransform: 'capitalize' }}>
                                {curveStrategy.replace('_', ' ')}
                            </MetricValue>
                        </MetricItem>
                    </MetricsContainer>

                    <ChartContainer>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border.default} />
                                <XAxis
                                    dataKey="input"
                                    stroke={theme.colors.text.secondary}
                                    label={{ value: 'Input Value', position: 'insideBottom', offset: -10 }}
                                    domain={[0, 100]}
                                    type="number"
                                />
                                <YAxis
                                    stroke={theme.colors.text.secondary}
                                    label={{ value: 'Output Value', angle: -90, position: 'insideLeft' }}
                                    domain={[0, 100]}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: theme.colors.background.secondary,
                                        borderColor: theme.colors.border.default,
                                        color: theme.colors.text.primary,
                                    }}
                                    formatter={(value: number) => value.toFixed(2)}
                                />
                                <ReferenceLine x={0} stroke={theme.colors.border.default} />
                                <ReferenceLine y={0} stroke={theme.colors.border.default} />

                                {/* Diagonal reference for linearity comparison */}
                                <Line
                                    type="monotone"
                                    dataKey="input"
                                    stroke={theme.colors.border.default}
                                    strokeDasharray="5 5"
                                    dot={false}
                                    strokeWidth={1}
                                    name="Linear Ref"
                                />

                                <Line
                                    type="monotone"
                                    dataKey="output"
                                    stroke={theme.colors.accent.primary}
                                    strokeWidth={3}
                                    dot={false}
                                    activeDot={{ r: 6 }}
                                    name="Curve"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </ChartContainer>
                </>
            )}

            {/* Navigation */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2rem' }}>
                <button
                    onClick={prevStep}
                    style={{
                        padding: '12px 24px',
                        backgroundColor: theme.colors.background.tertiary,
                        color: theme.colors.text.secondary,
                        border: `1px solid ${theme.colors.border.default}`,
                        borderRadius: theme.radii.md,
                        cursor: 'pointer',
                    }}
                >
                    Back
                </button>
                <button
                    onClick={nextStep}
                    disabled={!generatedCurve || isPending}
                    style={{
                        padding: '12px 24px',
                        backgroundColor: theme.colors.accent.primary,
                        color: theme.colors.text.inverse,
                        border: 'none',
                        borderRadius: theme.radii.md,
                        cursor: !generatedCurve || isPending ? 'not-allowed' : 'pointer',
                        fontWeight: 600,
                        opacity: !generatedCurve || isPending ? 0.5 : 1,
                    }}
                >
                    Continue to Export
                </button>
            </div>
        </Container>
    );
}
