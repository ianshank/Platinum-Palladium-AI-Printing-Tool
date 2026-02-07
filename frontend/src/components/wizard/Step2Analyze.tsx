import { useMemo } from 'react';
import styled, { useTheme } from 'styled-components';
import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { useStore } from '@/stores';
import { assessScanQuality } from '@/api/hooks/useScan';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[8]};
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${({ theme }) => theme.spacing[4]};
`;

const MetricCard = styled.div`
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  padding: ${({ theme }) => theme.spacing[4]};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const MetricLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const MetricValue = styled.div<{ $color?: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme, $color }) => $color || theme.colors.text.primary};
`;

const ChartContainer = styled.div`
  height: 300px;
  width: 100%;
  margin-top: ${({ theme }) => theme.spacing[4]};
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.lg};
`;

const QualityBadge = styled.span<{ $type: 'success' | 'warning' | 'error' }>`
  display: inline-flex;
  align-items: center;
  padding: ${({ theme }) => `${theme.spacing[1]} ${theme.spacing[3]}`};
  border-radius: ${({ theme }) => theme.radii.full};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  background-color: ${({ theme, $type }) =>
        $type === 'success'
            ? theme.colors.semantic.success + '20'
            : $type === 'warning'
                ? theme.colors.semantic.warning + '20'
                : theme.colors.semantic.error + '20'};
  color: ${({ theme, $type }) =>
        $type === 'success'
            ? theme.colors.semantic.success
            : $type === 'warning'
                ? theme.colors.semantic.warning
                : theme.colors.semantic.error};
`;

const IssuesList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};
`;

const IssueItem = styled.li<{ $type: 'warning' | 'error' }>`
  display: flex;
  align-items: flex-start;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme, $type }) =>
        $type === 'error'
            ? theme.colors.semantic.error + '10'
            : theme.colors.semantic.warning + '10'};
  border-radius: ${({ theme }) => theme.radii.md};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.primary};
`;

export function Step2Analyze() {
    const theme = useTheme();

    // Store
    const currentCalibration = useStore((state) => state.calibration.current);
    const nextStep = useStore((state) => state.calibration.nextStep);
    const prevStep = useStore((state) => state.calibration.previousStep);

    // Derive analysis result from calibration state
    const analysisResult = useMemo(() => {
        if (!currentCalibration || !currentCalibration.measurements.length) return null;

        const densities = currentCalibration.measurements.map(m => m.measuredDensity);
        const metadata = currentCalibration.metadata;

        return {
            densities,
            dmax: metadata?.dmax ?? Math.max(...densities),
            dmin: metadata?.dmin ?? Math.min(...densities),
            range: metadata?.range ?? (Math.max(...densities) - Math.min(...densities)),
            num_patches: metadata?.num_patches ?? densities.length,
        };
    }, [currentCalibration]);

    const qualityAssessment = useMemo(() => {
        if (!analysisResult) return null;
        return assessScanQuality(analysisResult);
    }, [analysisResult]);

    const chartData = useMemo(() => {
        if (!analysisResult) return [];
        return analysisResult.densities.map((d: number, i: number) => ({
            step: i + 1,
            density: d,
        }));
    }, [analysisResult]);

    if (!analysisResult) {
        return <div>No analysis result found. Please restart the wizard.</div>;
    }

    return (
        <Container>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3>Analysis Results</h3>
                {qualityAssessment && (
                    <QualityBadge
                        $type={
                            qualityAssessment.overall === 'excellent' || qualityAssessment.overall === 'good'
                                ? 'success'
                                : qualityAssessment.overall === 'acceptable'
                                    ? 'warning'
                                    : 'error'
                        }
                    >
                        Quality: {qualityAssessment.overall.toUpperCase()} ({qualityAssessment.score}/100)
                    </QualityBadge>
                )}
            </div>

            <MetricsGrid>
                <MetricCard>
                    <MetricLabel>Dmax (Black)</MetricLabel>
                    <MetricValue>{analysisResult.dmax.toFixed(2)}</MetricValue>
                </MetricCard>
                <MetricCard>
                    <MetricLabel>Dmin (Paper Base)</MetricLabel>
                    <MetricValue>{analysisResult.dmin.toFixed(2)}</MetricValue>
                </MetricCard>
                <MetricCard>
                    <MetricLabel>Density Range</MetricLabel>
                    <MetricValue>{analysisResult.range.toFixed(2)}</MetricValue>
                </MetricCard>
                <MetricCard>
                    <MetricLabel>Steps Detected</MetricLabel>
                    <MetricValue>{analysisResult.num_patches}</MetricValue>
                </MetricCard>
            </MetricsGrid>

            <ChartContainer>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border.default} />
                        <XAxis
                            dataKey="step"
                            stroke={theme.colors.text.secondary}
                            label={{ value: 'Step Number', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                            stroke={theme.colors.text.secondary}
                            label={{ value: 'Density', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: theme.colors.background.secondary,
                                borderColor: theme.colors.border.default,
                                color: theme.colors.text.primary,
                            }}
                        />
                        <Line
                            type="monotone"
                            dataKey="density"
                            stroke={theme.colors.accent.primary}
                            strokeWidth={2}
                            dot={{ fill: theme.colors.accent.primary }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </ChartContainer>

            {qualityAssessment && qualityAssessment.issues.length > 0 && (
                <div>
                    <h4>Quality Issues</h4>
                    <IssuesList>
                        {qualityAssessment.issues.map((issue: any, idx: number) => (
                            <IssueItem key={idx} $type={issue.type}>
                                <strong>{issue.type.toUpperCase()}:</strong> {issue.message}
                                {issue.suggestion && <span> — {issue.suggestion}</span>}
                            </IssueItem>
                        ))}
                    </IssuesList>
                </div>
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
                    style={{
                        padding: '12px 24px',
                        backgroundColor: theme.colors.accent.primary,
                        color: theme.colors.text.inverse,
                        border: 'none',
                        borderRadius: theme.radii.md,
                        cursor: 'pointer',
                        fontWeight: 600,
                    }}
                >
                    Continue to Configuration
                </button>
            </div>
        </Container>
    );
}
