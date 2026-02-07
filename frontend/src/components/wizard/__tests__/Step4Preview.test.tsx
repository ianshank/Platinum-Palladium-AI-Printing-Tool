import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { renderWithProviders } from '@/test-utils';
import { useStore } from '@/stores';
import { Step4Preview } from '../Step4Preview';
import type { DensityMeasurement } from '@/stores/slices/calibrationSlice';

// Mock Recharts
vi.mock('recharts', async (importOriginal) => {
    const actual = await importOriginal();
    return {
        ...actual as any,
        ResponsiveContainer: ({ children }: { children: any }) => <div style={{ width: 800, height: 600 }}>{children}</div>,
    };
});

// Mock ResizeObserver
beforeAll(() => {
    global.ResizeObserver = class ResizeObserver {
        observe() { }
        unobserve() { }
        disconnect() { }
    };
});

// Mock API hook
const mockMutate = vi.fn();
vi.mock('@/api/hooks', () => ({
    useGenerateCurve: () => ({
        mutate: mockMutate,
        isPending: false,
        error: null,
    }),
}));

const mockMeasurements: DensityMeasurement[] = Array.from({ length: 5 }, (_, i) => ({
    step: i + 1,
    targetDensity: i * 0.2,
    measuredDensity: i * 0.2,
}));

describe('Step4Preview', () => {
    const initialCalibration = useStore.getState().calibration;
    const initialCurve = useStore.getState().curve;

    beforeEach(() => {
        vi.clearAllMocks();
        useStore.setState({
            calibration: initialCalibration,
            curve: initialCurve
        });
    });

    it('triggers generation on mount if no curve exists', async () => {
        useStore.setState((state) => ({
            calibration: {
                ...state.calibration,
                current: {
                    ...state.calibration.current!,
                    measurements: mockMeasurements,
                    name: 'Test Curve',
                }
            },
            curve: { ...state.curve, current: null }
        }));

        renderWithProviders(<Step4Preview />);

        await waitFor(() => {
            expect(mockMutate).toHaveBeenCalledTimes(1);
        });

        expect(mockMutate).toHaveBeenCalledWith(
            expect.objectContaining({
                measurements: expect.any(Array),
                name: 'Test Curve',
                curve_type: 'monotonic'
            }),
            expect.any(Object)
        );
    });

    it('displays loading state while generating', () => {
        // Mock isPending true
        // We need to override the mock for this test or use a spy.
        // It's easier if we mock the whole module with a variable we can change, 
        // OR just render with "StatusMessage" expected text since "isPending" comes from hook.

        // Since we mocked the hook to return isPending: false, we can't easily change it here without closure manipulation.
        // Let's rely on text presence "Generating your calibration curve..." which appears if !generatedCurve (and no error).

        useStore.setState((state) => ({
            calibration: { ...state.calibration, current: { ...state.calibration.current!, measurements: mockMeasurements } },
            curve: { ...state.curve, current: null }
        }));

        renderWithProviders(<Step4Preview />);

        expect(screen.getByText(/Generating your calibration curve/i)).toBeInTheDocument();
    });

    it('displays chart and metrics when curve exists', () => {
        useStore.setState((state) => ({
            calibration: { ...state.calibration, current: { ...state.calibration.current!, measurements: mockMeasurements } },
            curve: {
                ...state.curve,
                current: {
                    id: 'curve-1',
                    name: 'Generated Curve',
                    type: 'linear',
                    measurements: [],
                    points: [{ x: 0, y: 0 }, { x: 100, y: 100 }],
                    createdAt: '',
                    updatedAt: '',
                },
                isModified: false,
                history: [],
                future: []
            }
        }));

        renderWithProviders(<Step4Preview />);

        expect(screen.getByText('Curve Preview')).toBeInTheDocument();

        expect(screen.getByText('Points')).toBeInTheDocument();
        expect(screen.getByText('2')).toBeInTheDocument();
    });

    it('calls regenerate when button is clicked', async () => {
        // Setup existing curve so button is enabled
        useStore.setState((state) => ({
            calibration: { ...state.calibration, current: { ...state.calibration.current!, measurements: mockMeasurements } },
            curve: {
                ...state.curve,
                current: {
                    id: 'curve-1',
                    name: 'Generated Curve',
                    type: 'linear',
                    measurements: [],
                    points: [{ x: 0, y: 0 }],
                    createdAt: '',
                    updatedAt: '',
                }
            }
        }));

        renderWithProviders(<Step4Preview />);

        // Clear initial call if any (from useEffect? useEffect checks !generatedCurve, so it won't fire)
        mockMutate.mockClear();

        const regenerateBtn = screen.getByText('Regenerate');
        fireEvent.click(regenerateBtn);

        await waitFor(() => {
            expect(mockMutate).toHaveBeenCalledTimes(1);
        });
    });
});
