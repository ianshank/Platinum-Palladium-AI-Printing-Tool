import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, screen } from '@testing-library/react';
import { renderWithProviders } from '@/test-utils';
import { useStore } from '@/stores';
import { Step2Analyze } from '../Step2Analyze';
import type {
  CalibrationData,
  DensityMeasurement,
} from '@/stores/slices/calibrationSlice';

// Mock Recharts
vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal();
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  return {
    ...(actual as any),
    ResponsiveContainer: ({ children }: { children: any }) => (
      <div style={{ width: 800, height: 600 }}>{children}</div>
    ),
  };
});

// Mock ResizeObserver
beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

// Mock the API hook
vi.mock('@/api/hooks', () => ({
  assessScanQuality: vi.fn(() => ({
    overall: 'good',
    score: 85,
    quality: 'good',
    issues: [],
  })),
}));

const mockMeasurements: DensityMeasurement[] = Array.from(
  { length: 21 },
  (_, i) => ({
    step: i + 1,
    targetDensity: i * 0.15,
    measuredDensity: i * 0.14 + 0.05,
  })
);

const mockCalibration: CalibrationData = {
  id: 'test-cal-1',
  name: 'Test Calibration',
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  tabletType: '21-step',
  measurements: mockMeasurements,
  metadata: {
    dmax: 2.85,
    dmin: 0.05,
    range: 2.8,
    num_patches: 21,
  },
};

describe('Step2Analyze', () => {
  const initialCalibrationState = useStore.getState().calibration;

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset full state
    useStore.setState({ calibration: initialCalibrationState });
  });

  it('renders "No analysis result" if no current calibration', () => {
    // Ensure state is empty (it should be by default/reset)
    useStore.setState((state) => ({
      calibration: { ...state.calibration, current: null },
    }));

    renderWithProviders(<Step2Analyze />);

    expect(screen.getByText(/No analysis result found/i)).toBeInTheDocument();
  });

  it('renders metrics and chart when measurements exist', () => {
    useStore.setState((state) => ({
      calibration: {
        ...state.calibration,
        current: mockCalibration,
      },
    }));

    renderWithProviders(<Step2Analyze />);

    expect(screen.getByText('Analysis Results')).toBeInTheDocument();
    expect(screen.getByText('Dmax (Black)')).toBeInTheDocument();
    expect(screen.getByText('2.85')).toBeInTheDocument(); // dmax
    expect(screen.getByText('0.05')).toBeInTheDocument(); // dmin
    expect(screen.getByText('2.80')).toBeInTheDocument(); // range
    expect(screen.getByText('21')).toBeInTheDocument(); // steps

    // Quality badge
    expect(screen.getByText(/GOOD/)).toBeInTheDocument();
    expect(screen.getByText(/\(85\/100\)/)).toBeInTheDocument();
  });

  it('navigates to next step on continue', () => {
    useStore.setState((state) => ({
      calibration: {
        ...state.calibration,
        current: mockCalibration,
        totalSteps: 5,
        currentStep: 1,
      },
    }));

    renderWithProviders(<Step2Analyze />);

    const continueBtn = screen.getByText('Continue to Configuration');
    fireEvent.click(continueBtn);

    expect(useStore.getState().calibration.currentStep).toBe(2);
  });

  it('navigates to previous step on back', () => {
    useStore.setState((state) => ({
      calibration: {
        ...state.calibration,
        current: mockCalibration,
        totalSteps: 5,
        currentStep: 2,
      },
    }));

    renderWithProviders(<Step2Analyze />);

    const backBtn = screen.getByText('Back');
    fireEvent.click(backBtn);

    expect(useStore.getState().calibration.currentStep).toBe(1);
  });
});
