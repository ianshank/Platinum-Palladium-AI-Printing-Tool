/* eslint-disable @typescript-eslint/no-unsafe-argument -- Test mocks use partial return types */
/**
 * ResultsDashboard component tests
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ResultsDashboard } from './ResultsDashboard';
import { useMCTSCalibration } from '@/hooks/useMCTSCalibration';
import type { MCTSEvaluateResponse, MCTSSearchResponse } from '@/types/mcts';

// Mock the hooks
vi.mock('@/hooks/useMCTSCalibration');
vi.mock('@/api/mctsHooks', () => ({
  useMCTSExport: () => ({
    mutate: vi.fn(),
    isPending: false,
  }),
}));

// Mock Plotly
vi.mock('react-plotly.js', () => ({
  default: () => <div data-testid="plotly-chart">Chart</div>,
}));

describe('ResultsDashboard', () => {
  const mockSearchResult: MCTSSearchResponse = {
    searchId: 'test-123',
    bestParameters: {
      exposure_time: 120,
      developer_dilution: 1.5,
    },
    predictedCurve: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    qualityScore: 0.85,
    alternatives: [],
    searchTimeSeconds: 12.5,
    numSimulations: 500,
  };

  const mockEvaluateResult: MCTSEvaluateResponse = {
    densityCurve: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    dmin: 0.05,
    dmax: 1.85,
    densityRange: 1.8,
    gamma: 2.2,
    qualityScore: 0.9,
  };

  beforeEach(() => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      currentResult: null,
      evaluateResult: null,
    } as any);
  });

  it('should show placeholder when no results', () => {
    render(<ResultsDashboard />);

    expect(screen.getByTestId('results-placeholder')).toBeInTheDocument();
    expect(screen.getByText('No Results Yet')).toBeInTheDocument();
  });

  it('should render search results when available', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      currentResult: mockSearchResult,
      evaluateResult: null,
    } as any);

    render(<ResultsDashboard />);

    expect(screen.getByTestId('results-dashboard')).toBeInTheDocument();
    expect(screen.getByText('Search Results')).toBeInTheDocument();
  });

  it('should display quality score', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      currentResult: mockSearchResult,
      evaluateResult: null,
    } as any);

    render(<ResultsDashboard />);

    expect(screen.getByText('Quality Score')).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument();
  });

  it('should display search time for search results', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      currentResult: mockSearchResult,
      evaluateResult: null,
    } as any);

    render(<ResultsDashboard />);

    expect(screen.getByText('Search Time')).toBeInTheDocument();
    expect(screen.getByText('12.50s')).toBeInTheDocument();
  });

  it('should render predicted density curve chart', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      currentResult: mockSearchResult,
      evaluateResult: null,
    } as any);

    render(<ResultsDashboard />);

    expect(screen.getByText('Predicted Density Curve')).toBeInTheDocument();
    expect(screen.getByTestId('plotly-chart')).toBeInTheDocument();
  });

  it('should display optimal parameters table', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      currentResult: mockSearchResult,
      evaluateResult: null,
    } as any);

    render(<ResultsDashboard />);

    expect(screen.getByText('Optimal Parameters')).toBeInTheDocument();
    expect(screen.getByText('exposure time')).toBeInTheDocument();
    expect(screen.getByText('developer dilution')).toBeInTheDocument();
  });

  it('should display metrics when evaluation result available', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      currentResult: null,
      evaluateResult: mockEvaluateResult,
    } as any);

    render(<ResultsDashboard />);

    expect(screen.getByText('Dmin')).toBeInTheDocument();
    expect(screen.getByText('0.050')).toBeInTheDocument();
    expect(screen.getByText('Dmax')).toBeInTheDocument();
    expect(screen.getByText('1.850')).toBeInTheDocument();
    expect(screen.getByText('Density Range')).toBeInTheDocument();
    expect(screen.getByText('1.800')).toBeInTheDocument();
    expect(screen.getByText('Gamma')).toBeInTheDocument();
    expect(screen.getByText('2.200')).toBeInTheDocument();
  });

  it('should show export button', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      currentResult: mockSearchResult,
      evaluateResult: null,
    } as any);

    render(<ResultsDashboard />);

    expect(
      screen.getByRole('button', { name: /Export Results/i })
    ).toBeInTheDocument();
  });
});
