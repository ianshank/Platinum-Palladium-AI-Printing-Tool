/* eslint-disable @typescript-eslint/no-unsafe-argument -- Test mocks use partial return types */
/**
 * SearchConfigPanel component tests
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SearchConfigPanel } from './SearchConfigPanel';
import { useMCTSCalibration } from '@/hooks/useMCTSCalibration';

// Mock the hook
vi.mock('@/hooks/useMCTSCalibration');

describe('SearchConfigPanel', () => {
  const mockUseMCTSCalibration = {
    searchConfig: {
      fixedParameters: {},
      targetAesthetics: {},
    },
    setSearchConfig: vi.fn(),
    setFixedParameter: vi.fn(),
    removeFixedParameter: vi.fn(),
    runSearch: vi.fn(),
    isSearching: false,
    isEngineReady: true,
    isTorchAvailable: true,
    parameterRanges: {
      exposure_time: { min: 60, max: 300, default: 120, unit: 'seconds' },
    },
  };

  beforeEach(() => {
    vi.mocked(useMCTSCalibration).mockReturnValue(mockUseMCTSCalibration as any);
  });

  it('should render panel with title', () => {
    render(<SearchConfigPanel />);

    expect(screen.getByText('Search Configuration')).toBeInTheDocument();
    expect(screen.getByTestId('search-config-panel')).toBeInTheDocument();
  });

  it('should show engine ready status', () => {
    render(<SearchConfigPanel />);

    expect(screen.getByText('Engine Ready')).toBeInTheDocument();
  });

  it('should show engine not ready when torch unavailable', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      ...mockUseMCTSCalibration,
      isEngineReady: false,
      isTorchAvailable: false,
    } as any);

    render(<SearchConfigPanel />);

    expect(screen.getByText('Engine Not Ready')).toBeInTheDocument();
    expect(screen.getByText('(PyTorch not available)')).toBeInTheDocument();
  });

  it('should render paper type dropdown', () => {
    render(<SearchConfigPanel />);

    const select = screen.getByLabelText('Paper Type');
    expect(select).toBeInTheDocument();
    expect(select).toHaveValue('');
  });

  it('should update paper type on selection', async () => {
    const user = userEvent.setup();
    render(<SearchConfigPanel />);

    const select = screen.getByLabelText('Paper Type');
    await user.selectOptions(select, 'arches_platine');

    expect(mockUseMCTSCalibration.setSearchConfig).toHaveBeenCalledWith({
      paperType: 'arches_platine',
    });
  });

  it('should render UV source dropdown', () => {
    render(<SearchConfigPanel />);

    const select = screen.getByLabelText('UV Source');
    expect(select).toBeInTheDocument();
  });

  it('should render simulations slider', () => {
    render(<SearchConfigPanel />);

    expect(screen.getByLabelText(/Simulations:/)).toBeInTheDocument();
  });

  it('should disable search button when engine not ready', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      ...mockUseMCTSCalibration,
      isEngineReady: false,
    } as any);

    render(<SearchConfigPanel />);

    const button = screen.getByRole('button', { name: /Run MCTS Search/i });
    expect(button).toBeDisabled();
  });

  it('should disable search button when searching', () => {
    vi.mocked(useMCTSCalibration).mockReturnValue({
      ...mockUseMCTSCalibration,
      isSearching: true,
    } as any);

    render(<SearchConfigPanel />);

    const button = screen.getByRole('button', { name: /Searching.../i });
    expect(button).toBeDisabled();
  });

  it('should call runSearch when search button clicked', async () => {
    const user = userEvent.setup();
    render(<SearchConfigPanel />);

    const button = screen.getByRole('button', { name: /Run MCTS Search/i });
    await user.click(button);

    expect(mockUseMCTSCalibration.runSearch).toHaveBeenCalled();
  });

  it('should render parameter controls when ranges available', () => {
    render(<SearchConfigPanel />);

    expect(screen.getByText('Fixed Parameters')).toBeInTheDocument();
    expect(screen.getByText('exposure time')).toBeInTheDocument();
  });
});
