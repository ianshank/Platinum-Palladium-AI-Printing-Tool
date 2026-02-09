import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { Step5Export } from './Step5Export';

const theme = {
  spacing: {
    0: '0',
    1: '0.25rem',
    2: '0.5rem',
    3: '0.75rem',
    4: '1rem',
    5: '1.25rem',
    6: '1.5rem',
    8: '2rem',
    10: '2.5rem',
    12: '3rem',
    16: '4rem',
    20: '5rem',
    24: '6rem',
    32: '8rem',
    40: '10rem',
    48: '12rem',
    56: '14rem',
    64: '16rem',
  },
  typography: {
    fontFamily: { sans: 'Inter, sans-serif', mono: 'monospace' },
    fontSize: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
    },
    fontWeight: { regular: 400, medium: 500, semibold: 600, bold: 700 },
  },
  colors: {
    text: {
      primary: '#111',
      secondary: '#666',
      inverse: '#fff',
      error: '#ef4444',
    },
    background: {
      primary: '#fff',
      secondary: '#f9fafb',
      tertiary: '#f3f4f6',
      hover: '#e5e7eb',
    },
    border: { default: '#e5e7eb', focus: '#3b82f6' },
    accent: {
      primary: '#3b82f6',
      primaryHover: '#2563eb',
      secondary: '#6366f1',
    },
    semantic: {
      success: '#10b981',
      warning: '#f59e0b',
      error: '#ef4444',
      info: '#3b82f6',
    },
  },
  radii: { sm: '0.25rem', md: '0.375rem', lg: '0.5rem', full: '9999px' },
  shadows: {
    sm: '0 1px 2px rgba(0,0,0,0.05)',
    md: '0 4px 6px rgba(0,0,0,0.1)',
    lg: '0 10px 15px rgba(0,0,0,0.1)',
  },
  transitions: { fast: '0.15s', normal: '0.3s', slow: '0.5s' },
  breakpoints: { tablet: '768px', desktop: '1024px', wide: '1280px' },
};

const mockResetCalibration = vi.fn();
const mockResetCurve = vi.fn();
const mockPreviousStep = vi.fn();
const mockExportMutate = vi.fn();

// Mock useStore
vi.mock('@/stores', () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- mock store selector with test state
  useStore: (selector: (state: any) => any) => {
    const state = {
      calibration: {
        current: { name: 'My Curve' },
        resetCalibration: mockResetCalibration,
        previousStep: mockPreviousStep,
      },
      curve: {
        current: { id: 'curve-123' },
        resetCurve: mockResetCurve,
      },
    };
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return selector(state);
  },
}));

// Mock useExportCurve
vi.mock('@/api/hooks', () => ({
  useExportCurve: () => ({
    mutate: mockExportMutate,
    isPending: false,
  }),
}));

// Mock tablet config
vi.mock('@/config/tablet.config', () => ({
  tabletConfig: {
    defaults: { tabletId: 'stouffer_21', exportFormat: 'qtr' },
    exportFormats: [
      {
        id: 'qtr',
        label: 'QuadTone RIP (.qtr)',
        extension: '.qtr',
        description: 'Standard QTR format',
      },
      {
        id: 'piezography',
        label: 'Piezography (.quad)',
        extension: '.quad',
        description: 'Piezography format',
      },
      {
        id: 'csv',
        label: 'CSV Data (.csv)',
        extension: '.csv',
        description: 'CSV data',
      },
      {
        id: 'json',
        label: 'JSON (.json)',
        extension: '.json',
        description: 'JSON data',
      },
    ],
  },
}));

function renderWithTheme(ui: React.ReactElement): ReturnType<typeof render> {
  return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

describe('Step5Export', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Provide window.alert and window.confirm for happy-dom
    vi.stubGlobal('alert', vi.fn());
    vi.stubGlobal('confirm', vi.fn());
  });

  it('renders success message', () => {
    renderWithTheme(<Step5Export />);

    expect(screen.getByText('Calibration Complete!')).toBeInTheDocument();
    expect(
      screen.getByText(/Your curve has been generated/)
    ).toBeInTheDocument();
  });

  it('renders export format options', () => {
    renderWithTheme(<Step5Export />);

    expect(screen.getByText('QuadTone RIP (.qtr)')).toBeInTheDocument();
    expect(screen.getByText('Piezography (.quad)')).toBeInTheDocument();
    expect(screen.getByText('CSV Data (.csv)')).toBeInTheDocument();
    expect(screen.getByText('JSON (.json)')).toBeInTheDocument();
  });

  it('renders action buttons', () => {
    renderWithTheme(<Step5Export />);

    expect(screen.getByText('Back')).toBeInTheDocument();
    expect(screen.getByText('Download Curve')).toBeInTheDocument();
    expect(screen.getByText('Start New Calibration')).toBeInTheDocument();
  });

  it('calls exportCurve on Download click with default format', () => {
    renderWithTheme(<Step5Export />);

    fireEvent.click(screen.getByText('Download Curve'));

    expect(mockExportMutate).toHaveBeenCalledWith(
      { curveId: 'curve-123', format: 'qtr' },
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment -- expect.any returns any for type matching
      expect.objectContaining({ onSuccess: expect.any(Function) })
    );
  });

  it('allows selecting different export format', () => {
    renderWithTheme(<Step5Export />);

    const csvRadio = screen.getByDisplayValue('csv');
    fireEvent.click(csvRadio);

    fireEvent.click(screen.getByText('Download Curve'));

    expect(mockExportMutate).toHaveBeenCalledWith(
      { curveId: 'curve-123', format: 'csv' },
      expect.any(Object)
    );
  });

  it('calls previousStep on Back click', () => {
    renderWithTheme(<Step5Export />);

    fireEvent.click(screen.getByText('Back'));
    expect(mockPreviousStep).toHaveBeenCalledTimes(1);
  });

  it('resets on Start New Calibration when confirmed', () => {
    vi.mocked(confirm).mockReturnValue(true);

    renderWithTheme(<Step5Export />);

    fireEvent.click(screen.getByText('Start New Calibration'));

    expect(mockResetCalibration).toHaveBeenCalledTimes(1);
    expect(mockResetCurve).toHaveBeenCalledTimes(1);
  });

  it('does not reset on Start New Calibration when cancelled', () => {
    vi.mocked(confirm).mockReturnValue(false);

    renderWithTheme(<Step5Export />);

    fireEvent.click(screen.getByText('Start New Calibration'));

    expect(mockResetCalibration).not.toHaveBeenCalled();
    expect(mockResetCurve).not.toHaveBeenCalled();
  });

  it('renders success icon', () => {
    renderWithTheme(<Step5Export />);

    expect(screen.getByText('✓')).toBeInTheDocument();
  });
});
