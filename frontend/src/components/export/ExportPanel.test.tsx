import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { type ExportFormat, ExportPanel } from './ExportPanel';

// Mock utilities
vi.mock('@/lib/utils', async () => {
  const actual = await vi.importActual('@/lib/utils');
  return {
    ...actual,
    downloadFile: vi.fn(),
  };
});

// Mock logger
vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Test data
const testFormats: ExportFormat[] = [
  {
    id: 'qtr',
    label: 'QuadTone RIP',
    extension: '.qtr',
    description: 'Standard QTR format',
  },
  {
    id: 'csv',
    label: 'CSV Data',
    extension: '.csv',
    description: 'Raw data',
  },
  {
    id: 'json',
    label: 'JSON',
    extension: '.json',
    description: 'Structured data',
  },
];

describe('ExportPanel', () => {
  let mockOnExport: ReturnType<typeof vi.fn>;
  let mockOnExportComplete: ReturnType<typeof vi.fn>;
  let mockOnExportError: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockOnExport = vi.fn();
    mockOnExportComplete = vi.fn();
    mockOnExportError = vi.fn();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Rendering', () => {
    it('renders with title and description', () => {
      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          title="Export Your Data"
          description="Choose a format to download"
        />
      );

      expect(screen.getByText('Export Your Data')).toBeInTheDocument();
      expect(screen.getByText('Choose a format to download')).toBeInTheDocument();
    });

    it('renders without description when not provided', () => {
      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          title="Export"
        />
      );

      expect(screen.getByText('Export')).toBeInTheDocument();
      expect(screen.queryByText('Choose a format')).not.toBeInTheDocument();
    });

    it('renders all format options from formats prop', () => {
      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      expect(screen.getByText('QuadTone RIP')).toBeInTheDocument();
      expect(screen.getByText('.qtr')).toBeInTheDocument();
      expect(screen.getByText('Standard QTR format')).toBeInTheDocument();

      expect(screen.getByText('CSV Data')).toBeInTheDocument();
      expect(screen.getByText('.csv')).toBeInTheDocument();
      expect(screen.getByText('Raw data')).toBeInTheDocument();

      expect(screen.getByText('JSON')).toBeInTheDocument();
      expect(screen.getByText('.json')).toBeInTheDocument();
      expect(screen.getByText('Structured data')).toBeInTheDocument();
    });

    it('renders with data-testid for testing', () => {
      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      expect(screen.getByTestId('export-panel')).toBeInTheDocument();
      expect(screen.getByTestId('export-button')).toBeInTheDocument();
      expect(screen.getByTestId('format-option-qtr')).toBeInTheDocument();
      expect(screen.getByTestId('format-option-csv')).toBeInTheDocument();
      expect(screen.getByTestId('format-option-json')).toBeInTheDocument();
    });
  });

  describe('Format Selection', () => {
    it('selects the first format by default when no defaultFormat provided', () => {
      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      const qtrRadio = screen.getByLabelText('QuadTone RIP .qtr');
      expect(qtrRadio).toBeChecked();
    });

    it('selects the defaultFormat when provided', () => {
      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          defaultFormat="csv"
        />
      );

      const csvRadio = screen.getByLabelText('CSV Data .csv');
      expect(csvRadio).toBeChecked();
    });

    it('allows changing format selection via click', () => {
      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      const csvRadio = screen.getByLabelText('CSV Data .csv');
      fireEvent.click(csvRadio);

      expect(csvRadio).toBeChecked();
    });

    it('displays checkmark on selected format', () => {
      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          defaultFormat="json"
        />
      );

      const jsonOption = screen.getByTestId('format-option-json');
      expect(jsonOption.querySelector('svg')).toBeInTheDocument();
    });
  });

  describe('Export Flow', () => {
    it('calls onExport with selected format ID when download button is clicked', () => {
      const mockBlob = new Blob(['test data'], { type: 'text/plain' });
      mockOnExport.mockResolvedValue(mockBlob);

      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          defaultFormat="csv"
        />
      );

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      expect(mockOnExport).toHaveBeenCalledWith('csv');
    });

    it('shows loading state during export', async () => {
      let resolveExport: (value: Blob) => void;
      mockOnExport.mockImplementation(
        () =>
          new Promise<Blob>((resolve) => {
            resolveExport = resolve;
          })
      );

      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      // Start export but don't resolve
      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(screen.getByText('Preparing download...')).toBeInTheDocument();
      });

      // Clean up
      await act(() => {
        resolveExport!(new Blob(['data']));
        return Promise.resolve();
      });
    });

    it('shows success state after successful export', async () => {
      const mockBlob = new Blob(['test data'], { type: 'text/plain' });
      mockOnExport.mockResolvedValue(mockBlob);

      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          fileName="test_curve"
        />
      );

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('export-success')).toBeInTheDocument();
      });

      expect(
        screen.getByText(/Export successful! File downloaded as/i)
      ).toBeInTheDocument();
    });

    it('calls onExportComplete after successful export', async () => {
      const mockBlob = new Blob(['test data'], { type: 'text/plain' });
      mockOnExport.mockResolvedValue(mockBlob);

      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          defaultFormat="json"
          onExportComplete={mockOnExportComplete}
        />
      );

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(mockOnExportComplete).toHaveBeenCalledWith('json');
      });
    });

    it('shows error state when export fails', async () => {
      const mockError = new Error('Network error');
      mockOnExport.mockRejectedValue(mockError);

      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('export-error')).toBeInTheDocument();
      });

      expect(screen.getByText(/Export failed:/i)).toBeInTheDocument();
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });

    it('calls onExportError when export fails', async () => {
      const mockError = new Error('Export error');
      mockOnExport.mockRejectedValue(mockError);

      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          onExportError={mockOnExportError}
        />
      );

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(mockOnExportError).toHaveBeenCalledWith(mockError);
      });
    });

    it('retry button calls onExport again after failure', async () => {
      const mockError = new Error('First attempt failed');
      const mockBlob = new Blob(['retry success'], { type: 'text/plain' });

      mockOnExport
        .mockRejectedValueOnce(mockError)
        .mockResolvedValueOnce(mockBlob);

      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      // First attempt - should fail
      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('export-error')).toBeInTheDocument();
      });

      // Click retry
      act(() => {
        fireEvent.click(screen.getByTestId('export-retry-button'));
      });

      // Should succeed on retry
      await waitFor(() => {
        expect(screen.getByTestId('export-success')).toBeInTheDocument();
      });

      expect(mockOnExport).toHaveBeenCalledTimes(2);
    });
  });

  describe('File Download', () => {
    it('triggers file download with correct filename and extension', async () => {
      const { downloadFile } = await import('@/lib/utils');
      const mockBlob = new Blob(['test data'], { type: 'text/plain' });
      mockOnExport.mockResolvedValue(mockBlob);

      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          fileName="my_calibration"
          defaultFormat="qtr"
        />
      );

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(downloadFile).toHaveBeenCalledWith(mockBlob, 'my_calibration.qtr');
      });
    });

    it('sanitizes filename by replacing spaces with underscores', async () => {
      const { downloadFile } = await import('@/lib/utils');
      const mockBlob = new Blob(['test data'], { type: 'text/plain' });
      mockOnExport.mockResolvedValue(mockBlob);

      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          fileName="my curve name"
          defaultFormat="csv"
        />
      );

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(downloadFile).toHaveBeenCalledWith(mockBlob, 'my_curve_name.csv');
      });
    });

    it('uses default filename when not provided', async () => {
      const { downloadFile } = await import('@/lib/utils');
      const mockBlob = new Blob(['test data'], { type: 'text/plain' });
      mockOnExport.mockResolvedValue(mockBlob);

      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          defaultFormat="json"
        />
      );

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(downloadFile).toHaveBeenCalledWith(mockBlob, 'export.json');
      });
    });
  });

  describe('Disabled State', () => {
    it('disables all interactions when disabled prop is true', () => {
      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          disabled={true}
        />
      );

      const downloadButton = screen.getByTestId('export-button');
      expect(downloadButton).toBeDisabled();

      const qtrRadio = screen.getByLabelText('QuadTone RIP .qtr');
      expect(qtrRadio).toBeDisabled();

      fireEvent.click(downloadButton);
      expect(mockOnExport).not.toHaveBeenCalled();
    });

    it('disables format selection during export', async () => {
      let resolveExport: (value: Blob) => void;
      mockOnExport.mockImplementation(
        () =>
          new Promise<Blob>((resolve) => {
            resolveExport = resolve;
          })
      );

      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        const csvRadio = screen.getByLabelText('CSV Data .csv');
        expect(csvRadio).toBeDisabled();
      });

      // Clean up
      await act(() => {
        resolveExport!(new Blob(['data']));
        return Promise.resolve();
      });
    });

    it('disables retry button when disabled prop is true', async () => {
      const mockError = new Error('Export failed');
      mockOnExport.mockRejectedValue(mockError);

      // First render without disabled so export can trigger
      const { rerender } = render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          disabled={false}
        />
      );

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('export-error')).toBeInTheDocument();
      });

      // Re-render with disabled=true
      rerender(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          disabled={true}
        />
      );

      const retryButton = screen.getByTestId('export-retry-button');
      expect(retryButton).toBeDisabled();
    });
  });

  describe('Accessibility', () => {
    it('uses radiogroup role for format selection', () => {
      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      const radioGroup = screen.getByRole('radiogroup', {
        name: 'Export format selection',
      });
      expect(radioGroup).toBeInTheDocument();
    });

    it('allows keyboard navigation through format options', () => {
      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      const qtrRadio = screen.getByLabelText('QuadTone RIP .qtr');
      qtrRadio.focus();
      expect(qtrRadio).toHaveFocus();
    });

    it('announces success state with aria-live', async () => {
      const mockBlob = new Blob(['test data'], { type: 'text/plain' });
      mockOnExport.mockResolvedValue(mockBlob);

      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        const successAlert = screen.getByRole('alert');
        expect(successAlert).toHaveAttribute('aria-live', 'polite');
      });
    });

    it('announces error state with aria-live', async () => {
      const mockError = new Error('Export failed');
      mockOnExport.mockRejectedValue(mockError);

      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      act(() => {
        fireEvent.click(screen.getByTestId('export-button'));
      });

      await waitFor(() => {
        const errorAlert = screen.getByRole('alert');
        expect(errorAlert).toHaveAttribute('aria-live', 'assertive');
      });
    });

    it('has proper aria-label for each format option', () => {
      render(<ExportPanel formats={testFormats} onExport={mockOnExport} />);

      expect(screen.getByLabelText('QuadTone RIP .qtr')).toBeInTheDocument();
      expect(screen.getByLabelText('CSV Data .csv')).toBeInTheDocument();
      expect(screen.getByLabelText('JSON .json')).toBeInTheDocument();
    });
  });

  describe('Controlled Mode', () => {
    it('uses controlled isExporting prop when provided', () => {
      const { rerender } = render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          isExporting={true}
        />
      );

      const downloadButton = screen.getByTestId('export-button');
      expect(downloadButton).toBeDisabled();
      expect(screen.getByText('Preparing download...')).toBeInTheDocument();

      rerender(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          isExporting={false}
        />
      );

      expect(downloadButton).not.toBeDisabled();
    });
  });

  describe('Custom Styling', () => {
    it('applies custom className', () => {
      render(
        <ExportPanel
          formats={testFormats}
          onExport={mockOnExport}
          className="custom-class"
        />
      );

      const panel = screen.getByTestId('export-panel');
      expect(panel).toHaveClass('custom-class');
    });
  });
});
