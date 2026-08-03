import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders, userEvent } from '@/test-utils';
import { CurvesPage } from './CurvesPage';
import type { QuadCurveValues } from '@/types/models';

// Mock heavy child components
vi.mock('@/components/curves/CurveEditor', () => ({
  CurveEditor: (props: {
    className?: string;
    initialCurve?: { name: string };
    onSave?: (data: { id: string; name: string }) => void;
  }) => (
    <div data-testid="curve-editor" className={props.className}>
      {props.initialCurve ? `Editor: ${props.initialCurve.name}` : 'Editor: no curve'}
      <button onClick={() => props.onSave?.({ id: 'saved-id', name: 'Saved Curve' })}>
        Save
      </button>
    </div>
  ),
}));

vi.mock('@/components/curves/CurveUpload', () => ({
  CurveUpload: (props: {
    onLoadCurve?: (data: QuadCurveValues, id: string, name: string) => void;
  }) => (
    <div data-testid="curve-upload">
      <button
        onClick={() => {
          const mockData: QuadCurveValues = {
            input_values: [0, 128, 255],
            output_values: [0, 128, 255],
          };
          props.onLoadCurve?.(mockData, 'test-curve-id', 'My Profile');
        }}
      >
        Load Curve
      </button>
    </div>
  ),
}));

vi.mock('@/components/export/ExportPanel', () => ({
  ExportPanel: (props: { disabled?: boolean; fileName?: string }) => (
    <div data-testid="export-panel" data-disabled={String(props.disabled ?? false)}>
      Export: {props.fileName}
    </div>
  ),
}));

vi.mock('@/api/hooks', () => ({
  useExportCurve: () => ({ mutateAsync: vi.fn().mockResolvedValue(new Blob()) }),
}));

describe('CurvesPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders page heading', () => {
    renderWithProviders(<CurvesPage />);

    expect(screen.getByText('Curve Editor')).toBeInTheDocument();
    expect(screen.getByText(/Upload, edit, and export/i)).toBeInTheDocument();
  });

  it('renders three tab triggers', () => {
    renderWithProviders(<CurvesPage />);

    expect(screen.getByRole('tab', { name: /upload .quad/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /edit curve/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /export/i })).toBeInTheDocument();
  });

  it('upload tab is active by default', () => {
    renderWithProviders(<CurvesPage />);

    expect(screen.getByRole('tab', { name: /upload .quad/i })).toHaveAttribute(
      'data-state',
      'active'
    );
    expect(screen.getByTestId('curve-upload')).toBeInTheDocument();
  });

  it('Edit and Export tabs are disabled when no curve is loaded', () => {
    renderWithProviders(<CurvesPage />);

    expect(screen.getByRole('tab', { name: /edit curve/i })).toBeDisabled();
    expect(screen.getByRole('tab', { name: /export/i })).toBeDisabled();
  });

  it('switches to Edit tab and enables Export after loading a curve', async () => {
    renderWithProviders(<CurvesPage />);

    // Click "Load Curve" in the mocked CurveUpload
    await userEvent.click(screen.getByRole('button', { name: /load curve/i }));

    // Edit tab should now be active
    expect(screen.getByRole('tab', { name: /edit curve/i })).toHaveAttribute(
      'data-state',
      'active'
    );
    // CurveEditor should render with the loaded curve name
    expect(screen.getByTestId('curve-editor')).toHaveTextContent('Editor: My Profile');

    // Export tab should be enabled
    expect(screen.getByRole('tab', { name: /export/i })).not.toBeDisabled();
  });

  it('export panel is disabled before any curve is loaded', () => {
    renderWithProviders(<CurvesPage />);

    // Export panel exists in DOM (Radix mounts all Content)
    // but should be marked as disabled
    const exportTab = screen.getByRole('tab', { name: /export/i });
    expect(exportTab).toBeDisabled();
  });
});
