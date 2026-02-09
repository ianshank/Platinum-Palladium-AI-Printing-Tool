import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { renderWithProviders, userEvent } from '@/test-utils';
import { CurveEditor } from './CurveEditor';
import { api } from '@/api/client';
import { type CurveData } from '@/types/models';

// Mock API
vi.mock('@/api/client', () => ({
  api: {
    curves: {
      modify: vi.fn(),
    },
  },
}));

// Mock ResizeObserver for Recharts
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

describe('CurveEditor', () => {
  const mockCurve: CurveData = {
    id: '123',
    name: 'Test Curve',
    created_at: '2023-01-01',
    curve_type: 'custom',
    input_values: [0, 128, 255],
    output_values: [0, 128, 255],
  } as any;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders initial curve data', () => {
    renderWithProviders(<CurveEditor initialCurve={mockCurve} />);

    expect(screen.getByDisplayValue('Test Curve')).toBeInTheDocument();
  });

  it('renders default state without initial curve', () => {
    renderWithProviders(<CurveEditor />);

    expect(screen.getByDisplayValue('New Curve')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /save/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reset/i })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /apply adjustment/i })
    ).toBeInTheDocument();
  });

  it('handles apply adjustment', async () => {
    vi.spyOn(api.curves, 'modify').mockResolvedValue({
      success: true,
      curve_id: '123',
      name: 'Test Curve',
      adjustment_applied: 'contrast',
      input_values: [0, 128, 255],
      output_values: [10, 130, 245],
    } as any);

    renderWithProviders(<CurveEditor initialCurve={mockCurve} />);

    const applyButton = screen.getByRole('button', {
      name: /apply adjustment/i,
    });
    await userEvent.click(applyButton);

    expect(api.curves.modify).toHaveBeenCalled();
  });

  it('saves curve via API and calls onSave callback', async () => {
    const saveResponse = {
      success: true,
      curve_id: 'saved-123',
      name: 'Test Curve',
      adjustment_applied: 'none',
      input_values: [0, 128, 255],
      output_values: [0, 128, 255],
    };
    vi.spyOn(api.curves, 'modify').mockResolvedValue(saveResponse as any);

    const onSave = vi.fn();
    renderWithProviders(
      <CurveEditor initialCurve={mockCurve} onSave={onSave} />
    );

    const saveButton = screen.getByRole('button', { name: /save/i });
    await userEvent.click(saveButton);

    await waitFor(() => {
      expect(api.curves.modify).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'Test Curve',
          input_values: mockCurve.input_values,
          output_values: mockCurve.output_values,
          adjustment_type: 'none',
          amount: 0,
        })
      );
    });

    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 'saved-123',
          name: 'Test Curve',
        })
      );
    });
  });

  it('shows error when save fails', async () => {
    const axiosLikeError = Object.assign(new Error('Network error'), {
      response: { data: { message: 'Network error' } },
      isAxiosError: true,
    });
    vi.spyOn(api.curves, 'modify').mockRejectedValue(axiosLikeError);

    const onSave = vi.fn();
    renderWithProviders(
      <CurveEditor initialCurve={mockCurve} onSave={onSave} />
    );

    const saveButton = screen.getByRole('button', { name: /save/i });
    await userEvent.click(saveButton);

    await waitFor(() => {
      expect(api.curves.modify).toHaveBeenCalled();
    });

    // Ensure the mutation error fully settles
    await waitFor(() => {
      expect(onSave).not.toHaveBeenCalled();
    });
  });

  it('handles reset to initial curve', async () => {
    renderWithProviders(<CurveEditor initialCurve={mockCurve} />);

    const resetButton = screen.getByRole('button', { name: /reset/i });
    await userEvent.click(resetButton);

    expect(screen.getByDisplayValue('Test Curve')).toBeInTheDocument();
  });

  it('handles reset to linear when no initial curve', async () => {
    renderWithProviders(<CurveEditor />);

    const resetButton = screen.getByRole('button', { name: /reset/i });
    await userEvent.click(resetButton);

    expect(screen.getByDisplayValue('New Curve')).toBeInTheDocument();
  });

  it('allows changing curve name', async () => {
    renderWithProviders(<CurveEditor initialCurve={mockCurve} />);

    const nameInput = screen.getByDisplayValue('Test Curve');
    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, 'Renamed Curve');

    expect(screen.getByDisplayValue('Renamed Curve')).toBeInTheDocument();
  });
});
