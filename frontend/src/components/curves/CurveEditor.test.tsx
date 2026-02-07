import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { userEvent, renderWithProviders } from '@/test-utils';
import { CurveEditor } from './CurveEditor';
import { api } from '@/api/client';
import { CurveData } from '@/types/models';

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
    observe() { }
    unobserve() { }
    disconnect() { }
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

    it('handles apply adjustment', async () => {
        // Mock modify response
        vi.spyOn(api.curves, 'modify').mockResolvedValue({
            success: true,
            curve_id: '123',
            name: 'Test Curve',
            adjustment_applied: 'contrast',
            input_values: [0, 128, 255],
            output_values: [10, 130, 245], // changed
        } as any);

        renderWithProviders(<CurveEditor initialCurve={mockCurve} />);

        // Change slider (difficult to test slider drag with userEvent, but we can try basic interaction or direct prop change if we could reach it)
        // For now, let's just click 'Apply Adjustment' which uses default amount 0 or initial state.
        // To change slider value, we might need to access the input if it's a native range, but it's Radix slider.
        // Radix slider usually exposes role="slider".

        const applyButton = screen.getByRole('button', { name: /apply adjustment/i });
        await userEvent.click(applyButton);

        expect(api.curves.modify).toHaveBeenCalled();
    });

    it('handles save', async () => {
        const onSave = vi.fn();
        renderWithProviders(<CurveEditor initialCurve={mockCurve} onSave={onSave} />);

        const saveButton = screen.getByRole('button', { name: /save/i });
        await userEvent.click(saveButton);

        expect(onSave).toHaveBeenCalledWith(expect.objectContaining({
            name: 'Test Curve',
            input_values: mockCurve.input_values,
            output_values: mockCurve.output_values,
        }));
    });
});
