import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { renderWithProviders, userEvent } from '@/test-utils';
import { CalibrationWizard } from './CalibrationWizard';
import { api } from '@/api/client';

// Mock API
vi.mock('@/api/client', () => ({
    api: {
        curves: {
            generate: vi.fn(),
        },
    },
}));

// Mock child components
vi.mock('./ScanUpload', () => ({
    ScanUpload: ({ onUploadComplete }: any) => (
        <div data-testid="scan-upload-mock">
            <button onClick={() => onUploadComplete({
                extraction_id: 'mock-extraction-id',
                densities: [0.1, 0.5, 0.9],
                success: true
            })}>
                Simulate Upload
            </button>
        </div>
    ),
}));

vi.mock('@/components/curves/CurveEditor', () => ({
    CurveEditor: ({ initialCurve }: any) => (
        <div data-testid="curve-editor-mock">
            Curve Editor: {initialCurve.name}
        </div>
    ),
}));

describe('CalibrationWizard', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('navigates through the wizard', async () => {
        vi.spyOn(api.curves, 'generate').mockResolvedValue({
            success: true,
            curve_id: 'new-curve-id',
            name: 'Generated Curve',
            num_points: 256,
            input_values: [],
            output_values: [],
        } as any);

        renderWithProviders(<CalibrationWizard />);

        // Step 1: Setup
        expect(screen.getByText('Process Setup')).toBeInTheDocument();

        // Fill required fields
        const paperInput = screen.getByPlaceholderText(/Arches Platine/i);
        await userEvent.type(paperInput, 'Test Paper');

        // Click Next
        await userEvent.click(screen.getByRole('button', { name: /next/i }));

        // Step 2: Print
        expect(screen.getByRole('heading', { name: 'Print Target' })).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /i have printed/i })); // "I have printed the target" or "Next"

        // Step 3: Scan
        expect(screen.getByRole('heading', { name: 'Scan Target' })).toBeInTheDocument();
        expect(screen.getByTestId('scan-upload-mock')).toBeInTheDocument();

        // Simulate Scan Completion
        await userEvent.click(screen.getByText('Simulate Upload'));

        // Step 4: Analyze
        expect(screen.getByText('Analysis')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /generate curve/i }));

        // Step 5: Generate (might be skipped if generate call is fast?)
        // Actually handleGenerateValues sets isGenerating then calls API then Next.
        // So we expect to land on Finish if successful.

        await waitFor(() => {
            expect(api.curves.generate).toHaveBeenCalled();
        });

        // We should be on Finish step now?
        // Wait, handleGenerateValues calls handleNext() on success.
        // Step 4 (Analyze) --(Generate Curve Click)--> API Call --> Next --> Step 5 (Generate?? No).
        // Let's check logic:
        // STEPS indices: 0:setup, 1:print, 2:scan, 3:analyze, 4:generate, 5:finish.
        // handleGenerateValues calls handleNext(). If current is 3 (Analyze), next is 4 (Generate).
        // Wait, renderStepContent for case 3 (Analyze) has button "Generate Curve" calling handleGenerateValues.
        // handleGenerateValues calls setCurveResult then handleNext().
        // Next step is 4 (Generate).
        // In case 4 (Generate), it shows "Curve Generated!" and Next button.

        expect(screen.getByText('Curve Generated!')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /next/i }));

        // Step 5: Finish (Index 5)
        expect(screen.getByText('Complete')).toBeInTheDocument();
        expect(screen.getByTestId('curve-editor-mock')).toBeInTheDocument();
    });

    it('preserves state when navigating back', async () => {
        renderWithProviders(<CalibrationWizard />);

        // Fill setup
        const paperInput = screen.getByPlaceholderText(/Arches Platine/i);
        await userEvent.type(paperInput, 'Special Paper');

        await userEvent.click(screen.getByRole('button', { name: /next/i }));

        // Go back
        await userEvent.click(screen.getByRole('button', { name: /back/i }));

        // Check if value is still there
        expect(screen.getByDisplayValue('Special Paper')).toBeInTheDocument();
    });
});
