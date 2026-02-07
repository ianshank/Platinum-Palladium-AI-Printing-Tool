import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, render } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { Step3Configure } from './Step3Configure';

// Setup theme for styled-components
const theme = {
    spacing: { 1: '0.25rem', 2: '0.5rem', 3: '0.75rem', 4: '1rem', 8: '2rem' },
    typography: {
        fontSize: { sm: '0.875rem', base: '1rem', lg: '1.125rem' },
        fontWeight: { medium: 500, semibold: 600 },
    },
    colors: {
        text: { primary: '#111', secondary: '#666', inverse: '#fff' },
        background: { tertiary: '#f3f4f6' },
        border: { default: '#e5e7eb' },
        accent: { primary: '#3b82f6' },
    },
    radii: { md: '0.375rem' },
    transitions: { fast: '0.15s' },
};

// Create mock functions
const mockNextStep = vi.fn();
const mockPreviousStep = vi.fn();
const mockSaveCalibration = vi.fn();
const mockUpdateMetadata = vi.fn();

// Mock useStore
vi.mock('@/stores', () => ({
    useStore: (selector: (state: any) => any) => {
        const state = {
            calibration: {
                current: {
                    name: 'Test Calibration',
                    notes: 'Some notes',
                    metadata: {
                        linearizationMode: 'linear',
                        targetResponse: 'qtr',
                        curveStrategy: 'monotonic',
                    },
                },
                nextStep: mockNextStep,
                previousStep: mockPreviousStep,
                saveCalibration: mockSaveCalibration,
                updateMetadata: mockUpdateMetadata,
            },
        };
        return selector(state);
    },
}));

// Mock tablet config
vi.mock('@/config/tablet.config', () => ({
    tabletConfig: {
        defaults: { tabletId: 'stouffer_21', exportFormat: 'qtr' },
        linearizationMethods: [
            { id: 'linear', label: 'Linear', description: 'Standard linear interpolation' },
            { id: 'cubic', label: 'Cubic Spline', description: 'Smooth cubic spline interpolation' },
            { id: 'monotonic', label: 'Monotonic', description: 'Preserves monotonicity (recommended)' },
        ],
        targetResponses: [
            { id: 'linear', label: 'Linear', description: 'Linear response (L*)' },
            { id: 's_curve', label: 'S-Curve', description: 'Contrast boosting S-curve' },
            { id: 'gamma_22', label: 'Gamma 2.2', description: 'Standard display gamma' },
        ],
    },
}));

function renderWithTheme(ui: React.ReactElement) {
    return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

describe('Step3Configure', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders section titles', () => {
        renderWithTheme(<Step3Configure />);

        expect(screen.getByText('Curve Name & Notes')).toBeInTheDocument();
        expect(screen.getByText('Linearization Method')).toBeInTheDocument();
        expect(screen.getByText('Target Response')).toBeInTheDocument();
    });

    it('renders calibration name input with current value', () => {
        renderWithTheme(<Step3Configure />);

        const nameInput = screen.getByPlaceholderText(/Arches Platine/i);
        expect(nameInput).toBeInTheDocument();
        expect(nameInput).toHaveValue('Test Calibration');
    });

    it('renders notes textarea', () => {
        renderWithTheme(<Step3Configure />);

        const notes = screen.getByPlaceholderText(/exposure time/i);
        expect(notes).toBeInTheDocument();
        expect(notes).toHaveValue('Some notes');
    });

    it('renders linearization method radio options', () => {
        renderWithTheme(<Step3Configure />);

        // 'Linear' appears in both Linearization Method and Target Response sections
        expect(screen.getAllByText('Linear')).toHaveLength(2);
        expect(screen.getByText('Cubic Spline')).toBeInTheDocument();
        expect(screen.getByText('Monotonic')).toBeInTheDocument();
    });

    it('renders target response options', () => {
        renderWithTheme(<Step3Configure />);

        expect(screen.getByText('S-Curve')).toBeInTheDocument();
        expect(screen.getByText('Gamma 2.2')).toBeInTheDocument();
    });

    it('renders Back and Generate Curve buttons', () => {
        renderWithTheme(<Step3Configure />);

        expect(screen.getByText('Back')).toBeInTheDocument();
        expect(screen.getByText('Generate Curve')).toBeInTheDocument();
    });

    it('calls nextStep on Generate Curve click', () => {
        renderWithTheme(<Step3Configure />);

        fireEvent.click(screen.getByText('Generate Curve'));
        expect(mockNextStep).toHaveBeenCalledTimes(1);
    });

    it('calls previousStep on Back click', () => {
        renderWithTheme(<Step3Configure />);

        fireEvent.click(screen.getByText('Back'));
        expect(mockPreviousStep).toHaveBeenCalledTimes(1);
    });

    it('calls saveCalibration on name change', () => {
        renderWithTheme(<Step3Configure />);

        const nameInput = screen.getByPlaceholderText(/Arches Platine/i);
        fireEvent.change(nameInput, { target: { value: 'Updated Name' } });

        expect(mockSaveCalibration).toHaveBeenCalledWith('Updated Name', 'Some notes');
    });

    it('calls saveCalibration on notes change', () => {
        renderWithTheme(<Step3Configure />);

        const notes = screen.getByPlaceholderText(/exposure time/i);
        fireEvent.change(notes, { target: { value: 'New notes' } });

        expect(mockSaveCalibration).toHaveBeenCalledWith('Test Calibration', 'New notes');
    });
});
