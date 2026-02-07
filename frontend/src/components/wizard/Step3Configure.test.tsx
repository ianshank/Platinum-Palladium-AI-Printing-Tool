import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { Step3Configure } from './Step3Configure';

// Setup theme for styled-components
const theme = {
    spacing: { 0: '0', 1: '0.25rem', 2: '0.5rem', 3: '0.75rem', 4: '1rem', 5: '1.25rem', 6: '1.5rem', 8: '2rem', 10: '2.5rem', 12: '3rem', 16: '4rem', 20: '5rem', 24: '6rem', 32: '8rem', 40: '10rem', 48: '12rem', 56: '14rem', 64: '16rem' },
    typography: {
        fontFamily: { sans: 'Inter, sans-serif', mono: 'monospace' },
        fontSize: { xs: '0.75rem', sm: '0.875rem', base: '1rem', lg: '1.125rem', xl: '1.25rem', '2xl': '1.5rem' },
        fontWeight: { regular: 400, medium: 500, semibold: 600, bold: 700 },
    },
    colors: {
        text: { primary: '#111', secondary: '#666', inverse: '#fff', error: '#ef4444' },
        background: { primary: '#fff', secondary: '#f9fafb', tertiary: '#f3f4f6', hover: '#e5e7eb' },
        border: { default: '#e5e7eb', focus: '#3b82f6' },
        accent: { primary: '#3b82f6', primaryHover: '#2563eb', secondary: '#6366f1' },
        semantic: { success: '#10b981', warning: '#f59e0b', error: '#ef4444', info: '#3b82f6' },
    },
    radii: { sm: '0.25rem', md: '0.375rem', lg: '0.5rem', full: '9999px' },
    shadows: { sm: '0 1px 2px rgba(0,0,0,0.05)', md: '0 4px 6px rgba(0,0,0,0.1)', lg: '0 10px 15px rgba(0,0,0,0.1)' },
    transitions: { fast: '0.15s', normal: '0.3s', slow: '0.5s' },
    breakpoints: { tablet: '768px', desktop: '1024px', wide: '1280px' },
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
