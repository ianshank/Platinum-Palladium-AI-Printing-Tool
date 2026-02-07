import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CalibrationPage } from './CalibrationPage';

vi.mock('@/components/calibration/CalibrationWizard', () => ({
    CalibrationWizard: () => <div data-testid="calibration-wizard">Wizard</div>,
}));

describe('CalibrationPage', () => {
    it('renders page heading', () => {
        render(<CalibrationPage />);

        expect(screen.getByText('Calibration Wizard')).toBeInTheDocument();
        expect(screen.getByText(/follow the steps/i)).toBeInTheDocument();
    });

    it('renders CalibrationWizard component', () => {
        render(<CalibrationPage />);

        expect(screen.getByTestId('calibration-wizard')).toBeInTheDocument();
    });
});
