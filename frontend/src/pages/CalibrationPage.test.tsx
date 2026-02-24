import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CalibrationPage } from './CalibrationPage';

vi.mock('@/components/calibration/CalibrationWizard', () => ({
  CalibrationWizard: () => <div data-testid="calibration-wizard">Wizard</div>,
}));

describe('CalibrationPage', () => {
  it('renders page instructions', () => {
    render(<CalibrationPage />);

    expect(screen.getByText(/follow the steps/i)).toBeInTheDocument();
  });

  it('renders CalibrationWizard component', () => {
    render(<CalibrationPage />);

    expect(screen.getByTestId('calibration-wizard')).toBeInTheDocument();
  });
});
