/**
 * ChemistryPage Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChemistryPage } from './ChemistryPage';

// Mock the ChemistryCalculator to test page isolation
vi.mock('@/components/chemistry/ChemistryCalculator', () => ({
    ChemistryCalculator: () => (
        <div data-testid="mock-chemistry-calculator">ChemistryCalculator</div>
    ),
}));

describe('ChemistryPage', () => {
    it('renders the page container', () => {
        render(<ChemistryPage />);
        expect(screen.getByTestId('chemistry-page')).toBeInTheDocument();
    });

    it('renders page heading', () => {
        render(<ChemistryPage />);
        expect(screen.getByText('Chemistry Calculator')).toBeInTheDocument();
    });

    it('renders page description', () => {
        render(<ChemistryPage />);
        expect(
            screen.getByText(/Calculate coating recipes/)
        ).toBeInTheDocument();
    });

    it('renders the ChemistryCalculator component', () => {
        render(<ChemistryPage />);
        expect(
            screen.getByTestId('mock-chemistry-calculator')
        ).toBeInTheDocument();
    });
});
