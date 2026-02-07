import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CurvesPage } from './CurvesPage';

vi.mock('@/components/curves/CurveEditor', () => ({
    CurveEditor: ({ className }: any) => <div data-testid="curve-editor" className={className}>Editor</div>,
}));

describe('CurvesPage', () => {
    it('renders page heading', () => {
        render(<CurvesPage />);

        expect(screen.getByText('Curve Editor')).toBeInTheDocument();
        expect(screen.getByText(/view, edit, and generate/i)).toBeInTheDocument();
    });

    it('renders CurveEditor component', () => {
        render(<CurvesPage />);

        expect(screen.getByTestId('curve-editor')).toBeInTheDocument();
    });
});
