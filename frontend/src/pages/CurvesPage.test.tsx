import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { CurvesPage } from './CurvesPage';

expect.extend(toHaveNoViolations);

vi.mock('@/components/curves/CurveEditor', () => ({
  CurveEditor: ({ className }: any) => (
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    <div data-testid="curve-editor" className={className}>
      Editor
    </div>
  ),
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

  describe('Accessibility', () => {
    it('has no accessibility violations', async () => {
      const { container } = render(<CurvesPage />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
});
