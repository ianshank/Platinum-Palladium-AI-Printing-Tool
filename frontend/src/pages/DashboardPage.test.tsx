/**
 * DashboardPage Tests
 *
 * Covers page-level rendering and routing integration.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { axe, toHaveNoViolations } from 'jest-axe';
import { DashboardPage } from './DashboardPage';

expect.extend(toHaveNoViolations);

// Mock the Dashboard component to test isolation at page level
vi.mock('@/components/dashboard/Dashboard', () => ({
  Dashboard: ({ autoRefresh }: { autoRefresh?: boolean }) => (
    <div data-testid="mock-dashboard" data-auto-refresh={String(autoRefresh)}>
      Dashboard Component
    </div>
  ),
}));

describe('DashboardPage', () => {
  it('renders the page container', () => {
    render(
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>
    );

    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
  });

  it('renders the Dashboard component', () => {
    render(
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>
    );

    expect(screen.getByTestId('mock-dashboard')).toBeInTheDocument();
  });

  it('passes autoRefresh=false to Dashboard', () => {
    render(
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>
    );

    expect(screen.getByTestId('mock-dashboard')).toHaveAttribute(
      'data-auto-refresh',
      'false'
    );
  });

  it('has container styling', () => {
    render(
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>
    );

    expect(screen.getByTestId('dashboard-page')).toHaveClass('container');
  });

  describe('Accessibility', () => {
    it('has no accessibility violations', async () => {
      const { container } = render(
        <MemoryRouter>
          <DashboardPage />
        </MemoryRouter>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
});
