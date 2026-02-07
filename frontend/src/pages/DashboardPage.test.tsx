/**
 * DashboardPage Tests
 *
 * Covers page-level rendering and routing integration.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { DashboardPage } from './DashboardPage';

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
});
