/**
 * Dashboard Component Tests
 *
 * Comprehensive test suite covering:
 * - Loading, error, and success states
 * - Statistics card rendering from API data
 * - Recent calibrations table
 * - Quick action navigation
 * - Refresh functionality
 * - Empty state handling
 * - Accessibility
 */

import { beforeEach, describe, expect, it, type Mock, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Dashboard } from './Dashboard';

// Mock the API client
vi.mock('@/api/client', () => ({
    api: {
        statistics: {
            get: vi.fn(),
        },
        calibrations: {
            list: vi.fn(),
        },
    },
}));

// Mock the logger
vi.mock('@/lib/logger', () => ({
    logger: {
        debug: vi.fn(),
        error: vi.fn(),
        warn: vi.fn(),
        info: vi.fn(),
    },
}));

// Mock react-router-dom's useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return {
        ...actual,
        useNavigate: () => mockNavigate,
    };
});

import { api } from '@/api/client';

// Test fixtures
const mockStatistics = {
    total_records: 15,
    paper_types: ['Hahnemühle Platinum', 'Arches Platine', 'Bergger COT320'],
    chemistry_types: ['platinum_palladium', 'pure_palladium'],
    date_range: ['2025-06-15T10:30:00', '2026-02-01T14:20:00'] as [string, string],
    exposure_range: [120, 480] as [number, number],
};

const mockCalibrations = {
    count: 3,
    records: [
        {
            id: 'cal-001',
            paper_type: 'Hahnemühle Platinum',
            exposure_time: 240,
            metal_ratio: 0.7,
            timestamp: '2026-02-01T14:20:00',
            dmax: 1.82,
        },
        {
            id: 'cal-002',
            paper_type: 'Arches Platine',
            exposure_time: 180,
            metal_ratio: 0.5,
            timestamp: '2026-01-28T09:15:00',
            dmax: 1.65,
        },
        {
            id: 'cal-003',
            paper_type: 'Bergger COT320',
            exposure_time: 360,
            metal_ratio: 1.0,
            timestamp: '2026-01-20T11:00:00',
            dmax: 1.95,
        },
    ],
};

const emptyStatistics = {
    total_records: 0,
    paper_types: [],
    chemistry_types: [],
};

const emptyCalibrations = { count: 0, records: [] };

/** Helper to render Dashboard inside a router context */
function renderDashboard(props: Partial<React.ComponentProps<typeof Dashboard>> = {}) {
    return render(
        <MemoryRouter>
            <Dashboard {...props} />
        </MemoryRouter>
    );
}

describe('Dashboard', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigate.mockClear();
    });

    describe('Loading State', () => {
        it('shows loading spinner on initial load', () => {
            // Never resolve to keep loading state
            (api.statistics.get as Mock).mockReturnValue(new Promise(() => { }));
            (api.calibrations.list as Mock).mockReturnValue(new Promise(() => { }));

            renderDashboard();

            expect(screen.getByTestId('dashboard-loading')).toBeInTheDocument();
            expect(screen.getByText('Loading dashboard data…')).toBeInTheDocument();
        });

        it('has accessible loading role', () => {
            (api.statistics.get as Mock).mockReturnValue(new Promise(() => { }));
            (api.calibrations.list as Mock).mockReturnValue(new Promise(() => { }));

            renderDashboard();

            expect(screen.getByRole('status')).toBeInTheDocument();
        });
    });

    describe('Error State', () => {
        it('shows error message when API fails', async () => {
            (api.statistics.get as Mock).mockRejectedValue(
                new Error('Network error')
            );
            (api.calibrations.list as Mock).mockRejectedValue(
                new Error('Network error')
            );

            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('dashboard-error')).toBeInTheDocument();
            });
            expect(screen.getByText('Network error')).toBeInTheDocument();
        });

        it('shows retry button on error', async () => {
            (api.statistics.get as Mock).mockRejectedValue(
                new Error('Server down')
            );
            (api.calibrations.list as Mock).mockRejectedValue(
                new Error('Server down')
            );

            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('dashboard-retry')).toBeInTheDocument();
            });
        });

        it('retries on retry button click', async () => {
            (api.statistics.get as Mock)
                .mockRejectedValueOnce(new Error('Fail once'))
                .mockResolvedValueOnce(mockStatistics);
            (api.calibrations.list as Mock)
                .mockRejectedValueOnce(new Error('Fail once'))
                .mockResolvedValueOnce(mockCalibrations);

            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('dashboard-retry')).toBeInTheDocument();
            });

            fireEvent.click(screen.getByTestId('dashboard-retry'));

            await waitFor(() => {
                expect(screen.getByTestId('dashboard')).toBeInTheDocument();
            });
        });

        it('has accessible error alert role', async () => {
            (api.statistics.get as Mock).mockRejectedValue(new Error('Oops'));
            (api.calibrations.list as Mock).mockRejectedValue(new Error('Oops'));

            renderDashboard();

            await waitFor(() => {
                expect(screen.getByRole('alert')).toBeInTheDocument();
            });
        });
    });

    describe('Success State — Statistics Cards', () => {
        beforeEach(() => {
            (api.statistics.get as Mock).mockResolvedValue(mockStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(mockCalibrations);
        });

        it('renders the dashboard container', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('dashboard')).toBeInTheDocument();
            });
        });

        it('shows total calibrations card', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('stat-total-calibrations')).toBeInTheDocument();
            });
            expect(screen.getByText('15')).toBeInTheDocument();
        });

        it('shows paper types card with count', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('stat-paper-types')).toBeInTheDocument();
            });
            expect(screen.getByText('3')).toBeInTheDocument();
        });

        it('shows chemistry types card with count', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('stat-chemistry-types')).toBeInTheDocument();
            });
            expect(screen.getByText('2')).toBeInTheDocument();
        });

        it('shows exposure range card', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('stat-exposure-range')).toBeInTheDocument();
            });
            expect(screen.getByText('120–480s')).toBeInTheDocument();
        });
    });

    describe('Success State — Recent Calibrations Table', () => {
        beforeEach(() => {
            (api.statistics.get as Mock).mockResolvedValue(mockStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(mockCalibrations);
        });

        it('renders the calibrations table', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('calibrations-table')).toBeInTheDocument();
            });
        });

        it('renders all calibration rows', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('calibration-row-cal-001')).toBeInTheDocument();
                expect(screen.getByTestId('calibration-row-cal-002')).toBeInTheDocument();
                expect(screen.getByTestId('calibration-row-cal-003')).toBeInTheDocument();
            });
        });

        it('shows paper type in each row', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByText('Hahnemühle Platinum')).toBeInTheDocument();
                expect(screen.getByText('Arches Platine')).toBeInTheDocument();
                expect(screen.getByText('Bergger COT320')).toBeInTheDocument();
            });
        });

        it('shows exposure time formatted with unit', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByText('240s')).toBeInTheDocument();
                expect(screen.getByText('180s')).toBeInTheDocument();
                expect(screen.getByText('360s')).toBeInTheDocument();
            });
        });

        it('shows dmax formatted to 2 decimal places', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByText('1.82')).toBeInTheDocument();
                expect(screen.getByText('1.65')).toBeInTheDocument();
                expect(screen.getByText('1.95')).toBeInTheDocument();
            });
        });

        it('navigates to calibration on row click', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('calibration-row-cal-001')).toBeInTheDocument();
            });

            fireEvent.click(screen.getByTestId('calibration-row-cal-001'));
            expect(mockNavigate).toHaveBeenCalledWith('/calibration');
        });
    });

    describe('Empty State', () => {
        beforeEach(() => {
            (api.statistics.get as Mock).mockResolvedValue(emptyStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(emptyCalibrations);
        });

        it('shows empty state message when no calibrations', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('no-calibrations')).toBeInTheDocument();
            });
            expect(screen.getByText(/No calibrations recorded/)).toBeInTheDocument();
        });

        it('shows "Start Calibration" CTA button', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('start-first-calibration')).toBeInTheDocument();
            });
        });

        it('navigates to calibration when CTA clicked', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('start-first-calibration')).toBeInTheDocument();
            });

            fireEvent.click(screen.getByTestId('start-first-calibration'));
            expect(mockNavigate).toHaveBeenCalledWith('/calibration');
        });

        it('shows zero in total calibrations card', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('stat-total-calibrations')).toBeInTheDocument();
            });
            // Multiple cards show '0'; scope to the specific card
            const totalCard = screen.getByTestId('stat-total-calibrations');
            expect(totalCard).toHaveTextContent('0');
        });
    });

    describe('Quick Actions', () => {
        beforeEach(() => {
            (api.statistics.get as Mock).mockResolvedValue(mockStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(mockCalibrations);
        });

        it('renders all 4 quick action cards', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('action-new-calibration')).toBeInTheDocument();
                expect(screen.getByTestId('action-edit-curves')).toBeInTheDocument();
                expect(screen.getByTestId('action-upload-scan')).toBeInTheDocument();
                expect(screen.getByTestId('action-ai-assistant')).toBeInTheDocument();
            });
        });

        it('navigates to /calibration on "New Calibration" click', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('action-new-calibration')).toBeInTheDocument();
            });

            fireEvent.click(screen.getByTestId('action-new-calibration'));
            expect(mockNavigate).toHaveBeenCalledWith('/calibration');
        });

        it('navigates to /curves on "Edit Curves" click', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('action-edit-curves')).toBeInTheDocument();
            });

            fireEvent.click(screen.getByTestId('action-edit-curves'));
            expect(mockNavigate).toHaveBeenCalledWith('/curves');
        });

        it('navigates to /assistant on "AI Assistant" click', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('action-ai-assistant')).toBeInTheDocument();
            });

            fireEvent.click(screen.getByTestId('action-ai-assistant'));
            expect(mockNavigate).toHaveBeenCalledWith('/assistant');
        });
    });

    describe('Refresh', () => {
        it('shows refresh button', async () => {
            (api.statistics.get as Mock).mockResolvedValue(mockStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(mockCalibrations);

            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('dashboard-refresh')).toBeInTheDocument();
            });
        });

        it('calls API again on refresh click', async () => {
            (api.statistics.get as Mock).mockResolvedValue(mockStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(mockCalibrations);

            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('dashboard')).toBeInTheDocument();
            });

            // Reset call counts after initial load
            (api.statistics.get as Mock).mockClear();
            (api.calibrations.list as Mock).mockClear();
            (api.statistics.get as Mock).mockResolvedValue(mockStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(mockCalibrations);

            fireEvent.click(screen.getByTestId('dashboard-refresh'));

            await waitFor(() => {
                expect(api.statistics.get).toHaveBeenCalledTimes(1);
                expect(api.calibrations.list).toHaveBeenCalledTimes(1);
            });
        });

        it('refresh button has accessible label', async () => {
            (api.statistics.get as Mock).mockResolvedValue(mockStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(mockCalibrations);

            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('dashboard-refresh')).toBeInTheDocument();
            });

            expect(screen.getByTestId('dashboard-refresh')).toHaveAttribute(
                'aria-label',
                'Refresh dashboard data'
            );
        });
    });

    describe('Accessibility', () => {
        beforeEach(() => {
            (api.statistics.get as Mock).mockResolvedValue(mockStatistics);
            (api.calibrations.list as Mock).mockResolvedValue(mockCalibrations);
        });

        it('has labeled sections', async () => {
            renderDashboard();

            await waitFor(() => {
                expect(screen.getByTestId('dashboard')).toBeInTheDocument();
            });

            // Check that sections have aria-labels
            const sections = screen.getAllByRole('region');
            const sectionLabels = sections.map((s) => s.getAttribute('aria-label'));
            expect(sectionLabels).toContain('Statistics');
            expect(sectionLabels).toContain('Recent calibrations');
            expect(sectionLabels).toContain('Quick actions');
        });
    });
});
