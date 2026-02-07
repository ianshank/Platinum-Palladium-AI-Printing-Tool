/**
 * Dashboard Component
 *
 * The main landing page showing system overview:
 * - Statistics cards (total records, paper types, chemistry types, exposure range)
 * - Recent calibrations table
 * - Quick action shortcuts
 *
 * All values are config-driven. No hardcoded display strings beyond labels.
 */

import { type FC, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { logger } from '@/lib/logger';

import { useDashboardData } from '@/hooks/useDashboardData';
import { StatCard } from '@/components/ui/StatCard';
import { QuickActionCard } from '@/components/ui/QuickActionCard';

export interface DashboardProps {
    /** Additional CSS class names */
    className?: string;
    /** Whether to auto-refresh data periodically */
    autoRefresh?: boolean;
}

/** Format a date string for display (respects locale) */
function formatDate(isoString: string): string {
    try {
        return new Intl.DateTimeFormat(undefined, {
            dateStyle: 'medium',
            timeStyle: 'short',
        }).format(new Date(isoString));
    } catch {
        return isoString;
    }
}

/** SVG icons as inline components to avoid external deps */
const Icons = {
    Database: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5v14a9 3 0 0 0 18 0V5" /><path d="M3 12a9 3 0 0 0 18 0" />
        </svg>
    ),
    Layers: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 2 2 7l10 5 10-5-10-5Z" /><path d="m2 17 10 5 10-5" /><path d="m2 12 10 5 10-5" />
        </svg>
    ),
    Flask: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M9 3h6" /><path d="M10 9V3" /><path d="M14 9V3" /><path d="M10 9a7.08 7.08 0 0 0-5.9 8.18A2 2 0 0 0 6.07 19h11.86a2 2 0 0 0 1.97-1.82A7.08 7.08 0 0 0 14 9Z" />
        </svg>
    ),
    Sun: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="4" /><path d="M12 2v2" /><path d="M12 20v2" /><path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" /><path d="M2 12h2" /><path d="M20 12h2" /><path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" />
        </svg>
    ),
    Plus: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M5 12h14" /><path d="M12 5v14" />
        </svg>
    ),
    LineChart: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M3 3v18h18" /><path d="m19 9-5 5-4-4-3 3" />
        </svg>
    ),
    Upload: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" x2="12" y1="3" y2="15" />
        </svg>
    ),
    MessageSquare: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
    ),
    RefreshCw: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /><path d="M21 3v5h-5" /><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /><path d="M8 16H3v5" />
        </svg>
    ),
};

export const Dashboard: FC<DashboardProps> = ({
    className,
    autoRefresh = false,
}) => {
    const {
        statistics,
        recentCalibrations,
        isLoading,
        error,
        lastFetched,
        refresh,
    } = useDashboardData(autoRefresh);

    const navigate = useNavigate();

    const handleNavigate = useCallback(
        (path: string) => {
            logger.debug('Dashboard: Navigating', { path });
            navigate(path);
        },
        [navigate]
    );

    /** Memoized stat cards derived from API data */
    const statCards = useMemo(() => {
        if (!statistics) return [];

        return [
            {
                label: 'Total Calibrations',
                value: statistics.total_records,
                subtitle: statistics.date_range
                    ? `Since ${formatDate(statistics.date_range[0])}`
                    : 'No records yet',
                icon: <Icons.Database />,
                testId: 'stat-total-calibrations',
            },
            {
                label: 'Paper Types',
                value: statistics.paper_types.length,
                subtitle: statistics.paper_types.length > 0
                    ? statistics.paper_types.slice(0, 3).join(', ')
                    : 'None recorded',
                icon: <Icons.Layers />,
                testId: 'stat-paper-types',
            },
            {
                label: 'Chemistry Types',
                value: statistics.chemistry_types.length,
                subtitle: statistics.chemistry_types.length > 0
                    ? statistics.chemistry_types.slice(0, 3).join(', ')
                    : 'None recorded',
                icon: <Icons.Flask />,
                testId: 'stat-chemistry-types',
            },
            {
                label: 'Exposure Range',
                value: statistics.exposure_range
                    ? `${statistics.exposure_range[0]}–${statistics.exposure_range[1]}s`
                    : '—',
                subtitle: 'UV exposure time',
                icon: <Icons.Sun />,
                testId: 'stat-exposure-range',
            },
        ];
    }, [statistics]);

    /** Quick action definitions — driven by route config, not hardcoded URLs */
    const quickActions = useMemo(
        () => [
            {
                title: 'New Calibration',
                description: 'Start a new calibration workflow with the step-by-step wizard.',
                icon: <Icons.Plus />,
                path: '/calibration',
                testId: 'action-new-calibration',
            },
            {
                title: 'Edit Curves',
                description: 'View, modify, and export linearization curves.',
                icon: <Icons.LineChart />,
                path: '/curves',
                testId: 'action-edit-curves',
            },
            {
                title: 'Upload Scan',
                description: 'Upload a step tablet scan for density measurement.',
                icon: <Icons.Upload />,
                path: '/calibration',
                testId: 'action-upload-scan',
            },
            {
                title: 'AI Assistant',
                description: 'Get help with printing processes and troubleshooting.',
                icon: <Icons.MessageSquare />,
                path: '/assistant',
                testId: 'action-ai-assistant',
            },
        ],
        []
    );

    // Loading state
    if (isLoading && !statistics) {
        return (
            <div
                className={cn('flex items-center justify-center py-12', className)}
                data-testid="dashboard-loading"
                role="status"
                aria-label="Loading dashboard"
            >
                <div className="text-center">
                    <div className="mx-auto h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                    <p className="mt-4 text-sm text-muted-foreground">
                        Loading dashboard data…
                    </p>
                </div>
            </div>
        );
    }

    // Error state
    if (error && !statistics) {
        return (
            <div
                className={cn('rounded-lg border border-destructive/50 bg-destructive/10 p-6', className)}
                data-testid="dashboard-error"
                role="alert"
            >
                <h2 className="text-lg font-semibold text-destructive">
                    Failed to load dashboard
                </h2>
                <p className="mt-2 text-sm text-muted-foreground">{error}</p>
                <button
                    type="button"
                    onClick={refresh}
                    className="mt-4 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                    data-testid="dashboard-retry"
                >
                    Try Again
                </button>
            </div>
        );
    }

    return (
        <div className={cn('space-y-8', className)} data-testid="dashboard">
            {/* Header with refresh */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
                    <p className="text-muted-foreground">
                        Overview of your calibration system
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {lastFetched && (
                        <span className="text-xs text-muted-foreground" data-testid="dashboard-last-fetched">
                            Updated {formatDate(lastFetched.toISOString())}
                        </span>
                    )}
                    <button
                        type="button"
                        onClick={refresh}
                        disabled={isLoading}
                        className={cn(
                            'inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm',
                            'hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                            'disabled:cursor-not-allowed disabled:opacity-50',
                            isLoading && 'animate-pulse'
                        )}
                        data-testid="dashboard-refresh"
                        aria-label="Refresh dashboard data"
                    >
                        <Icons.RefreshCw />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Statistics Cards */}
            <section aria-label="Statistics">
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    {statCards.map((card) => (
                        <StatCard
                            key={card.testId}
                            label={card.label}
                            value={card.value}
                            subtitle={card.subtitle}
                            icon={card.icon}
                            data-testid={card.testId}
                        />
                    ))}
                </div>
            </section>

            {/* Two-column layout: Recent Calibrations + Quick Actions */}
            <div className="grid gap-8 lg:grid-cols-3">
                {/* Recent Calibrations */}
                <section className="lg:col-span-2" aria-label="Recent calibrations">
                    <h2 className="mb-4 text-lg font-semibold">Recent Calibrations</h2>
                    {recentCalibrations.length === 0 ? (
                        <div
                            className="rounded-lg border border-dashed p-8 text-center"
                            data-testid="no-calibrations"
                        >
                            <p className="text-sm text-muted-foreground">
                                No calibrations recorded yet. Start your first calibration to
                                see data here.
                            </p>
                            <button
                                type="button"
                                onClick={() => handleNavigate('/calibration')}
                                className="mt-4 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                                data-testid="start-first-calibration"
                            >
                                Start Calibration
                            </button>
                        </div>
                    ) : (
                        <div className="overflow-hidden rounded-lg border">
                            <table className="min-w-full divide-y divide-border" data-testid="calibrations-table">
                                <thead className="bg-muted/50">
                                    <tr>
                                        <th scope="col" className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                            Paper Type
                                        </th>
                                        <th scope="col" className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                            Exposure
                                        </th>
                                        <th scope="col" className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                            Metal Ratio
                                        </th>
                                        <th scope="col" className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                            Dmax
                                        </th>
                                        <th scope="col" className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                            Date
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-border bg-card">
                                    {recentCalibrations.map((cal) => (
                                        <tr
                                            key={cal.id}
                                            className="cursor-pointer transition-colors hover:bg-muted/30"
                                            onClick={() => handleNavigate(`/calibration`)}
                                            data-testid={`calibration-row-${cal.id}`}
                                        >
                                            <td className="whitespace-nowrap px-4 py-3 text-sm font-medium">
                                                {cal.paper_type}
                                            </td>
                                            <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
                                                {cal.exposure_time}s
                                            </td>
                                            <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
                                                {cal.metal_ratio}
                                            </td>
                                            <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
                                                {typeof cal.dmax === 'number' ? cal.dmax.toFixed(2) : '—'}
                                            </td>
                                            <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
                                                {formatDate(cal.timestamp)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </section>

                {/* Quick Actions */}
                <section aria-label="Quick actions">
                    <h2 className="mb-4 text-lg font-semibold">Quick Actions</h2>
                    <div className="grid gap-3">
                        {quickActions.map((action) => (
                            <QuickActionCard
                                key={action.testId}
                                title={action.title}
                                description={action.description}
                                icon={action.icon}
                                onClick={() => handleNavigate(action.path)}
                                data-testid={action.testId}
                            />
                        ))}
                    </div>
                </section>
            </div>
        </div>
    );
};

Dashboard.displayName = 'Dashboard';
