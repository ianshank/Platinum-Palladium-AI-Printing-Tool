/**
 * SessionLog Component
 *
 * Displays a table of print session records with filtering and stats.
 * Reads from and writes to the session Zustand store slice.
 *
 * Features:
 * - Session statistics summary (total, success rate, avg exposure, most-used paper)
 * - Filterable record table (by result type)
 * - Empty state with call-to-action
 * - Delete record action
 * - Export records
 * - Clear all records
 */

import { type FC, useCallback, useMemo } from 'react';
import { cn } from '@/lib/utils';
import { logger } from '@/lib/logger';
import { useStore } from '@/stores';
import type { PrintRecord } from '@/stores/slices/sessionSlice';

export interface SessionLogProps {
    className?: string;
}

const RESULT_LABELS: Record<PrintRecord['result'], string> = {
    success: '✓ Success',
    partial: '◐ Partial',
    failure: '✗ Failure',
};

const RESULT_BADGE_STYLES: Record<PrintRecord['result'], string> = {
    success: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    partial: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
    failure: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
};

function formatDate(iso: string): string {
    try {
        return new Intl.DateTimeFormat(undefined, {
            dateStyle: 'medium',
            timeStyle: 'short',
        }).format(new Date(iso));
    } catch {
        return iso;
    }
}

export const SessionLog: FC<SessionLogProps> = ({ className }) => {
    const records = useStore((s) => s.session.records);
    const stats = useStore((s) => s.session.stats);
    const filterResult = useStore((s) => s.session.filterResult);

    const setResultFilter = useStore((s) => s.session.setResultFilter);
    const clearFilters = useStore((s) => s.session.clearFilters);
    const deleteRecord = useStore((s) => s.session.deleteRecord);
    const exportRecords = useStore((s) => s.session.exportRecords);
    const clearRecords = useStore((s) => s.session.clearRecords);

    const filteredRecords = useMemo(() => {
        if (filterResult === 'all') return records;
        return records.filter((r) => r.result === filterResult);
    }, [records, filterResult]);

    const handleExport = useCallback(() => {
        const data = exportRecords();
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json',
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `session-records-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        logger.info('SessionLog: Exported records', { count: data.length });
    }, [exportRecords]);

    const handleClearAll = useCallback(() => {
        logger.warn('SessionLog: Clearing all records');
        clearRecords();
    }, [clearRecords]);

    const hasRecords = records.length > 0;

    return (
        <div className={cn('space-y-6', className)} data-testid="session-log">
            {/* Stats Bar */}
            {hasRecords && (
                <div
                    className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
                    data-testid="session-stats"
                >
                    <StatItem label="Total Prints" value={String(stats.totalPrints)} testId="stat-total" />
                    <StatItem
                        label="Success Rate"
                        value={`${Math.round(stats.successRate * 100)}%`}
                        testId="stat-success"
                    />
                    <StatItem
                        label="Avg Exposure"
                        value={`${Math.round(stats.averageExposure)}s`}
                        testId="stat-exposure"
                    />
                    <StatItem
                        label="Most Used Paper"
                        value={stats.mostUsedPaper ?? '—'}
                        testId="stat-paper"
                    />
                </div>
            )}

            {/* Toolbar */}
            <div className="flex items-center justify-between">
                <div className="flex gap-2">
                    {(['all', 'success', 'partial', 'failure'] as const).map((f) => (
                        <button
                            key={f}
                            type="button"
                            onClick={() => (f === 'all' ? clearFilters() : setResultFilter(f))}
                            className={cn(
                                'rounded-md border px-3 py-1.5 text-xs transition-colors',
                                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                                filterResult === f
                                    ? 'border-primary bg-primary/10 font-medium text-primary'
                                    : 'hover:bg-accent'
                            )}
                            data-testid={`filter-${f}`}
                        >
                            {f === 'all' ? 'All' : RESULT_LABELS[f]}
                        </button>
                    ))}
                </div>
                <div className="flex gap-2">
                    {hasRecords && (
                        <>
                            <button
                                type="button"
                                onClick={handleExport}
                                className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                                data-testid="export-btn"
                            >
                                Export JSON
                            </button>
                            <button
                                type="button"
                                onClick={handleClearAll}
                                className="rounded-md border px-3 py-1.5 text-xs text-destructive hover:bg-destructive/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                                data-testid="clear-all-btn"
                            >
                                Clear All
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* Empty State */}
            {!hasRecords && (
                <div
                    className="flex flex-col items-center justify-center rounded-lg border border-dashed p-12 text-center"
                    data-testid="empty-session"
                >
                    <p className="text-sm font-medium">No print records yet</p>
                    <p className="text-xs text-muted-foreground mt-1">
                        Records are created when you complete a calibration print.
                    </p>
                </div>
            )}

            {/* Records Table */}
            {hasRecords && (
                <div className="overflow-x-auto rounded-lg border" data-testid="records-table">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b bg-muted/50">
                                <th className="px-4 py-3 text-left font-medium">Date</th>
                                <th className="px-4 py-3 text-left font-medium">Paper</th>
                                <th className="px-4 py-3 text-left font-medium">Exposure</th>
                                <th className="px-4 py-3 text-left font-medium">Ratio</th>
                                <th className="px-4 py-3 text-left font-medium">Result</th>
                                <th className="px-4 py-3 text-right font-medium">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredRecords.map((record) => (
                                <tr
                                    key={record.id}
                                    className="border-b last:border-b-0 hover:bg-muted/30 transition-colors"
                                    data-testid={`record-${record.id}`}
                                >
                                    <td className="px-4 py-3 whitespace-nowrap">
                                        {formatDate(record.timestamp)}
                                    </td>
                                    <td className="px-4 py-3">{record.paperType}</td>
                                    <td className="px-4 py-3">{record.exposure.timeSeconds}s</td>
                                    <td className="px-4 py-3">
                                        {Math.round(record.chemistry.metalRatio * 100)}% Pt
                                    </td>
                                    <td className="px-4 py-3">
                                        <span
                                            className={cn(
                                                'inline-block rounded-full px-2 py-0.5 text-xs font-medium',
                                                RESULT_BADGE_STYLES[record.result]
                                            )}
                                        >
                                            {RESULT_LABELS[record.result]}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-right">
                                        <button
                                            type="button"
                                            onClick={() => deleteRecord(record.id)}
                                            className="text-xs text-muted-foreground hover:text-destructive transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-sm px-1"
                                            data-testid={`delete-${record.id}`}
                                            aria-label={`Delete record ${record.id}`}
                                        >
                                            Delete
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

SessionLog.displayName = 'SessionLog';

/** Mini stat card for the bar */
const StatItem: FC<{ label: string; value: string; testId: string }> = ({
    label,
    value,
    testId,
}) => (
    <div
        className="rounded-lg border bg-card p-4"
        data-testid={testId}
    >
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="mt-1 text-xl font-bold">{value}</p>
    </div>
);
StatItem.displayName = 'StatItem';
