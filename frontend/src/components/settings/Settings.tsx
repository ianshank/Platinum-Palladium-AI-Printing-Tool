/**
 * Settings Component
 *
 * Application configuration panel.
 * Reads from and writes to the UI Zustand store slice.
 *
 * Features:
 * - Theme toggle (light/dark)
 * - Sidebar open/close toggle
 * - Reset UI state
 * - Displays current config values (read-only)
 */

import { type FC, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { logger } from '@/lib/logger';
import { useStore } from '@/stores';
import { config } from '@/config';

export interface SettingsProps {
    className?: string;
}

export const Settings: FC<SettingsProps> = ({ className }) => {
    const theme = useStore((s) => s.ui.theme);
    const sidebarOpen = useStore((s) => s.ui.sidebarOpen);

    const setTheme = useStore((s) => s.ui.setTheme);
    const setSidebarOpen = useStore((s) => s.ui.setSidebarOpen);
    const resetUI = useStore((s) => s.ui.resetUI);

    const handleResetUI = useCallback(() => {
        logger.info('Settings: resetUI');
        resetUI();
    }, [resetUI]);

    return (
        <div className={cn('space-y-8', className)} data-testid="settings">
            {/* Appearance */}
            <section>
                <h3 className="text-lg font-semibold mb-4">Appearance</h3>
                <div className="space-y-4">
                    {/* Theme */}
                    <div className="flex items-center justify-between rounded-lg border p-4">
                        <div>
                            <p className="text-sm font-medium">Theme</p>
                            <p className="text-xs text-muted-foreground">
                                Switch between dark and light mode
                            </p>
                        </div>
                        <div className="flex gap-2">
                            <button
                                type="button"
                                onClick={() => setTheme('light')}
                                className={cn(
                                    'rounded-md border px-3 py-1.5 text-xs transition-colors',
                                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                                    theme === 'light'
                                        ? 'border-primary bg-primary/10 font-medium text-primary'
                                        : 'hover:bg-accent'
                                )}
                                data-testid="theme-light-btn"
                                aria-pressed={theme === 'light'}
                            >
                                ☀️ Light
                            </button>
                            <button
                                type="button"
                                onClick={() => setTheme('dark')}
                                className={cn(
                                    'rounded-md border px-3 py-1.5 text-xs transition-colors',
                                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                                    theme === 'dark'
                                        ? 'border-primary bg-primary/10 font-medium text-primary'
                                        : 'hover:bg-accent'
                                )}
                                data-testid="theme-dark-btn"
                                aria-pressed={theme === 'dark'}
                            >
                                🌙 Dark
                            </button>
                        </div>
                    </div>

                    {/* Sidebar */}
                    <div className="flex items-center justify-between rounded-lg border p-4">
                        <div>
                            <p className="text-sm font-medium">Sidebar</p>
                            <p className="text-xs text-muted-foreground">
                                Toggle the navigation sidebar
                            </p>
                        </div>
                        <button
                            type="button"
                            role="switch"
                            aria-checked={sidebarOpen}
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            className={cn(
                                'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
                                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                                sidebarOpen ? 'bg-primary' : 'bg-muted'
                            )}
                            data-testid="sidebar-toggle"
                        >
                            <span
                                className={cn(
                                    'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                                    sidebarOpen ? 'translate-x-6' : 'translate-x-1'
                                )}
                            />
                        </button>
                    </div>
                </div>
            </section>

            {/* Application Info */}
            <section>
                <h3 className="text-lg font-semibold mb-4">Application</h3>
                <div className="rounded-lg border divide-y">
                    <InfoRow label="API URL" value={config.api.baseUrl} testId="info-api-url" />
                    <InfoRow
                        label="Auto-refresh Interval"
                        value={`${config.api.staleTime / 1000}s`}
                        testId="info-stale-time"
                    />
                    <InfoRow
                        label="Default Steps"
                        value={String(config.calibration.defaultSteps)}
                        testId="info-default-steps"
                    />
                    <InfoRow
                        label="DevTools"
                        value={config.features.devtools ? 'Enabled' : 'Disabled'}
                        testId="info-devtools"
                    />
                </div>
            </section>

            {/* Danger Zone */}
            <section>
                <h3 className="text-lg font-semibold mb-4 text-destructive">
                    Danger Zone
                </h3>
                <div className="rounded-lg border border-destructive/30 p-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium">Reset UI</p>
                            <p className="text-xs text-muted-foreground">
                                Reset tabs, toasts, and modals to defaults
                            </p>
                        </div>
                        <button
                            type="button"
                            onClick={handleResetUI}
                            className="rounded-md border border-destructive px-3 py-1.5 text-xs text-destructive hover:bg-destructive/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                            data-testid="reset-ui-btn"
                        >
                            Reset
                        </button>
                    </div>
                </div>
            </section>
        </div>
    );
};

Settings.displayName = 'Settings';

const InfoRow: FC<{ label: string; value: string; testId: string }> = ({
    label,
    value,
    testId,
}) => (
    <div className="flex items-center justify-between px-4 py-3" data-testid={testId}>
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className="text-sm font-mono">{value}</span>
    </div>
);
InfoRow.displayName = 'InfoRow';
