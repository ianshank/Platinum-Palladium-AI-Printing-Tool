/**
 * SettingsPage — Route page wrapping the Settings component.
 */

import { type FC } from 'react';
import { Settings } from '@/components/settings/Settings';

export const SettingsPage: FC = () => (
    <div className="container mx-auto py-6" data-testid="settings-page">
        <div className="mb-6">
            <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
            <p className="text-muted-foreground">
                Application configuration and preferences
            </p>
        </div>
        <Settings />
    </div>
);

SettingsPage.displayName = 'SettingsPage';
