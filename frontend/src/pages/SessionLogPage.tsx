/**
 * SessionLogPage — Route page wrapping the SessionLog component.
 */

import { type FC } from 'react';
import { SessionLog } from '@/components/session/SessionLog';

export const SessionLogPage: FC = () => (
  <div className="container mx-auto px-4 py-6 sm:px-6 lg:px-8" data-testid="session-page">
    <div className="mb-6">
      <h1 className="text-3xl font-bold tracking-tight">Session Log</h1>
      <p className="text-muted-foreground">
        Print history, statistics, and records
      </p>
    </div>
    <SessionLog />
  </div>
);

SessionLogPage.displayName = 'SessionLogPage';
