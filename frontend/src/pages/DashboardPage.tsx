/**
 * DashboardPage — Route page wrapping the Dashboard component.
 *
 * Keeps the page layer thin; all logic lives in Dashboard and its hook.
 */

import { type FC } from 'react';
import { Dashboard } from '@/components/dashboard/Dashboard';

export const DashboardPage: FC = () => (
  <div className="container mx-auto py-6" data-testid="dashboard-page">
    <Dashboard autoRefresh={false} />
  </div>
);

DashboardPage.displayName = 'DashboardPage';
