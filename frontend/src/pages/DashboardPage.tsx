/**
 * DashboardPage — Route page wrapping the Dashboard component.
 *
 * Keeps the page layer thin; all logic lives in Dashboard and its hook.
 */

import { type FC } from 'react';
import { Dashboard } from '@/components/dashboard/Dashboard';

export const DashboardPage: FC = () => (
  <div className="container mx-auto px-4 py-6 sm:px-6 lg:px-8" data-testid="dashboard-page">
    <Dashboard autoRefresh={false} />
  </div>
);

DashboardPage.displayName = 'DashboardPage';
