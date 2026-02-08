/**
 * useDashboardData hook
 *
 * Fetches statistics and recent calibrations from the API.
 * Encapsulates loading, error, and refresh logic for the Dashboard page.
 * No hardcoded values — poll interval and limits come from config.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api/client';
import { logger } from '@/lib/logger';
import { config } from '@/config';
import type { CalibrationSummary, StatisticsResponse } from '@/types/models';

/** Shape of the dashboard data bundle */
export interface DashboardData {
  statistics: StatisticsResponse | null;
  recentCalibrations: CalibrationSummary[];
  isLoading: boolean;
  error: string | null;
  lastFetched: Date | null;
}

/** Default empty state */
const INITIAL_STATE: DashboardData = {
  statistics: null,
  recentCalibrations: [],
  isLoading: false,
  error: null,
  lastFetched: null,
};

/**
 * Custom hook for Dashboard data fetching.
 *
 * @param autoRefresh - Whether to automatically refresh data on a timer
 * @returns Dashboard data, refresh function, and loading state
 */
export function useDashboardData(autoRefresh = false) {
  const [data, setData] = useState<DashboardData>(INITIAL_STATE);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isMountedRef = useRef(true);

  const fetchDashboardData = useCallback(async () => {
    logger.debug('Dashboard: Fetching data');
    setData((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const [statistics, calibrationResponse] = await Promise.all([
        api.statistics.get(),
        api.calibrations.list(undefined, config.calibration.defaultSteps),
      ]);

      if (!isMountedRef.current) return;

      logger.debug('Dashboard: Data fetched', {
        totalRecords: statistics.total_records,
        calibrationCount: calibrationResponse.count,
      });

      setData({
        statistics,
        recentCalibrations: calibrationResponse.records,
        isLoading: false,
        error: null,
        lastFetched: new Date(),
      });
    } catch (err) {
      if (!isMountedRef.current) return;

      const message =
        err instanceof Error ? err.message : 'Failed to load dashboard data';
      logger.error('Dashboard: Fetch failed', { error: message });

      setData((prev) => ({
        ...prev,
        isLoading: false,
        error: message,
      }));
    }
  }, []);

  const refresh = useCallback(() => {
    logger.debug('Dashboard: Manual refresh triggered');
    return fetchDashboardData();
  }, [fetchDashboardData]);

  // Initial fetch
  useEffect(() => {
    isMountedRef.current = true;
    fetchDashboardData();

    return () => {
      isMountedRef.current = false;
    };
  }, [fetchDashboardData]);

  // Auto-refresh timer (driven by config staleTime, not hardcoded)
  useEffect(() => {
    if (!autoRefresh) return;

    const intervalMs = config.api.staleTime;
    logger.debug('Dashboard: Auto-refresh enabled', { intervalMs });

    intervalRef.current = setInterval(() => {
      fetchDashboardData();
    }, intervalMs);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [autoRefresh, fetchDashboardData]);

  return { ...data, refresh };
}
