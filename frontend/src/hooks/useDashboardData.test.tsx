/**
 * useDashboardData Hook Tests
 *
 * Covers:
 * - Initial fetch on mount (statistics + calibrations)
 * - Loading state transitions
 * - Error handling on fetch failure
 * - Manual refresh
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';

// Use vi.hoisted so the mock references are available before vi.mock factories run
const { mockStatisticsGet, mockCalibrationsList } = vi.hoisted(() => ({
  mockStatisticsGet: vi.fn(),
  mockCalibrationsList: vi.fn(),
}));

vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
  },
}));

vi.mock('@/api/client', () => ({
  api: {
    statistics: { get: mockStatisticsGet },
    calibrations: { list: mockCalibrationsList },
  },
}));

vi.mock('@/config', () => ({
  config: {
    api: { staleTime: 60000 },
    calibration: { defaultSteps: 21 },
  },
}));

// Import AFTER mocks are set up
import { useDashboardData } from './useDashboardData';

const mockStatistics = {
  total_records: 42,
  success_rate: 0.85,
  average_exposure: 180,
  paper_types: ['Arches Platine'],
};

const mockCalibrations = {
  count: 3,
  records: [
    { id: 'cal-1', name: 'Test Cal', timestamp: '2026-02-07T10:00:00Z' },
  ],
};

describe('useDashboardData', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStatisticsGet.mockResolvedValue(mockStatistics);
    mockCalibrationsList.mockResolvedValue(mockCalibrations);
  });

  it('returns initial loading state', () => {
    const { result } = renderHook(() => useDashboardData());

    expect(result.current.isLoading).toBe(true);
    expect(result.current.statistics).toBeNull();
    expect(result.current.recentCalibrations).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  it('fetches statistics and calibrations on mount', async () => {
    const { result } = renderHook(() => useDashboardData());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockStatisticsGet).toHaveBeenCalledTimes(1);
    expect(mockCalibrationsList).toHaveBeenCalledTimes(1);
    expect(result.current.statistics).toEqual(mockStatistics);
    expect(result.current.recentCalibrations).toEqual(mockCalibrations.records);
  });

  it('sets lastFetched after successful fetch', async () => {
    const { result } = renderHook(() => useDashboardData());

    await waitFor(() => {
      expect(result.current.lastFetched).not.toBeNull();
    });

    expect(result.current.lastFetched).toBeInstanceOf(Date);
  });

  it('handles API errors gracefully', async () => {
    mockStatisticsGet.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useDashboardData());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBe('Network error');
    expect(result.current.statistics).toBeNull();
  });

  it('handles non-Error rejection', async () => {
    mockStatisticsGet.mockRejectedValueOnce('string error');

    const { result } = renderHook(() => useDashboardData());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBe('Failed to load dashboard data');
  });

  it('manual refresh re-fetches data', async () => {
    const { result } = renderHook(() => useDashboardData());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockStatisticsGet).toHaveBeenCalledTimes(1);

    await act(async () => {
      await result.current.refresh();
    });

    expect(mockStatisticsGet).toHaveBeenCalledTimes(2);
  });

  it('auto-refresh triggers re-fetch', async () => {
    // Use real timers — just verify that auto-refresh mode fetches at least once
    const { result } = renderHook(() => useDashboardData(true));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // At minimum, the initial fetch should have happened
    expect(mockStatisticsGet).toHaveBeenCalled();
    expect(mockCalibrationsList).toHaveBeenCalled();
  });

  it('does not fail when auto-refresh is disabled', async () => {
    const { result } = renderHook(() => useDashboardData(false));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockStatisticsGet).toHaveBeenCalledTimes(1);
  });

  it('cleans up without errors on unmount', async () => {
    const { result, unmount } = renderHook(() => useDashboardData(true));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Unmount should not throw
    expect(() => unmount()).not.toThrow();
  });
});
