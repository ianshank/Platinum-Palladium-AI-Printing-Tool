/**
 * Session state slice
 * Manages print session records and statistics
 */

import type { StateCreator } from 'zustand';
import { logger } from '@/lib/logger';

export interface PrintRecord {
  id: string;
  timestamp: string;
  paperType: string;
  paperSize: string;
  chemistry: {
    platinumMl: number;
    palladiumMl: number;
    metalRatio: number;
  };
  exposure: {
    timeSeconds: number;
    uvSource: string;
  };
  result: 'success' | 'partial' | 'failure';
  notes?: string;
  imageUrl?: string;
  calibrationId?: string;
  curveId?: string;
}

export interface SessionStats {
  totalPrints: number;
  successRate: number;
  averageExposure: number;
  mostUsedPaper: string | null;
  recentActivity: {
    date: string;
    count: number;
  }[];
}

export interface SessionSlice {
  // State
  records: PrintRecord[];
  stats: SessionStats;
  selectedRecordId: string | null;
  filterDateRange: { start: string | null; end: string | null };
  filterResult: PrintRecord['result'] | 'all';
  isLoading: boolean;

  // Actions
  addRecord: (record: Omit<PrintRecord, 'id' | 'timestamp'>) => void;
  updateRecord: (id: string, updates: Partial<PrintRecord>) => void;
  deleteRecord: (id: string) => void;
  selectRecord: (id: string | null) => void;

  // Filtering
  setDateRange: (start: string | null, end: string | null) => void;
  setResultFilter: (result: PrintRecord['result'] | 'all') => void;
  clearFilters: () => void;

  // Stats
  calculateStats: () => void;

  // Bulk operations
  importRecords: (records: PrintRecord[]) => void;
  exportRecords: () => PrintRecord[];
  clearRecords: () => void;

  // Reset
  resetSession: () => void;
}

const initialState = {
  records: [] as PrintRecord[],
  stats: {
    totalPrints: 0,
    successRate: 0,
    averageExposure: 0,
    mostUsedPaper: null,
    recentActivity: [],
  } as SessionStats,
  selectedRecordId: null as string | null,
  filterDateRange: { start: null, end: null } as { start: string | null; end: string | null },
  filterResult: 'all' as PrintRecord['result'] | 'all',
  isLoading: false,
};

export const createSessionSlice: StateCreator<
  { session: SessionSlice },
  [['zustand/immer', never]],
  [],
  SessionSlice
> = (set, get) => ({
  ...initialState,

  addRecord: (record) => {
    const id = `rec-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    const timestamp = new Date().toISOString();

    logger.debug('Session: addRecord', { id, result: record.result });

    set((state) => {
      state.session.records.unshift({
        ...record,
        id,
        timestamp,
      });
    });

    get().session.calculateStats();
  },

  updateRecord: (id, updates) => {
    logger.debug('Session: updateRecord', { id, updates: Object.keys(updates) });
    set((state) => {
      const record = state.session.records.find((r) => r.id === id);
      if (record) {
        Object.assign(record, updates);
      }
    });

    get().session.calculateStats();
  },

  deleteRecord: (id) => {
    logger.debug('Session: deleteRecord', { id });
    set((state) => {
      state.session.records = state.session.records.filter((r) => r.id !== id);
      if (state.session.selectedRecordId === id) {
        state.session.selectedRecordId = null;
      }
    });

    get().session.calculateStats();
  },

  selectRecord: (id) => {
    logger.debug('Session: selectRecord', { id });
    set((state) => {
      state.session.selectedRecordId = id;
    });
  },

  setDateRange: (start, end) => {
    logger.debug('Session: setDateRange', { start, end });
    set((state) => {
      state.session.filterDateRange = { start, end };
    });
  },

  setResultFilter: (result) => {
    logger.debug('Session: setResultFilter', { result });
    set((state) => {
      state.session.filterResult = result;
    });
  },

  clearFilters: () => {
    logger.debug('Session: clearFilters');
    set((state) => {
      state.session.filterDateRange = { start: null, end: null };
      state.session.filterResult = 'all';
    });
  },

  calculateStats: () => {
    const records = get().session.records;

    if (records.length === 0) {
      set((state) => {
        state.session.stats = initialState.stats;
      });
      return;
    }

    // Calculate success rate
    const successCount = records.filter((r) => r.result === 'success').length;
    const successRate = successCount / records.length;

    // Calculate average exposure
    const exposureTimes = records.map((r) => r.exposure.timeSeconds);
    const averageExposure = exposureTimes.reduce((a, b) => a + b, 0) / exposureTimes.length;

    // Find most used paper
    const paperCounts = records.reduce((acc, r) => {
      acc[r.paperType] = (acc[r.paperType] ?? 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    let mostUsedPaper: string | null = null;
    let maxCount = 0;
    for (const [paper, count] of Object.entries(paperCounts)) {
      if (count > maxCount) {
        maxCount = count;
        mostUsedPaper = paper;
      }
    }

    // Calculate recent activity (last 7 days)
    const now = new Date();
    const recentActivity: SessionStats['recentActivity'] = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0]!;
      const count = records.filter((r) => r.timestamp.startsWith(dateStr)).length;
      recentActivity.push({ date: dateStr, count });
    }

    logger.debug('Session: calculateStats', {
      totalPrints: records.length,
      successRate,
      averageExposure,
      mostUsedPaper,
    });

    set((state) => {
      state.session.stats = {
        totalPrints: records.length,
        successRate,
        averageExposure,
        mostUsedPaper,
        recentActivity,
      };
    });
  },

  importRecords: (records) => {
    logger.info('Session: importRecords', { count: records.length });
    set((state) => {
      // Merge, avoiding duplicates by ID
      const existingIds = new Set(state.session.records.map((r) => r.id));
      const newRecords = records.filter((r) => !existingIds.has(r.id));
      state.session.records = [...state.session.records, ...newRecords];
    });

    get().session.calculateStats();
  },

  exportRecords: () => {
    logger.info('Session: exportRecords');
    return get().session.records;
  },

  clearRecords: () => {
    logger.warn('Session: clearRecords');
    set((state) => {
      state.session.records = [];
      state.session.stats = initialState.stats;
      state.session.selectedRecordId = null;
    });
  },

  resetSession: () => {
    logger.debug('Session: resetSession');
    set((state) => {
      Object.assign(state.session, initialState);
    });
  },
});
