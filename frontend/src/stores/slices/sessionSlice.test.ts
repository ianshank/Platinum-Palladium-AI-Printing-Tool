import { beforeEach, describe, expect, it } from 'vitest';
import { createStore } from '@/stores';
import type { PrintRecord } from './sessionSlice';

describe('sessionSlice', () => {
  let store: ReturnType<typeof createStore>;

  const mockRecord: Omit<PrintRecord, 'id' | 'timestamp'> = {
    paperType: 'Arches Platine',
    paperSize: '8x10',
    chemistry: { platinumMl: 0.5, palladiumMl: 0.5, metalRatio: 0.5 },
    exposure: { timeSeconds: 180, uvSource: 'uv-led' },
    result: 'success',
  };

  beforeEach(() => {
    store = createStore();
  });

  describe('Initial State', () => {
    it('has correct initial values', () => {
      const state = store.getState();
      expect(state.session.records).toEqual([]);
      expect(state.session.stats.totalPrints).toBe(0);
      expect(state.session.stats.successRate).toBe(0);
      expect(state.session.selectedRecordId).toBeNull();
      expect(state.session.filterResult).toBe('all');
      expect(state.session.isLoading).toBe(false);
    });
  });

  describe('Record CRUD', () => {
    it('addRecord creates a record with id and timestamp', () => {
      store.getState().session.addRecord(mockRecord);

      const records = store.getState().session.records;
      expect(records).toHaveLength(1);
      expect(records[0]?.id).toBeDefined();
      expect(records[0]?.timestamp).toBeDefined();
      expect(records[0]?.paperType).toBe('Arches Platine');
    });

    it('addRecord prepends (most recent first)', () => {
      store.getState().session.addRecord({ ...mockRecord, paperType: 'First' });
      store
        .getState()
        .session.addRecord({ ...mockRecord, paperType: 'Second' });

      expect(store.getState().session.records[0]?.paperType).toBe('Second');
    });

    it('updateRecord modifies existing record', () => {
      store.getState().session.addRecord(mockRecord);
      const id = store.getState().session.records[0]?.id;

      if (id) {
        store.getState().session.updateRecord(id, { notes: 'Great print' });
        expect(store.getState().session.records[0]?.notes).toBe('Great print');
      }
    });

    it('deleteRecord removes record and clears selection', () => {
      store.getState().session.addRecord(mockRecord);
      const id = store.getState().session.records[0]?.id;

      if (id) {
        store.getState().session.selectRecord(id);
        store.getState().session.deleteRecord(id);

        expect(store.getState().session.records).toHaveLength(0);
        expect(store.getState().session.selectedRecordId).toBeNull();
      }
    });

    it('selectRecord sets selectedRecordId', () => {
      store.getState().session.addRecord(mockRecord);
      const id = store.getState().session.records[0]?.id;

      if (id) {
        store.getState().session.selectRecord(id);
        expect(store.getState().session.selectedRecordId).toBe(id);
      }
    });
  });

  describe('Filtering', () => {
    it('setDateRange updates filter', () => {
      store.getState().session.setDateRange('2026-01-01', '2026-02-01');

      const range = store.getState().session.filterDateRange;
      expect(range.start).toBe('2026-01-01');
      expect(range.end).toBe('2026-02-01');
    });

    it('setResultFilter updates filter', () => {
      store.getState().session.setResultFilter('failure');
      expect(store.getState().session.filterResult).toBe('failure');
    });

    it('clearFilters resets all filters', () => {
      store.getState().session.setDateRange('2026-01-01', '2026-02-01');
      store.getState().session.setResultFilter('success');
      store.getState().session.clearFilters();

      expect(store.getState().session.filterDateRange).toEqual({
        start: null,
        end: null,
      });
      expect(store.getState().session.filterResult).toBe('all');
    });
  });

  describe('Statistics', () => {
    it('calculateStats computes correct stats for records', () => {
      store.getState().session.addRecord({ ...mockRecord, result: 'success' });
      store.getState().session.addRecord({ ...mockRecord, result: 'failure' });

      const stats = store.getState().session.stats;
      expect(stats.totalPrints).toBe(2);
      expect(stats.successRate).toBe(0.5);
      expect(stats.averageExposure).toBe(180);
      expect(stats.mostUsedPaper).toBe('Arches Platine');
    });

    it('calculateStats returns initial stats for empty records', () => {
      store.getState().session.calculateStats();

      const stats = store.getState().session.stats;
      expect(stats.totalPrints).toBe(0);
      expect(stats.successRate).toBe(0);
    });

    it('calculateStats updates after record delete', () => {
      store.getState().session.addRecord(mockRecord);
      const id = store.getState().session.records[0]?.id;
      if (id) {
        store.getState().session.deleteRecord(id);
      }

      expect(store.getState().session.stats.totalPrints).toBe(0);
    });
  });

  describe('Import/Export', () => {
    it('importRecords merges without duplicates', () => {
      store.getState().session.addRecord(mockRecord);
      const existing = store.getState().session.records[0]!;

      const imported: PrintRecord = {
        ...mockRecord,
        id: 'imported-1',
        timestamp: '2026-01-15T00:00:00Z',
      };

      store.getState().session.importRecords([existing, imported]);

      // Should have 2 total (1 original + 1 new import, existing deduplicated)
      expect(store.getState().session.records).toHaveLength(2);
    });

    it('exportRecords returns all records', () => {
      store.getState().session.addRecord(mockRecord);
      store
        .getState()
        .session.addRecord({ ...mockRecord, paperType: 'Bergger COT320' });

      const exported = store.getState().session.exportRecords();
      expect(exported).toHaveLength(2);
    });
  });

  describe('Reset', () => {
    it('clearRecords empties records and resets stats', () => {
      store.getState().session.addRecord(mockRecord);
      store.getState().session.clearRecords();

      expect(store.getState().session.records).toEqual([]);
      expect(store.getState().session.stats.totalPrints).toBe(0);
    });

    it('resetSession restores all initial state', () => {
      store.getState().session.addRecord(mockRecord);
      store.getState().session.setResultFilter('failure');
      store.getState().session.resetSession();

      const state = store.getState();
      expect(state.session.records).toEqual([]);
      expect(state.session.filterResult).toBe('all');
      expect(state.session.isLoading).toBe(false);
    });
  });
});
