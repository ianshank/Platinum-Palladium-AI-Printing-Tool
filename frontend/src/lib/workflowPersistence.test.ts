/**
 * Tests for workflow persistence utility
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  cleanupExpiredCheckpoints,
  clearCheckpoint,
  loadCheckpoint,
  saveCheckpoint,
  type WorkflowCheckpoint,
} from './workflowPersistence';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string): string | null => store[key] ?? null,
    setItem: (key: string, value: string): void => {
      store[key] = value;
    },
    removeItem: (key: string): void => {
      delete store[key];
    },
    clear: (): void => {
      store = {};
    },
    get length(): number {
      return Object.keys(store).length;
    },
    key: (index: number): string | null => {
      const keys = Object.keys(store);
      return keys[index] ?? null;
    },
  };
})();

// Assign mock to global
global.localStorage = localStorageMock as unknown as Storage;

describe('workflowPersistence', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  describe('saveCheckpoint', () => {
    it('should save checkpoint to localStorage', () => {
      const data = { step: 1, value: 'test' };
      saveCheckpoint('test-workflow', 1, data);

      const raw = localStorage.getItem('ptpd-workflow:test-workflow');
      expect(raw).toBeTruthy();

      const checkpoint: WorkflowCheckpoint<typeof data> = JSON.parse(raw!);
      expect(checkpoint.id).toBe('test-workflow');
      expect(checkpoint.step).toBe(1);
      expect(checkpoint.data).toEqual(data);
      expect(checkpoint.savedAt).toBeTruthy();
      expect(checkpoint.expiresAt).toBeTruthy();
    });

    it('should set expiration based on TTL', () => {
      const data = { value: 'test' };
      const ttlMs = 1000; // 1 second
      const beforeSave = Date.now();

      saveCheckpoint('test-workflow', 1, data, ttlMs);

      const raw = localStorage.getItem('ptpd-workflow:test-workflow');
      const checkpoint: WorkflowCheckpoint<typeof data> = JSON.parse(raw!);

      const savedAt = new Date(checkpoint.savedAt).getTime();
      const expiresAt = new Date(checkpoint.expiresAt).getTime();

      // Check that expiration is approximately TTL after save
      expect(expiresAt - savedAt).toBeGreaterThanOrEqual(ttlMs);
      expect(savedAt).toBeGreaterThanOrEqual(beforeSave);
    });

    it('should overwrite existing checkpoint', () => {
      const data1 = { value: 'first' };
      const data2 = { value: 'second' };

      saveCheckpoint('test-workflow', 1, data1);
      saveCheckpoint('test-workflow', 2, data2);

      const raw = localStorage.getItem('ptpd-workflow:test-workflow');
      const checkpoint: WorkflowCheckpoint<typeof data2> = JSON.parse(raw!);

      expect(checkpoint.step).toBe(2);
      expect(checkpoint.data.value).toBe('second');
    });

    it('should handle localStorage quota errors gracefully', () => {
      // Mock setItem to throw quota error
      const mockSetItem = vi.fn(() => {
        throw new Error('QuotaExceededError');
      });
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const originalSetItem = localStorage.setItem;
      localStorage.setItem = mockSetItem;

      // Should not throw
      expect(() => {
        saveCheckpoint('test-workflow', 1, { large: 'data' });
      }).not.toThrow();

      // Restore
      localStorage.setItem = originalSetItem;
    });
  });

  describe('loadCheckpoint', () => {
    it('should load valid checkpoint from localStorage', () => {
      const data = { step: 1, value: 'test' };
      saveCheckpoint('test-workflow', 1, data);

      const checkpoint = loadCheckpoint<typeof data>('test-workflow');

      expect(checkpoint).toBeTruthy();
      expect(checkpoint?.id).toBe('test-workflow');
      expect(checkpoint?.step).toBe(1);
      expect(checkpoint?.data).toEqual(data);
    });

    it('should return null for non-existent checkpoint', () => {
      const checkpoint = loadCheckpoint('non-existent');
      expect(checkpoint).toBeNull();
    });

    it('should return null and remove expired checkpoint', () => {
      const data = { value: 'test' };

      // Manually create expired checkpoint
      const now = new Date();
      const checkpoint: WorkflowCheckpoint<typeof data> = {
        id: 'test-workflow',
        step: 1,
        data,
        savedAt: new Date(now.getTime() - 2000).toISOString(),
        expiresAt: new Date(now.getTime() - 1000).toISOString(),
      };
      localStorage.setItem('ptpd-workflow:test-workflow', JSON.stringify(checkpoint));

      const loaded = loadCheckpoint<typeof data>('test-workflow');
      expect(loaded).toBeNull();

      // Should be removed from localStorage
      expect(localStorage.getItem('ptpd-workflow:test-workflow')).toBeNull();
    });

    it('should handle corrupted JSON gracefully', () => {
      localStorage.setItem('ptpd-workflow:test-workflow', 'invalid-json{');

      const checkpoint = loadCheckpoint('test-workflow');
      expect(checkpoint).toBeNull();
    });

    it('should handle missing fields in checkpoint', () => {
      const invalid = { id: 'test', step: 1 }; // Missing required fields
      localStorage.setItem('ptpd-workflow:test-workflow', JSON.stringify(invalid));

      // Should not throw, but may return unexpected result
      expect(() => {
        loadCheckpoint('test-workflow');
      }).not.toThrow();
    });
  });

  describe('clearCheckpoint', () => {
    it('should remove checkpoint from localStorage', () => {
      const data = { value: 'test' };
      saveCheckpoint('test-workflow', 1, data);

      expect(localStorage.getItem('ptpd-workflow:test-workflow')).toBeTruthy();

      clearCheckpoint('test-workflow');

      expect(localStorage.getItem('ptpd-workflow:test-workflow')).toBeNull();
    });

    it('should not throw if checkpoint does not exist', () => {
      expect(() => {
        clearCheckpoint('non-existent');
      }).not.toThrow();
    });
  });

  describe('cleanupExpiredCheckpoints', () => {
    it('should remove all expired checkpoints', () => {
      const now = new Date();

      // Create expired checkpoint
      const expired: WorkflowCheckpoint<{ value: string }> = {
        id: 'expired',
        step: 1,
        data: { value: 'expired' },
        savedAt: new Date(now.getTime() - 2000).toISOString(),
        expiresAt: new Date(now.getTime() - 1000).toISOString(),
      };
      localStorage.setItem('ptpd-workflow:expired', JSON.stringify(expired));

      // Create valid checkpoint
      const valid: WorkflowCheckpoint<{ value: string }> = {
        id: 'valid',
        step: 1,
        data: { value: 'valid' },
        savedAt: now.toISOString(),
        expiresAt: new Date(now.getTime() + 1000).toISOString(),
      };
      localStorage.setItem('ptpd-workflow:valid', JSON.stringify(valid));

      cleanupExpiredCheckpoints();

      expect(localStorage.getItem('ptpd-workflow:expired')).toBeNull();
      expect(localStorage.getItem('ptpd-workflow:valid')).toBeTruthy();
    });

    it('should remove corrupted checkpoints', () => {
      localStorage.setItem('ptpd-workflow:corrupted', 'invalid-json{');
      localStorage.setItem('other-key', 'value'); // Should not be touched

      cleanupExpiredCheckpoints();

      expect(localStorage.getItem('ptpd-workflow:corrupted')).toBeNull();
      expect(localStorage.getItem('other-key')).toBe('value');
    });

    it('should not remove checkpoints from other prefixes', () => {
      localStorage.setItem('other-prefix:workflow', 'value');
      localStorage.setItem('ptpd-other:workflow', 'value');

      cleanupExpiredCheckpoints();

      expect(localStorage.getItem('other-prefix:workflow')).toBe('value');
      expect(localStorage.getItem('ptpd-other:workflow')).toBe('value');
    });

    it('should handle empty localStorage', () => {
      expect(() => {
        cleanupExpiredCheckpoints();
      }).not.toThrow();
    });

    it('should handle multiple expired checkpoints', () => {
      // Use a fixed past timestamp to ensure they're clearly expired
      const pastTime = Date.now() - 10000; // 10 seconds ago

      for (let i = 0; i < 5; i++) {
        const expired: WorkflowCheckpoint<{ value: number }> = {
          id: `expired-${i}`,
          step: i,
          data: { value: i },
          savedAt: new Date(pastTime - 2000).toISOString(),
          expiresAt: new Date(pastTime).toISOString(), // Clearly in the past
        };
        localStorage.setItem(`ptpd-workflow:expired-${i}`, JSON.stringify(expired));
      }

      cleanupExpiredCheckpoints();

      // All should be removed
      for (let i = 0; i < 5; i++) {
        expect(localStorage.getItem(`ptpd-workflow:expired-${i}`)).toBeNull();
      }
    });
  });

  describe('integration', () => {
    it('should handle complete save-load-clear workflow', () => {
      const data = {
        calibrationId: 'cal-123',
        step: 2,
        measurements: [
          { step: 1, target: 0.1, measured: 0.12 },
          { step: 2, target: 0.2, measured: 0.19 },
        ],
      };

      // Save
      saveCheckpoint('test-workflow', 2, data);

      // Load
      const loaded = loadCheckpoint<typeof data>('test-workflow');
      expect(loaded).toBeTruthy();
      expect(loaded?.data).toEqual(data);

      // Clear
      clearCheckpoint('test-workflow');
      const afterClear = loadCheckpoint<typeof data>('test-workflow');
      expect(afterClear).toBeNull();
    });

    it('should handle multiple workflows simultaneously', () => {
      const workflow1 = { value: 'workflow1' };
      const workflow2 = { value: 'workflow2' };

      saveCheckpoint('workflow-1', 1, workflow1);
      saveCheckpoint('workflow-2', 2, workflow2);

      const loaded1 = loadCheckpoint<typeof workflow1>('workflow-1');
      const loaded2 = loadCheckpoint<typeof workflow2>('workflow-2');

      expect(loaded1?.data.value).toBe('workflow1');
      expect(loaded2?.data.value).toBe('workflow2');
    });
  });
});
