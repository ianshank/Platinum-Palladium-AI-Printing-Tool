/**
 * Feature Flags Tests
 *
 * Covers:
 * - isEnabled (default resolution, env override, runtime override)
 * - setFlag / clearFlag / clearAllFlags
 * - getAllFlags snapshot
 * - Override precedence (runtime > env > default)
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearAllFlags,
  clearFlag,
  FeatureFlag,
  getAllFlags,
  isEnabled,
  setFlag,
} from './featureFlags';

// Mock getBoolEnv via config index
vi.mock('./index', () => ({
  getBoolEnv: vi.fn((key: string, fallback: boolean) => {
    // Simulate env-driven override for specific flag
    if (key === 'VITE_FF_CURVE_EDITOR_V2') return true;
    return fallback;
  }),
}));

describe('featureFlags', () => {
  beforeEach(() => {
    clearAllFlags();
  });

  describe('isEnabled', () => {
    it('returns false for flags with default=false and no env override', () => {
      // BATCH_PROCESSING defaults to false and our mock doesn't override it
      expect(isEnabled(FeatureFlag.BATCH_PROCESSING)).toBe(false);
    });

    it('returns true for flags with default=true', () => {
      // AI_ASSISTANT defaults to true
      expect(isEnabled(FeatureFlag.AI_ASSISTANT)).toBe(true);
    });

    it('resolves env var override (CURVE_EDITOR_V2 mock returns true)', () => {
      // Our mock returns true for VITE_FF_CURVE_EDITOR_V2, default is false
      expect(isEnabled(FeatureFlag.CURVE_EDITOR_V2)).toBe(true);
    });
  });

  describe('setFlag', () => {
    it('overrides the resolved value', () => {
      // BATCH_PROCESSING defaults to false
      expect(isEnabled(FeatureFlag.BATCH_PROCESSING)).toBe(false);
      setFlag(FeatureFlag.BATCH_PROCESSING, true);
      expect(isEnabled(FeatureFlag.BATCH_PROCESSING)).toBe(true);
    });

    it('can disable a default-true flag', () => {
      expect(isEnabled(FeatureFlag.AI_ASSISTANT)).toBe(true);
      setFlag(FeatureFlag.AI_ASSISTANT, false);
      expect(isEnabled(FeatureFlag.AI_ASSISTANT)).toBe(false);
    });
  });

  describe('clearFlag', () => {
    it('reverts to env/default after clearing', () => {
      setFlag(FeatureFlag.BATCH_PROCESSING, true);
      expect(isEnabled(FeatureFlag.BATCH_PROCESSING)).toBe(true);

      clearFlag(FeatureFlag.BATCH_PROCESSING);
      expect(isEnabled(FeatureFlag.BATCH_PROCESSING)).toBe(false);
    });
  });

  describe('clearAllFlags', () => {
    it('reverts all runtime overrides', () => {
      setFlag(FeatureFlag.BATCH_PROCESSING, true);
      setFlag(FeatureFlag.AI_ASSISTANT, false);
      setFlag(FeatureFlag.DARK_MODE, false);

      clearAllFlags();

      expect(isEnabled(FeatureFlag.BATCH_PROCESSING)).toBe(false);
      expect(isEnabled(FeatureFlag.AI_ASSISTANT)).toBe(true);
      expect(isEnabled(FeatureFlag.DARK_MODE)).toBe(true);
    });
  });

  describe('getAllFlags', () => {
    it('returns a record with all flags', () => {
      const flags = getAllFlags();
      expect(Object.keys(flags)).toHaveLength(
        Object.values(FeatureFlag).length
      );
    });

    it('reflects runtime overrides', () => {
      setFlag(FeatureFlag.OFFLINE_MODE, true);
      const flags = getAllFlags();
      expect(flags[FeatureFlag.OFFLINE_MODE]).toBe(true);
    });

    it('each flag has a boolean value', () => {
      const flags = getAllFlags();
      Object.values(flags).forEach((v) => {
        expect(typeof v).toBe('boolean');
      });
    });
  });

  describe('override precedence', () => {
    it('runtime override wins over env var', () => {
      // CURVE_EDITOR_V2: env mock returns true, but runtime override false
      expect(isEnabled(FeatureFlag.CURVE_EDITOR_V2)).toBe(true);
      setFlag(FeatureFlag.CURVE_EDITOR_V2, false);
      expect(isEnabled(FeatureFlag.CURVE_EDITOR_V2)).toBe(false);
    });
  });
});
