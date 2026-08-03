/**
 * Dynamic Feature Flags System
 *
 * All flags are driven by VITE_FF_* environment variables.
 * No hardcoded values — defaults are provided as fallbacks only.
 *
 * Usage:
 *   import { isEnabled, FeatureFlag } from '@/config/featureFlags';
 *   if (isEnabled(FeatureFlag.CURVE_EDITOR_V2)) { ... }
 */

import { getBoolEnv } from './index';

/**
 * Centralized feature flag registry.
 * Add new flags here — they map to VITE_FF_{FLAG_NAME} env vars.
 */
export const FeatureFlag = {
  /** Enable the new React curve editor (vs legacy Gradio) */
  CURVE_EDITOR_V2: 'CURVE_EDITOR_V2',
  /** Enable Vertex AI cloud enhancement for curves */
  VERTEX_AI_ENHANCE: 'VERTEX_AI_ENHANCE',
  /** Enable batch processing queue UI */
  BATCH_PROCESSING: 'BATCH_PROCESSING',
  /** Enable AI assistant chat panel */
  AI_ASSISTANT: 'AI_ASSISTANT',
  /** Enable PWA offline mode */
  OFFLINE_MODE: 'OFFLINE_MODE',
  /** Enable cloud profile sync */
  CLOUD_SYNC: 'CLOUD_SYNC',
  /** Enable undo/redo history for curve edits */
  UNDO_REDO: 'UNDO_REDO',
  /** Enable dark mode toggle */
  DARK_MODE: 'DARK_MODE',
  /** Enable step tablet scanner integration */
  SCANNER_INTEGRATION: 'SCANNER_INTEGRATION',
} as const;

export type FeatureFlagKey = (typeof FeatureFlag)[keyof typeof FeatureFlag];

/**
 * Default flag states — used when no env var is set.
 * Keep defaults conservative (off) for experimental features.
 */
const FLAG_DEFAULTS: Record<FeatureFlagKey, boolean> = {
  [FeatureFlag.CURVE_EDITOR_V2]: false,
  [FeatureFlag.VERTEX_AI_ENHANCE]: false,
  [FeatureFlag.BATCH_PROCESSING]: false,
  [FeatureFlag.AI_ASSISTANT]: true,
  [FeatureFlag.OFFLINE_MODE]: false,
  [FeatureFlag.CLOUD_SYNC]: false,
  [FeatureFlag.UNDO_REDO]: false,
  [FeatureFlag.DARK_MODE]: true,
  [FeatureFlag.SCANNER_INTEGRATION]: false,
};

/**
 * Runtime override map — allows programmatic toggling in tests/dev.
 * Not persisted; resets on page reload.
 */
const runtimeOverrides = new Map<FeatureFlagKey, boolean>();

/**
 * Check if a feature flag is enabled.
 *
 * Resolution order:
 * 1. Runtime override (set via `setFlag`)
 * 2. Environment variable `VITE_FF_{FLAG_NAME}`
 * 3. Default from `FLAG_DEFAULTS`
 */
export function isEnabled(flag: FeatureFlagKey): boolean {
  if (runtimeOverrides.has(flag)) {
    return runtimeOverrides.get(flag)!;
  }
  return getBoolEnv(`VITE_FF_${flag}`, FLAG_DEFAULTS[flag] ?? false);
}

/**
 * Set a runtime override for a feature flag.
 * Useful for testing and development.
 */
export function setFlag(flag: FeatureFlagKey, value: boolean): void {
  runtimeOverrides.set(flag, value);
}

/**
 * Clear a runtime override, reverting to env/default.
 */
export function clearFlag(flag: FeatureFlagKey): void {
  runtimeOverrides.delete(flag);
}

/**
 * Clear all runtime overrides.
 */
export function clearAllFlags(): void {
  runtimeOverrides.clear();
}

/**
 * Get all flag states (for debugging / devtools display).
 */
export function getAllFlags(): Record<FeatureFlagKey, boolean> {
  const flags = {} as Record<FeatureFlagKey, boolean>;
  for (const key of Object.values(FeatureFlag)) {
    flags[key] = isEnabled(key);
  }
  return flags;
}
