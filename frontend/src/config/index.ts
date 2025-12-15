/**
 * Configuration exports.
 * All configuration is centralized here for easy access.
 */

export { env, getEnv } from './env';
export type { EnvConfig } from './env';

export { apiConfig } from './api.config';
export type { ApiEndpoints } from './api.config';

export { chemistryConfig } from './chemistry.config';
export type {
  PaperAbsorbency,
  CoatingMethod,
  MetalPresetId,
  PaperSizeId,
} from './chemistry.config';

export { tabletConfig } from './tablet.config';
export type {
  TabletTypeId,
  LinearizationMethodId,
  TargetResponseId,
  ExportFormatId,
  AdjustmentTypeId,
} from './tablet.config';

/**
 * Application-wide constants.
 * These should not be environment-specific.
 */
export const APP_CONSTANTS = {
  // Application info
  APP_NAME: 'PTPD Calibration Studio',
  APP_VERSION: __APP_VERSION__ || '1.0.0',

  // UI defaults
  DEFAULT_CURVE_POINTS: 256,
  MAX_FILE_SIZE_MB: 50,
  TOAST_DURATION_MS: 5000,

  // Keyboard shortcuts
  SHORTCUTS: {
    DASHBOARD: 'Ctrl+1',
    CALIBRATION: 'Ctrl+2',
    CHEMISTRY: 'Ctrl+3',
    ASSISTANT: 'Ctrl+4',
    SESSIONS: 'Ctrl+5',
    SAVE: 'Ctrl+S',
    UNDO: 'Ctrl+Z',
  },

  // Storage keys
  STORAGE_KEYS: {
    THEME: 'ptpd-theme',
    WIZARD_STATE: 'ptpd-wizard-state',
    RECENT_CURVES: 'ptpd-recent-curves',
    USER_PREFERENCES: 'ptpd-user-preferences',
  },
} as const;

declare const __APP_VERSION__: string | undefined;
