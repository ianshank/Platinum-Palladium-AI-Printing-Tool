/**
 * Application configuration
 * All dynamic values are loaded from environment variables
 * No hardcoded values - everything is configurable
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface AppConfig {
  app: {
    name: string;
    version: string;
    environment: 'development' | 'production' | 'test';
  };
  api: {
    baseUrl: string;
    wsUrl: string;
    timeout: number;
    retryAttempts: number;
    retryDelay: number;
    staleTime: number;
    gcTime: number;
  };
  features: {
    devtools: boolean;
    mockApi: boolean;
    darkMode: boolean;
    offlineMode: boolean;
  };
  logging: {
    level: LogLevel;
    enableConsole: boolean;
    enableRemote: boolean;
    maxLogSize: number;
  };
  ui: {
    defaultTab: string;
    animationDuration: number;
    toastDuration: number;
    debounceDelay: number;
  };
  calibration: {
    defaultSteps: number;
    maxCurvePoints: number;
    smoothingDefault: number;
  };
}

/**
 * Get environment variable with fallback
 */
function getEnv(key: string, fallback: string): string {
  // Vite exposes env vars with VITE_ prefix
  const value = import.meta.env[key];
  return typeof value === 'string' && value.length > 0 ? value : fallback;
}

/**
 * Parse boolean from environment variable
 */
function getBoolEnv(key: string, fallback: boolean): boolean {
  const value = import.meta.env[key];
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    return value.toLowerCase() === 'true' || value === '1';
  }
  return fallback;
}

/**
 * Parse number from environment variable
 */
function getNumEnv(key: string, fallback: number): number {
  const value = import.meta.env[key];
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/**
 * Determine current environment
 */
function getEnvironment(): 'development' | 'production' | 'test' {
  if (import.meta.env.MODE === 'test') return 'test';
  if (import.meta.env.PROD) return 'production';
  return 'development';
}

/**
 * Application configuration object
 * All values are derived from environment variables with sensible defaults
 */
export const config: AppConfig = {
  app: {
    name: getEnv('VITE_APP_NAME', 'Platinum-Palladium AI Printing Tool'),
    version: getEnv('VITE_APP_VERSION', '0.1.0'),
    environment: getEnvironment(),
  },
  api: {
    baseUrl: getEnv('VITE_API_URL', 'http://localhost:8000'),
    wsUrl: getEnv('VITE_WS_URL', 'ws://localhost:8000/ws'),
    timeout: getNumEnv('VITE_API_TIMEOUT', 30000),
    retryAttempts: getNumEnv('VITE_API_RETRY_ATTEMPTS', 3),
    retryDelay: getNumEnv('VITE_API_RETRY_DELAY', 1000),
    staleTime: getNumEnv('VITE_API_STALE_TIME', 5 * 60 * 1000), // 5 minutes
    gcTime: getNumEnv('VITE_API_GC_TIME', 10 * 60 * 1000), // 10 minutes
  },
  features: {
    devtools: getBoolEnv('VITE_ENABLE_DEVTOOLS', true),
    mockApi: getBoolEnv('VITE_ENABLE_MOCK_API', false),
    darkMode: getBoolEnv('VITE_ENABLE_DARK_MODE', true),
    offlineMode: getBoolEnv('VITE_ENABLE_OFFLINE_MODE', false),
  },
  logging: {
    level: getEnv('VITE_LOG_LEVEL', 'debug') as LogLevel,
    enableConsole: getBoolEnv('VITE_LOG_CONSOLE', true),
    enableRemote: getBoolEnv('VITE_LOG_REMOTE', false),
    maxLogSize: getNumEnv('VITE_LOG_MAX_SIZE', 1000),
  },
  ui: {
    defaultTab: getEnv('VITE_DEFAULT_TAB', 'dashboard'),
    animationDuration: getNumEnv('VITE_ANIMATION_DURATION', 200),
    toastDuration: getNumEnv('VITE_TOAST_DURATION', 5000),
    debounceDelay: getNumEnv('VITE_DEBOUNCE_DELAY', 300),
  },
  calibration: {
    defaultSteps: getNumEnv('VITE_DEFAULT_STEPS', 21),
    maxCurvePoints: getNumEnv('VITE_MAX_CURVE_POINTS', 256),
    smoothingDefault: getNumEnv('VITE_SMOOTHING_DEFAULT', 0.5),
  },
} as const;

/**
 * Check if running in development mode
 */
export const isDev = config.app.environment === 'development';

/**
 * Check if running in production mode
 */
export const isProd = config.app.environment === 'production';

/**
 * Check if running in test mode
 */
export const isTest = config.app.environment === 'test';

// Type export for use in other files
export type { AppConfig, LogLevel };
