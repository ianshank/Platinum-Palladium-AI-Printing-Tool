/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_WS_URL: string;
  readonly VITE_ENABLE_DEVTOOLS: string;
  readonly VITE_ENABLE_MOCK_API: string;
  readonly VITE_LOG_LEVEL: string;
  readonly VITE_APP_NAME: string;
  readonly VITE_APP_VERSION: string;
  readonly VITE_API_TIMEOUT: string;
  readonly VITE_API_RETRY_ATTEMPTS: string;
  readonly VITE_API_RETRY_DELAY: string;
  readonly VITE_API_STALE_TIME: string;
  readonly VITE_API_GC_TIME: string;
  readonly VITE_ENABLE_DARK_MODE: string;
  readonly VITE_ENABLE_OFFLINE_MODE: string;
  readonly VITE_LOG_CONSOLE: string;
  readonly VITE_LOG_REMOTE: string;
  readonly VITE_LOG_MAX_SIZE: string;
  readonly VITE_DEFAULT_TAB: string;
  readonly VITE_ANIMATION_DURATION: string;
  readonly VITE_TOAST_DURATION: string;
  readonly VITE_DEBOUNCE_DELAY: string;
  readonly VITE_DEFAULT_STEPS: string;
  readonly VITE_MAX_CURVE_POINTS: string;
  readonly VITE_SMOOTHING_DEFAULT: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// Global type declarations
declare const __APP_VERSION__: string;
declare const __DEV__: boolean;
