/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_API_TIMEOUT: string;
  readonly VITE_ENABLE_AI_ASSISTANT: string;
  readonly VITE_ENABLE_DEEP_LEARNING: string;
  readonly VITE_ENABLE_CLOUD_SYNC: string;
  readonly VITE_DEFAULT_THEME: string;
  readonly VITE_ANIMATION_ENABLED: string;
  readonly VITE_DEBUG_MODE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare const __APP_VERSION__: string | undefined;
