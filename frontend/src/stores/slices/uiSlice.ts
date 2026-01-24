/**
 * UI state slice
 * Manages application-wide UI state like navigation, loading states, and theme
 */

import type { StateCreator } from 'zustand';
import { logger } from '@/lib/logger';
import { config } from '@/config';

export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant?: 'default' | 'success' | 'warning' | 'error';
  duration?: number;
}

export interface UISlice {
  // State
  activeTab: string;
  sidebarOpen: boolean;
  isProcessing: boolean;
  isInitialized: boolean;
  theme: 'light' | 'dark';
  toasts: Toast[];
  modals: Record<string, boolean>;

  // Actions
  setActiveTab: (tab: string) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setProcessing: (processing: boolean) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleTheme: () => void;
  initializeApp: () => void;

  // Toast actions
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;

  // Modal actions
  openModal: (modalId: string) => void;
  closeModal: (modalId: string) => void;
  toggleModal: (modalId: string) => void;

  // Reset
  resetUI: () => void;
}

const initialState = {
  activeTab: config.ui.defaultTab,
  sidebarOpen: true,
  isProcessing: false,
  isInitialized: false,
  theme: 'dark' as const,
  toasts: [] as Toast[],
  modals: {} as Record<string, boolean>,
};

let toastIdCounter = 0;

export const createUISlice: StateCreator<
  { ui: UISlice },
  [['zustand/immer', never]],
  [],
  UISlice
> = (set, get) => ({
  ...initialState,

  setActiveTab: (tab) => {
    logger.debug('UI: setActiveTab', { tab });
    set((state) => {
      state.ui.activeTab = tab;
    });
  },

  toggleSidebar: () => {
    const current = get().ui.sidebarOpen;
    logger.debug('UI: toggleSidebar', { current, next: !current });
    set((state) => {
      state.ui.sidebarOpen = !state.ui.sidebarOpen;
    });
  },

  setSidebarOpen: (open) => {
    logger.debug('UI: setSidebarOpen', { open });
    set((state) => {
      state.ui.sidebarOpen = open;
    });
  },

  setProcessing: (processing) => {
    logger.debug('UI: setProcessing', { processing });
    set((state) => {
      state.ui.isProcessing = processing;
    });
  },

  setTheme: (theme) => {
    logger.debug('UI: setTheme', { theme });
    set((state) => {
      state.ui.theme = theme;
    });
    // Update document class for theme
    document.documentElement.classList.toggle('light', theme === 'light');
    document.documentElement.classList.toggle('dark', theme === 'dark');
  },

  toggleTheme: () => {
    const current = get().ui.theme;
    const next = current === 'dark' ? 'light' : 'dark';
    get().ui.setTheme(next);
  },

  initializeApp: () => {
    logger.info('UI: Initializing application');

    // Apply saved theme
    const theme = get().ui.theme;
    document.documentElement.classList.toggle('dark', theme === 'dark');

    set((state) => {
      state.ui.isInitialized = true;
    });

    logger.info('UI: Application initialized');
  },

  addToast: (toast) => {
    const id = `toast-${++toastIdCounter}`;
    const duration = toast.duration ?? config.ui.toastDuration;

    logger.debug('UI: addToast', { id, ...toast });

    set((state) => {
      state.ui.toasts.push({ ...toast, id });
    });

    // Auto-remove after duration
    if (duration > 0) {
      setTimeout(() => {
        get().ui.removeToast(id);
      }, duration);
    }
  },

  removeToast: (id) => {
    logger.debug('UI: removeToast', { id });
    set((state) => {
      state.ui.toasts = state.ui.toasts.filter((t) => t.id !== id);
    });
  },

  clearToasts: () => {
    logger.debug('UI: clearToasts');
    set((state) => {
      state.ui.toasts = [];
    });
  },

  openModal: (modalId) => {
    logger.debug('UI: openModal', { modalId });
    set((state) => {
      state.ui.modals[modalId] = true;
    });
  },

  closeModal: (modalId) => {
    logger.debug('UI: closeModal', { modalId });
    set((state) => {
      state.ui.modals[modalId] = false;
    });
  },

  toggleModal: (modalId) => {
    const current = get().ui.modals[modalId] ?? false;
    logger.debug('UI: toggleModal', { modalId, current, next: !current });
    set((state) => {
      state.ui.modals[modalId] = !current;
    });
  },

  resetUI: () => {
    logger.debug('UI: resetUI');
    set((state) => {
      state.ui.activeTab = initialState.activeTab;
      state.ui.isProcessing = initialState.isProcessing;
      state.ui.toasts = [];
      state.ui.modals = {};
    });
  },
});
