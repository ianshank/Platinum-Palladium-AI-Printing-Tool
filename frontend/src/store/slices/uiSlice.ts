/**
 * UI state management with Zustand.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
}

interface UIState {
  // Sidebar
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;

  // Theme
  theme: 'dark' | 'light' | 'system';
  setTheme: (theme: 'dark' | 'light' | 'system') => void;

  // Modals
  activeModal: string | null;
  modalData: Record<string, unknown>;
  openModal: (modalId: string, data?: Record<string, unknown>) => void;
  closeModal: () => void;

  // Toasts
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;

  // Loading states
  globalLoading: boolean;
  setGlobalLoading: (loading: boolean) => void;

  // Confirmation dialog
  confirmDialog: {
    isOpen: boolean;
    title: string;
    message: string;
    confirmLabel: string;
    cancelLabel: string;
    onConfirm: () => void;
    onCancel: () => void;
    variant: 'danger' | 'warning' | 'info';
  } | null;
  showConfirmDialog: (config: Omit<NonNullable<UIState['confirmDialog']>, 'isOpen'>) => void;
  hideConfirmDialog: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set) => ({
        // Sidebar
        sidebarCollapsed: false,
        toggleSidebar: () =>
          set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
        setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

        // Theme
        theme: 'dark',
        setTheme: (theme) => set({ theme }),

        // Modals
        activeModal: null,
        modalData: {},
        openModal: (modalId, data = {}) =>
          set({ activeModal: modalId, modalData: data }),
        closeModal: () => set({ activeModal: null, modalData: {} }),

        // Toasts
        toasts: [],
        addToast: (toast) =>
          set((state) => ({
            toasts: [
              ...state.toasts,
              {
                ...toast,
                id: `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              },
            ],
          })),
        removeToast: (id) =>
          set((state) => ({
            toasts: state.toasts.filter((t) => t.id !== id),
          })),
        clearToasts: () => set({ toasts: [] }),

        // Loading
        globalLoading: false,
        setGlobalLoading: (loading) => set({ globalLoading: loading }),

        // Confirmation dialog
        confirmDialog: null,
        showConfirmDialog: (config) =>
          set({ confirmDialog: { ...config, isOpen: true } }),
        hideConfirmDialog: () => set({ confirmDialog: null }),
      }),
      {
        name: 'ptpd-ui-storage',
        partialize: (state) => ({
          sidebarCollapsed: state.sidebarCollapsed,
          theme: state.theme,
        }),
      }
    ),
    { name: 'UIStore' }
  )
);
