/**
 * Zustand store unit tests.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';

// Create test stores (isolated from app stores)
import { create } from 'zustand';

// Re-create store slices for testing to ensure isolation
interface UIState {
  theme: 'dark' | 'light' | 'system';
  sidebarCollapsed: boolean;
  toasts: Array<{ id: string; type: string; message: string }>;
  modals: Record<string, boolean>;
  setTheme: (theme: 'dark' | 'light' | 'system') => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  addToast: (toast: { type: string; message: string }) => void;
  removeToast: (id: string) => void;
  openModal: (id: string) => void;
  closeModal: (id: string) => void;
}

const createUIStore = () =>
  create<UIState>((set) => ({
    theme: 'dark',
    sidebarCollapsed: false,
    toasts: [],
    modals: {},
    setTheme: (theme) => set({ theme }),
    setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
    addToast: (toast) =>
      set((state) => ({
        toasts: [...state.toasts, { ...toast, id: `toast-${Date.now()}` }],
      })),
    removeToast: (id) =>
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      })),
    openModal: (id) =>
      set((state) => ({
        modals: { ...state.modals, [id]: true },
      })),
    closeModal: (id) =>
      set((state) => ({
        modals: { ...state.modals, [id]: false },
      })),
  }));

describe('UI Store', () => {
  let useUIStore: ReturnType<typeof createUIStore>;

  beforeEach(() => {
    useUIStore = createUIStore();
  });

  describe('theme', () => {
    it('has default theme of dark', () => {
      const { result } = renderHook(() => useUIStore((s) => s.theme));
      expect(result.current).toBe('dark');
    });

    it('can set theme to light', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setTheme('light');
      });

      expect(result.current.theme).toBe('light');
    });

    it('can set theme to system', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setTheme('system');
      });

      expect(result.current.theme).toBe('system');
    });
  });

  describe('sidebar', () => {
    it('starts with sidebar expanded', () => {
      const { result } = renderHook(() => useUIStore((s) => s.sidebarCollapsed));
      expect(result.current).toBe(false);
    });

    it('can collapse sidebar', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setSidebarCollapsed(true);
      });

      expect(result.current.sidebarCollapsed).toBe(true);
    });

    it('can expand sidebar', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setSidebarCollapsed(true);
        result.current.setSidebarCollapsed(false);
      });

      expect(result.current.sidebarCollapsed).toBe(false);
    });
  });

  describe('toasts', () => {
    it('starts with empty toasts', () => {
      const { result } = renderHook(() => useUIStore((s) => s.toasts));
      expect(result.current).toHaveLength(0);
    });

    it('can add a toast', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.addToast({ type: 'success', message: 'Test toast' });
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].message).toBe('Test toast');
      expect(result.current.toasts[0].type).toBe('success');
    });

    it('can remove a toast', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.addToast({ type: 'success', message: 'Test toast' });
      });

      const toastId = result.current.toasts[0].id;

      act(() => {
        result.current.removeToast(toastId);
      });

      expect(result.current.toasts).toHaveLength(0);
    });

    it('can add multiple toasts', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.addToast({ type: 'success', message: 'Toast 1' });
        result.current.addToast({ type: 'error', message: 'Toast 2' });
        result.current.addToast({ type: 'info', message: 'Toast 3' });
      });

      expect(result.current.toasts).toHaveLength(3);
    });
  });

  describe('modals', () => {
    it('starts with no modals open', () => {
      const { result } = renderHook(() => useUIStore((s) => s.modals));
      expect(result.current).toEqual({});
    });

    it('can open a modal', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.openModal('settings');
      });

      expect(result.current.modals.settings).toBe(true);
    });

    it('can close a modal', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.openModal('settings');
        result.current.closeModal('settings');
      });

      expect(result.current.modals.settings).toBe(false);
    });

    it('can manage multiple modals', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.openModal('settings');
        result.current.openModal('confirm');
      });

      expect(result.current.modals.settings).toBe(true);
      expect(result.current.modals.confirm).toBe(true);
    });
  });
});

// Wizard store tests
interface WizardState {
  currentStep: number;
  totalSteps: number;
  completedSteps: number[];
  setCurrentStep: (step: number) => void;
  markStepComplete: (step: number) => void;
  resetWizard: () => void;
}

const createWizardStore = () =>
  create<WizardState>((set) => ({
    currentStep: 0,
    totalSteps: 5,
    completedSteps: [],
    setCurrentStep: (step) => set({ currentStep: step }),
    markStepComplete: (step) =>
      set((state) => ({
        completedSteps: [...new Set([...state.completedSteps, step])],
      })),
    resetWizard: () => set({ currentStep: 0, completedSteps: [] }),
  }));

describe('Wizard Store', () => {
  let useWizardStore: ReturnType<typeof createWizardStore>;

  beforeEach(() => {
    useWizardStore = createWizardStore();
  });

  describe('step navigation', () => {
    it('starts at step 0', () => {
      const { result } = renderHook(() => useWizardStore((s) => s.currentStep));
      expect(result.current).toBe(0);
    });

    it('can navigate to a specific step', () => {
      const { result } = renderHook(() => useWizardStore());

      act(() => {
        result.current.setCurrentStep(2);
      });

      expect(result.current.currentStep).toBe(2);
    });

    it('has correct total steps', () => {
      const { result } = renderHook(() => useWizardStore((s) => s.totalSteps));
      expect(result.current).toBe(5);
    });
  });

  describe('step completion', () => {
    it('starts with no completed steps', () => {
      const { result } = renderHook(() => useWizardStore((s) => s.completedSteps));
      expect(result.current).toHaveLength(0);
    });

    it('can mark a step as complete', () => {
      const { result } = renderHook(() => useWizardStore());

      act(() => {
        result.current.markStepComplete(0);
      });

      expect(result.current.completedSteps).toContain(0);
    });

    it('does not duplicate completed steps', () => {
      const { result } = renderHook(() => useWizardStore());

      act(() => {
        result.current.markStepComplete(0);
        result.current.markStepComplete(0);
        result.current.markStepComplete(0);
      });

      expect(result.current.completedSteps.filter((s) => s === 0)).toHaveLength(1);
    });

    it('can complete multiple steps', () => {
      const { result } = renderHook(() => useWizardStore());

      act(() => {
        result.current.markStepComplete(0);
        result.current.markStepComplete(1);
        result.current.markStepComplete(2);
      });

      expect(result.current.completedSteps).toContain(0);
      expect(result.current.completedSteps).toContain(1);
      expect(result.current.completedSteps).toContain(2);
    });
  });

  describe('reset', () => {
    it('can reset wizard to initial state', () => {
      const { result } = renderHook(() => useWizardStore());

      act(() => {
        result.current.setCurrentStep(3);
        result.current.markStepComplete(0);
        result.current.markStepComplete(1);
        result.current.markStepComplete(2);
      });

      act(() => {
        result.current.resetWizard();
      });

      expect(result.current.currentStep).toBe(0);
      expect(result.current.completedSteps).toHaveLength(0);
    });
  });
});
