import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createStore } from '@/stores';

describe('uiSlice', () => {
  let store: ReturnType<typeof createStore>;

  beforeEach(() => {
    store = createStore();
    vi.clearAllMocks();
  });

  describe('Initial State', () => {
    it('has correct initial values', () => {
      const state = store.getState();
      expect(state.ui.activeTab).toBe('dashboard');
      expect(state.ui.sidebarOpen).toBe(true);
      expect(state.ui.isProcessing).toBe(false);
      expect(state.ui.isInitialized).toBe(false);
      expect(state.ui.theme).toBe('dark');
      expect(state.ui.toasts).toEqual([]);
      expect(state.ui.modals).toEqual({});
    });
  });

  describe('Tab Navigation', () => {
    it('setActiveTab updates active tab', () => {
      store.getState().ui.setActiveTab('calibration');
      expect(store.getState().ui.activeTab).toBe('calibration');
    });
  });

  describe('Sidebar', () => {
    it('toggleSidebar toggles sidebar state', () => {
      expect(store.getState().ui.sidebarOpen).toBe(true);
      store.getState().ui.toggleSidebar();
      expect(store.getState().ui.sidebarOpen).toBe(false);
      store.getState().ui.toggleSidebar();
      expect(store.getState().ui.sidebarOpen).toBe(true);
    });

    it('setSidebarOpen sets sidebar state', () => {
      store.getState().ui.setSidebarOpen(false);
      expect(store.getState().ui.sidebarOpen).toBe(false);
      store.getState().ui.setSidebarOpen(true);
      expect(store.getState().ui.sidebarOpen).toBe(true);
    });
  });

  describe('Processing State', () => {
    it('setProcessing updates processing state', () => {
      store.getState().ui.setProcessing(true);
      expect(store.getState().ui.isProcessing).toBe(true);
      store.getState().ui.setProcessing(false);
      expect(store.getState().ui.isProcessing).toBe(false);
    });
  });

  describe('Theme', () => {
    it('setTheme updates theme', () => {
      store.getState().ui.setTheme('light');
      expect(store.getState().ui.theme).toBe('light');
      store.getState().ui.setTheme('dark');
      expect(store.getState().ui.theme).toBe('dark');
    });

    it('toggleTheme switches between light and dark', () => {
      expect(store.getState().ui.theme).toBe('dark');
      store.getState().ui.toggleTheme();
      expect(store.getState().ui.theme).toBe('light');
      store.getState().ui.toggleTheme();
      expect(store.getState().ui.theme).toBe('dark');
    });
  });

  describe('Toasts', () => {
    it('addToast adds a toast', () => {
      store.getState().ui.addToast({
        title: 'Test Toast',
        description: 'This is a test',
        variant: 'success',
      });

      const toasts = store.getState().ui.toasts;
      expect(toasts).toHaveLength(1);
      expect(toasts[0]?.title).toBe('Test Toast');
      expect(toasts[0]?.description).toBe('This is a test');
      expect(toasts[0]?.variant).toBe('success');
      expect(toasts[0]?.id).toBeDefined();
    });

    it('removeToast removes a toast by id', () => {
      store.getState().ui.addToast({ title: 'Toast 1' });
      store.getState().ui.addToast({ title: 'Toast 2' });

      const toasts = store.getState().ui.toasts;
      const toastId = toasts[0]?.id;

      if (toastId) {
        store.getState().ui.removeToast(toastId);
      }

      expect(store.getState().ui.toasts).toHaveLength(1);
      expect(store.getState().ui.toasts[0]?.title).toBe('Toast 2');
    });

    it('clearToasts removes all toasts', () => {
      store.getState().ui.addToast({ title: 'Toast 1' });
      store.getState().ui.addToast({ title: 'Toast 2' });
      store.getState().ui.clearToasts();

      expect(store.getState().ui.toasts).toHaveLength(0);
    });
  });

  describe('Modals', () => {
    it('openModal opens a modal', () => {
      store.getState().ui.openModal('settings');
      expect(store.getState().ui.modals['settings']).toBe(true);
    });

    it('closeModal closes a modal', () => {
      store.getState().ui.openModal('settings');
      store.getState().ui.closeModal('settings');
      expect(store.getState().ui.modals['settings']).toBe(false);
    });

    it('toggleModal toggles a modal', () => {
      store.getState().ui.toggleModal('settings');
      expect(store.getState().ui.modals['settings']).toBe(true);
      store.getState().ui.toggleModal('settings');
      expect(store.getState().ui.modals['settings']).toBe(false);
    });
  });

  describe('Initialization', () => {
    it('initializeApp sets isInitialized to true', () => {
      expect(store.getState().ui.isInitialized).toBe(false);
      store.getState().ui.initializeApp();
      expect(store.getState().ui.isInitialized).toBe(true);
    });
  });

  describe('Reset', () => {
    it('resetUI resets UI state', () => {
      store.getState().ui.setActiveTab('curves');
      store.getState().ui.setProcessing(true);
      store.getState().ui.addToast({ title: 'Test' });
      store.getState().ui.openModal('test');

      store.getState().ui.resetUI();

      expect(store.getState().ui.activeTab).toBe('dashboard');
      expect(store.getState().ui.isProcessing).toBe(false);
      expect(store.getState().ui.toasts).toHaveLength(0);
      expect(store.getState().ui.modals).toEqual({});
    });
  });
});
