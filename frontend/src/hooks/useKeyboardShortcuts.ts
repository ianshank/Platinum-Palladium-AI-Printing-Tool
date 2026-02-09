/**
 * Keyboard shortcuts hook
 * Provides keyboard navigation matching legacy Gradio app (Ctrl+1-5 for tabs)
 */

import { useCallback, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';

interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  alt?: boolean;
  shift?: boolean;
  action: () => void;
  description: string;
}

/**
 * Register keyboard shortcuts
 */
export function useKeyboardShortcuts(shortcuts: ShortcutConfig[]): void {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement ||
        event.target instanceof HTMLSelectElement ||
        (event.target as HTMLElement).isContentEditable
      ) {
        return;
      }

      for (const shortcut of shortcuts) {
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlOrMeta = event.ctrlKey || event.metaKey;
        const ctrlMatch = ctrlOrMeta === !!shortcut.ctrl;
        const altMatch = event.altKey === !!shortcut.alt;
        const shiftMatch = event.shiftKey === !!shortcut.shift;

        if (keyMatch && ctrlMatch && altMatch && shiftMatch) {
          event.preventDefault();
          logger.debug('Keyboard shortcut triggered', {
            key: shortcut.key,
            description: shortcut.description,
          });
          shortcut.action();
          return;
        }
      }
    },
    [shortcuts]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

/**
 * Default application shortcuts matching legacy Gradio app
 */
export function useAppShortcuts(): void {
  const navigate = useNavigate();
  const undo = useStore((state) => state.curve.undo);
  const redo = useStore((state) => state.curve.redo);
  const canUndo = useStore((state) => state.curve.canUndo);
  const canRedo = useStore((state) => state.curve.canRedo);
  const addToast = useStore((state) => state.ui.addToast);

  const shortcuts: ShortcutConfig[] = useMemo(
    () => [
      // Tab navigation (matching Gradio Ctrl+1-5)
      {
        key: '1',
        ctrl: true,
        action: () => navigate('/'),
        description: 'Navigate to Dashboard',
      },
      {
        key: '2',
        ctrl: true,
        action: () => navigate('/calibration'),
        description: 'Navigate to Calibration',
      },
      {
        key: '3',
        ctrl: true,
        action: () => navigate('/curves'),
        description: 'Navigate to Curves',
      },
      {
        key: '4',
        ctrl: true,
        action: () => navigate('/chemistry'),
        description: 'Navigate to Chemistry',
      },
      {
        key: '5',
        ctrl: true,
        action: () => navigate('/assistant'),
        description: 'Navigate to AI Assistant',
      },
      // Undo/Redo
      {
        key: 'z',
        ctrl: true,
        action: () => {
          if (canUndo()) {
            undo();
            addToast({ title: 'Undo', variant: 'default', duration: 1000 });
          }
        },
        description: 'Undo',
      },
      {
        key: 'z',
        ctrl: true,
        shift: true,
        action: () => {
          if (canRedo()) {
            redo();
            addToast({ title: 'Redo', variant: 'default', duration: 1000 });
          }
        },
        description: 'Redo',
      },
      {
        key: 'y',
        ctrl: true,
        action: () => {
          if (canRedo()) {
            redo();
            addToast({ title: 'Redo', variant: 'default', duration: 1000 });
          }
        },
        description: 'Redo (alternative)',
      },
    ],
    [navigate, undo, redo, canUndo, canRedo, addToast]
  );

  useKeyboardShortcuts(shortcuts);
}

export type { ShortcutConfig };
