/**
 * Keyboard shortcuts hook
 * Provides keyboard navigation matching legacy Gradio app (Ctrl+1-5 for tabs)
 */

import { useCallback, useEffect } from 'react';
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
 * Tag names of form controls that own keyboard input. Shortcuts are
 * suppressed while one of these has focus so typing (or e.g. Ctrl+Z inside a
 * field) is never hijacked. Add new element types here, one line each.
 */
const EDITABLE_TAG_NAMES: ReadonlySet<string> = new Set([
  'INPUT',
  'TEXTAREA',
  'SELECT',
]);

/**
 * Whether a keyboard event target is an editable control (input, textarea,
 * select or a contentEditable region) that must keep its keystrokes.
 */
export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) {
    return false;
  }
  if (EDITABLE_TAG_NAMES.has(target.tagName.toUpperCase())) {
    return true;
  }
  return target instanceof HTMLElement && target.isContentEditable === true;
}

/**
 * Whether a keydown event matches a shortcut's key and modifier combination.
 * Ctrl and Meta (Cmd) are treated as the same modifier.
 */
function matchesShortcut(
  event: KeyboardEvent,
  shortcut: ShortcutConfig
): boolean {
  const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
  const ctrlPressed = event.ctrlKey || event.metaKey;
  const requiresCtrl = shortcut.ctrl ?? false;
  const ctrlMatch = requiresCtrl ? ctrlPressed : !ctrlPressed;
  const altMatch = event.altKey === !!shortcut.alt;
  const shiftMatch = event.shiftKey === !!shortcut.shift;
  return keyMatch && ctrlMatch && altMatch && shiftMatch;
}

/**
 * Register keyboard shortcuts
 */
export function useKeyboardShortcuts(shortcuts: ShortcutConfig[]): void {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const shortcut = shortcuts.find((candidate) =>
        matchesShortcut(event, candidate)
      );
      if (!shortcut) {
        return;
      }

      // Don't trigger shortcuts when typing in inputs, selects or editable regions
      if (isEditableTarget(event.target)) {
        logger.debug('Keyboard shortcut suppressed: editable target focused', {
          key: shortcut.key,
          description: shortcut.description,
          target: (event.target as Element).tagName,
        });
        return;
      }

      event.preventDefault();
      logger.debug('Keyboard shortcut triggered', {
        key: shortcut.key,
        description: shortcut.description,
      });
      shortcut.action();
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

  const shortcuts: ShortcutConfig[] = [
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
  ];

  useKeyboardShortcuts(shortcuts);
}

export type { ShortcutConfig };
