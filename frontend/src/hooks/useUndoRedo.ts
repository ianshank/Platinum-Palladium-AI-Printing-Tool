/**
 * Generic undo/redo hook for managing state history
 * Used by CurveEditor to track output value changes
 */

import { useCallback, useState } from 'react';
import { config } from '@/config';
import { logger } from '@/lib/logger';

interface UndoRedoState<T> {
  current: T;
  undoStack: T[];
  redoStack: T[];
}

export interface UseUndoRedoReturn<T> {
  state: T;
  setState: (value: T) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  reset: (value: T) => void;
}

/** Enforce a maximum stack size, trimming from the oldest end. */
function capStack<T>(stack: T[], limit: number): T[] {
  if (stack.length > limit) {
    return stack.slice(stack.length - limit);
  }
  return stack;
}

/**
 * Hook that wraps a state value with undo/redo history tracking.
 *
 * @param initialValue - The initial state value
 * @param maxHistory - Maximum number of undo steps to retain (defaults to config.ui.undoHistoryLimit)
 */
export function useUndoRedo<T>(
  initialValue: T,
  maxHistory: number = config.ui.undoHistoryLimit
): UseUndoRedoReturn<T> {
  const safeMax = Math.max(1, Math.floor(maxHistory));

  const [history, setHistory] = useState<UndoRedoState<T>>({
    current: initialValue,
    undoStack: [],
    redoStack: [],
  });

  const setState = useCallback(
    (value: T) => {
      setHistory((prev) => {
        const undoStack = capStack([...prev.undoStack, prev.current], safeMax);
        logger.debug('UndoRedo: setState', {
          undoDepth: undoStack.length,
        });
        return {
          current: value,
          undoStack,
          redoStack: [], // Clear redo on new change
        };
      });
    },
    [safeMax]
  );

  const undo = useCallback(() => {
    setHistory((prev) => {
      if (prev.undoStack.length === 0) return prev;
      const undoStack = [...prev.undoStack];
      const previous = undoStack.pop()!;
      logger.debug('UndoRedo: undo', {
        undoDepth: undoStack.length,
        redoDepth: prev.redoStack.length + 1,
      });
      return {
        current: previous,
        undoStack,
        redoStack: capStack([...prev.redoStack, prev.current], safeMax),
      };
    });
  }, [safeMax]);

  const redo = useCallback(() => {
    setHistory((prev) => {
      if (prev.redoStack.length === 0) return prev;
      const redoStack = [...prev.redoStack];
      const next = redoStack.pop()!;
      logger.debug('UndoRedo: redo', {
        undoDepth: prev.undoStack.length + 1,
        redoDepth: redoStack.length,
      });
      return {
        current: next,
        undoStack: capStack([...prev.undoStack, prev.current], safeMax),
        redoStack,
      };
    });
  }, [safeMax]);

  const reset = useCallback((value: T) => {
    logger.debug('UndoRedo: reset');
    setHistory({
      current: value,
      undoStack: [],
      redoStack: [],
    });
  }, []);

  return {
    state: history.current,
    setState,
    undo,
    redo,
    canUndo: history.undoStack.length > 0,
    canRedo: history.redoStack.length > 0,
    reset,
  };
}
