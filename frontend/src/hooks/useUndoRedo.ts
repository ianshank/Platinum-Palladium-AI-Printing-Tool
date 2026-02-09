/**
 * Generic undo/redo hook for managing state history
 * Used by CurveEditor to track output value changes
 */

import { useCallback, useRef, useState } from 'react';

interface UndoRedoState<T> {
  current: T;
  undoStack: T[];
  redoStack: T[];
}

interface UseUndoRedoReturn<T> {
  state: T;
  setState: (value: T) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  reset: (value: T) => void;
}

/**
 * Hook that wraps a state value with undo/redo history tracking.
 *
 * @param initialValue - The initial state value
 * @param maxHistory - Maximum number of undo steps to retain (default 50)
 */
export function useUndoRedo<T>(
  initialValue: T,
  maxHistory = 50
): UseUndoRedoReturn<T> {
  const [history, setHistory] = useState<UndoRedoState<T>>({
    current: initialValue,
    undoStack: [],
    redoStack: [],
  });

  // Track whether we've received initial value changes from props
  const initialRef = useRef(initialValue);

  const setState = useCallback(
    (value: T) => {
      setHistory((prev) => {
        const undoStack = [...prev.undoStack, prev.current];
        if (undoStack.length > maxHistory) {
          undoStack.shift();
        }
        return {
          current: value,
          undoStack,
          redoStack: [], // Clear redo on new change
        };
      });
    },
    [maxHistory]
  );

  const undo = useCallback(() => {
    setHistory((prev) => {
      if (prev.undoStack.length === 0) return prev;
      const undoStack = [...prev.undoStack];
      const previous = undoStack.pop()!;
      return {
        current: previous,
        undoStack,
        redoStack: [...prev.redoStack, prev.current],
      };
    });
  }, []);

  const redo = useCallback(() => {
    setHistory((prev) => {
      if (prev.redoStack.length === 0) return prev;
      const redoStack = [...prev.redoStack];
      const next = redoStack.pop()!;
      return {
        current: next,
        undoStack: [...prev.undoStack, prev.current],
        redoStack,
      };
    });
  }, []);

  const reset = useCallback((value: T) => {
    initialRef.current = value;
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
