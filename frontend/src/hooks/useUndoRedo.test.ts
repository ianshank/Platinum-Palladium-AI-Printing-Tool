import { describe, expect, it } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useUndoRedo } from './useUndoRedo';

describe('useUndoRedo', () => {
  it('initializes with the given value', () => {
    const { result } = renderHook(() => useUndoRedo([1, 2, 3]));
    expect(result.current.state).toEqual([1, 2, 3]);
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
  });

  it('updates state and tracks history', () => {
    const { result } = renderHook(() => useUndoRedo([0, 0, 0]));

    act(() => {
      result.current.setState([1, 1, 1]);
    });

    expect(result.current.state).toEqual([1, 1, 1]);
    expect(result.current.canUndo).toBe(true);
    expect(result.current.canRedo).toBe(false);
  });

  it('undoes to previous state', () => {
    const { result } = renderHook(() => useUndoRedo('A'));

    act(() => {
      result.current.setState('B');
    });
    act(() => {
      result.current.setState('C');
    });

    expect(result.current.state).toBe('C');

    act(() => {
      result.current.undo();
    });

    expect(result.current.state).toBe('B');
    expect(result.current.canUndo).toBe(true);
    expect(result.current.canRedo).toBe(true);

    act(() => {
      result.current.undo();
    });

    expect(result.current.state).toBe('A');
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(true);
  });

  it('redoes to next state', () => {
    const { result } = renderHook(() => useUndoRedo(0));

    act(() => {
      result.current.setState(10);
    });
    act(() => {
      result.current.setState(20);
    });
    act(() => {
      result.current.undo();
    });
    act(() => {
      result.current.undo();
    });

    expect(result.current.state).toBe(0);

    act(() => {
      result.current.redo();
    });

    expect(result.current.state).toBe(10);

    act(() => {
      result.current.redo();
    });

    expect(result.current.state).toBe(20);
    expect(result.current.canRedo).toBe(false);
  });

  it('clears redo stack on new change after undo', () => {
    const { result } = renderHook(() => useUndoRedo(0));

    act(() => {
      result.current.setState(10);
    });
    act(() => {
      result.current.setState(20);
    });
    act(() => {
      result.current.undo();
    });

    expect(result.current.canRedo).toBe(true);

    // New change should clear redo
    act(() => {
      result.current.setState(15);
    });

    expect(result.current.state).toBe(15);
    expect(result.current.canRedo).toBe(false);
    expect(result.current.canUndo).toBe(true);
  });

  it('does nothing when undoing with empty stack', () => {
    const { result } = renderHook(() => useUndoRedo('initial'));

    act(() => {
      result.current.undo();
    });

    expect(result.current.state).toBe('initial');
  });

  it('does nothing when redoing with empty stack', () => {
    const { result } = renderHook(() => useUndoRedo('initial'));

    act(() => {
      result.current.redo();
    });

    expect(result.current.state).toBe('initial');
  });

  it('respects maxHistory limit', () => {
    const { result } = renderHook(() => useUndoRedo(0, 3));

    // Push 5 states, but maxHistory is 3
    for (let i = 1; i <= 5; i++) {
      act(() => {
        result.current.setState(i);
      });
    }

    expect(result.current.state).toBe(5);

    // Should only be able to undo 3 times (maxHistory)
    act(() => result.current.undo());
    expect(result.current.state).toBe(4);

    act(() => result.current.undo());
    expect(result.current.state).toBe(3);

    act(() => result.current.undo());
    expect(result.current.state).toBe(2);

    // Can't undo further
    act(() => result.current.undo());
    expect(result.current.state).toBe(2);
    expect(result.current.canUndo).toBe(false);
  });

  it('resets state and clears all history', () => {
    const { result } = renderHook(() => useUndoRedo([0, 0]));

    act(() => {
      result.current.setState([1, 1]);
    });
    act(() => {
      result.current.setState([2, 2]);
    });

    expect(result.current.canUndo).toBe(true);

    act(() => {
      result.current.reset([5, 5]);
    });

    expect(result.current.state).toEqual([5, 5]);
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
  });

  it('works with complex objects', () => {
    const initial = { points: [{ x: 0, y: 0 }], name: 'test' };
    const { result } = renderHook(() => useUndoRedo(initial));

    const updated = { points: [{ x: 0, y: 0 }, { x: 1, y: 1 }], name: 'test' };
    act(() => {
      result.current.setState(updated);
    });

    expect(result.current.state).toEqual(updated);

    act(() => {
      result.current.undo();
    });

    expect(result.current.state).toEqual(initial);
  });

  it('handles rapid undo/redo cycles correctly', () => {
    const { result } = renderHook(() => useUndoRedo(0));

    act(() => result.current.setState(1));
    act(() => result.current.setState(2));
    act(() => result.current.setState(3));

    // Undo all
    act(() => result.current.undo());
    act(() => result.current.undo());
    act(() => result.current.undo());
    expect(result.current.state).toBe(0);

    // Redo all
    act(() => result.current.redo());
    act(() => result.current.redo());
    act(() => result.current.redo());
    expect(result.current.state).toBe(3);

    // Undo one, set new
    act(() => result.current.undo());
    act(() => result.current.setState(99));
    expect(result.current.state).toBe(99);
    expect(result.current.canRedo).toBe(false);

    act(() => result.current.undo());
    expect(result.current.state).toBe(2);
  });
});
