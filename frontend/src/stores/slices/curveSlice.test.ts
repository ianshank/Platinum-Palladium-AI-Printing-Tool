import { describe, it, expect, beforeEach } from 'vitest';
import { createStore } from '@/stores';
import type { CurveData, CurvePoint } from './curveSlice';

describe('curveSlice', () => {
  let store: ReturnType<typeof createStore>;

  const mockCurve: CurveData = {
    id: 'test-curve-1',
    name: 'Test Curve',
    createdAt: '2026-01-24T00:00:00Z',
    updatedAt: '2026-01-24T00:00:00Z',
    type: 'linear',
    points: [
      { x: 0, y: 0 },
      { x: 0.5, y: 0.5 },
      { x: 1, y: 1 },
    ],
  };

  beforeEach(() => {
    store = createStore();
  });

  describe('Initial State', () => {
    it('has correct initial values', () => {
      const state = store.getState();
      expect(state.curve.current).toBeNull();
      expect(state.curve.points).toEqual([]);
      expect(state.curve.isModified).toBe(false);
      expect(state.curve.modifications).toEqual([]);
      expect(state.curve.undoStack).toEqual([]);
      expect(state.curve.redoStack).toEqual([]);
      expect(state.curve.selectedPointIndex).toBeNull();
      expect(state.curve.isEditing).toBe(false);
    });
  });

  describe('Curve Management', () => {
    it('setCurve sets the current curve', () => {
      store.getState().curve.setCurve(mockCurve);

      expect(store.getState().curve.current).toEqual(mockCurve);
      expect(store.getState().curve.points).toEqual(mockCurve.points);
      expect(store.getState().curve.isModified).toBe(false);
    });

    it('setPoints updates curve points', () => {
      store.getState().curve.setCurve(mockCurve);
      const newPoints: CurvePoint[] = [
        { x: 0, y: 0 },
        { x: 0.25, y: 0.3 },
        { x: 0.75, y: 0.7 },
        { x: 1, y: 1 },
      ];

      store.getState().curve.setPoints(newPoints);

      expect(store.getState().curve.points).toEqual(newPoints);
      expect(store.getState().curve.isModified).toBe(true);
    });

    it('addPoint adds a point in sorted order', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.addPoint({ x: 0.25, y: 0.25 });

      const points = store.getState().curve.points;
      expect(points).toHaveLength(4);
      expect(points[1]).toEqual({ x: 0.25, y: 0.25 });
    });

    it('updatePoint updates a specific point', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.updatePoint(1, { y: 0.6 });

      expect(store.getState().curve.points[1]?.y).toBe(0.6);
      expect(store.getState().curve.isModified).toBe(true);
    });

    it('updatePoint clamps values to 0-1 range', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.updatePoint(1, { x: 1.5, y: -0.5 });

      expect(store.getState().curve.points[1]?.x).toBe(1);
      expect(store.getState().curve.points[1]?.y).toBe(0);
    });

    it('removePoint removes a point (not endpoints)', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.removePoint(1);

      expect(store.getState().curve.points).toHaveLength(2);
    });

    it('removePoint does not remove endpoints', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.removePoint(0); // Try to remove first point

      expect(store.getState().curve.points).toHaveLength(3);
    });
  });

  describe('Point Selection', () => {
    it('selectPoint selects a point', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.selectPoint(1);

      expect(store.getState().curve.selectedPointIndex).toBe(1);
    });

    it('selectPoint with null deselects', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.selectPoint(1);
      store.getState().curve.selectPoint(null);

      expect(store.getState().curve.selectedPointIndex).toBeNull();
    });
  });

  describe('Curve Modifications', () => {
    it('applyBrightness adjusts y values', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyBrightness(0.1);

      const points = store.getState().curve.points;
      expect(points[0]?.y).toBe(0.1);
      expect(points[1]?.y).toBe(0.6);
      expect(points[2]?.y).toBe(1); // Clamped to 1
    });

    it('applyContrast adjusts contrast', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyContrast(0.5);

      const points = store.getState().curve.points;
      expect(points[1]?.y).toBe(0.5); // Midpoint should stay at 0.5
    });

    it('applyGamma adjusts gamma curve', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyGamma(2.2);

      const points = store.getState().curve.points;
      expect(points[0]?.y).toBe(0); // Endpoints should remain
      expect(points[2]?.y).toBe(1);
    });

    it('applySmoothing smooths the curve', () => {
      const jaggedCurve: CurveData = {
        ...mockCurve,
        points: [
          { x: 0, y: 0 },
          { x: 0.25, y: 0.5 },
          { x: 0.5, y: 0.3 },
          { x: 0.75, y: 0.7 },
          { x: 1, y: 1 },
        ],
      };
      store.getState().curve.setCurve(jaggedCurve);
      store.getState().curve.applySmoothing(0.5);

      // Smoothing should reduce the difference between adjacent points
      const points = store.getState().curve.points;
      expect(points[0]?.y).toBe(0); // Endpoints unchanged
      expect(points[4]?.y).toBe(1);
    });

    it('records modifications', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyBrightness(0.1);
      store.getState().curve.applyContrast(0.2);

      const mods = store.getState().curve.modifications;
      expect(mods).toHaveLength(2);
      expect(mods[0]?.type).toBe('brightness');
      expect(mods[1]?.type).toBe('contrast');
    });
  });

  describe('Undo/Redo', () => {
    it('undo reverts to previous state', () => {
      store.getState().curve.setCurve(mockCurve);
      const originalPoints = [...store.getState().curve.points];

      store.getState().curve.applyBrightness(0.1);
      store.getState().curve.undo();

      expect(store.getState().curve.points).toEqual(originalPoints);
    });

    it('redo restores undone state', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyBrightness(0.1);
      const modifiedPoints = [...store.getState().curve.points];

      store.getState().curve.undo();
      store.getState().curve.redo();

      expect(store.getState().curve.points).toEqual(modifiedPoints);
    });

    it('canUndo returns correct value', () => {
      store.getState().curve.setCurve(mockCurve);
      expect(store.getState().curve.canUndo()).toBe(false);

      store.getState().curve.applyBrightness(0.1);
      expect(store.getState().curve.canUndo()).toBe(true);
    });

    it('canRedo returns correct value', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyBrightness(0.1);
      expect(store.getState().curve.canRedo()).toBe(false);

      store.getState().curve.undo();
      expect(store.getState().curve.canRedo()).toBe(true);
    });

    it('clearHistory clears undo/redo stacks', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyBrightness(0.1);
      store.getState().curve.undo();

      store.getState().curve.clearHistory();

      expect(store.getState().curve.undoStack).toEqual([]);
      expect(store.getState().curve.redoStack).toEqual([]);
    });
  });

  describe('Reset Operations', () => {
    it('revertToOriginal restores original curve', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyBrightness(0.1);
      store.getState().curve.applyContrast(0.2);

      store.getState().curve.revertToOriginal();

      expect(store.getState().curve.points).toEqual(mockCurve.points);
      expect(store.getState().curve.isModified).toBe(false);
      expect(store.getState().curve.modifications).toEqual([]);
    });

    it('clearCurve clears current curve', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.clearCurve();

      expect(store.getState().curve.current).toBeNull();
      expect(store.getState().curve.points).toEqual([]);
    });

    it('resetCurve resets all curve state', () => {
      store.getState().curve.setCurve(mockCurve);
      store.getState().curve.applyBrightness(0.1);
      store.getState().curve.selectPoint(1);

      store.getState().curve.resetCurve();

      expect(store.getState().curve.current).toBeNull();
      expect(store.getState().curve.points).toEqual([]);
      expect(store.getState().curve.isModified).toBe(false);
      expect(store.getState().curve.selectedPointIndex).toBeNull();
    });
  });
});
