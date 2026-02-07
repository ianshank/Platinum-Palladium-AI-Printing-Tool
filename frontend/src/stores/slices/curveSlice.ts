/**
 * Curve state slice
 * Manages calibration curves including points, modifications, and history
 */

import type { StateCreator } from 'zustand';
import { logger } from '@/lib/logger';

export interface CurvePoint {
  x: number;
  y: number;
  isControlPoint?: boolean;
}

export interface CurveData {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  type: 'linear' | 'cubic' | 'monotonic' | 'pchip';
  points: CurvePoint[];
  calibrationId?: string;
  metadata?: Record<string, unknown>;
}

export interface CurveModification {
  type: 'brightness' | 'contrast' | 'gamma' | 'smooth' | 'manual';
  value: number;
  timestamp: string;
}

export interface CurveSlice {
  // State
  current: CurveData | null;
  points: CurvePoint[];
  isModified: boolean;
  modifications: CurveModification[];
  undoStack: CurvePoint[][];
  redoStack: CurvePoint[][];
  maxUndoSteps: number;
  selectedPointIndex: number | null;
  isEditing: boolean;

  // Actions
  setCurve: (curve: CurveData) => void;
  setPoints: (points: CurvePoint[]) => void;
  addPoint: (point: CurvePoint) => void;
  updatePoint: (index: number, point: Partial<CurvePoint>) => void;
  removePoint: (index: number) => void;
  selectPoint: (index: number | null) => void;
  setEditing: (editing: boolean) => void;

  // Modifications
  applyBrightness: (value: number) => void;
  applyContrast: (value: number) => void;
  applyGamma: (value: number) => void;
  applySmoothing: (amount: number) => void;

  // History
  pushUndo: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
  clearHistory: () => void;

  // Reset
  revertToOriginal: () => void;
  clearCurve: () => void;
  resetCurve: () => void;
}

const initialState = {
  current: null as CurveData | null,
  points: [] as CurvePoint[],
  isModified: false,
  modifications: [] as CurveModification[],
  undoStack: [] as CurvePoint[][],
  redoStack: [] as CurvePoint[][],
  maxUndoSteps: 50,
  selectedPointIndex: null as number | null,
  isEditing: false,
};

/**
 * Clamp a value between 0 and 1
 */
function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export const createCurveSlice: StateCreator<
  { curve: CurveSlice },
  [['zustand/immer', never]],
  [],
  CurveSlice
> = (set, get) => ({
  ...initialState,

  setCurve: (curve) => {
    logger.debug('Curve: setCurve', { id: curve.id, pointCount: curve.points.length });
    set((state) => {
      state.curve.current = curve;
      state.curve.points = [...curve.points];
      state.curve.isModified = false;
      state.curve.modifications = [];
      state.curve.undoStack = [];
      state.curve.redoStack = [];
      state.curve.selectedPointIndex = null;
    });
  },

  setPoints: (points) => {
    logger.debug('Curve: setPoints', { count: points.length });
    get().curve.pushUndo();
    set((state) => {
      state.curve.points = points;
      state.curve.isModified = true;
      state.curve.redoStack = [];
    });
  },

  addPoint: (point) => {
    logger.debug('Curve: addPoint', { x: point.x, y: point.y });
    get().curve.pushUndo();
    set((state) => {
      // Insert point in sorted order by x
      const index = state.curve.points.findIndex((p) => p.x > point.x);
      if (index === -1) {
        state.curve.points.push(point);
      } else {
        state.curve.points.splice(index, 0, point);
      }
      state.curve.isModified = true;
      state.curve.redoStack = [];
    });
  },

  updatePoint: (index, point) => {
    logger.debug('Curve: updatePoint', { index, ...point });
    get().curve.pushUndo();
    set((state) => {
      if (index >= 0 && index < state.curve.points.length) {
        const currentPoint = state.curve.points[index];
        if (!currentPoint) return;
        state.curve.points[index] = {
          ...currentPoint,
          ...point,
          x: point.x !== undefined ? clamp01(point.x) : currentPoint.x,
          y: point.y !== undefined ? clamp01(point.y) : currentPoint.y,
        };
        state.curve.isModified = true;
        state.curve.redoStack = [];
      }
    });
  },

  removePoint: (index) => {
    logger.debug('Curve: removePoint', { index });
    const points = get().curve.points;
    // Don't allow removing endpoints
    if (index <= 0 || index >= points.length - 1) {
      logger.warn('Curve: cannot remove endpoint');
      return;
    }

    get().curve.pushUndo();
    set((state) => {
      state.curve.points.splice(index, 1);
      state.curve.isModified = true;
      state.curve.redoStack = [];
      if (state.curve.selectedPointIndex === index) {
        state.curve.selectedPointIndex = null;
      }
    });
  },

  selectPoint: (index) => {
    logger.debug('Curve: selectPoint', { index });
    set((state) => {
      state.curve.selectedPointIndex = index;
    });
  },

  setEditing: (editing) => {
    logger.debug('Curve: setEditing', { editing });
    set((state) => {
      state.curve.isEditing = editing;
    });
  },

  applyBrightness: (value) => {
    logger.debug('Curve: applyBrightness', { value });
    get().curve.pushUndo();
    set((state) => {
      state.curve.points = state.curve.points.map((p) => ({
        ...p,
        y: clamp01(p.y + value),
      }));
      state.curve.isModified = true;
      state.curve.modifications.push({
        type: 'brightness',
        value,
        timestamp: new Date().toISOString(),
      });
      state.curve.redoStack = [];
    });
  },

  applyContrast: (value) => {
    logger.debug('Curve: applyContrast', { value });
    get().curve.pushUndo();
    set((state) => {
      const factor = (1 + value);
      state.curve.points = state.curve.points.map((p) => ({
        ...p,
        y: clamp01((p.y - 0.5) * factor + 0.5),
      }));
      state.curve.isModified = true;
      state.curve.modifications.push({
        type: 'contrast',
        value,
        timestamp: new Date().toISOString(),
      });
      state.curve.redoStack = [];
    });
  },

  applyGamma: (value) => {
    logger.debug('Curve: applyGamma', { value });
    get().curve.pushUndo();
    set((state) => {
      const gamma = value;
      state.curve.points = state.curve.points.map((p) => ({
        ...p,
        y: clamp01(Math.pow(p.y, 1 / gamma)),
      }));
      state.curve.isModified = true;
      state.curve.modifications.push({
        type: 'gamma',
        value,
        timestamp: new Date().toISOString(),
      });
      state.curve.redoStack = [];
    });
  },

  applySmoothing: (amount) => {
    logger.debug('Curve: applySmoothing', { amount });
    get().curve.pushUndo();
    set((state) => {
      const points = state.curve.points;
      if (points.length < 3) return;

      // Simple moving average smoothing
      const smoothed = points.map((p, i) => {
        if (i === 0 || i === points.length - 1) return p; // Keep endpoints
        const prev = points[i - 1];
        const next = points[i + 1];
        if (!prev || !next) return p;
        return {
          ...p,
          y: clamp01(p.y * (1 - amount) + ((prev.y + next.y) / 2) * amount),
        };
      });

      state.curve.points = smoothed;
      state.curve.isModified = true;
      state.curve.modifications.push({
        type: 'smooth',
        value: amount,
        timestamp: new Date().toISOString(),
      });
      state.curve.redoStack = [];
    });
  },

  pushUndo: () => {
    const currentPoints = get().curve.points;
    set((state) => {
      state.curve.undoStack.push([...currentPoints]);
      if (state.curve.undoStack.length > state.curve.maxUndoSteps) {
        state.curve.undoStack.shift();
      }
    });
  },

  undo: () => {
    const undoStack = get().curve.undoStack;
    if (undoStack.length === 0) return;

    logger.debug('Curve: undo');
    const currentPoints = get().curve.points;
    const previousPoints = undoStack[undoStack.length - 1];

    if (previousPoints) {
      set((state) => {
        state.curve.redoStack.push([...currentPoints]);
        state.curve.points = previousPoints;
        state.curve.undoStack.pop();
        state.curve.isModified = true;
      });
    }
  },

  redo: () => {
    const redoStack = get().curve.redoStack;
    if (redoStack.length === 0) return;

    logger.debug('Curve: redo');
    const currentPoints = get().curve.points;
    const nextPoints = redoStack[redoStack.length - 1];

    if (nextPoints) {
      set((state) => {
        state.curve.undoStack.push([...currentPoints]);
        state.curve.points = nextPoints;
        state.curve.redoStack.pop();
        state.curve.isModified = true;
      });
    }
  },

  canUndo: () => get().curve.undoStack.length > 0,
  canRedo: () => get().curve.redoStack.length > 0,

  clearHistory: () => {
    logger.debug('Curve: clearHistory');
    set((state) => {
      state.curve.undoStack = [];
      state.curve.redoStack = [];
    });
  },

  revertToOriginal: () => {
    const current = get().curve.current;
    if (!current) return;

    logger.info('Curve: revertToOriginal');
    set((state) => {
      state.curve.points = [...current.points];
      state.curve.isModified = false;
      state.curve.modifications = [];
      state.curve.undoStack = [];
      state.curve.redoStack = [];
    });
  },

  clearCurve: () => {
    logger.debug('Curve: clearCurve');
    set((state) => {
      state.curve.current = null;
      state.curve.points = [];
      state.curve.isModified = false;
      state.curve.modifications = [];
      state.curve.selectedPointIndex = null;
    });
  },

  resetCurve: () => {
    logger.debug('Curve: resetCurve');
    set((state) => {
      Object.assign(state.curve, initialState);
    });
  },
});
