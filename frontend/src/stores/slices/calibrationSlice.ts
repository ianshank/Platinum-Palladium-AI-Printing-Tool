/**
 * Calibration state slice
 * Manages calibration workflow state including step tablet data and calibration history
 */

import type { StateCreator } from 'zustand';
import { logger } from '@/lib/logger';

export interface DensityMeasurement {
  step: number;
  targetDensity: number;
  measuredDensity: number;
  deltaE?: number;
}

/**
 * Typed metadata for calibration workflow.
 * Known keys are typed explicitly; additional keys are allowed via index signature.
 */
export interface CalibrationMetadata {
  // Scan analysis metrics
  dmin?: number;
  dmax?: number;
  range?: number;
  num_patches?: number;
  originalFileName?: string;

  // Configuration preferences
  linearizationMode?: string;
  targetResponse?: string;
  curveStrategy?: string;

  // Extensible for future keys
  [key: string]: unknown;
}

export interface CalibrationData {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  tabletType: '21-step' | '31-step' | '41-step' | 'custom';
  measurements: DensityMeasurement[];
  curveId?: string | undefined;
  notes?: string | undefined;
  metadata?: CalibrationMetadata | undefined;
}

export interface CalibrationSlice {
  // State
  current: CalibrationData | null;
  currentStep: number;
  totalSteps: number;
  history: CalibrationData[];
  isAnalyzing: boolean;
  analysisProgress: number;
  error: string | null;

  // Actions
  startCalibration: (tabletType: CalibrationData['tabletType']) => void;
  setCurrentStep: (step: number) => void;
  nextStep: () => void;
  previousStep: () => void;
  addMeasurement: (measurement: DensityMeasurement) => void;
  updateMeasurement: (step: number, measurement: Partial<DensityMeasurement>) => void;
  setMeasurements: (measurements: DensityMeasurement[]) => void;
  setAnalyzing: (analyzing: boolean) => void;
  setAnalysisProgress: (progress: number) => void;
  setError: (error: string | null) => void;
  saveCalibration: (name: string, notes?: string) => void;
  loadCalibration: (id: string) => void;
  deleteCalibration: (id: string) => void;
  clearCurrent: () => void;
  resetCalibration: () => void;
  updateMetadata: (metadata: CalibrationMetadata) => void;
}

const WIZARD_STEPS = 5; // Total wizard steps

const initialState = {
  current: null as CalibrationData | null,
  currentStep: 0,
  totalSteps: WIZARD_STEPS,
  history: [] as CalibrationData[],
  isAnalyzing: false,
  analysisProgress: 0,
  error: null as string | null,
};

export const createCalibrationSlice: StateCreator<
  { calibration: CalibrationSlice },
  [['zustand/immer', never]],
  [],
  CalibrationSlice
> = (set, get) => ({
  ...initialState,

  startCalibration: (tabletType) => {
    logger.debug('Calibration: startCalibration', { tabletType });
    const id = `cal-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    const now = new Date().toISOString();

    set((state) => {
      state.calibration.current = {
        id,
        name: '',
        createdAt: now,
        updatedAt: now,
        tabletType,
        measurements: [],
      };
      state.calibration.currentStep = 0;
      state.calibration.error = null;
    });
  },

  setCurrentStep: (step) => {
    const total = get().calibration.totalSteps;
    const clampedStep = Math.max(0, Math.min(step, total - 1));
    logger.debug('Calibration: setCurrentStep', { step, clampedStep });
    set((state) => {
      state.calibration.currentStep = clampedStep;
    });
  },

  nextStep: () => {
    const current = get().calibration.currentStep;
    const total = get().calibration.totalSteps;
    if (current < total - 1) {
      logger.debug('Calibration: nextStep', { from: current, to: current + 1 });
      set((state) => {
        state.calibration.currentStep = current + 1;
      });
    }
  },

  previousStep: () => {
    const current = get().calibration.currentStep;
    if (current > 0) {
      logger.debug('Calibration: previousStep', { from: current, to: current - 1 });
      set((state) => {
        state.calibration.currentStep = current - 1;
      });
    }
  },

  addMeasurement: (measurement) => {
    logger.debug('Calibration: addMeasurement', { step: measurement.step });
    set((state) => {
      if (state.calibration.current) {
        state.calibration.current.measurements.push(measurement);
        state.calibration.current.updatedAt = new Date().toISOString();
      }
    });
  },

  updateMeasurement: (step, measurement) => {
    logger.debug('Calibration: updateMeasurement', { step, measurement });
    set((state) => {
      if (state.calibration.current) {
        const index = state.calibration.current.measurements.findIndex(
          (m) => m.step === step
        );
        if (index !== -1) {
          state.calibration.current.measurements[index] = {
            ...state.calibration.current.measurements[index],
            ...measurement as DensityMeasurement,
          };
          state.calibration.current.updatedAt = new Date().toISOString();
        }
      }
    });
  },

  setMeasurements: (measurements) => {
    logger.debug('Calibration: setMeasurements', { count: measurements.length });
    set((state) => {
      if (state.calibration.current) {
        state.calibration.current.measurements = measurements;
        state.calibration.current.updatedAt = new Date().toISOString();
      }
    });
  },

  setAnalyzing: (analyzing) => {
    logger.debug('Calibration: setAnalyzing', { analyzing });
    set((state) => {
      state.calibration.isAnalyzing = analyzing;
      if (!analyzing) {
        state.calibration.analysisProgress = 0;
      }
    });
  },

  setAnalysisProgress: (progress) => {
    set((state) => {
      state.calibration.analysisProgress = Math.max(0, Math.min(100, progress));
    });
  },

  setError: (error) => {
    if (error) {
      logger.error('Calibration: error', { error });
    }
    set((state) => {
      state.calibration.error = error;
    });
  },

  saveCalibration: (name, notes) => {
    const current = get().calibration.current;
    if (!current) {
      logger.warn('Calibration: saveCalibration - no current calibration');
      return;
    }

    logger.info('Calibration: saveCalibration', { name });
    set((state) => {
      if (state.calibration.current) {
        state.calibration.current.name = name;
        state.calibration.current.notes = notes;
        state.calibration.current.updatedAt = new Date().toISOString();

        // Add to history
        const existingIndex = state.calibration.history.findIndex(
          (c) => c.id === state.calibration.current!.id
        );
        if (existingIndex !== -1) {
          state.calibration.history[existingIndex] = state.calibration.current;
        } else {
          state.calibration.history.unshift(state.calibration.current);
        }
      }
    });
  },

  loadCalibration: (id) => {
    logger.debug('Calibration: loadCalibration', { id });
    const calibration = get().calibration.history.find((c) => c.id === id);
    if (calibration) {
      set((state) => {
        state.calibration.current = { ...calibration };
        state.calibration.currentStep = 0;
        state.calibration.error = null;
      });
    } else {
      logger.warn('Calibration: loadCalibration - not found', { id });
    }
  },

  deleteCalibration: (id) => {
    logger.info('Calibration: deleteCalibration', { id });
    set((state) => {
      state.calibration.history = state.calibration.history.filter(
        (c) => c.id !== id
      );
      if (state.calibration.current?.id === id) {
        state.calibration.current = null;
      }
    });
  },

  clearCurrent: () => {
    logger.debug('Calibration: clearCurrent');
    set((state) => {
      state.calibration.current = null;
      state.calibration.currentStep = 0;
      state.calibration.error = null;
    });
  },

  resetCalibration: () => {
    logger.debug('Calibration: resetCalibration');
    set((state) => {
      state.calibration.current = initialState.current;
      state.calibration.currentStep = initialState.currentStep;
      state.calibration.isAnalyzing = initialState.isAnalyzing;
      state.calibration.analysisProgress = initialState.analysisProgress;
      state.calibration.error = initialState.error;
    });
  },

  updateMetadata: (metadata) => {
    logger.debug('Calibration: updateMetadata', { metadata });
    set((state) => {
      if (state.calibration.current) {
        state.calibration.current.metadata = {
          ...state.calibration.current.metadata,
          ...metadata
        };
        state.calibration.current.updatedAt = new Date().toISOString();
      }
    });
  },
});
