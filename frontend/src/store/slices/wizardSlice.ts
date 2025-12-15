/**
 * Calibration wizard state management with Zustand.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { CurveData, ScanUploadResponse } from '@/types';

interface WizardState {
  // Current step (1-5)
  currentStep: number;
  maxStepReached: number;

  // Step 1: Upload
  uploadedFile: File | null;
  uploadedFileName: string | null;
  tabletType: string;

  // Step 2: Analysis result
  analysisResult: ScanUploadResponse | null;
  densities: number[];

  // Step 3: Configuration
  linearizationMode: string;
  targetResponse: string;
  curveStrategy: string;
  paperPreset: string;
  curveName: string;
  metalRatio: number;
  notes: string;

  // Step 4: Generated curve
  generatedCurve: CurveData | null;
  curveAdjustments: {
    brightness: number;
    contrast: number;
    gamma: number;
    highlights: number;
    shadows: number;
  };

  // Step 5: Export
  exportFormat: string;
  exportedFilePath: string | null;

  // Actions
  setStep: (step: number) => void;
  nextStep: () => void;
  prevStep: () => void;

  setUploadedFile: (file: File | null) => void;
  setTabletType: (type: string) => void;

  setAnalysisResult: (result: ScanUploadResponse | null) => void;
  setDensities: (densities: number[]) => void;

  setConfiguration: (config: Partial<WizardState>) => void;
  setCurveName: (name: string) => void;
  setMetalRatio: (ratio: number) => void;
  setNotes: (notes: string) => void;

  setGeneratedCurve: (curve: CurveData | null) => void;
  setCurveAdjustments: (adjustments: Partial<WizardState['curveAdjustments']>) => void;

  setExportFormat: (format: string) => void;
  setExportedFilePath: (path: string | null) => void;

  reset: () => void;
  resetToStep: (step: number) => void;
}

const initialState = {
  currentStep: 1,
  maxStepReached: 1,

  uploadedFile: null,
  uploadedFileName: null,
  tabletType: 'stouffer_21',

  analysisResult: null,
  densities: [],

  linearizationMode: 'single_curve',
  targetResponse: 'linear',
  curveStrategy: 'spline_fit',
  paperPreset: '',
  curveName: 'Calibration Curve',
  metalRatio: 0.5,
  notes: '',

  generatedCurve: null,
  curveAdjustments: {
    brightness: 0,
    contrast: 0,
    gamma: 1.0,
    highlights: 0,
    shadows: 0,
  },

  exportFormat: 'qtr',
  exportedFilePath: null,
};

export const useWizardStore = create<WizardState>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        setStep: (step) =>
          set((state) => ({
            currentStep: step,
            maxStepReached: Math.max(state.maxStepReached, step),
          })),

        nextStep: () =>
          set((state) => {
            const nextStep = Math.min(state.currentStep + 1, 5);
            return {
              currentStep: nextStep,
              maxStepReached: Math.max(state.maxStepReached, nextStep),
            };
          }),

        prevStep: () =>
          set((state) => ({
            currentStep: Math.max(state.currentStep - 1, 1),
          })),

        setUploadedFile: (file) =>
          set({
            uploadedFile: file,
            uploadedFileName: file?.name || null,
            // Reset downstream state when file changes
            analysisResult: null,
            densities: [],
            generatedCurve: null,
          }),

        setTabletType: (type) => set({ tabletType: type }),

        setAnalysisResult: (result) =>
          set({
            analysisResult: result,
            densities: result?.densities || [],
          }),

        setDensities: (densities) => set({ densities }),

        setConfiguration: (config) => set(config),

        setCurveName: (name) => set({ curveName: name }),

        setMetalRatio: (ratio) => set({ metalRatio: ratio }),

        setNotes: (notes) => set({ notes }),

        setGeneratedCurve: (curve) => set({ generatedCurve: curve }),

        setCurveAdjustments: (adjustments) =>
          set((state) => ({
            curveAdjustments: { ...state.curveAdjustments, ...adjustments },
          })),

        setExportFormat: (format) => set({ exportFormat: format }),

        setExportedFilePath: (path) => set({ exportedFilePath: path }),

        reset: () => set(initialState),

        resetToStep: (step) => {
          const state = get();
          const newState = { ...state, currentStep: step };

          // Reset state for steps after the target step
          if (step <= 1) {
            Object.assign(newState, {
              uploadedFile: null,
              uploadedFileName: null,
              analysisResult: null,
              densities: [],
              generatedCurve: null,
              exportedFilePath: null,
            });
          } else if (step <= 2) {
            Object.assign(newState, {
              generatedCurve: null,
              exportedFilePath: null,
            });
          } else if (step <= 3) {
            Object.assign(newState, {
              generatedCurve: null,
              exportedFilePath: null,
            });
          } else if (step <= 4) {
            Object.assign(newState, {
              exportedFilePath: null,
            });
          }

          set(newState);
        },
      }),
      {
        name: 'ptpd-wizard-storage',
        partialize: (state) => ({
          tabletType: state.tabletType,
          linearizationMode: state.linearizationMode,
          targetResponse: state.targetResponse,
          curveStrategy: state.curveStrategy,
          exportFormat: state.exportFormat,
        }),
      }
    ),
    { name: 'WizardStore' }
  )
);
