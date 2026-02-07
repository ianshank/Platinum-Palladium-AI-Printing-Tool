/**
 * Central Zustand store composition
 * Uses slice pattern for modularity and maintainability
 */

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

import { createUISlice, type UISlice } from './slices/uiSlice';
import { createCalibrationSlice, type CalibrationSlice } from './slices/calibrationSlice';
import { createCurveSlice, type CurveSlice } from './slices/curveSlice';
import { createChemistrySlice, type ChemistrySlice } from './slices/chemistrySlice';
import { createChatSlice, type ChatSlice } from './slices/chatSlice';
import { createSessionSlice, type SessionSlice } from './slices/sessionSlice';
import { createImageSlice, type ImageSlice } from './slices/imageSlice';
import { config } from '@/config';

/**
 * Combined store state type
 */
export type StoreState = {
  ui: UISlice;
  calibration: CalibrationSlice;
  curve: CurveSlice;
  chemistry: ChemistrySlice;
  chat: ChatSlice;
  session: SessionSlice;
  image: ImageSlice;
};

/**
 * Store middleware configuration
 */
/**
 * Store middleware configuration
 */
// Explicitly type the middleware to avoid complex inference issues
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type StoreMiddleware = (f: any) => any;

const storeMiddleware: StoreMiddleware = (
  f
) =>
  devtools(
    subscribeWithSelector(
      persist(
        immer(f) as any,
        {
          name: 'ptpd-store',
          partialize: (state: any) => ({
            // Only persist UI preferences
            ui: {
              activeTab: state.ui.activeTab,
              sidebarOpen: state.ui.sidebarOpen,
              theme: state.ui.theme,
              isProcessing: false,
              isInitialized: true,
            },
          }),
        }
      )
    ),
    {
      name: 'PtPdPrintingTool',
      enabled: config.features.devtools,
    }
  );

/**
 * Main application store
 */
export const useStore = create<StoreState>()(
  storeMiddleware((set: any, get: any, store: any) => ({
    ui: createUISlice(set, get, store),
    calibration: createCalibrationSlice(set, get, store),
    curve: createCurveSlice(set, get, store),
    chemistry: createChemistrySlice(set, get, store),
    chat: createChatSlice(set, get, store),
    session: createSessionSlice(set, get, store),
    image: createImageSlice(set, get, store),
  }))
);

/**
 * Create a store instance for testing
 * Allows creating isolated stores for tests
 */
export const createStore = (): typeof useStore => {
  return create<StoreState>()(
    immer((set, get, store) => ({
      ui: createUISlice(set, get, store),
      calibration: createCalibrationSlice(set, get, store),
      curve: createCurveSlice(set, get, store),
      chemistry: createChemistrySlice(set, get, store),
      chat: createChatSlice(set, get, store),
      session: createSessionSlice(set, get, store),
      image: createImageSlice(set, get, store),
    }))
  );
};

// ============================================================================
// Typed Selectors (define outside components to prevent re-renders)
// ============================================================================

// UI Selectors
export const selectActiveTab = (state: StoreState): string => state.ui.activeTab;
export const selectIsProcessing = (state: StoreState): boolean => state.ui.isProcessing;
export const selectSidebarOpen = (state: StoreState): boolean => state.ui.sidebarOpen;
export const selectTheme = (state: StoreState): 'light' | 'dark' => state.ui.theme;
export const selectIsInitialized = (state: StoreState): boolean => state.ui.isInitialized;

// Calibration Selectors
export const selectCurrentCalibration = (state: StoreState) => state.calibration.current;
export const selectCalibrationStep = (state: StoreState): number => state.calibration.currentStep;
export const selectCalibrationHistory = (state: StoreState) => state.calibration.history;

// Curve Selectors
export const selectCurrentCurve = (state: StoreState) => state.curve.current;
export const selectCurvePoints = (state: StoreState) => state.curve.points;
export const selectCurveModified = (state: StoreState): boolean => state.curve.isModified;

// Chemistry Selectors
export const selectChemistryRecipe = (state: StoreState) => state.chemistry.recipe;
export const selectPaperSize = (state: StoreState) => state.chemistry.paperSize;
export const selectMetalRatio = (state: StoreState): number => state.chemistry.metalRatio;

// Chat Selectors
export const selectChatMessages = (state: StoreState) => state.chat.messages;
export const selectChatLoading = (state: StoreState): boolean => state.chat.isLoading;

// Session Selectors
export const selectSessionRecords = (state: StoreState) => state.session.records;
export const selectSessionStats = (state: StoreState) => state.session.stats;

// Image Selectors
export const selectCurrentImage = (state: StoreState) => state.image.current;
export const selectImagePreview = (state: StoreState) => state.image.preview;
export const selectUploadProgress = (state: StoreState): number => state.image.uploadProgress;

// Re-export slice types
export type { UISlice, CalibrationSlice, CurveSlice, ChemistrySlice, ChatSlice, SessionSlice, ImageSlice };
