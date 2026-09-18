/**
 * Central Zustand store composition
 * Uses slice pattern for modularity and maintainability
 */

import {
  create,
  type StateCreator,
  type StoreApi,
  type UseBoundStore,
} from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

import { createUISlice, type UISlice } from './slices/uiSlice';
import {
  type CalibrationSlice,
  createCalibrationSlice,
} from './slices/calibrationSlice';
import { createCurveSlice, type CurveSlice } from './slices/curveSlice';
import {
  type ChemistrySlice,
  createChemistrySlice,
} from './slices/chemistrySlice';
import { type ChatSlice, createChatSlice } from './slices/chatSlice';
import { createSessionSlice, type SessionSlice } from './slices/sessionSlice';
import { createImageSlice, type ImageSlice } from './slices/imageSlice';
import { createMCTSSlice, type MCTSSlice } from './slices/mctsSlice';
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
  mcts: MCTSSlice;
};

/**
 * Subset of the state written to storage by the persist middleware.
 * Only UI preferences are persisted; runtime flags are reset on load.
 */
type PersistedState = {
  ui: Pick<UISlice, 'activeTab' | 'sidebarOpen' | 'theme' | 'isProcessing'>;
};

/**
 * Middleware stack applied to the application store, outermost first
 * (devtools → subscribeWithSelector → persist → immer). Each slice creator is
 * typed against its own sub-state with the immer mutator only; the wider
 * `set`/`get`/`store` of the full store satisfy those signatures structurally.
 * The persist entry is `unknown` here because that is what `persist()` expects
 * on its initializer; the persisted shape is inferred from `partialize`.
 */
type StoreMutators = [
  ['zustand/devtools', never],
  ['zustand/subscribeWithSelector', never],
  ['zustand/persist', unknown],
  ['zustand/immer', never],
];

/**
 * Store API shared by the application store and isolated test stores.
 */
export type AppStore = UseBoundStore<StoreApi<StoreState>>;

/**
 * Compose every slice into the root state.
 */
const createRootState: StateCreator<StoreState, StoreMutators, []> = (
  set,
  get,
  store
) => ({
  ui: createUISlice(set, get, store),
  calibration: createCalibrationSlice(set, get, store),
  curve: createCurveSlice(set, get, store),
  chemistry: createChemistrySlice(set, get, store),
  chat: createChatSlice(set, get, store),
  session: createSessionSlice(set, get, store),
  image: createImageSlice(set, get, store),
  mcts: createMCTSSlice(set, get, store),
});

/**
 * Main application store
 */
export const useStore = create<StoreState>()(
  devtools(
    subscribeWithSelector(
      persist(immer(createRootState), {
        name: 'ptpd-store',
        partialize: (state): PersistedState => ({
          // Only persist UI preferences
          ui: {
            activeTab: state.ui.activeTab,
            sidebarOpen: state.ui.sidebarOpen,
            theme: state.ui.theme,
            // Do not persist runtime flags; ensure processing is reset on load
            isProcessing: false,
          },
        }),
        merge: (persistedState, currentState): StoreState => {
          const persisted = persistedState as
            | Partial<PersistedState>
            | undefined;
          if (!persisted?.ui) {
            return currentState;
          }

          return {
            ...currentState,
            ui: {
              ...currentState.ui,
              ...persisted.ui,
            },
          };
        },
      })
    ),
    {
      name: 'PtPdPrintingTool',
      enabled: config.features.devtools,
    }
  )
);

/**
 * Create a store instance for testing
 * Allows creating isolated stores for tests
 */
export const createStore = (): AppStore => {
  return create<StoreState>()(
    immer((set, get, store) => ({
      ui: createUISlice(set, get, store),
      calibration: createCalibrationSlice(set, get, store),
      curve: createCurveSlice(set, get, store),
      chemistry: createChemistrySlice(set, get, store),
      chat: createChatSlice(set, get, store),
      session: createSessionSlice(set, get, store),
      image: createImageSlice(set, get, store),
      mcts: createMCTSSlice(set, get, store),
    }))
  );
};

// ============================================================================
// Typed Selectors (define outside components to prevent re-renders)
// ============================================================================

// UI Selectors
export const selectActiveTab = (state: StoreState): string =>
  state.ui.activeTab;
export const selectIsProcessing = (state: StoreState): boolean =>
  state.ui.isProcessing;
export const selectSidebarOpen = (state: StoreState): boolean =>
  state.ui.sidebarOpen;
export const selectTheme = (state: StoreState): 'light' | 'dark' =>
  state.ui.theme;
export const selectIsInitialized = (state: StoreState): boolean =>
  state.ui.isInitialized;

// Calibration Selectors
export const selectCurrentCalibration = (
  state: StoreState
): CalibrationSlice['current'] => state.calibration.current;
export const selectCalibrationStep = (state: StoreState): number =>
  state.calibration.currentStep;
export const selectCalibrationHistory = (
  state: StoreState
): CalibrationSlice['history'] => state.calibration.history;

// Curve Selectors
export const selectCurrentCurve = (state: StoreState): CurveSlice['current'] =>
  state.curve.current;
export const selectCurvePoints = (state: StoreState): CurveSlice['points'] =>
  state.curve.points;
export const selectCurveModified = (state: StoreState): boolean =>
  state.curve.isModified;

// Chemistry Selectors
export const selectChemistryRecipe = (
  state: StoreState
): ChemistrySlice['recipe'] => state.chemistry.recipe;
export const selectPaperSize = (
  state: StoreState
): ChemistrySlice['paperSize'] => state.chemistry.paperSize;
export const selectMetalRatio = (state: StoreState): number =>
  state.chemistry.metalRatio;

// Chat Selectors
export const selectChatMessages = (state: StoreState): ChatSlice['messages'] =>
  state.chat.messages;
export const selectChatLoading = (state: StoreState): boolean =>
  state.chat.isLoading;

// Session Selectors
export const selectSessionRecords = (
  state: StoreState
): SessionSlice['records'] => state.session.records;
export const selectSessionStats = (state: StoreState): SessionSlice['stats'] =>
  state.session.stats;

// Image Selectors
export const selectCurrentImage = (state: StoreState): ImageSlice['current'] =>
  state.image.current;
export const selectImagePreview = (state: StoreState): ImageSlice['preview'] =>
  state.image.preview;
export const selectUploadProgress = (state: StoreState): number =>
  state.image.uploadProgress;

// Re-export slice types
export type {
  UISlice,
  CalibrationSlice,
  CurveSlice,
  ChemistrySlice,
  ChatSlice,
  SessionSlice,
  ImageSlice,
  MCTSSlice,
};
