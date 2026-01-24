/**
 * Chemistry state slice
 * Manages coating recipe calculations and chemistry settings
 */

import type { StateCreator } from 'zustand';
import { logger } from '@/lib/logger';

export interface PaperSize {
  name: string;
  widthInches: number;
  heightInches: number;
  custom?: boolean;
}

export interface ChemistryRecipe {
  totalVolume: number;
  platinumMl: number;
  palladiumMl: number;
  ferricOxalateMl: number;
  contrastAgent?: {
    type: 'na2' | 'dichromate' | 'none';
    amount: number;
  };
  developer: {
    type: 'potassium_oxalate' | 'ammonium_citrate';
    concentration: number;
    temperatureC: number;
  };
}

export interface ChemistrySlice {
  // State
  paperSize: PaperSize;
  customSizes: PaperSize[];
  metalRatio: number; // 0 = all Pd, 1 = all Pt, 0.5 = 50/50
  coatingMethod: 'brush' | 'rod' | 'puddle';
  contrastLevel: number; // 0-5 grade
  recipe: ChemistryRecipe | null;
  developer: ChemistryRecipe['developer'];

  // Actions
  setPaperSize: (size: PaperSize) => void;
  addCustomSize: (size: PaperSize) => void;
  removeCustomSize: (name: string) => void;
  setMetalRatio: (ratio: number) => void;
  setCoatingMethod: (method: ChemistrySlice['coatingMethod']) => void;
  setContrastLevel: (level: number) => void;
  setDeveloper: (developer: Partial<ChemistryRecipe['developer']>) => void;
  calculateRecipe: () => void;
  clearRecipe: () => void;
  resetChemistry: () => void;
}

// Standard paper sizes
export const STANDARD_PAPER_SIZES: PaperSize[] = [
  { name: '4x5', widthInches: 4, heightInches: 5 },
  { name: '5x7', widthInches: 5, heightInches: 7 },
  { name: '8x10', widthInches: 8, heightInches: 10 },
  { name: '11x14', widthInches: 11, heightInches: 14 },
  { name: '16x20', widthInches: 16, heightInches: 20 },
  { name: 'A4', widthInches: 8.27, heightInches: 11.69 },
  { name: 'A3', widthInches: 11.69, heightInches: 16.54 },
];

const initialState = {
  paperSize: STANDARD_PAPER_SIZES[2]!, // 8x10 default
  customSizes: [] as PaperSize[],
  metalRatio: 0.5, // 50% Pt, 50% Pd
  coatingMethod: 'brush' as const,
  contrastLevel: 2,
  recipe: null as ChemistryRecipe | null,
  developer: {
    type: 'potassium_oxalate' as const,
    concentration: 25, // percent
    temperatureC: 20,
  },
};

// Coating constants (ml per square inch)
const COATING_FACTORS: Record<ChemistrySlice['coatingMethod'], number> = {
  brush: 0.035,
  rod: 0.030,
  puddle: 0.040,
};

export const createChemistrySlice: StateCreator<
  { chemistry: ChemistrySlice },
  [['zustand/immer', never]],
  [],
  ChemistrySlice
> = (set, get) => ({
  ...initialState,

  setPaperSize: (size) => {
    logger.debug('Chemistry: setPaperSize', { size: size.name });
    set((state) => {
      state.chemistry.paperSize = size;
      state.chemistry.recipe = null; // Clear recipe when size changes
    });
  },

  addCustomSize: (size) => {
    logger.debug('Chemistry: addCustomSize', { name: size.name });
    set((state) => {
      state.chemistry.customSizes.push({ ...size, custom: true });
    });
  },

  removeCustomSize: (name) => {
    logger.debug('Chemistry: removeCustomSize', { name });
    set((state) => {
      state.chemistry.customSizes = state.chemistry.customSizes.filter(
        (s) => s.name !== name
      );
    });
  },

  setMetalRatio: (ratio) => {
    const clampedRatio = Math.max(0, Math.min(1, ratio));
    logger.debug('Chemistry: setMetalRatio', { ratio: clampedRatio });
    set((state) => {
      state.chemistry.metalRatio = clampedRatio;
      state.chemistry.recipe = null;
    });
  },

  setCoatingMethod: (method) => {
    logger.debug('Chemistry: setCoatingMethod', { method });
    set((state) => {
      state.chemistry.coatingMethod = method;
      state.chemistry.recipe = null;
    });
  },

  setContrastLevel: (level) => {
    const clampedLevel = Math.max(0, Math.min(5, level));
    logger.debug('Chemistry: setContrastLevel', { level: clampedLevel });
    set((state) => {
      state.chemistry.contrastLevel = clampedLevel;
      state.chemistry.recipe = null;
    });
  },

  setDeveloper: (developer) => {
    logger.debug('Chemistry: setDeveloper', developer);
    set((state) => {
      state.chemistry.developer = {
        ...state.chemistry.developer,
        ...developer,
      };
      state.chemistry.recipe = null;
    });
  },

  calculateRecipe: () => {
    const { paperSize, metalRatio, coatingMethod, contrastLevel, developer } = get().chemistry;

    logger.debug('Chemistry: calculateRecipe', {
      paperSize: paperSize.name,
      metalRatio,
      coatingMethod,
      contrastLevel,
    });

    // Calculate coating area
    const areaSquareInches = paperSize.widthInches * paperSize.heightInches;

    // Calculate total solution volume needed
    const coatingFactor = COATING_FACTORS[coatingMethod];
    const totalVolume = areaSquareInches * coatingFactor;

    // Metal solution is typically 40% of total
    const metalVolume = totalVolume * 0.4;
    const platinumMl = metalVolume * metalRatio;
    const palladiumMl = metalVolume * (1 - metalRatio);

    // Ferric oxalate is 60% of total
    const ferricOxalateMl = totalVolume * 0.6;

    // Calculate contrast agent based on level
    let contrastAgent: ChemistryRecipe['contrastAgent'];
    if (contrastLevel > 0) {
      contrastAgent = {
        type: 'na2',
        amount: contrastLevel * 0.5, // drops per 10ml
      };
    }

    const recipe: ChemistryRecipe = {
      totalVolume: Math.round(totalVolume * 100) / 100,
      platinumMl: Math.round(platinumMl * 100) / 100,
      palladiumMl: Math.round(palladiumMl * 100) / 100,
      ferricOxalateMl: Math.round(ferricOxalateMl * 100) / 100,
      contrastAgent,
      developer: { ...developer },
    };

    logger.info('Chemistry: recipe calculated', recipe);

    set((state) => {
      state.chemistry.recipe = recipe;
    });
  },

  clearRecipe: () => {
    logger.debug('Chemistry: clearRecipe');
    set((state) => {
      state.chemistry.recipe = null;
    });
  },

  resetChemistry: () => {
    logger.debug('Chemistry: resetChemistry');
    set((state) => {
      Object.assign(state.chemistry, initialState);
    });
  },
});
