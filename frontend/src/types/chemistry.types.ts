/**
 * Chemistry-related types.
 * Types for coating calculations, recipes, and chemistry management.
 */

/**
 * Paper absorbency levels
 */
export type PaperAbsorbency = 'low' | 'medium' | 'high';

/**
 * Coating application methods
 */
export type CoatingMethod = 'brush' | 'rod' | 'puddle_pusher';

/**
 * Contrast agent types
 */
export type ContrastAgent = 'none' | 'na2' | 'potassium_chlorate';

/**
 * Developer types
 */
export type DeveloperType =
  | 'potassium_oxalate'
  | 'ammonium_citrate'
  | 'sodium_acetate'
  | 'other';

/**
 * Chemistry process types
 */
export type ChemistryType =
  | 'platinum_palladium'
  | 'cyanotype'
  | 'vandyke'
  | 'kallitype'
  | 'silver_gelatin';

/**
 * Print dimensions
 */
export interface PrintDimensions {
  widthInches: number;
  heightInches: number;
  marginInches?: number;
}

/**
 * Coating area calculation result
 */
export interface CoatingArea {
  coatingWidthInches: number;
  coatingHeightInches: number;
  coatingAreaSqInches: number;
}

/**
 * Solution amounts (both drops and milliliters)
 */
export interface SolutionAmounts {
  ferricOxalate1: { drops: number; ml: number };
  ferricOxalate2Contrast: { drops: number; ml: number };
  palladium: { drops: number; ml: number };
  platinum: { drops: number; ml: number };
  na2: { drops: number; ml: number };
  total: { drops: number; ml: number };
}

/**
 * Metal ratio settings
 */
export interface MetalRatio {
  platinumPercent: number;
  palladiumPercent: number;
}

/**
 * Complete chemistry recipe
 */
export interface ChemistryRecipe {
  id?: string;
  name?: string;
  printDimensions: PrintDimensions;
  coatingArea: CoatingArea;
  solutionAmounts: SolutionAmounts;
  metalRatio: MetalRatio;
  paperAbsorbency: PaperAbsorbency;
  coatingMethod: CoatingMethod;
  contrastBoost: number;
  estimatedCostUsd: number | null;
  notes: string[];
  createdAt?: string;
}

/**
 * Chemistry preset
 */
export interface ChemistryPreset {
  id: string;
  name: string;
  description: string;
  platinumRatio: number;
  contrastAgent: ContrastAgent;
  contrastAmount: number;
  developer: DeveloperType;
  notes: string;
}

/**
 * Paper profile
 */
export interface PaperProfile {
  id: string;
  name: string;
  manufacturer?: string;
  weightGsm?: number;
  sizing: 'internal' | 'external' | 'none';
  absorbency: PaperAbsorbency;
  baseDensity?: number;
  maxDensity?: number;
  recommendedExposureFactor: number;
  notes?: string;
}

/**
 * Chemistry inventory item
 */
export interface ChemistryInventory {
  id: string;
  name: string;
  type: 'ferric_oxalate' | 'palladium' | 'platinum' | 'na2' | 'developer' | 'other';
  currentVolumeMl: number;
  originalVolumeMl: number;
  purchaseDate: string;
  expirationDate: string;
  costPerMl: number;
  batchNumber?: string;
  notes?: string;
}

/**
 * Chemistry calculation input
 */
export interface ChemistryCalculationInput {
  dimensions: PrintDimensions;
  platinumRatio: number;
  paperAbsorbency: PaperAbsorbency;
  coatingMethod: CoatingMethod;
  contrastBoost: number;
  na2Ratio?: number;
  includeCost: boolean;
}

/**
 * Exposure calculation input
 */
export interface ExposureCalculationInput {
  negativeDensity: number;
  uvSource: 'bl_tubes' | 'mercury_vapor' | 'sunlight' | 'led_uv' | 'other';
  paperType: string;
  metalRatio: number;
  humidity?: number;
  temperature?: number;
}

/**
 * Exposure calculation result
 */
export interface ExposureCalculationResult {
  recommendedTimeSeconds: number;
  confidenceLevel: 'high' | 'medium' | 'low';
  adjustmentFactors: {
    factor: string;
    adjustment: number;
    reason: string;
  }[];
  notes: string[];
}
