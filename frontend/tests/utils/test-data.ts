/**
 * Test data factories for consistent test data generation.
 */

import type { CurveData, CurveType } from '@/types/curve.types';
import type { ChemistryRecipe, PaperProfile } from '@/types/chemistry.types';
import type { PrintSession, PrintRecord, DashboardMetrics } from '@/types/session.types';

/**
 * Factory for creating curve test data.
 */
export const createCurve = (overrides: Partial<CurveData> = {}): CurveData => ({
  id: `curve-${Date.now()}`,
  name: 'Test Curve',
  type: 'contrast' as CurveType,
  input_values: Array.from({ length: 21 }, (_, i) => i * 5),
  output_values: Array.from({ length: 21 }, (_, i) => Math.round(i * 4.76)),
  created_at: new Date().toISOString(),
  modified_at: new Date().toISOString(),
  metadata: {
    paper_type: 'Arches Platine',
    chemistry: 'Pt/Pd Classic',
  },
  ...overrides,
});

/**
 * Factory for creating chemistry recipe test data.
 */
export const createRecipe = (overrides: Partial<ChemistryRecipe> = {}): ChemistryRecipe => ({
  id: `recipe-${Date.now()}`,
  name: 'Test Recipe',
  platinum_ml: 10,
  palladium_ml: 10,
  ferric_oxalate_ml: 20,
  contrast_agent_ml: 0,
  contrast_agent_type: 'none',
  total_volume_ml: 40,
  metal_ratio: 0.5,
  coverage_area_sq_in: 80,
  created_at: new Date().toISOString(),
  ...overrides,
});

/**
 * Factory for creating paper profile test data.
 */
export const createPaperProfile = (overrides: Partial<PaperProfile> = {}): PaperProfile => ({
  id: `paper-${Date.now()}`,
  name: 'Arches Platine',
  manufacturer: 'Arches',
  weight_gsm: 310,
  texture: 'smooth',
  sizing: 'internal',
  base_exposure_factor: 1.0,
  recommended_chemistry: ['Pt/Pd Classic', 'Na2 Developer'],
  notes: 'Excellent for platinum printing',
  ...overrides,
});

/**
 * Factory for creating print record test data.
 */
export const createPrintRecord = (overrides: Partial<PrintRecord> = {}): PrintRecord => ({
  id: `print-${Date.now()}`,
  session_id: 'session-1',
  image_name: 'test-image.tiff',
  paper_profile: createPaperProfile(),
  chemistry_recipe: createRecipe(),
  curve_id: 'curve-1',
  exposure_time_seconds: 300,
  uv_intensity: 100,
  humidity_percent: 50,
  temperature_celsius: 20,
  notes: 'Test print',
  rating: 4,
  created_at: new Date().toISOString(),
  ...overrides,
});

/**
 * Factory for creating print session test data.
 */
export const createSession = (overrides: Partial<PrintSession> = {}): PrintSession => ({
  id: `session-${Date.now()}`,
  name: 'Test Session',
  date: new Date().toISOString(),
  prints: [],
  notes: '',
  ...overrides,
});

/**
 * Factory for creating dashboard metrics test data.
 */
export const createDashboardMetrics = (overrides: Partial<DashboardMetrics> = {}): DashboardMetrics => ({
  total_prints: 42,
  total_sessions: 8,
  prints_this_week: 5,
  prints_this_month: 15,
  average_rating: 3.8,
  most_used_paper: 'Arches Platine',
  most_used_chemistry: 'Pt/Pd Classic',
  active_curves: 5,
  recent_activity: [],
  ...overrides,
});

/**
 * Creates mock step tablet measurements.
 */
export const createStepTabletMeasurements = (steps = 21) =>
  Array.from({ length: steps }, (_, i) => ({
    step: i,
    target_density: i * 0.15,
    measured_density: i * 0.15 + (Math.random() - 0.5) * 0.05,
    lab: {
      l: 100 - i * 4.5,
      a: (Math.random() - 0.5) * 2,
      b: (Math.random() - 0.5) * 2,
    },
  }));

/**
 * Creates mock scan result.
 */
export const createScanResult = () => ({
  id: `scan-${Date.now()}`,
  filename: 'test-scan.tiff',
  quality_score: 0.85,
  measurements: createStepTabletMeasurements(),
  issues: [],
  analyzed_at: new Date().toISOString(),
});
