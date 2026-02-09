/**
 * Equivalence Test Framework Setup
 *
 * Provides utilities for verifying that migrated React components
 * produce functionally equivalent results to the legacy Gradio implementation.
 *
 * Tests compare API responses and data transformations between the legacy
 * Python backend (shared) and the React frontend's API layer.
 */

import { api } from '@/api/client';

/**
 * Tolerance for floating-point curve value comparisons.
 * Allows for minor precision differences between implementations.
 */
export const CURVE_TOLERANCE = 0.001;

/**
 * Compare two arrays of numbers within a given tolerance.
 */
export function arraysMatchWithTolerance(
  a: number[],
  b: number[],
  tolerance: number = CURVE_TOLERANCE
): { match: boolean; maxDiff: number; diffIndex: number } {
  if (a.length !== b.length) {
    return { match: false, maxDiff: Infinity, diffIndex: -1 };
  }

  let maxDiff = 0;
  let diffIndex = -1;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i] ?? 0;
    const bVal = b[i] ?? 0;
    const diff = Math.abs(aVal - bVal);
    if (diff > maxDiff) {
      maxDiff = diff;
      diffIndex = i;
    }
  }

  return {
    match: maxDiff <= tolerance,
    maxDiff,
    diffIndex,
  };
}

/**
 * Standard test curve inputs for equivalence testing.
 * These represent common calibration scenarios.
 */
export const TEST_FIXTURES = {
  linear: {
    name: 'Linear Reference',
    input_values: Array.from({ length: 256 }, (_, i) => i),
    output_values: Array.from({ length: 256 }, (_, i) => i),
  },

  typicalCalibration: {
    name: 'Typical Pt/Pd Calibration',
    input_values: Array.from({ length: 21 }, (_, i) => Math.round((i / 20) * 255)),
    output_values: [0, 8, 18, 30, 45, 62, 80, 100, 118, 135, 150, 163, 175, 186, 196, 205, 214, 223, 232, 243, 255],
  },

  highContrast: {
    name: 'High Contrast',
    input_values: Array.from({ length: 256 }, (_, i) => i),
    output_values: Array.from({ length: 256 }, (_, i) =>
      Math.round(Math.min(255, Math.max(0, (i - 128) * 1.5 + 128)))
    ),
  },

  densityMeasurements: {
    stouffer21: [0.05, 0.15, 0.25, 0.35, 0.48, 0.60, 0.72, 0.85, 0.98, 1.10, 1.22, 1.35, 1.48, 1.58, 1.68, 1.75, 1.82, 1.88, 1.92, 1.95, 1.98],
    stouffer31: Array.from({ length: 31 }, (_, i) => 0.05 + (i / 30) * 1.95),
  },

  scanQuality: {
    excellent: {
      densities: Array.from({ length: 21 }, (_, i) => 0.05 + (i / 20) * 1.95),
      dmax: 2.0,
      dmin: 0.05,
      range: 1.95,
      num_patches: 21,
    },
    poor: {
      densities: Array.from({ length: 8 }, (_, i) => 0.1 + (i / 7) * 0.3),
      dmax: 0.4,
      dmin: 0.1,
      range: 0.3,
      num_patches: 8,
    },
  },
} as const;

/**
 * Helper to call the backend API and validate response shape.
 * Used to compare legacy and new API behavior.
 */
export async function callCurveModifyEndpoint(params: {
  name: string;
  input_values: number[];
  output_values: number[];
  adjustment_type: string;
  amount: number;
}) {
  return api.curves.modify(params);
}

/**
 * Custom matcher: checks if two curve arrays match within tolerance.
 */
export function expectCurvesMatch(
  actual: number[],
  expected: number[],
  tolerance: number = CURVE_TOLERANCE
): void {
  const result = arraysMatchWithTolerance(actual, expected, tolerance);
  if (!result.match) {
    throw new Error(
      `Curves differ by ${result.maxDiff} at index ${result.diffIndex} (tolerance: ${tolerance}). ` +
      `Expected[${result.diffIndex}]=${expected[result.diffIndex]}, Got[${result.diffIndex}]=${actual[result.diffIndex]}`
    );
  }
}
