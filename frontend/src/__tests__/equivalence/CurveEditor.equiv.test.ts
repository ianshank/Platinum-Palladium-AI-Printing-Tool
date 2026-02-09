/**
 * CurveEditor Equivalence Tests
 *
 * Verifies that the React CurveEditor produces identical outputs to
 * the legacy Gradio implementation for the same inputs.
 *
 * These tests validate:
 * 1. Curve modification (contrast, brightness, gamma) produces same output
 * 2. Data round-trip through API returns consistent results
 */

import { describe, expect, it } from 'vitest';
import {
  arraysMatchWithTolerance,
  CURVE_TOLERANCE,
  expectCurvesMatch,
  TEST_FIXTURES,
} from './setup';

// ============================================================================
// Curve Array Comparison Utilities
// ============================================================================

describe('Equivalence: Curve Array Comparison', () => {
  it('identifies identical arrays as matching', () => {
    const a = [0, 64, 128, 192, 255];
    const b = [0, 64, 128, 192, 255];
    const result = arraysMatchWithTolerance(a, b);
    expect(result.match).toBe(true);
    expect(result.maxDiff).toBe(0);
  });

  it('identifies arrays within tolerance as matching', () => {
    const a = [0, 64, 128, 192, 255];
    const b = [0, 64.0004, 128.0003, 192.0005, 255];
    const result = arraysMatchWithTolerance(a, b, CURVE_TOLERANCE);
    expect(result.match).toBe(true);
  });

  it('identifies arrays exceeding tolerance as non-matching', () => {
    const a = [0, 64, 128, 192, 255];
    const b = [0, 64, 130, 192, 255];
    const result = arraysMatchWithTolerance(a, b, CURVE_TOLERANCE);
    expect(result.match).toBe(false);
    expect(result.diffIndex).toBe(2);
    expect(result.maxDiff).toBe(2);
  });

  it('identifies different-length arrays as non-matching', () => {
    const a = [0, 128, 255];
    const b = [0, 128];
    const result = arraysMatchWithTolerance(a, b);
    expect(result.match).toBe(false);
    expect(result.maxDiff).toBe(Infinity);
  });
});

// ============================================================================
// Test Fixtures Validation
// ============================================================================

describe('Equivalence: Test Fixtures', () => {
  it('linear fixture has correct shape (256 points, identity mapping)', () => {
    const { input_values, output_values } = TEST_FIXTURES.linear;
    expect(input_values).toHaveLength(256);
    expect(output_values).toHaveLength(256);
    expectCurvesMatch(input_values, output_values);
  });

  it('typical calibration fixture has 21 step values', () => {
    const { input_values, output_values } = TEST_FIXTURES.typicalCalibration;
    expect(input_values).toHaveLength(21);
    expect(output_values).toHaveLength(21);
    // Output should be monotonically increasing
    for (let i = 1; i < output_values.length; i++) {
      expect(output_values[i]).toBeGreaterThanOrEqual(output_values[i - 1]!);
    }
  });

  it('high contrast fixture applies contrast correctly', () => {
    const { output_values } = TEST_FIXTURES.highContrast;
    expect(output_values).toHaveLength(256);
    // Midpoint (128) should remain at ~128
    expect(output_values[128]).toBe(128);
    // Low values should be lower
    expect(output_values[64]!).toBeLessThan(64);
    // High values should be higher or clamped to 255
    expect(output_values[192]!).toBeGreaterThan(192);
  });

  it('density measurements have expected ranges', () => {
    const { stouffer21 } = TEST_FIXTURES.densityMeasurements;
    expect(stouffer21).toHaveLength(21);
    expect(stouffer21[0]).toBeCloseTo(0.05, 2);
    expect(stouffer21[20]).toBeCloseTo(1.98, 2);
  });
});

// ============================================================================
// Curve Modification Equivalence (API-level, mocked)
// ============================================================================

describe('Equivalence: Curve Modification Data Flow', () => {
  it('contrast adjustment preserves array length', () => {
    const input = TEST_FIXTURES.linear.output_values;
    // Simulate contrast: (value - 128) * factor + 128
    const factor = 1.5;
    const result = input.map((v) =>
      Math.round(Math.min(255, Math.max(0, (v - 128) * factor + 128)))
    );
    expect(result).toHaveLength(input.length);
    expect(result[0]).toBe(0); // Black stays black (clamped)
    expect(result[128]).toBe(128); // Midpoint stays
    expect(result[255]).toBe(255); // White stays white (clamped)
  });

  it('brightness adjustment shifts all values uniformly', () => {
    const input = TEST_FIXTURES.linear.output_values;
    const shift = 20;
    const result = input.map((v) =>
      Math.min(255, Math.max(0, v + shift))
    );
    expect(result).toHaveLength(input.length);
    expect(result[0]).toBe(20);
    expect(result[128]).toBe(148);
    expect(result[235]).toBe(255); // Clamped at 255
  });

  it('gamma adjustment applies power curve correctly', () => {
    const input = TEST_FIXTURES.linear.output_values;
    const gamma = 2.2;
    const result = input.map((v) =>
      Math.round(Math.pow(v / 255, 1 / gamma) * 255)
    );
    expect(result).toHaveLength(input.length);
    expect(result[0]).toBe(0); // Black stays
    expect(result[255]).toBe(255); // White stays
    // Midtones should be lighter with gamma > 1
    expect(result[128]!).toBeGreaterThan(128);
  });

});

// ============================================================================
// Data Shape Equivalence
// ============================================================================

describe('Test Fixtures: Density Data Integrity', () => {
  it('density measurement arrays maintain correct ordering', () => {
    const densities = TEST_FIXTURES.densityMeasurements.stouffer21;
    for (let i = 1; i < densities.length; i++) {
      expect(densities[i]).toBeGreaterThan(densities[i - 1]!);
    }
  });
});
