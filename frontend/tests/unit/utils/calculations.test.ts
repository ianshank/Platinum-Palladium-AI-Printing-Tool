/**
 * Chemistry calculation utility tests.
 */

import { describe, it, expect } from 'vitest';

// Chemistry calculation functions (extracted for testing)
interface ChemistryInputs {
  width: number;
  height: number;
  metalRatio: number;
  contrastAgent: string;
}

interface ChemistryResult {
  platinum_ml: number;
  palladium_ml: number;
  ferric_oxalate_ml: number;
  contrast_agent_ml: number;
  total_volume_ml: number;
  coverage_area_sq_in: number;
}

// Configuration (would normally come from config)
const COATING_ML_PER_SQ_INCH = 0.5;
const FERRIC_OXALATE_RATIO = 2.0;
const CONTRAST_AGENT_RATIO = 0.1;
const MINIMUM_VOLUME_ML = 2.0;
const ROUNDING_PRECISION = 1;

const round = (value: number, precision: number): number => {
  const factor = Math.pow(10, precision);
  return Math.round(value * factor) / factor;
};

const calculateChemistry = (inputs: ChemistryInputs): ChemistryResult => {
  const area = inputs.width * inputs.height;
  const totalMetal = Math.max(area * COATING_ML_PER_SQ_INCH, MINIMUM_VOLUME_ML);

  const platinum = totalMetal * inputs.metalRatio;
  const palladium = totalMetal * (1 - inputs.metalRatio);
  const ferricOxalate = totalMetal * FERRIC_OXALATE_RATIO;
  const contrastAgent = inputs.contrastAgent !== 'none' ? totalMetal * CONTRAST_AGENT_RATIO : 0;

  const total = platinum + palladium + ferricOxalate + contrastAgent;

  return {
    platinum_ml: round(platinum, ROUNDING_PRECISION),
    palladium_ml: round(palladium, ROUNDING_PRECISION),
    ferric_oxalate_ml: round(ferricOxalate, ROUNDING_PRECISION),
    contrast_agent_ml: round(contrastAgent, ROUNDING_PRECISION),
    total_volume_ml: round(total, ROUNDING_PRECISION),
    coverage_area_sq_in: area,
  };
};

describe('Chemistry Calculations', () => {
  describe('calculateChemistry', () => {
    it('calculates correct volumes for 8x10 print with neutral ratio', () => {
      const result = calculateChemistry({
        width: 8,
        height: 10,
        metalRatio: 0.5,
        contrastAgent: 'none',
      });

      expect(result.coverage_area_sq_in).toBe(80);
      expect(result.platinum_ml).toBe(20);
      expect(result.palladium_ml).toBe(20);
      expect(result.ferric_oxalate_ml).toBe(80);
      expect(result.contrast_agent_ml).toBe(0);
      expect(result.total_volume_ml).toBe(120);
    });

    it('calculates correct volumes for pure platinum', () => {
      const result = calculateChemistry({
        width: 8,
        height: 10,
        metalRatio: 1.0,
        contrastAgent: 'none',
      });

      expect(result.platinum_ml).toBe(40);
      expect(result.palladium_ml).toBe(0);
    });

    it('calculates correct volumes for pure palladium', () => {
      const result = calculateChemistry({
        width: 8,
        height: 10,
        metalRatio: 0.0,
        contrastAgent: 'none',
      });

      expect(result.platinum_ml).toBe(0);
      expect(result.palladium_ml).toBe(40);
    });

    it('includes contrast agent when specified', () => {
      const result = calculateChemistry({
        width: 8,
        height: 10,
        metalRatio: 0.5,
        contrastAgent: 'na2',
      });

      expect(result.contrast_agent_ml).toBe(4);
      expect(result.total_volume_ml).toBe(124);
    });

    it('applies minimum volume for small prints', () => {
      const result = calculateChemistry({
        width: 1,
        height: 1,
        metalRatio: 0.5,
        contrastAgent: 'none',
      });

      // 1 sq inch * 0.5 ml = 0.5 ml, but minimum is 2 ml
      expect(result.platinum_ml).toBe(1);
      expect(result.palladium_ml).toBe(1);
    });

    it('scales correctly for large prints', () => {
      const result = calculateChemistry({
        width: 20,
        height: 24,
        metalRatio: 0.5,
        contrastAgent: 'none',
      });

      expect(result.coverage_area_sq_in).toBe(480);
      expect(result.platinum_ml).toBe(120);
      expect(result.palladium_ml).toBe(120);
    });

    it('handles warm tone preset (30% platinum)', () => {
      const result = calculateChemistry({
        width: 8,
        height: 10,
        metalRatio: 0.3,
        contrastAgent: 'none',
      });

      expect(result.platinum_ml).toBe(12);
      expect(result.palladium_ml).toBe(28);
    });

    it('handles cool tone preset (70% platinum)', () => {
      const result = calculateChemistry({
        width: 8,
        height: 10,
        metalRatio: 0.7,
        contrastAgent: 'none',
      });

      expect(result.platinum_ml).toBe(28);
      expect(result.palladium_ml).toBe(12);
    });
  });

  describe('round utility', () => {
    it('rounds to specified precision', () => {
      expect(round(1.234, 1)).toBe(1.2);
      expect(round(1.256, 1)).toBe(1.3);
      expect(round(1.234, 2)).toBe(1.23);
      expect(round(1.235, 2)).toBe(1.24);
    });

    it('handles zero precision', () => {
      expect(round(1.5, 0)).toBe(2);
      expect(round(1.4, 0)).toBe(1);
    });

    it('handles negative numbers', () => {
      expect(round(-1.25, 1)).toBe(-1.2);
      expect(round(-1.35, 1)).toBe(-1.3);
    });
  });
});

describe('Step Tablet Calculations', () => {
  const calculateDensity = (step: number, totalSteps: number, dmax: number): number => {
    return (step / (totalSteps - 1)) * dmax;
  };

  const calculateLabL = (density: number): number => {
    // Simplified conversion: higher density = lower L*
    return Math.max(0, 100 - density * 45);
  };

  it('calculates linear density progression', () => {
    const dmax = 2.1;
    const steps = 21;

    const step0 = calculateDensity(0, steps, dmax);
    const step10 = calculateDensity(10, steps, dmax);
    const step20 = calculateDensity(20, steps, dmax);

    expect(step0).toBe(0);
    expect(step10).toBeCloseTo(1.05, 2);
    expect(step20).toBeCloseTo(2.1, 2);
  });

  it('converts density to Lab L* value', () => {
    expect(calculateLabL(0)).toBe(100); // Paper white
    expect(calculateLabL(2.1)).toBeCloseTo(5.5, 1); // Dmax
  });

  it('handles 11-step tablet', () => {
    const dmax = 2.0;
    const steps = 11;

    const values = Array.from({ length: steps }, (_, i) => calculateDensity(i, steps, dmax));

    expect(values[0]).toBe(0);
    expect(values[5]).toBeCloseTo(1.0, 2);
    expect(values[10]).toBe(2.0);
  });
});
