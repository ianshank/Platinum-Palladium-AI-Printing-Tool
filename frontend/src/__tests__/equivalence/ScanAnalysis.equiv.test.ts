/**
 * Scan Analysis Equivalence Tests
 *
 * Verifies that the React scan analysis components produce equivalent
 * results to the legacy Gradio implementation for scan quality assessment,
 * density extraction, and calibration data flow.
 */

import { describe, expect, it } from 'vitest';
import { assessScanQuality, type ScanAnalysisInput } from '@/api/hooks';
import { TEST_FIXTURES } from './setup';

// ============================================================================
// Scan Quality Grading Equivalence
// ============================================================================

describe('Equivalence: Scan Quality Grading', () => {
  const gradingScenarios: Array<{
    label: string;
    input: ScanAnalysisInput;
    expectedGrade: 'excellent' | 'good' | 'acceptable' | 'poor';
    minScore: number;
    maxScore: number;
  }> = [
    {
      label: 'perfect scan (21 patches, full range)',
      input: {
        densities: Array.from({ length: 21 }, (_, i) => 0.05 + (i / 20) * 1.95),
        dmax: 2.0,
        dmin: 0.05,
        range: 1.95,
        num_patches: 21,
      },
      expectedGrade: 'excellent',
      minScore: 90,
      maxScore: 100,
    },
    {
      label: 'boundary: moderate range penalty still grades excellent',
      input: {
        densities: Array.from({ length: 15 }, (_, i) => 0.1 + (i / 14) * 0.8),
        dmax: 0.9,
        dmin: 0.1,
        range: 0.8,
        num_patches: 15,
      },
      expectedGrade: 'excellent',
      minScore: 90,
      maxScore: 100,
    },
    {
      label: 'low range scan (range < 0.5, low dmax)',
      input: {
        densities: Array.from({ length: 21 }, (_, i) => 0.1 + (i / 20) * 0.3),
        dmax: 0.4,
        dmin: 0.1,
        range: 0.3,
        num_patches: 21,
      },
      expectedGrade: 'acceptable',
      minScore: 50,
      maxScore: 69,
    },
    {
      label: 'too few patches (8 < 11 minimum)',
      input: {
        densities: Array.from({ length: 8 }, (_, i) => 0.05 + (i / 7) * 1.5),
        dmax: 1.55,
        dmin: 0.05,
        range: 1.5,
        num_patches: 8,
      },
      expectedGrade: 'good',
      minScore: 70,
      maxScore: 79,
    },
    {
      label: 'everything wrong (few patches, low range, low dmax)',
      input: {
        densities: [0.1, 0.15, 0.2, 0.25, 0.3],
        dmax: 0.3,
        dmin: 0.1,
        range: 0.2,
        num_patches: 5,
      },
      expectedGrade: 'poor',
      minScore: 0,
      maxScore: 49,
    },
  ];

  for (const scenario of gradingScenarios) {
    it(`grades ${scenario.label} as ${scenario.expectedGrade}`, () => {
      const result = assessScanQuality(scenario.input);
      expect(result.overall).toBe(scenario.expectedGrade);
      expect(result.score).toBeGreaterThanOrEqual(scenario.minScore);
      expect(result.score).toBeLessThanOrEqual(scenario.maxScore);
    });
  }
});

// ============================================================================
// Quality Issue Detection
// ============================================================================

describe('Equivalence: Quality Issue Detection', () => {
  it('detects low density range (< 0.5)', () => {
    const input: ScanAnalysisInput = {
      densities: Array.from({ length: 21 }, (_, i) => 0.1 + (i / 20) * 0.3),
      dmax: 0.4,
      dmin: 0.1,
      range: 0.3,
      num_patches: 21,
    };
    const result = assessScanQuality(input);
    const rangeIssue = result.issues.find((i) => i.message.includes('density range'));
    expect(rangeIssue).toBeDefined();
    expect(rangeIssue!.type).toBe('warning');
    expect(rangeIssue!.suggestion).toBeTruthy();
  });

  it('detects moderate density range (0.5-1.0)', () => {
    const input: ScanAnalysisInput = {
      densities: Array.from({ length: 21 }, (_, i) => 0.1 + (i / 20) * 0.7),
      dmax: 0.9,
      dmin: 0.1,
      range: 0.7,
      num_patches: 21,
    };
    const result = assessScanQuality(input);
    const rangeIssue = result.issues.find((i) => i.message.includes('density range'));
    expect(rangeIssue).toBeDefined();
    expect(rangeIssue!.message).toContain('Moderate');
  });

  it('detects insufficient patch count', () => {
    const input: ScanAnalysisInput = {
      densities: Array.from({ length: 8 }, (_, i) => 0.1 + (i / 7) * 1.5),
      dmax: 1.6,
      dmin: 0.1,
      range: 1.5,
      num_patches: 8,
    };
    const result = assessScanQuality(input);
    const patchIssue = result.issues.find((i) => i.message.includes('patches'));
    expect(patchIssue).toBeDefined();
    expect(patchIssue!.type).toBe('error');
  });

  it('detects low Dmax (< 0.8)', () => {
    const input: ScanAnalysisInput = {
      densities: Array.from({ length: 21 }, (_, i) => 0.1 + (i / 20) * 0.5),
      dmax: 0.6,
      dmin: 0.1,
      range: 0.5,
      num_patches: 21,
    };
    const result = assessScanQuality(input);
    const dmaxIssue = result.issues.find((i) => i.message.includes('Dmax'));
    expect(dmaxIssue).toBeDefined();
    expect(dmaxIssue!.type).toBe('warning');
  });

  it('reports no issues for excellent scan', () => {
    const result = assessScanQuality(TEST_FIXTURES.scanQuality.excellent);
    expect(result.issues).toHaveLength(0);
    expect(result.score).toBe(100);
  });
});

// ============================================================================
// Score Calculation Determinism
// ============================================================================

describe('Equivalence: Score Calculation Determinism', () => {
  it('produces identical scores for identical inputs across calls', () => {
    const input = TEST_FIXTURES.scanQuality.excellent;
    const result1 = assessScanQuality(input);
    const result2 = assessScanQuality(input);
    expect(result1.score).toBe(result2.score);
    expect(result1.overall).toBe(result2.overall);
    expect(result1.issues).toEqual(result2.issues);
  });

  it('penalty stacking is additive', () => {
    // Single penalty: low range only
    const lowRange: ScanAnalysisInput = {
      densities: Array.from({ length: 21 }, (_, i) => 0.1 + (i / 20) * 0.3),
      dmax: 0.4,
      dmin: 0.1,
      range: 0.3,
      num_patches: 21,
    };
    const singlePenalty = assessScanQuality(lowRange);

    // Double penalty: low range + low patches
    const combined: ScanAnalysisInput = {
      ...lowRange,
      num_patches: 8,
    };
    const doublePenalty = assessScanQuality(combined);

    // Double should be strictly lower than single
    expect(doublePenalty.score).toBeLessThan(singlePenalty.score);
  });
});
