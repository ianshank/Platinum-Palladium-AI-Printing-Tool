/**
 * Step tablet configuration.
 * Values are synchronized with backend TabletType and WedgeAnalysisSettings.
 */

export const tabletConfig = {
  // Tablet types
  types: [
    {
      id: 'stouffer_21',
      label: 'Stouffer 21-Step',
      steps: 21,
      densityIncrement: 0.15,
    },
    {
      id: 'stouffer_31',
      label: 'Stouffer 31-Step',
      steps: 31,
      densityIncrement: 0.1,
    },
    {
      id: 'stouffer_41',
      label: 'Stouffer 41-Step',
      steps: 41,
      densityIncrement: 0.075,
    },
    {
      id: 'custom',
      label: 'Custom',
      steps: 0,
      densityIncrement: 0,
    },
  ] as const,

  // Quality thresholds (from backend WedgeAnalysisSettings)
  quality: {
    minDensityRange: 1.5,
    maxDmin: 0.15,
    minDmax: 1.8,
    uniformityThreshold: 0.7,
    maxReversalTolerance: 0.02,
  },

  // Linearization methods
  linearizationMethods: [
    {
      id: 'spline_fit',
      label: 'Smooth spline (recommended)',
      description:
        'Creates smooth, continuous curves using cubic spline interpolation.',
    },
    {
      id: 'polynomial_fit',
      label: 'Polynomial fit',
      description:
        'Fits a polynomial curve to the data. Good for simple responses.',
    },
    {
      id: 'iterative',
      label: 'Iterative refinement',
      description:
        'Refines curve iteratively. Best for difficult linearizations.',
    },
    {
      id: 'hybrid',
      label: 'Hybrid (spline + iterative)',
      description:
        'Combines spline fitting with iterative refinement for optimal results.',
    },
    {
      id: 'direct_inversion',
      label: 'Direct inversion (simple)',
      description: 'Simple mathematical inversion. Fast but less smooth.',
    },
  ] as const,

  // Target response curves
  targetResponses: [
    {
      id: 'linear',
      label: 'Linear (even tonal steps)',
      description: 'Equal density steps from highlight to shadow.',
    },
    {
      id: 'gamma_22',
      label: 'Gamma 2.2 (sRGB match)',
      description:
        'Matches sRGB display gamma for print/screen consistency.',
    },
    {
      id: 'gamma_18',
      label: 'Gamma 1.8 (Mac display)',
      description: 'Traditional Mac display gamma.',
    },
    {
      id: 'paper_white',
      label: 'Paper white preserve',
      description: 'Holds paper white in highlights for maximum D-range.',
    },
    {
      id: 'perceptual',
      label: 'Perceptually uniform (L* curve)',
      description: 'Based on CIE L* for perceptually equal steps.',
    },
  ] as const,

  // Export formats
  exportFormats: [
    {
      id: 'qtr',
      label: 'QuadTone RIP (.quad)',
      extension: '.quad',
      mimeType: 'text/plain',
    },
    {
      id: 'piezography',
      label: 'Piezography (.ppt)',
      extension: '.ppt',
      mimeType: 'text/plain',
    },
    {
      id: 'csv',
      label: 'CSV (.csv)',
      extension: '.csv',
      mimeType: 'text/csv',
    },
    {
      id: 'json',
      label: 'JSON (.json)',
      extension: '.json',
      mimeType: 'application/json',
    },
  ] as const,

  // Curve adjustment types
  adjustmentTypes: [
    { id: 'brightness', label: 'Brightness', range: [-0.5, 0.5], default: 0 },
    { id: 'contrast', label: 'Contrast', range: [-0.5, 0.5], default: 0 },
    { id: 'gamma', label: 'Gamma', range: [0.5, 2.5], default: 1.0 },
    { id: 'highlights', label: 'Highlights', range: [-0.5, 0.5], default: 0 },
    { id: 'shadows', label: 'Shadows', range: [-0.5, 0.5], default: 0 },
    { id: 'midtones', label: 'Midtones', range: [-0.5, 0.5], default: 0 },
  ] as const,
} as const;

export type TabletTypeId = (typeof tabletConfig.types)[number]['id'];
export type LinearizationMethodId =
  (typeof tabletConfig.linearizationMethods)[number]['id'];
export type TargetResponseId =
  (typeof tabletConfig.targetResponses)[number]['id'];
export type ExportFormatId = (typeof tabletConfig.exportFormats)[number]['id'];
export type AdjustmentTypeId =
  (typeof tabletConfig.adjustmentTypes)[number]['id'];
