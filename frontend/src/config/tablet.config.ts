/**
 * Tablet and calibration configuration
 */

export const tabletConfig = {
  // Step tablet options
  stepTablets: [
    {
      id: 'stouffer_21',
      name: 'Stouffer T2115 (21 Steps)',
      steps: 21,
      stepSize: 0.15,
      range: 3.0,
    },
    {
      id: 'stouffer_31',
      name: 'Stouffer T3110 (31 Steps)',
      steps: 31,
      stepSize: 0.1,
      range: 3.0,
    },
    {
      id: 'stouffer_41',
      name: 'Stouffer T4105 (41 Steps)',
      steps: 41,
      stepSize: 0.05,
      range: 2.0,
    },
  ],

  // Export formats matching backend ExportFormat enum
  exportFormats: [
    {
      id: 'qtr',
      label: 'QuadTone RIP (.qtr)',
      extension: '.qtr',
      description: 'Standard format for QuadTone RIP',
    },
    {
      id: 'piezography',
      label: 'Piezography (.quad)',
      extension: '.quad',
      description: 'Format for Piezography systems',
    },
    {
      id: 'csv',
      label: 'CSV Data (.csv)',
      extension: '.csv',
      description: 'Raw data for spreadsheet analysis',
    },
    {
      id: 'json',
      label: 'JSON (.json)',
      extension: '.json',
      description: 'Structured data for machine processing',
    },
  ],

  defaults: {
    tabletId: 'stouffer_21',
    exportFormat: 'qtr',
  },

  linearizationMethods: [
    {
      id: 'linear',
      label: 'Linear',
      description: 'Standard linear interpolation',
    },
    {
      id: 'cubic',
      label: 'Cubic Spline',
      description: 'Smooth cubic spline interpolation',
    },
    {
      id: 'monotonic',
      label: 'Monotonic',
      description: 'Preserves monotonicity (recommended)',
    },
  ],

  targetResponses: [
    {
      id: 'linear',
      label: 'Linear',
      description: 'Linear response (L*)',
    },
    {
      id: 's_curve',
      label: 'S-Curve',
      description: 'Contrast boosting S-curve',
    },
    {
      id: 'gamma_22',
      label: 'Gamma 2.2',
      description: 'Standard display gamma',
    },
  ],
};
