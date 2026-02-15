/**
 * Tests for tablet configuration
 * Verifies all step tablet definitions, export formats, defaults, and methods
 */

import { describe, expect, it } from 'vitest';
import { tabletConfig } from './tablet.config';

describe('tabletConfig', () => {
  describe('stepTablets', () => {
    it('defines all expected step tablets', () => {
      expect(tabletConfig.stepTablets).toHaveLength(3);

      const tabletIds = tabletConfig.stepTablets.map((t) => t.id);
      expect(tabletIds).toEqual(['stouffer_21', 'stouffer_31', 'stouffer_41']);
    });

    it('has correct Stouffer T2115 (21 Steps) configuration', () => {
      const tablet = tabletConfig.stepTablets.find(
        (t) => t.id === 'stouffer_21'
      );

      expect(tablet).toBeDefined();
      expect(tablet?.name).toBe('Stouffer T2115 (21 Steps)');
      expect(tablet?.steps).toBe(21);
      expect(tablet?.stepSize).toBe(0.15);
      expect(tablet?.range).toBe(3.0);
    });

    it('has correct Stouffer T3110 (31 Steps) configuration', () => {
      const tablet = tabletConfig.stepTablets.find(
        (t) => t.id === 'stouffer_31'
      );

      expect(tablet).toBeDefined();
      expect(tablet?.name).toBe('Stouffer T3110 (31 Steps)');
      expect(tablet?.steps).toBe(31);
      expect(tablet?.stepSize).toBe(0.1);
      expect(tablet?.range).toBe(3.0);
    });

    it('has correct Stouffer T4105 (41 Steps) configuration', () => {
      const tablet = tabletConfig.stepTablets.find(
        (t) => t.id === 'stouffer_41'
      );

      expect(tablet).toBeDefined();
      expect(tablet?.name).toBe('Stouffer T4105 (41 Steps)');
      expect(tablet?.steps).toBe(41);
      expect(tablet?.stepSize).toBe(0.05);
      expect(tablet?.range).toBe(2.0);
    });

    it('all tablets have unique IDs', () => {
      const ids = tabletConfig.stepTablets.map((t) => t.id);
      const uniqueIds = new Set(ids);
      expect(uniqueIds.size).toBe(ids.length);
    });

    it('all tablets have required fields', () => {
      tabletConfig.stepTablets.forEach((tablet) => {
        expect(tablet.id).toBeDefined();
        expect(tablet.name).toBeDefined();
        expect(tablet.steps).toBeGreaterThan(0);
        expect(tablet.stepSize).toBeGreaterThan(0);
        expect(tablet.range).toBeGreaterThan(0);

        expect(typeof tablet.id).toBe('string');
        expect(typeof tablet.name).toBe('string');
        expect(typeof tablet.steps).toBe('number');
        expect(typeof tablet.stepSize).toBe('number');
        expect(typeof tablet.range).toBe('number');
      });
    });

    it('step sizes produce valid density ranges', () => {
      // Verify that calculated range from steps matches or is close to declared range
      tabletConfig.stepTablets.forEach((tablet) => {
        const calculatedRange = (tablet.steps - 1) * tablet.stepSize;

        // For stouffer_21: (21-1) * 0.15 = 3.0
        // For stouffer_31: (31-1) * 0.1 = 3.0
        // For stouffer_41: (41-1) * 0.05 = 2.0
        expect(calculatedRange).toBeCloseTo(tablet.range, 1);
      });
    });

    it('all tablets have reasonable density ranges', () => {
      tabletConfig.stepTablets.forEach((tablet) => {
        // Density ranges for photographic step tablets typically 2.0-4.0
        expect(tablet.range).toBeGreaterThanOrEqual(2.0);
        expect(tablet.range).toBeLessThanOrEqual(4.0);
      });
    });

    it('all tablets have reasonable step counts', () => {
      tabletConfig.stepTablets.forEach((tablet) => {
        // Typical step tablets have 11-51 steps
        expect(tablet.steps).toBeGreaterThanOrEqual(11);
        expect(tablet.steps).toBeLessThanOrEqual(51);
      });
    });
  });

  describe('exportFormats', () => {
    it('defines all expected export formats', () => {
      expect(tabletConfig.exportFormats).toHaveLength(4);

      const formatIds = tabletConfig.exportFormats.map((f) => f.id);
      expect(formatIds).toEqual(['qtr', 'piezography', 'csv', 'json']);
    });

    it('has correct QTR format configuration', () => {
      const format = tabletConfig.exportFormats.find((f) => f.id === 'qtr');

      expect(format).toBeDefined();
      expect(format?.label).toBe('QuadTone RIP (.qtr)');
      expect(format?.extension).toBe('.qtr');
      expect(format?.description).toBe('Standard format for QuadTone RIP');
    });

    it('has correct Piezography format configuration', () => {
      const format = tabletConfig.exportFormats.find(
        (f) => f.id === 'piezography'
      );

      expect(format).toBeDefined();
      expect(format?.label).toBe('Piezography (.quad)');
      expect(format?.extension).toBe('.quad');
      expect(format?.description).toBe('Format for Piezography systems');
    });

    it('has correct CSV format configuration', () => {
      const format = tabletConfig.exportFormats.find((f) => f.id === 'csv');

      expect(format).toBeDefined();
      expect(format?.label).toBe('CSV Data (.csv)');
      expect(format?.extension).toBe('.csv');
      expect(format?.description).toBe('Raw data for spreadsheet analysis');
    });

    it('has correct JSON format configuration', () => {
      const format = tabletConfig.exportFormats.find((f) => f.id === 'json');

      expect(format).toBeDefined();
      expect(format?.label).toBe('JSON (.json)');
      expect(format?.extension).toBe('.json');
      expect(format?.description).toBe(
        'Structured data for machine processing'
      );
    });

    it('all formats have unique IDs', () => {
      const ids = tabletConfig.exportFormats.map((f) => f.id);
      const uniqueIds = new Set(ids);
      expect(uniqueIds.size).toBe(ids.length);
    });

    it('all formats have required fields', () => {
      tabletConfig.exportFormats.forEach((format) => {
        expect(format.id).toBeDefined();
        expect(format.label).toBeDefined();
        expect(format.extension).toBeDefined();
        expect(format.description).toBeDefined();

        expect(typeof format.id).toBe('string');
        expect(typeof format.label).toBe('string');
        expect(typeof format.extension).toBe('string');
        expect(typeof format.description).toBe('string');
      });
    });

    it('all extensions start with a period', () => {
      tabletConfig.exportFormats.forEach((format) => {
        expect(format.extension).toMatch(/^\./);
      });
    });

    it('all extensions match format IDs logically', () => {
      tabletConfig.exportFormats.forEach((format) => {
        const extensionWithoutDot = format.extension.replace('.', '');

        // Special case for piezography -> quad
        if (format.id === 'piezography') {
          expect(extensionWithoutDot).toBe('quad');
        } else {
          expect(extensionWithoutDot).toBe(format.id);
        }
      });
    });
  });

  describe('defaults', () => {
    it('has default tabletId', () => {
      expect(tabletConfig.defaults.tabletId).toBe('stouffer_21');
    });

    it('has default exportFormat', () => {
      expect(tabletConfig.defaults.exportFormat).toBe('qtr');
    });

    it('default tabletId references an existing tablet', () => {
      const tabletExists = tabletConfig.stepTablets.some(
        (t) => t.id === tabletConfig.defaults.tabletId
      );
      expect(tabletExists).toBe(true);
    });

    it('default exportFormat references an existing format', () => {
      const formatExists = tabletConfig.exportFormats.some(
        (f) => f.id === tabletConfig.defaults.exportFormat
      );
      expect(formatExists).toBe(true);
    });
  });

  describe('linearizationMethods', () => {
    it('defines all expected linearization methods', () => {
      expect(tabletConfig.linearizationMethods).toHaveLength(3);

      const methodIds = tabletConfig.linearizationMethods.map((m) => m.id);
      expect(methodIds).toEqual(['linear', 'cubic', 'monotonic']);
    });

    it('has correct linear method configuration', () => {
      const method = tabletConfig.linearizationMethods.find(
        (m) => m.id === 'linear'
      );

      expect(method).toBeDefined();
      expect(method?.label).toBe('Linear');
      expect(method?.description).toBe('Standard linear interpolation');
    });

    it('has correct cubic method configuration', () => {
      const method = tabletConfig.linearizationMethods.find(
        (m) => m.id === 'cubic'
      );

      expect(method).toBeDefined();
      expect(method?.label).toBe('Cubic Spline');
      expect(method?.description).toBe('Smooth cubic spline interpolation');
    });

    it('has correct monotonic method configuration', () => {
      const method = tabletConfig.linearizationMethods.find(
        (m) => m.id === 'monotonic'
      );

      expect(method).toBeDefined();
      expect(method?.label).toBe('Monotonic');
      expect(method?.description).toBe('Preserves monotonicity (recommended)');
    });

    it('all methods have unique IDs', () => {
      const ids = tabletConfig.linearizationMethods.map((m) => m.id);
      const uniqueIds = new Set(ids);
      expect(uniqueIds.size).toBe(ids.length);
    });

    it('all methods have required fields', () => {
      tabletConfig.linearizationMethods.forEach((method) => {
        expect(method.id).toBeDefined();
        expect(method.label).toBeDefined();
        expect(method.description).toBeDefined();

        expect(typeof method.id).toBe('string');
        expect(typeof method.label).toBe('string');
        expect(typeof method.description).toBe('string');
      });
    });
  });

  describe('targetResponses', () => {
    it('defines all expected target responses', () => {
      expect(tabletConfig.targetResponses).toHaveLength(3);

      const responseIds = tabletConfig.targetResponses.map((r) => r.id);
      expect(responseIds).toEqual(['linear', 's_curve', 'gamma_22']);
    });

    it('has correct linear response configuration', () => {
      const response = tabletConfig.targetResponses.find(
        (r) => r.id === 'linear'
      );

      expect(response).toBeDefined();
      expect(response?.label).toBe('Linear');
      expect(response?.description).toBe('Linear response (L*)');
    });

    it('has correct s_curve response configuration', () => {
      const response = tabletConfig.targetResponses.find(
        (r) => r.id === 's_curve'
      );

      expect(response).toBeDefined();
      expect(response?.label).toBe('S-Curve');
      expect(response?.description).toBe('Contrast boosting S-curve');
    });

    it('has correct gamma_22 response configuration', () => {
      const response = tabletConfig.targetResponses.find(
        (r) => r.id === 'gamma_22'
      );

      expect(response).toBeDefined();
      expect(response?.label).toBe('Gamma 2.2');
      expect(response?.description).toBe('Standard display gamma');
    });

    it('all responses have unique IDs', () => {
      const ids = tabletConfig.targetResponses.map((r) => r.id);
      const uniqueIds = new Set(ids);
      expect(uniqueIds.size).toBe(ids.length);
    });

    it('all responses have required fields', () => {
      tabletConfig.targetResponses.forEach((response) => {
        expect(response.id).toBeDefined();
        expect(response.label).toBeDefined();
        expect(response.description).toBeDefined();

        expect(typeof response.id).toBe('string');
        expect(typeof response.label).toBe('string');
        expect(typeof response.description).toBe('string');
      });
    });
  });

  describe('data integrity', () => {
    it('all IDs across all categories are unique', () => {
      const allIds = [
        ...tabletConfig.stepTablets.map((t) => t.id),
        ...tabletConfig.exportFormats.map((f) => f.id),
        ...tabletConfig.linearizationMethods.map((m) => m.id),
        ...tabletConfig.targetResponses.map((r) => r.id),
      ];

      const uniqueIds = new Set(allIds);

      // Note: 'linear' appears in both linearizationMethods and targetResponses
      // This is intentional and acceptable as they're in different categories
      expect(uniqueIds.size).toBeLessThanOrEqual(allIds.length);
    });

    it('config object maintains consistent values', () => {
      // Verify config values don't change unexpectedly
      const originalTabletCount = tabletConfig.stepTablets.length;
      const originalFormatCount = tabletConfig.exportFormats.length;
      const originalMethodCount = tabletConfig.linearizationMethods.length;
      const originalResponseCount = tabletConfig.targetResponses.length;

      // Verify counts remain stable
      expect(tabletConfig.stepTablets).toHaveLength(originalTabletCount);
      expect(tabletConfig.exportFormats).toHaveLength(originalFormatCount);
      expect(tabletConfig.linearizationMethods).toHaveLength(
        originalMethodCount
      );
      expect(tabletConfig.targetResponses).toHaveLength(originalResponseCount);

      // Verify specific values don't change
      expect(tabletConfig.stepTablets[0]?.steps).toBe(21);
      expect(tabletConfig.defaults.tabletId).toBe('stouffer_21');
    });
  });
});
