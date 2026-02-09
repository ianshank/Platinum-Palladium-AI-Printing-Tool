import { describe, expect, it } from 'vitest';
import { theme } from './theme';

describe('theme', () => {
  describe('Colors', () => {
    it('has complete background color palette', () => {
      expect(theme.colors.background).toEqual(
        expect.objectContaining({
          primary: expect.any(String),
          secondary: expect.any(String),
          tertiary: expect.any(String),
          hover: expect.any(String),
        })
      );
    });

    it('has complete text color palette', () => {
      expect(theme.colors.text).toEqual(
        expect.objectContaining({
          primary: expect.any(String),
          secondary: expect.any(String),
          inverse: expect.any(String),
          error: expect.any(String),
        })
      );
    });

    it('has border colors', () => {
      expect(theme.colors.border.default).toBeDefined();
      expect(theme.colors.border.focus).toBeDefined();
    });

    it('has accent colors', () => {
      expect(theme.colors.accent.primary).toBeDefined();
      expect(theme.colors.accent.primaryHover).toBeDefined();
      expect(theme.colors.accent.secondary).toBeDefined();
    });

    it('has semantic colors', () => {
      expect(theme.colors.semantic).toEqual(
        expect.objectContaining({
          success: expect.any(String),
          warning: expect.any(String),
          error: expect.any(String),
          info: expect.any(String),
        })
      );
    });
  });

  describe('Typography', () => {
    it('has font families', () => {
      expect(theme.typography.fontFamily.sans).toContain('Inter');
      expect(theme.typography.fontFamily.mono).toBeDefined();
    });

    it('has font sizes from xs to 2xl', () => {
      const sizes = theme.typography.fontSize;
      expect(sizes.xs).toBeDefined();
      expect(sizes.sm).toBeDefined();
      expect(sizes.base).toBeDefined();
      expect(sizes.lg).toBeDefined();
      expect(sizes.xl).toBeDefined();
      expect(sizes['2xl']).toBeDefined();
    });

    it('has font weights', () => {
      expect(theme.typography.fontWeight.regular).toBe(400);
      expect(theme.typography.fontWeight.medium).toBe(500);
      expect(theme.typography.fontWeight.semibold).toBe(600);
      expect(theme.typography.fontWeight.bold).toBe(700);
    });
  });

  describe('Spacing', () => {
    it('has spacing scale from 0 to 64', () => {
      expect(theme.spacing[0]).toBe('0px');
      expect(theme.spacing[4]).toBe('1rem');
      expect(theme.spacing[8]).toBe('2rem');
      expect(theme.spacing[64]).toBe('16rem');
    });
  });

  describe('Other tokens', () => {
    it('has border radius tokens', () => {
      expect(theme.radii.sm).toBeDefined();
      expect(theme.radii.md).toBeDefined();
      expect(theme.radii.lg).toBeDefined();
      expect(theme.radii.full).toBe('9999px');
    });

    it('has shadow tokens', () => {
      expect(theme.shadows.sm).toBeDefined();
      expect(theme.shadows.md).toBeDefined();
      expect(theme.shadows.lg).toBeDefined();
    });

    it('has transition tokens', () => {
      expect(theme.transitions.fast).toContain('150ms');
      expect(theme.transitions.normal).toContain('300ms');
      expect(theme.transitions.slow).toContain('500ms');
    });

    it('has breakpoint tokens', () => {
      expect(theme.breakpoints.tablet).toBe('640px');
      expect(theme.breakpoints.desktop).toBe('1024px');
      expect(theme.breakpoints.wide).toBe('1280px');
    });
  });
});
