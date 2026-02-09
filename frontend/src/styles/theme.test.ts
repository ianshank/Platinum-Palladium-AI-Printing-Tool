import { describe, expect, it } from 'vitest';
import { theme } from './theme';

describe('theme', () => {
  describe('Colors', () => {
    it('has complete background color palette', () => {
      const bg = theme.colors.background;
      expect(typeof bg.primary).toBe('string');
      expect(typeof bg.secondary).toBe('string');
      expect(typeof bg.tertiary).toBe('string');
      expect(typeof bg.hover).toBe('string');
    });

    it('has complete text color palette', () => {
      const text = theme.colors.text;
      expect(typeof text.primary).toBe('string');
      expect(typeof text.secondary).toBe('string');
      expect(typeof text.inverse).toBe('string');
      expect(typeof text.error).toBe('string');
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
      const semantic = theme.colors.semantic;
      expect(typeof semantic.success).toBe('string');
      expect(typeof semantic.warning).toBe('string');
      expect(typeof semantic.error).toBe('string');
      expect(typeof semantic.info).toBe('string');
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
