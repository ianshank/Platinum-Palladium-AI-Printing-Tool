/**
 * Darkroom theme definition.
 * Combines design tokens with component-specific styling.
 */

import {
  colors,
  spacing,
  typography,
  radii,
  shadows,
  transitions,
  zIndex,
  breakpoints,
} from './tokens';

export const darkroomTheme = {
  colors,
  spacing,
  typography,
  radii,
  shadows,
  transitions,
  zIndex,
  breakpoints,

  // Component-specific tokens
  components: {
    button: {
      primary: {
        bg: colors.accent.primary,
        bgHover: colors.accent.primaryHover,
        bgActive: colors.accent.primaryActive,
        text: colors.text.inverse,
        border: 'transparent',
      },
      secondary: {
        bg: colors.background.tertiary,
        bgHover: colors.background.hover,
        bgActive: colors.background.secondary,
        text: colors.text.secondary,
        border: colors.border.default,
      },
      ghost: {
        bg: 'transparent',
        bgHover: colors.background.hover,
        bgActive: colors.background.tertiary,
        text: colors.text.secondary,
        border: 'transparent',
      },
      danger: {
        bg: colors.semantic.error,
        bgHover: '#DC2626',
        bgActive: '#B91C1C',
        text: '#FFFFFF',
        border: 'transparent',
      },
    },

    input: {
      bg: colors.background.tertiary,
      bgFocus: colors.background.tertiary,
      border: colors.border.subtle,
      borderFocus: colors.accent.primary,
      borderError: colors.semantic.error,
      text: colors.text.primary,
      placeholder: colors.text.disabled,
      label: colors.text.secondary,
    },

    card: {
      bg: colors.background.secondary,
      bgHover: colors.background.hover,
      border: colors.border.default,
      shadow: shadows.md,
    },

    tabs: {
      bg: '#0A0A0A',
      border: colors.border.default,
      activeText: colors.accent.primary,
      activeBorder: colors.accent.primary,
      inactiveText: colors.text.secondary,
      hoverBg: colors.background.hover,
    },

    modal: {
      bg: colors.background.elevated,
      overlay: 'rgba(0, 0, 0, 0.75)',
      border: colors.border.default,
      shadow: shadows['2xl'],
    },

    tooltip: {
      bg: colors.background.elevated,
      text: colors.text.primary,
      border: colors.border.subtle,
      shadow: shadows.lg,
    },

    toast: {
      success: {
        bg: colors.semantic.successBg,
        border: colors.semantic.success,
        icon: colors.semantic.success,
      },
      error: {
        bg: colors.semantic.errorBg,
        border: colors.semantic.error,
        icon: colors.semantic.error,
      },
      warning: {
        bg: colors.semantic.warningBg,
        border: colors.semantic.warning,
        icon: colors.semantic.warning,
      },
      info: {
        bg: colors.semantic.infoBg,
        border: colors.semantic.info,
        icon: colors.semantic.info,
      },
    },

    sidebar: {
      bg: colors.background.primary,
      border: colors.border.default,
      itemHover: colors.background.hover,
      itemActive: colors.background.tertiary,
      itemActiveText: colors.accent.primary,
    },

    table: {
      headerBg: colors.background.tertiary,
      headerText: colors.text.secondary,
      rowBg: colors.background.secondary,
      rowHover: colors.background.hover,
      rowAlternate: colors.background.primary,
      border: colors.border.default,
    },

    slider: {
      track: colors.background.tertiary,
      trackFilled: colors.accent.primary,
      thumb: colors.accent.primary,
      thumbBorder: colors.background.primary,
    },

    chart: {
      bg: colors.background.secondary,
      grid: colors.chart.grid,
      axis: colors.chart.axis,
      tooltipBg: colors.background.elevated,
      colors: [
        colors.chart.curve1,
        colors.chart.curve2,
        colors.chart.curve3,
        colors.chart.curve4,
        colors.chart.curve5,
      ],
    },
  },
} as const;

export type Theme = typeof darkroomTheme;
export type ThemeColors = typeof colors;
export type ThemeSpacing = keyof typeof spacing;
